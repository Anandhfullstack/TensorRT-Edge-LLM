/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! OpenAI-compatible transcription server for the Whisper Small pipeline.
//!
//! Engines and the tokenizer are deserialized once and shared; each pipeline
//! slot owns only its execution contexts, caches, stream and CUDA graph. The
//! decoder carries per-request cache state and a graph bound to fixed
//! addresses, so a slot serves exactly one request at a time. See
//! SERVER_HOST.md.

#include "whisper_audio_processor.h"
#include "whisper_chunking.h"
#include "whisper_decoder_runner.h"
#include "whisper_encoder_inference.h"

#include "common/logger.h"
#include "common/trtUtils.h"
#include "tokenizer/tokenizer.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

//! The LOG_* macros expand to an unqualified ``format::fmtstr``, which resolves
//! only inside namespace ``trt_edgellm``. This file's helpers and ``main`` sit at
//! global scope, so bring the namespace into reach.
namespace format = trt_edgellm::format;

namespace
{

using json = nlohmann::json;
using namespace trt_edgellm::examples::whisper;

//! Matches the documented OpenAI upload cap. The multipart body may exceed this
//! by its framing overhead, which is what the server's payload limit allows for.
constexpr std::size_t kMaxUploadBytes = 25ULL * 1024ULL * 1024ULL;

//! Longest clip accepted by default. Whisper Small sees one 30-second window at
//! a time, so anything longer is decoded as a sequence of overlapping windows
//! and stitched. The cap is what stops a single upload from occupying a slot
//! for minutes: at ~0.35 s per window and a 25 s stride, 600 s of audio is
//! 24 windows, roughly 8 s of work.
constexpr double kDefaultMaxAudioSeconds = 600.0;

//! Hard ceiling imposed by the shared audio loader (``kMaxDecodedSeconds`` in
//! cpp/runtime/audioLoader.cpp), which refuses to buffer a longer decode.
//! Duration is only knowable after decoding, so that limit fires first and the
//! server cannot accept more however it is configured.
constexpr double kLoaderMaxDecodedSeconds = 600.0;

//! Socket threads beyond the admission capacity. Keeps a shed request on a
//! worker long enough to answer 503 rather than stalling in the accept backlog.
constexpr int kSocketThreadHeadroom = 8;

struct ServerOptions
{
    std::string encoderEngine;
    std::string crossKvEngine;
    std::string decoderEngine;
    std::string tokenizerDir;

    std::string host{"127.0.0.1"};
    int port{8000};

    //! Concurrent pipeline instances. Each costs its own execution contexts and
    //! caches (~470 MiB); engine weights are shared. Default 1 so the server
    //! starts on the smallest supported device.
    int slots{1};

    //! Total requests allowed in the server at once: `slots` running, the rest
    //! waiting for a slot. Beyond this the server sheds load with 503.
    int queueDepth{16};

    int maxNewTokens{128};

    //! Longest accepted clip, in seconds. Beyond this the upload is refused
    //! with 413 rather than tying up a slot.
    double maxAudioSeconds{kDefaultMaxAudioSeconds};

    //! Overlap between consecutive windows, in seconds. Used only by the
    //! fixed-stride fallback; timestamp mode needs no overlap.
    double overlapSeconds{kDefaultOverlapSeconds};

    //! Long-audio strategy. Timestamp mode lets the model mark segment
    //! boundaries and seeks to the last one, which is Whisper's own long-form
    //! approach and needs no text de-duplication. The fixed-stride fallback is
    //! kept as an escape hatch for bisecting timestamp-decoding problems.
    bool timestampSeek{true};
};

void printUsage(
    char const* program)
{
    std::cerr
        << "Usage:\n"
        << "  " << program
        << " <encoder.engine> <cross_kv.engine> <decoder.engine> <tokenizer_dir>\n"
        << "      [--host 127.0.0.1] [--port 8000] [--slots 1]\n"
        << "      [--queue-depth 16] [--max-new-tokens 128]\n"
        << "      [--max-audio-seconds 600] [--overlap-seconds 5]\n"
        << "      [--long-audio-mode timestamps|overlap]\n";
}

bool parseArgs(
    int argc,
    char** argv,
    ServerOptions& options)
{
    if (argc < 5)
    {
        printUsage(argv[0]);

        return false;
    }

    options.encoderEngine = argv[1];
    options.crossKvEngine = argv[2];
    options.decoderEngine = argv[3];
    options.tokenizerDir = argv[4];

    for (int i = 5; i < argc; ++i)
    {
        std::string const flag = argv[i];

        auto next = [&](char const* name) -> char const*
        {
            if (i + 1 >= argc)
            {
                std::cerr << "Missing value for " << name << '\n';

                return nullptr;
            }

            return argv[++i];
        };

        if (flag == "--host")
        {
            char const* value = next("--host");

            if (value == nullptr)
            {
                return false;
            }

            options.host = value;
        }
        else if (flag == "--port")
        {
            char const* value = next("--port");

            if (value == nullptr)
            {
                return false;
            }

            options.port = std::atoi(value);
        }
        else if (flag == "--slots")
        {
            char const* value = next("--slots");

            if (value == nullptr)
            {
                return false;
            }

            options.slots = std::atoi(value);
        }
        else if (flag == "--queue-depth")
        {
            char const* value = next("--queue-depth");

            if (value == nullptr)
            {
                return false;
            }

            options.queueDepth = std::atoi(value);
        }
        else if (flag == "--max-new-tokens")
        {
            char const* value = next("--max-new-tokens");

            if (value == nullptr)
            {
                return false;
            }

            options.maxNewTokens = std::atoi(value);
        }
        else if (flag == "--max-audio-seconds")
        {
            char const* value = next("--max-audio-seconds");

            if (value == nullptr)
            {
                return false;
            }

            options.maxAudioSeconds = std::atof(value);
        }
        else if (flag == "--overlap-seconds")
        {
            char const* value = next("--overlap-seconds");

            if (value == nullptr)
            {
                return false;
            }

            options.overlapSeconds = std::atof(value);
        }
        else if (flag == "--long-audio-mode")
        {
            char const* value = next("--long-audio-mode");

            if (value == nullptr)
            {
                return false;
            }

            std::string const mode = value;

            if (mode == "timestamps")
            {
                options.timestampSeek = true;
            }
            else if (mode == "overlap")
            {
                options.timestampSeek = false;
            }
            else
            {
                std::cerr << "Unknown long-audio mode: " << mode
                          << " (use timestamps or overlap)\n";

                return false;
            }
        }
        else
        {
            std::cerr << "Unknown argument: " << flag << '\n';
            printUsage(argv[0]);

            return false;
        }
    }

    if (options.port <= 0 || options.port > 65535)
    {
        std::cerr << "Port out of range: " << options.port << '\n';

        return false;
    }

    if (options.slots < 1)
    {
        std::cerr << "Slots must be >= 1\n";

        return false;
    }

    if (options.queueDepth < options.slots)
    {
        std::cerr << "Queue depth must be >= slots\n";

        return false;
    }

    if (options.maxNewTokens < 1)
    {
        std::cerr << "Max new tokens must be >= 1\n";

        return false;
    }

    if (options.maxAudioSeconds < kWhisperChunkSeconds)
    {
        std::cerr << "Max audio seconds must be >= " << kWhisperChunkSeconds << '\n';

        return false;
    }

    if (options.maxAudioSeconds > kLoaderMaxDecodedSeconds)
    {
        std::cerr
            << "Max audio seconds must be <= "
            << kLoaderMaxDecodedSeconds
            << ": the audio loader refuses to decode anything longer\n";

        return false;
    }

    if (options.overlapSeconds < 0.0 || options.overlapSeconds >= kWhisperChunkSeconds)
    {
        std::cerr << "Overlap seconds must be in [0, " << kWhisperChunkSeconds << ")\n";

        return false;
    }

    return true;
}

// ============================================================
// Shared, read-only resources
// ============================================================

//! Engine weights and the tokenizer, deserialized once and shared by every
//! slot. A TensorRT ICudaEngine owns the weights; an IExecutionContext owns the
//! per-inference scratch, so N slots over one engine cost one copy of the
//! ~545 MiB of weights instead of N.
class SharedResources
{
public:
    bool load(
        ServerOptions const& options)
    {
        mRuntime.reset(
            nvinfer1::createInferRuntime(trt_edgellm::gLogger));

        if (!mRuntime)
        {
            LOG_ERROR("Failed to create TensorRT runtime");

            return false;
        }

        if (!loadEngine(options.encoderEngine, mEncoderEngine, "encoder")
            || !loadEngine(options.crossKvEngine, mCrossKvEngine, "cross-KV")
            || !loadEngine(options.decoderEngine, mDecoderEngine, "decoder"))
        {
            return false;
        }

        if (!mTokenizer.loadFromHF(options.tokenizerDir, false))
        {
            LOG_ERROR("Failed to load Whisper tokenizer from: %s", options.tokenizerDir.c_str());

            return false;
        }

        return true;
    }

    std::shared_ptr<nvinfer1::IRuntime> const& runtime() const noexcept
    {
        return mRuntime;
    }

    std::shared_ptr<nvinfer1::ICudaEngine> const& encoderEngine() const noexcept
    {
        return mEncoderEngine;
    }

    std::shared_ptr<nvinfer1::ICudaEngine> const& crossKvEngine() const noexcept
    {
        return mCrossKvEngine;
    }

    std::shared_ptr<nvinfer1::ICudaEngine> const& decoderEngine() const noexcept
    {
        return mDecoderEngine;
    }

    //! Tokenizer::decode is const, so all slots may share one instance.
    trt_edgellm::tokenizer::Tokenizer const& tokenizer() const noexcept
    {
        return mTokenizer;
    }

private:
    bool loadEngine(
        std::string const& path,
        std::shared_ptr<nvinfer1::ICudaEngine>& target,
        char const* label)
    {
        target = trt_edgellm::deserializeCudaEngineFromFile(*mRuntime, path);

        if (!target)
        {
            LOG_ERROR("Failed to load %s engine: %s", label, path.c_str());

            return false;
        }

        return true;
    }

private:
    // Declared before the engines so the engines are destroyed first.
    std::shared_ptr<nvinfer1::IRuntime> mRuntime;

    std::shared_ptr<nvinfer1::ICudaEngine> mEncoderEngine;
    std::shared_ptr<nvinfer1::ICudaEngine> mCrossKvEngine;
    std::shared_ptr<nvinfer1::ICudaEngine> mDecoderEngine;

    trt_edgellm::tokenizer::Tokenizer mTokenizer;
};

// ============================================================
// Forced decoder prompt (language / task)
// ============================================================

//! Builds Whisper's 4-token forced prefix from the request's language and task.
//!
//! Token ids come from the tokenizer's special-token map rather than a
//! hard-coded table, so the set of accepted languages is exactly the set the
//! loaded checkpoint supports.
class PromptBuilder
{
public:
    bool initialize(
        trt_edgellm::tokenizer::Tokenizer const& tokenizer)
    {
        mSpecialTokens = &tokenizer.getSpecialTokensEncoder();

        return lookup("<|startoftranscript|>", mStartToken)
            && lookup("<|notimestamps|>", mNoTimestampsToken);
    }

    //! \return false when the language or task is not in the checkpoint's
    //!         vocabulary; the caller turns that into a 400.
    //! \param withTimestamps Omit ``<|notimestamps|>`` so the model is free to
    //!        emit timestamp tokens.
    bool build(
        std::string const& language,
        std::string const& task,
        bool const withTimestamps,
        std::vector<int64_t>& prompt) const
    {
        int64_t languageToken = 0;
        int64_t taskToken = 0;

        if (!lookup("<|" + language + "|>", languageToken)
            || !lookup("<|" + task + "|>", taskToken))
        {
            return false;
        }

        prompt = {mStartToken, languageToken, taskToken};

        if (!withTimestamps)
        {
            prompt.push_back(mNoTimestampsToken);
        }

        return true;
    }

private:
    bool lookup(
        std::string const& token,
        int64_t& id) const
    {
        auto const found = mSpecialTokens->find(token);

        if (found == mSpecialTokens->end())
        {
            return false;
        }

        id = static_cast<int64_t>(found->second);

        return true;
    }

private:
    trt_edgellm::tokenizer::TokenToRanks const* mSpecialTokens{nullptr};

    int64_t mStartToken{0};
    int64_t mNoTimestampsToken{0};
};

// ============================================================
// One pipeline slot
// ============================================================

//! Mel front-end, encoder, cross-KV projection and decoder for one slot. Not
//! thread-safe: the decoder holds per-request cache state and a CUDA graph
//! bound to this instance's buffers, so the pool hands out one slot per request.
class WhisperPipeline
{
public:
    WhisperPipeline(
        SharedResources const& shared,
        ServerOptions const& options)
        : mEncoder(shared.runtime(), shared.encoderEngine())
        , mDecoder(shared.runtime(), shared.crossKvEngine(), shared.decoderEngine())
        , mTokenizer(shared.tokenizer())
        , mMaxNewTokens(options.maxNewTokens)
        , mOverlapSamples(static_cast<std::size_t>(
              options.overlapSeconds * kWhisperSampleRate))
        , mMaxOverlapWords(maxOverlapWordsFor(options.overlapSeconds))
        , mTimestampSeek(options.timestampSeek)
    {
    }

    bool initialize()
    {
        if (!mEncoder.initialize())
        {
            LOG_ERROR("Failed to create encoder execution context");

            return false;
        }

        if (!mDecoder.initialize())
        {
            LOG_ERROR("Failed to create decoder execution contexts");

            return false;
        }

        return true;
    }

    //! Drive one utterance end to end. `pcm` is padded or trimmed in place.
    //! `prompt` is the 4-token forced prefix; nullptr uses the built-in English
    //! transcribe prompt.
    bool transcribe(
        trt_edgellm::rt::audio::AudioPCM& pcm,
        std::vector<int64_t> const* prompt,
        std::string& text,
        bool const emitTimestamps = false,
        std::vector<int64_t>* rawTokens = nullptr)
    {
        mFeatures.clear();

        if (!mProcessor.processPcm(pcm, mFeatures))
        {
            return false;
        }

        mEncoderOutput.clear();

        if (!mEncoder.run(mFeatures, mEncoderOutput))
        {
            return false;
        }

        mTokens.clear();

        if (!mDecoder.generate(mEncoderOutput, mTokens, mMaxNewTokens, prompt, emitTimestamps))
        {
            return false;
        }

        if (rawTokens != nullptr)
        {
            *rawTokens = mTokens;
        }

        // Tokenizer expects int32_t Rank.
        std::vector<trt_edgellm::tokenizer::Rank> decodeTokens;

        decodeTokens.reserve(mTokens.size());

        for (int64_t token : mTokens)
        {
            decodeTokens.push_back(
                static_cast<trt_edgellm::tokenizer::Rank>(token));
        }

        text = mTokenizer.decode(decodeTokens, true);

        return true;
    }

    //! Transcribe a clip of any length.
    //!
    //! Audio that fits one window takes the single-window path unchanged, so
    //! short-clip output stays bit-for-bit what it was. Longer audio is decoded
    //! as overlapping windows and stitched; windows run sequentially on this
    //! slot, so one long request occupies one slot for the whole of it.
    //!
    //! \param windowCount Windows actually decoded, for logging. May be null.
    bool transcribeAny(
        trt_edgellm::rt::audio::AudioPCM& pcm,
        std::vector<int64_t> const* shortPrompt,
        std::vector<int64_t> const* timestampPrompt,
        std::string& text,
        std::size_t* windowCount = nullptr)
    {
        if (pcm.samples.size() <= kWhisperPcmSamples)
        {
            if (windowCount != nullptr)
            {
                *windowCount = 1;
            }

            return transcribe(pcm, shortPrompt, text);
        }

        return mTimestampSeek
            ? transcribeByTimestampSeek(pcm, timestampPrompt, text, windowCount)
            : transcribeByFixedStride(pcm, shortPrompt, text, windowCount);
    }

private:
    //! Whisper's own long-form strategy: decode a window, seek to the last
    //! timestamp the model emitted, repeat. Windows do not overlap, so there is
    //! nothing to de-duplicate and genuinely repeated speech survives.
    bool transcribeByTimestampSeek(
        trt_edgellm::rt::audio::AudioPCM& pcm,
        std::vector<int64_t> const* prompt,
        std::string& text,
        std::size_t* windowCount)
    {
        std::size_t const total = pcm.samples.size();
        std::size_t seek = 0;
        std::size_t windows = 0;

        std::string joined;

        while (seek < total)
        {
            std::size_t const count = std::min(kWhisperPcmSamples, total - seek);

            mWindowPcm.sampleRate = pcm.sampleRate;
            mWindowPcm.numChannels = pcm.numChannels;
            mWindowPcm.samples.assign(
                pcm.samples.begin() + static_cast<std::ptrdiff_t>(seek),
                pcm.samples.begin() + static_cast<std::ptrdiff_t>(seek + count));

            std::string windowText;
            std::vector<int64_t> windowTokens;

            if (!transcribe(mWindowPcm, prompt, windowText, true, &windowTokens))
            {
                return false;
            }

            ++windows;

            bool const isFinalWindow = count < kWhisperPcmSamples;

            WindowSeek const seekPlan = planWindowSeek(windowTokens,
                kWhisperTimestampBeginToken, kWhisperTimestampPrecision,
                static_cast<double>(kWhisperChunkSeconds));

            // A cut-off tail is dropped here and re-decoded by the next window;
            // emitting it as well is what duplicates text. The final window has
            // no successor, so nothing may be dropped from it.
            std::string const settled = isFinalWindow
                ? stripEndOfTextMarker(windowText)
                : stripEndOfTextMarker(decodePrefix(windowTokens, seekPlan.keepTokens));

            if (!settled.empty())
            {
                if (!joined.empty() && settled.front() != ' ')
                {
                    joined.push_back(' ');
                }

                joined.append(settled);
            }

            if (isFinalWindow)
            {
                break;
            }

            auto step = static_cast<std::size_t>(seekPlan.advanceSeconds * kWhisperSampleRate);

            // planWindowSeek never returns 0, but the seek must stay monotonic
            // even if that ever changes.
            if (step == 0)
            {
                step = kWhisperPcmSamples;
            }

            seek += step;
        }

        if (windowCount != nullptr)
        {
            *windowCount = windows;
        }

        text = joined;

        return true;
    }

    //! Fixed-stride windows joined by text-overlap matching. Superseded by
    //! timestamp seeking; kept so timestamp decoding can be bisected against a
    //! known-good path.
    bool transcribeByFixedStride(
        trt_edgellm::rt::audio::AudioPCM& pcm,
        std::vector<int64_t> const* prompt,
        std::string& text,
        std::size_t* windowCount)
    {
        std::vector<AudioWindow> const windows
            = planWindows(pcm.samples.size(), kWhisperPcmSamples, mOverlapSamples);

        if (windowCount != nullptr)
        {
            *windowCount = windows.size();
        }

        std::string stitched;

        for (AudioWindow const& window : windows)
        {
            mWindowPcm.sampleRate = pcm.sampleRate;
            mWindowPcm.numChannels = pcm.numChannels;
            mWindowPcm.samples.assign(
                pcm.samples.begin() + static_cast<std::ptrdiff_t>(window.startSample),
                pcm.samples.begin()
                    + static_cast<std::ptrdiff_t>(window.startSample + window.sampleCount));

            std::string windowText;

            if (!transcribe(mWindowPcm, prompt, windowText))
            {
                return false;
            }

            appendWindowText(stitched, windowText, mMaxOverlapWords);
        }

        text = stitched;

        return true;
    }

public:

    //! Decode the first \p count token ids to text. Whisper special tokens,
    //! timestamps included, are dropped by the tokenizer.
    std::string decodePrefix(
        std::vector<int64_t> const& tokens,
        std::size_t const count) const
    {
        std::vector<trt_edgellm::tokenizer::Rank> ranks;

        ranks.reserve(count);

        for (std::size_t index = 0; index < count && index < tokens.size(); ++index)
        {
            ranks.push_back(static_cast<trt_edgellm::tokenizer::Rank>(tokens[index]));
        }

        return mTokenizer.decode(ranks, true);
    }

    //! Exercise every stage once so the first real request does not pay lazy
    //! TensorRT setup. Silence decodes to a handful of tokens.
    bool warmUp()
    {
        trt_edgellm::rt::audio::AudioPCM silence;

        silence.sampleRate = kWhisperSampleRate;
        silence.numChannels = 1;
        silence.samples.assign(kWhisperPcmSamples, 0.0f);

        std::string text;

        return transcribe(silence, nullptr, text);  // short path only
    }

private:
    WhisperAudioProcessor mProcessor;
    WhisperEncoderInference mEncoder;
    WhisperDecoderRunner mDecoder;
    trt_edgellm::tokenizer::Tokenizer const& mTokenizer;

    // Reused across requests so steady-state serving does not reallocate.
    std::vector<__half> mFeatures;
    std::vector<__half> mEncoderOutput;
    std::vector<int64_t> mTokens;
    trt_edgellm::rt::audio::AudioPCM mWindowPcm;

    int mMaxNewTokens;
    std::size_t mOverlapSamples;
    std::size_t mMaxOverlapWords;
    bool mTimestampSeek;
};

// ============================================================
// Slot pool
// ============================================================

//! Fixed set of pipeline slots with a blocking acquire. Admission (below) caps
//! how many requests may wait here, so this queue is bounded by construction.
class PipelinePool
{
public:
    bool initialize(
        SharedResources const& shared,
        ServerOptions const& options)
    {
        // Slots are built one at a time on purpose: each one captures its own
        // decode CUDA graph, and serialising capture keeps concurrent
        // stream-capture out of the startup path entirely.
        for (int i = 0; i < options.slots; ++i)
        {
            auto slot = std::make_unique<WhisperPipeline>(shared, options);

            if (!slot->initialize() || !slot->warmUp())
            {
                LOG_ERROR("Failed to initialize pipeline slot %d", i);

                return false;
            }

            LOG_INFO("Pipeline slot %d ready", i);

            mSlots.push_back(std::move(slot));
            mFree.push_back(i);
        }

        return true;
    }

    int acquire()
    {
        std::unique_lock<std::mutex> lock(mMutex);

        mAvailable.wait(lock, [this] { return !mFree.empty(); });

        int const index = mFree.back();

        mFree.pop_back();

        return index;
    }

    void release(
        int index)
    {
        {
            std::lock_guard<std::mutex> const lock(mMutex);

            mFree.push_back(index);
        }

        mAvailable.notify_one();
    }

    WhisperPipeline& at(
        int index) noexcept
    {
        return *mSlots[static_cast<std::size_t>(index)];
    }

    int size() const noexcept
    {
        return static_cast<int>(mSlots.size());
    }

private:
    std::vector<std::unique_ptr<WhisperPipeline>> mSlots;
    std::vector<int> mFree;

    std::mutex mMutex;
    std::condition_variable mAvailable;
};

//! Borrows a slot for the lifetime of one request.
class PipelineLease
{
public:
    explicit PipelineLease(
        PipelinePool& pool)
        : mPool(pool)
        , mIndex(pool.acquire())
    {
    }

    ~PipelineLease()
    {
        mPool.release(mIndex);
    }

    PipelineLease(PipelineLease const&) = delete;
    PipelineLease& operator=(PipelineLease const&) = delete;

    WhisperPipeline& pipeline() noexcept
    {
        return mPool.at(mIndex);
    }

private:
    PipelinePool& mPool;
    int mIndex;
};

// ============================================================
// Admission
// ============================================================

//! Bounded in-flight counter. Requests that cannot claim a ticket are shed with
//! 503 instead of queueing without limit in front of the slot pool.
class Admission
{
public:
    explicit Admission(
        int capacity) noexcept
        : mCapacity(capacity)
    {
    }

    bool tryAcquire() noexcept
    {
        int current = mCount.load(std::memory_order_relaxed);

        while (current < mCapacity)
        {
            if (mCount.compare_exchange_weak(
                    current,
                    current + 1,
                    std::memory_order_acquire,
                    std::memory_order_relaxed))
            {
                return true;
            }
        }

        return false;
    }

    void release() noexcept
    {
        mCount.fetch_sub(1, std::memory_order_release);
    }

    int inFlight() const noexcept
    {
        return mCount.load(std::memory_order_relaxed);
    }

private:
    std::atomic<int> mCount{0};
    int const mCapacity;
};

class AdmissionTicket
{
public:
    explicit AdmissionTicket(
        Admission& admission) noexcept
        : mAdmission(admission)
    {
    }

    ~AdmissionTicket()
    {
        mAdmission.release();
    }

    AdmissionTicket(AdmissionTicket const&) = delete;
    AdmissionTicket& operator=(AdmissionTicket const&) = delete;

private:
    Admission& mAdmission;
};

// ============================================================
// HTTP helpers
// ============================================================

void sendError(
    httplib::Response& response,
    int status,
    std::string const& message)
{
    json const body = {{"error", message}};

    response.status = status;
    response.set_content(body.dump(), "application/json");
}

std::string formField(
    httplib::Request const& request,
    char const* name,
    std::string const& fallback = "")
{
    return request.has_file(name)
        ? request.get_file_value(name).content
        : fallback;
}

std::string toLower(
    std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    return value;
}

httplib::Server* gServer = nullptr;

void handleSignal(
    int)
{
    if (gServer != nullptr)
    {
        gServer->stop();
    }
}

} // namespace

int main(
    int argc,
    char** argv)
{
    ServerOptions options;

    if (!parseArgs(argc, argv, options))
    {
        return 1;
    }

    // Keep plugins loaded for the full process lifetime.
    auto pluginHandle = trt_edgellm::loadEdgellmPluginLib();

    if (!pluginHandle)
    {
        std::cerr << "Failed to load Edge-LLM plugins\n";

        return 1;
    }

    // Create the CUDA context explicitly so the first request does not pay for
    // it. On Tegra this is ~1 sec of driver setup.
    cudaError_t const cudaInitStatus = cudaFree(nullptr);

    if (cudaInitStatus != cudaSuccess)
    {
        std::cerr
            << "CUDA context init failed: "
            << cudaGetErrorString(cudaInitStatus)
            << '\n';

        return 1;
    }

    SharedResources shared;

    if (!shared.load(options))
    {
        return 1;
    }

    PromptBuilder promptBuilder;

    if (!promptBuilder.initialize(shared.tokenizer()))
    {
        LOG_ERROR("Tokenizer is missing Whisper's control tokens; is this a Whisper checkpoint?");

        return 1;
    }

    PipelinePool pool;

    if (!pool.initialize(shared, options))
    {
        return 1;
    }

    Admission admission(options.queueDepth);

    httplib::Server server;

    gServer = &server;
    std::signal(SIGINT, handleSignal);
    std::signal(SIGTERM, handleSignal);

    // Leave headroom over the file cap for multipart framing; httplib rejects a
    // larger body itself, before it is buffered.
    server.set_payload_max_length(kMaxUploadBytes + (1ULL * 1024ULL * 1024ULL));

    // The socket pool must be strictly larger than the admission capacity, or
    // admission never becomes the binding constraint and the 503 path is
    // unreachable: httplib's default pool is max(8, ncpu-1), so with a larger
    // queue depth excess requests would wait in the accept backlog instead of
    // being shed. Sockets are cheap; a pipeline slot is not.
    int const socketThreads = options.queueDepth + kSocketThreadHeadroom;

    server.new_task_queue = [socketThreads] { return new httplib::ThreadPool(socketThreads); };

    server.Get("/health",
        [&](httplib::Request const&, httplib::Response& response)
        {
            json const body = {
                {"status", "ok"},
                {"slots", pool.size()},
                {"busy", admission.inFlight()},
                {"queue_depth", options.queueDepth},
            };

            response.set_content(body.dump(), "application/json");
        });

    server.Get("/v1/models",
        [&](httplib::Request const&, httplib::Response& response)
        {
            json const body = {
                {"object", "list"},
                {"data", json::array({json{
                             {"id", "whisper-small"},
                             {"object", "model"},
                             {"owned_by", "tensorrt-edgellm"},
                         }})},
            };

            response.set_content(body.dump(), "application/json");
        });

    server.Post("/v1/audio/transcriptions",
        [&](httplib::Request const& request, httplib::Response& response)
        {
            std::string const responseFormat = formField(request, "response_format", "json");

            if (responseFormat != "json" && responseFormat != "text")
            {
                sendError(response, 400,
                    "unsupported response_format '" + responseFormat + "'; use json or text");

                return;
            }

            // Whisper's forced prefix is <|sot|> <|lang|> <|task|> <|notimestamps|>.
            // Both ids are resolved against the checkpoint's own vocabulary.
            std::string const language = toLower(formField(request, "language", "en"));
            std::string const task = toLower(formField(request, "task", "transcribe"));

            if (task != "transcribe" && task != "translate")
            {
                sendError(response, 400,
                    "unsupported task '" + task + "'; use transcribe or translate");

                return;
            }

            // Two prefixes: the short path keeps <|notimestamps|> so its output
            // stays exactly what it was, while long audio drops it so the model
            // can mark segment boundaries for the seek.
            std::vector<int64_t> shortPrompt;
            std::vector<int64_t> timestampPrompt;

            if (!promptBuilder.build(language, task, false, shortPrompt)
                || !promptBuilder.build(language, task, true, timestampPrompt))
            {
                sendError(response, 400,
                    "unsupported language '" + language
                        + "'; expected an ISO-639-1 code the model supports");

                return;
            }

            if (!request.has_file("file"))
            {
                sendError(response, 400, "missing required form field 'file'");

                return;
            }

            auto const& upload = request.get_file_value("file");

            if (upload.content.empty())
            {
                sendError(response, 400, "empty audio file");

                return;
            }

            if (upload.content.size() > kMaxUploadBytes)
            {
                sendError(response, 413,
                    "audio upload exceeds the supported maximum of "
                        + std::to_string(kMaxUploadBytes) + " bytes");

                return;
            }

            // Decode and validate before claiming a slot, so a malformed or
            // over-long upload never occupies the pipeline.
            trt_edgellm::rt::audio::AudioPCM pcm;

            if (!WhisperAudioProcessor::decodeBytes(
                    reinterpret_cast<uint8_t const*>(upload.content.data()),
                    upload.content.size(),
                    pcm))
            {
                // The loader also refuses a decode longer than its own cap, and
                // reports that the same way as a malformed container, so the
                // message has to cover both.
                sendError(response, 400,
                    "could not decode audio; expected wav, mp3 or flac no longer than "
                        + std::to_string(static_cast<int>(kLoaderMaxDecodedSeconds)) + " s");

                return;
            }

            double const durationSeconds
                = static_cast<double>(pcm.samples.size())
                / static_cast<double>(pcm.sampleRate);

            if (durationSeconds > options.maxAudioSeconds)
            {
                sendError(response, 413,
                    "audio is " + std::to_string(durationSeconds)
                        + " s; this server accepts up to "
                        + std::to_string(static_cast<int>(options.maxAudioSeconds)) + " s");

                return;
            }

            if (!admission.tryAcquire())
            {
                response.set_header("Retry-After", "1");
                sendError(response, 503, "server overloaded: request queue is full");

                return;
            }

            AdmissionTicket ticket(admission);

            std::string text;

            {
                PipelineLease lease(pool);

                std::size_t windows = 0;

                if (!lease.pipeline().transcribeAny(
                        pcm, &shortPrompt, &timestampPrompt, text, &windows))
                {
                    sendError(response, 500, "transcription failed");

                    return;
                }

                LOG_DEBUG("transcribed %.2f s in %zu window(s)", durationSeconds, windows);
            }

            if (responseFormat == "text")
            {
                response.set_content(text, "text/plain");

                return;
            }

            json const body = {{"text", text}};

            response.set_content(body.dump(), "application/json");
        });

    LOG_INFO("Whisper server listening on %s:%d (slots %d, queue depth %d, max new tokens %d, "
             "max audio %.0f s, long-audio mode %s)",
        options.host.c_str(), options.port, pool.size(), options.queueDepth, options.maxNewTokens,
        options.maxAudioSeconds, options.timestampSeek ? "timestamps" : "overlap");

    if (!server.listen(options.host, options.port))
    {
        std::cerr
            << "Failed to bind "
            << options.host
            << ':'
            << options.port
            << '\n';

        return 1;
    }

    LOG_INFO("Whisper server stopped");

    return 0;
}
