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

#include "whisper_audio_processor.h"
#include "whisper_decoder_runner.h"
#include "whisper_encoder_inference.h"
#include "tokenizer/tokenizer.h"
#include "common/trtUtils.h"

#include <cuda_fp16.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <algorithm>
#include <chrono>
#include <iomanip>

#include <cuda_runtime.h>

namespace
{

using Clock = std::chrono::steady_clock;

double elapsedSeconds(
    Clock::time_point const start,
    Clock::time_point const end)
{
    return std::chrono::duration<double>(
        end - start)
        .count();
}

void printRow(
    char const* label,
    double const seconds)
{
    std::cout
        << "  "
        << std::left
        << std::setw(20)
        << label
        << std::right
        << std::setw(9)
        << seconds
        << " sec\n";
}

} // namespace

int main(
    int argc,
    char** argv)
{
    using namespace
        trt_edgellm::examples::whisper;

    if (argc < 6 || argc > 7)
    {
        std::cerr
            << "Usage:\n"
            << argv[0]
            << " <encoder.engine>"
            << " <cross_kv.engine>"
            << " <decoder.engine>"
            << " <tokenizer_dir>"
            << " <audio.wav>"
            << " [max_new_tokens]\n";

        return 1;
    }

    std::string const encoderEngine
    = argv[1];

    std::string const crossKvEngine
        = argv[2];

    std::string const decoderEngine
        = argv[3];

    std::string const tokenizerDir
        = argv[4];

    std::string const audioPath
        = argv[5];

    int32_t const maxNewTokens
        = argc == 7
        ? std::stoi(argv[6])
        : 32;

    std::cout
        << "\n================================\n"
        << " Whisper TensorRT Runtime\n"
        << "================================\n";

    // ========================================================
    // 0. One-time startup
    //
    // Everything here is paid once per process, not per
    // utterance. It is timed separately so it never inflates
    // the per-utterance stage numbers or the RTF.
    // ========================================================

    auto const pluginStart = Clock::now();

    // Keep plugin loaded for full runtime lifetime.
    auto pluginHandle
        = trt_edgellm::loadEdgellmPluginLib();

    if (!pluginHandle)
    {
        std::cerr
            << "Failed to load Edge-LLM plugins\n";

        return 1;
    }

    auto const pluginEnd = Clock::now();

    // Create the CUDA context explicitly. Otherwise the first CUDA call in
    // the process pays for it, and on Tegra that is ~1 sec of driver setup
    // charged to whichever stage happens to run first -- preprocessing,
    // since CPU mel tensors are allocated with cudaMallocHost.
    auto const cudaInitStart = Clock::now();

    cudaError_t const cudaInitStatus = cudaFree(nullptr);

    if (cudaInitStatus != cudaSuccess)
    {
        std::cerr
            << "CUDA context init failed: "
            << cudaGetErrorString(cudaInitStatus)
            << '\n';

        return 1;
    }

    auto const cudaInitEnd = Clock::now();

    // Both runners deserialize their engine lazily on first use. Force it
    // here so engine load is not billed to the encoder / decoder stage.
    auto const encoderLoadStart = Clock::now();

    WhisperEncoderInference encoder(encoderEngine);

    if (!encoder.initialize())
    {
        std::cerr
            << "Failed to load encoder engine: "
            << encoderEngine
            << '\n';

        return 1;
    }

    auto const encoderLoadEnd = Clock::now();

    // WhisperDecoderRunner decoder(decoderEngine);
    WhisperDecoderRunner decoder(
        crossKvEngine,
        decoderEngine);

    if (!decoder.initialize())
    {
        std::cerr
            << "Failed to load decoder engine: "
            << decoderEngine
            << '\n';

        return 1;
    }

    auto const decoderLoadEnd = Clock::now();

    trt_edgellm::tokenizer::Tokenizer tokenizer;

    if (!tokenizer.loadFromHF(
            tokenizerDir,
            false))
    {
        std::cerr
            << "Failed to load Whisper tokenizer from: "
            << tokenizerDir
            << '\n';

        return 1;
    }

    auto const tokenizerLoadEnd = Clock::now();

    double const pluginSeconds = elapsedSeconds(pluginStart, pluginEnd);
    double const cudaInitSeconds = elapsedSeconds(cudaInitStart, cudaInitEnd);
    double const encoderLoadSeconds = elapsedSeconds(encoderLoadStart, encoderLoadEnd);
    double const decoderLoadSeconds = elapsedSeconds(encoderLoadEnd, decoderLoadEnd);
    double const tokenizerLoadSeconds = elapsedSeconds(decoderLoadEnd, tokenizerLoadEnd);

    double const startupSeconds
        = pluginSeconds
        + cudaInitSeconds
        + encoderLoadSeconds
        + decoderLoadSeconds
        + tokenizerLoadSeconds;

    // ========================================================
    // 1. WAV -> Whisper Mel features
    // ========================================================

    WhisperAudioProcessor processor;

    std::vector<__half> inputFeatures;

    auto const preprocessStart = Clock::now();

    if (!processor.processFile(
            audioPath,
            inputFeatures))
    {
        std::cerr
            << "Audio preprocessing failed\n";

        return 1;
    }

    auto const preprocessEnd = Clock::now();

    double const preprocessSeconds = elapsedSeconds(preprocessStart, preprocessEnd);

    std::cout
        << "\nAudio preprocessing complete\n"
        << "Input: [1,80,3000]\n";

    // ========================================================
    // 2. Encoder
    // ========================================================

    std::vector<__half> encoderOutput;

    auto const encoderStart = Clock::now();

    if (!encoder.run(
            inputFeatures,
            encoderOutput))
    {
        std::cerr
            << "Encoder inference failed\n";

        return 1;
    }

    cudaDeviceSynchronize();

    auto const encoderEnd = Clock::now();

    double const encoderSeconds = elapsedSeconds(encoderStart, encoderEnd);

    std::cout
        << "\nEncoder complete\n"
        << "Encoder output: [1,1500,768]\n";

    // ========================================================
    // 3. Decoder
    //
    // NO FILE:
    //
    // encoderOutput
    //      ↓
    // decoder
    // ========================================================

    std::vector<int64_t> tokens;

    auto const decoderStart = Clock::now();

    if (!decoder.generate(encoderOutput, tokens, maxNewTokens))
    {
        std::cerr
            << "Decoder inference failed\n";

        return 1;
    }

    cudaDeviceSynchronize();

    auto const decoderEnd = Clock::now();

    double const decoderSeconds = elapsedSeconds(decoderStart, decoderEnd);

    // ========================================================
    // 4. Decode Whisper token IDs -> text
    // ========================================================

    // Tokenizer expects int32_t Rank.
    std::vector<trt_edgellm::tokenizer::Rank> decodeTokens;

    decodeTokens.reserve(
        tokens.size());

    for (int64_t token : tokens)
    {
        decodeTokens.push_back(
            static_cast<trt_edgellm::tokenizer::Rank>(
                token));
    }

    auto const detokenizeStart = Clock::now();

    // true = remove Whisper special tokens such as:
    // <|en|>
    // <|transcribe|>
    // <|notimestamps|>
    // <|endoftext|>
    std::string const text
        = tokenizer.decode(
            decodeTokens,
            true);

    auto const detokenizeEnd = Clock::now();

    double const detokenizeSeconds = elapsedSeconds(detokenizeStart, detokenizeEnd);

    std::cout
        << "\n================================\n"
        << " Transcription\n"
        << "================================\n"
        << text
        << '\n';

    // ========================================================
    // 5. Report
    // ========================================================

    double const originalAudioSeconds = processor.getLastAudioDurationSeconds();
    // Whisper currently processes at most one 30-second chunk.
    double const processedAudioSeconds
        = std::min(originalAudioSeconds, static_cast<double>(kWhisperChunkSeconds));

    double const pipelineSeconds
        = preprocessSeconds
        + encoderSeconds
        + decoderSeconds
        + detokenizeSeconds;

    double const rtf = pipelineSeconds / processedAudioSeconds;
    double const realtimeFactor = processedAudioSeconds / pipelineSeconds;
    double const coldStartSeconds = startupSeconds + pipelineSeconds;

    std::cout
        << std::fixed
        << std::setprecision(4)
        << "\n================================\n"
        << " Performance\n"
        << "================================\n"
        << "Audio duration      : "
        << processedAudioSeconds
        << " sec\n"
        << "\nStartup (one-time, excluded from RTF)\n";

    printRow("Plugin load", pluginSeconds);
    printRow("CUDA context init", cudaInitSeconds);
    printRow("Encoder engine", encoderLoadSeconds);
    printRow("Decoder engine", decoderLoadSeconds);
    printRow("Tokenizer", tokenizerLoadSeconds);
    printRow("Startup total", startupSeconds);

    std::cout
        << "\nPer-utterance pipeline\n";

    printRow("Preprocessing", preprocessSeconds);
    printRow("Encoder", encoderSeconds);
    printRow("Decoder", decoderSeconds);
    printRow("Detokenize", detokenizeSeconds);
    printRow("Pipeline total", pipelineSeconds);

    std::cout
        << "\nRTF (pipeline)      : "
        << rtf
        << '\n'
        << "Realtime speed      : "
        << realtimeFactor
        << "x\n"
        << "Cold start total    : "
        << coldStartSeconds
        << " sec\n";

    return 0;
}
