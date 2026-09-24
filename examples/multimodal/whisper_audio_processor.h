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

#pragma once

#include "runtime/audioLoader.h"
#include "runtime/melSpectrogram.h"

#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace trt_edgellm
{
namespace examples
{
namespace whisper
{

constexpr int32_t kWhisperSampleRate = 16000;
constexpr int32_t kWhisperChunkSeconds = 30;

constexpr std::size_t kWhisperPcmSamples = static_cast<std::size_t>(kWhisperSampleRate * kWhisperChunkSeconds);

constexpr int32_t kWhisperMelBins = 80;
constexpr int32_t kWhisperMelFrames = 3000;

constexpr std::size_t kWhisperFeatureElements
    = static_cast<std::size_t>(
        kWhisperMelBins * kWhisperMelFrames);

class WhisperAudioProcessor
{
public:
    WhisperAudioProcessor();

    bool processFile(
        std::filesystem::path const& audioPath,
        std::vector<__half>& outputFeatures);

    bool processBytes(
        uint8_t const* bytes,
        std::size_t size,
        std::vector<__half>& outputFeatures);

    //! Mel-extract PCM the caller already decoded. Split out from the decode so a
    //! server can measure duration and reject an over-long or undecodable upload
    //! before it occupies an inference slot.
    //!
    //! \param pcm Mono FP32 at ``kWhisperSampleRate``; padded or trimmed in place
    //!            to the single 30-second window.
    bool processPcm(
        rt::audio::AudioPCM& pcm,
        std::vector<__half>& outputFeatures);

    //! Decode an encoded blob (wav / mp3 / flac) to mono FP32 at
    //! ``kWhisperSampleRate``. Does not pad or trim.
    static bool decodeBytes(
        uint8_t const* bytes,
        std::size_t size,
        rt::audio::AudioPCM& out);

    //! Duration of the last input *before* padding or trimming, in seconds.
    double getLastAudioDurationSeconds() const
    {
        return mLastAudioDurationSeconds;
    }

private:
    static void padOrTrimPcm(
        rt::audio::AudioPCM& pcm);

private:
    rt::audio::MelExtractor mMelExtractor;

    double mLastAudioDurationSeconds{0.0};
};

} // namespace whisper
} // namespace examples
} // namespace trt_edgellm
