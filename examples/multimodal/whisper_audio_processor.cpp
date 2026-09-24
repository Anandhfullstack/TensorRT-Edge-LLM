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

#include "common/logger.h"

namespace trt_edgellm
{
namespace examples
{
namespace whisper
{

WhisperAudioProcessor::WhisperAudioProcessor()
    : mMelExtractor(
          rt::audio::makeWhisperSmallExtractor())
{
}

void WhisperAudioProcessor::padOrTrimPcm(
    rt::audio::AudioPCM& pcm)
{
    std::size_t const originalSamples
        = pcm.samples.size();

    if (originalSamples < kWhisperPcmSamples)
    {
        pcm.samples.resize(
            kWhisperPcmSamples,
            0.0f);

        LOG_DEBUG("PCM %zu samples -> zero-pad to %zu", originalSamples, kWhisperPcmSamples);
    }
    else if (originalSamples > kWhisperPcmSamples)
    {
        pcm.samples.resize(
            kWhisperPcmSamples);

        LOG_DEBUG("PCM %zu samples -> truncate to %zu", originalSamples, kWhisperPcmSamples);
    }
    else
    {
        LOG_DEBUG("PCM %zu samples -> exact window", originalSamples);
    }
}

bool WhisperAudioProcessor::decodeBytes(
    uint8_t const* bytes,
    std::size_t size,
    rt::audio::AudioPCM& out)
{
    if (bytes == nullptr || size == 0)
    {
        LOG_ERROR("Empty audio blob");

        return false;
    }

    if (!rt::audio::loadAudioBytes(
            bytes,
            size,
            kWhisperSampleRate,
            out))
    {
        LOG_ERROR("Failed to decode audio blob of %zu bytes", size);

        return false;
    }

    return true;
}

bool WhisperAudioProcessor::processFile(
    std::filesystem::path const& audioPath,
    std::vector<__half>& outputFeatures)
{
    LOG_DEBUG("Loading audio: %s", audioPath.string().c_str());

    rt::audio::AudioPCM pcm;

    if (!rt::audio::loadAudioFile(
            audioPath,
            kWhisperSampleRate,
            pcm))
    {
        LOG_ERROR("Failed to load audio file: %s", audioPath.string().c_str());

        return false;
    }

    return processPcm(
        pcm,
        outputFeatures);
}

bool WhisperAudioProcessor::processBytes(
    uint8_t const* bytes,
    std::size_t size,
    std::vector<__half>& outputFeatures)
{
    rt::audio::AudioPCM pcm;

    if (!decodeBytes(
            bytes,
            size,
            pcm))
    {
        return false;
    }

    return processPcm(
        pcm,
        outputFeatures);
}

bool WhisperAudioProcessor::processPcm(
    rt::audio::AudioPCM& pcm,
    std::vector<__half>& outputFeatures)
{
    if (pcm.sampleRate != kWhisperSampleRate)
    {
        LOG_ERROR("Unexpected sample rate: %d (expected %d)", pcm.sampleRate, kWhisperSampleRate);

        return false;
    }

    // --------------------------------------------------------
    // Whisper Small expects one 30-second input chunk.
    //
    // IMPORTANT:
    // Padding is performed on PCM, NOT on the mel tensor.
    // --------------------------------------------------------
    mLastAudioDurationSeconds
        = static_cast<double>(pcm.samples.size())
        / static_cast<double>(pcm.sampleRate);

    padOrTrimPcm(pcm);

    // --------------------------------------------------------
    // Existing TensorRT-Edge-LLM MelExtractor
    //
    // Output expected:
    //
    // FP32 CPU [80, 3000]
    // --------------------------------------------------------
    rt::Tensor melTensor;

    if (!mMelExtractor.extract(
            pcm,
            melTensor))
    {
        LOG_ERROR("Whisper mel extraction failed");

        return false;
    }

    rt::Coords const shape
        = melTensor.getShape();

    if (shape.getNumDims() != 2
        || shape[0] != kWhisperMelBins
        || shape[1] != kWhisperMelFrames)
    {
        LOG_ERROR("Unexpected mel shape %s (expected [80, 3000])", shape.formatString().c_str());

        return false;
    }

    if (melTensor.getDeviceType()
        != rt::DeviceType::kCPU)
    {
        LOG_ERROR("Expected CPU mel tensor");

        return false;
    }

    if (melTensor.getDataType()
        != nvinfer1::DataType::kFLOAT)
    {
        LOG_ERROR("Expected FP32 mel tensor");

        return false;
    }

    // --------------------------------------------------------
    // TensorRT engine expects FP16.
    //
    // [80,3000] FP32
    //       ↓
    // [80,3000] FP16
    // --------------------------------------------------------

    float const* melFp32
        = melTensor.dataPointer<float>();

    outputFeatures.resize(
        kWhisperFeatureElements);

    for (std::size_t i = 0;
         i < kWhisperFeatureElements;
         ++i)
    {
        outputFeatures[i]
            = __float2half(melFp32[i]);
    }

    return true;
}

} // namespace whisper
} // namespace examples
} // namespace trt_edgellm
