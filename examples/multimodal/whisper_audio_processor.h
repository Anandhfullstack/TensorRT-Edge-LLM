#pragma once

#include "runtime/melSpectrogram.h"

#include <cuda_fp16.h>

#include <cstddef>
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
}// namespace whisper
} // namespace examples
} // namespace trt_edgellm