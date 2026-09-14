#include "whisper_audio_processor.h"

#include "runtime/audioUtils.h"

#include <iostream>

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

    std::cout
        << "Original PCM samples : "
        << originalSamples
        << '\n';

    std::cout
        << "Original duration    : "
        << static_cast<double>(originalSamples)
            / static_cast<double>(kWhisperSampleRate)
        << " sec\n";

    if (originalSamples < kWhisperPcmSamples)
    {
        pcm.samples.resize(
            kWhisperPcmSamples,
            0.0f);

        std::cout
            << "PCM action           : zero-pad\n";
    }
    else if (originalSamples > kWhisperPcmSamples)
    {
        pcm.samples.resize(
            kWhisperPcmSamples);

        std::cout
            << "PCM action           : truncate\n";
    }
    else
    {
        std::cout
            << "PCM action           : none\n";
    }

    std::cout
        << "Final PCM samples    : "
        << pcm.samples.size()
        << '\n';
}

bool WhisperAudioProcessor::processFile(
    std::filesystem::path const& audioPath,
    std::vector<__half>& outputFeatures)
{
    std::cout
        << "Loading audio: "
        << audioPath.string()
        << '\n';

    rt::audioUtils::AudioData audioData;

    if (!rt::audioUtils::loadAudioDataFromFile(
            audioPath,
            kWhisperSampleRate,
            audioData))
    {
        std::cerr
            << "Failed to load audio file\n";

        return false;
    }

    if (!audioData.pcm)
    {
        std::cerr
            << "Audio loader returned null PCM\n";

        return false;
    }

    auto& pcm = *audioData.pcm;

    std::cout
        << "Loaded sample rate    : "
        << pcm.sampleRate
        << '\n';

    std::cout
        << "Loaded channels       : "
        << pcm.numChannels
        << '\n';

    if (pcm.sampleRate != kWhisperSampleRate)
    {
        std::cerr
            << "Unexpected sample rate: "
            << pcm.sampleRate
            << '\n';

        return false;
    }

    // --------------------------------------------------------
    // Whisper Small expects one 30-second input chunk.
    //
    // IMPORTANT:
    // Padding is performed on PCM, NOT on the mel tensor.
    // --------------------------------------------------------
    mLastAudioDurationSeconds   = static_cast<double>(pcm.samples.size()) / static_cast<double>(pcm.sampleRate);
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
        std::cerr
            << "Whisper mel extraction failed\n";

        return false;
    }

    rt::Coords const shape
        = melTensor.getShape();

    std::cout
        << "Mel shape            : "
        << shape.formatString()
        << '\n';

    if (shape.getNumDims() != 2
        || shape[0] != kWhisperMelBins
        || shape[1] != kWhisperMelFrames)
    {
        std::cerr
            << "Unexpected mel shape. "
            << "Expected [80, 3000]\n";

        return false;
    }

    if (melTensor.getDeviceType()
        != rt::DeviceType::kCPU)
    {
        std::cerr
            << "Expected CPU mel tensor\n";

        return false;
    }

    if (melTensor.getDataType()
        != nvinfer1::DataType::kFLOAT)
    {
        std::cerr
            << "Expected FP32 mel tensor\n";

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

    std::cout
        << "Output elements      : "
        << outputFeatures.size()
        << '\n';

    std::cout
        << "Output bytes         : "
        << outputFeatures.size()
               * sizeof(__half)
        << '\n';

    return true;
}

} // namespace whisper
} // namespace examples
} // namespace trt_edgellm