#include "whisper_audio_processor.h"
#include "whisper_encoder_inference.h"

#include <cuda_fp16.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>


namespace
{

bool saveFp16(
    std::string const& path,
    std::vector<__half> const& data)
{
    std::ofstream file(
        path,
        std::ios::binary);

    if (!file)
    {
        return false;
    }

    file.write(
        reinterpret_cast<char const*>(
            data.data()),
        static_cast<std::streamsize>(
            data.size() * sizeof(__half)));

    return static_cast<bool>(file);
}

} // namespace


int main(
    int argc,
    char** argv)
{
    using namespace
        trt_edgellm::examples::whisper;

    if (argc < 3 || argc > 4)
    {
        std::cerr
            << "Usage:\n\n"
            << argv[0]
            << " <engine>"
            << " <audio.wav>"
            << " [encoder_output_fp16.bin]\n";

        return 1;
    }

    std::string const enginePath
        = argv[1];

    std::string const audioPath
        = argv[2];

    std::string const outputPath
        = argc == 4
        ? argv[3]
        : "whisper_audio_encoder_output_fp16.bin";


    std::cout
        << "================================\n"
        << " Whisper Audio -> TensorRT Encoder\n"
        << "================================\n";


    // ========================================================
    // 1. Audio preprocessing
    // ========================================================

    WhisperAudioProcessor processor;

    std::vector<__half> inputFeatures;

    if (!processor.processFile(
            audioPath,
            inputFeatures))
    {
        std::cerr
            << "Whisper audio preprocessing failed\n";

        return 1;
    }


    std::cout
        << "\nPreprocessed input elements: "
        << inputFeatures.size()
        << '\n';


    if (inputFeatures.size() != kInputElements)
    {
        std::cerr
            << "Unexpected input feature size\n";

        return 1;
    }


    // ========================================================
    // 2. TensorRT encoder
    // ========================================================

    WhisperEncoderInference encoder(
        enginePath);


    std::vector<__half> encoderOutput;


    if (!encoder.run(
            inputFeatures,
            encoderOutput))
    {
        std::cerr
            << "Whisper TensorRT encoder failed\n";

        return 1;
    }


    // ========================================================
    // 3. Validate output shape
    // ========================================================

    if (encoderOutput.size()
        != kOutputElements)
    {
        std::cerr
            << "Unexpected encoder output size: "
            << encoderOutput.size()
            << '\n';

        return 1;
    }


    std::cout
        << "\n================================\n"
        << " Encoder completed\n"
        << "================================\n";

    std::cout
        << "Input shape  : [1, 80, 3000]\n"
        << "Output shape : [1, 1500, 768]\n"
        << "Output elems : "
        << encoderOutput.size()
        << '\n';


    // ========================================================
    // 4. Save output temporarily for validation
    // ========================================================

    if (!saveFp16(
            outputPath,
            encoderOutput))
    {
        std::cerr
            << "Failed to save encoder output\n";

        return 1;
    }


    std::cout
        << "Saved output : "
        << outputPath
        << '\n';


    return 0;
}