#include "whisper_encoder_inference.h"

#include <iostream>
#include <string>

int main(
    int argc,
    char** argv)
{
    if (argc < 4 || argc > 5)
    {
        std::cerr
            << "Usage:\n\n"
            << argv[0]
            << " <engine>"
            << " <input_fp16.bin>"
            << " <python_trt_output_fp16.bin>"
            << " [cpp_output_fp16.bin]\n";

        return 1;
    }

    std::string const enginePath = argv[1];
    std::string const inputPath = argv[2];
    std::string const referencePath = argv[3];

    std::string const outputPath
        = argc == 5
        ? argv[4]
        : "whisper_cpp_trt_output_fp16.bin";

    trt_edgellm::examples::whisper::
        WhisperEncoderInference runner(
            enginePath,
            inputPath,
            referencePath,
            outputPath);

    return runner.run()
        ? 0
        : 1;
}