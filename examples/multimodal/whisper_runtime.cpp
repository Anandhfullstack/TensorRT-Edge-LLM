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

int main(
    int argc,
    char** argv)
{
    using namespace
        trt_edgellm::examples::whisper;

    if (argc < 5 || argc > 6)
    {
        std::cerr
            << "Usage:\n"
            << argv[0]
            << " <encoder.engine>"
            << " <decoder.engine>"
            << " <tokenizer_dir>"
            << " <audio.wav>"
            << " [max_new_tokens]\n";

        return 1;
    }

    std::string const encoderEngine
        = argv[1];

    std::string const decoderEngine
        = argv[2];

    std::string const tokenizerDir = argv[3];
    std::string const audioPath
        = argv[4];

    int32_t const maxNewTokens
        = argc == 6
        ? std::stoi(argv[5])
        : 32;

    // Keep plugin loaded for full runtime lifetime.
    auto pluginHandle
        = trt_edgellm::loadEdgellmPluginLib();

    if (!pluginHandle)
    {
        std::cerr
            << "Failed to load Edge-LLM plugins\n";

        return 1;
    }

    std::cout
        << "\n================================\n"
        << " Whisper TensorRT Runtime\n"
        << "================================\n";

    // ========================================================
    // 1. WAV -> Whisper Mel features
    // ========================================================

    WhisperAudioProcessor processor;

    std::vector<__half> inputFeatures;

    if (!processor.processFile(
            audioPath,
            inputFeatures))
    {
        std::cerr
            << "Audio preprocessing failed\n";

        return 1;
    }

    std::cout
        << "\nAudio preprocessing complete\n"
        << "Input: [1,80,3000]\n";

    // ========================================================
    // 2. Encoder
    // ========================================================

    WhisperEncoderInference encoder(
        encoderEngine);

    std::vector<__half> encoderOutput;

    if (!encoder.run(
            inputFeatures,
            encoderOutput))
    {
        std::cerr
            << "Encoder inference failed\n";

        return 1;
    }

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

    WhisperDecoderRunner decoder(
        decoderEngine);

    std::vector<int64_t> tokens;

    if (!decoder.generate(
            encoderOutput,
            tokens,
            maxNewTokens))
    {
        std::cerr
            << "Decoder inference failed\n";

        return 1;
    }

    // ========================================================
    // 4. Decode Whisper token IDs -> text
    // ========================================================

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

    // true = remove Whisper special tokens such as:
    // <|en|>
    // <|transcribe|>
    // <|notimestamps|>
    // <|endoftext|>
    std::string const text
        = tokenizer.decode(
            decodeTokens,
            true);

    std::cout
        << "\n================================\n"
        << " Transcription\n"
        << "================================\n"
        << text
        << '\n';

    return 0;
}