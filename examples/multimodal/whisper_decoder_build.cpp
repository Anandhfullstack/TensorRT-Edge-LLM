#include "builder/whisperDecoderBuilder.h"
#include "common/logger.h"

#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <string>

using namespace trt_edgellm;

struct WhisperDecoderBuildArgs
{
    std::string onnxDir;
    std::string engineDir;

    int64_t minInputLen{1};
    int64_t optInputLen{16};
    int64_t maxInputLen{448};

    bool debug{false};
    bool help{false};
};

void printUsage(char const* programName)
{
    std::cerr
        << "Usage: " << programName
        << " --onnxDir <path>"
        << " --engineDir <path>"
        << " [--minInputLen 1]"
        << " [--optInputLen 16]"
        << " [--maxInputLen 448]"
        << " [--debug]"
        << std::endl;
}

bool parseArgs(
    WhisperDecoderBuildArgs& args,
    int argc,
    char* argv[])
{
    enum OptionId
    {
        OPT_MIN_INPUT_LEN = 1000,
        OPT_OPT_INPUT_LEN,
        OPT_MAX_INPUT_LEN
    };

    static struct option longOptions[] = {
        {"help", no_argument, nullptr, 'h'},
        {"onnxDir", required_argument, nullptr, 'o'},
        {"engineDir", required_argument, nullptr, 'e'},
        {"debug", no_argument, nullptr, 'd'},

        {"minInputLen", required_argument, nullptr, OPT_MIN_INPUT_LEN},
        {"optInputLen", required_argument, nullptr, OPT_OPT_INPUT_LEN},
        {"maxInputLen", required_argument, nullptr, OPT_MAX_INPUT_LEN},

        {nullptr, 0, nullptr, 0}
    };

    int optionIndex = 0;
    int opt;

    while ((opt = getopt_long(
                argc,
                argv,
                "ho:e:d",
                longOptions,
                &optionIndex)) != -1)
    {
        switch (opt)
        {
        case 'h':
            args.help = true;
            return true;

        case 'o':
            args.onnxDir = optarg;
            break;

        case 'e':
            args.engineDir = optarg;
            break;

        case 'd':
            args.debug = true;
            break;

        case OPT_MIN_INPUT_LEN:
            args.minInputLen = std::stoll(optarg);
            break;

        case OPT_OPT_INPUT_LEN:
            args.optInputLen = std::stoll(optarg);
            break;

        case OPT_MAX_INPUT_LEN:
            args.maxInputLen = std::stoll(optarg);
            break;

        default:
            return false;
        }
    }

    if (args.help)
    {
        return true;
    }

    if (args.onnxDir.empty()
        || args.engineDir.empty())
    {
        std::cerr
            << "--onnxDir and --engineDir are required."
            << std::endl;

        return false;
    }

    return true;
}

int main(int argc, char* argv[])
{
    WhisperDecoderBuildArgs args;

    if (!parseArgs(args, argc, argv))
    {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }

    if (args.help)
    {
        printUsage(argv[0]);
        return EXIT_SUCCESS;
    }

    if (args.debug)
    {
        gLogger.setLevel(
            nvinfer1::ILogger::Severity::kVERBOSE);
    }
    else
    {
        gLogger.setLevel(
            nvinfer1::ILogger::Severity::kINFO);
    }

    builder::WhisperDecoderBuilderConfig config;

    config.minInputLen = args.minInputLen;
    config.optInputLen = args.optInputLen;
    config.maxInputLen = args.maxInputLen;

    LOG_INFO(
        "Whisper decoder ONNX directory: %s",
        args.onnxDir.c_str());

    LOG_INFO(
        "Whisper decoder engine directory: %s",
        args.engineDir.c_str());

    LOG_INFO(
        "Decoder sequence profile: min=%ld opt=%ld max=%ld",
        config.minInputLen,
        config.optInputLen,
        config.maxInputLen);

    builder::WhisperDecoderBuilder decoderBuilder(
        args.onnxDir,
        args.engineDir,
        config);

    if (!decoderBuilder.build())
    {
        LOG_ERROR(
            "Failed to build Whisper decoder engine.");

        return EXIT_FAILURE;
    }

    LOG_INFO(
        "Whisper decoder engine build completed.");

    return EXIT_SUCCESS;
}