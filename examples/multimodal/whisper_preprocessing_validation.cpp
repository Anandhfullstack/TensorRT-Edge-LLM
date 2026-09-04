#include "whisper_audio_processor.h"

#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

std::vector<__half> readHalfFile(
    std::string const& path,
    std::size_t expectedElements)
{
    std::ifstream file(
        path,
        std::ios::binary | std::ios::ate);

    if (!file)
    {
        throw std::runtime_error(
            "Failed to open: " + path);
    }

    std::streamsize const fileSize
        = file.tellg();

    std::size_t const expectedBytes
        = expectedElements * sizeof(__half);

    if (fileSize
        != static_cast<std::streamsize>(
            expectedBytes))
    {
        throw std::runtime_error(
            "Unexpected reference file size: "
            + std::to_string(fileSize)
            + " bytes, expected "
            + std::to_string(expectedBytes));
    }

    file.seekg(
        0,
        std::ios::beg);

    std::vector<__half> data(
        expectedElements);

    file.read(
        reinterpret_cast<char*>(
            data.data()),
        fileSize);

    if (!file)
    {
        throw std::runtime_error(
            "Failed reading: " + path);
    }

    return data;
}


void writeHalfFile(
    std::string const& path,
    std::vector<__half> const& data)
{
    std::ofstream file(
        path,
        std::ios::binary);

    if (!file)
    {
        throw std::runtime_error(
            "Failed to create: " + path);
    }

    file.write(
        reinterpret_cast<char const*>(
            data.data()),
        static_cast<std::streamsize>(
            data.size() * sizeof(__half)));
}


uint16_t halfBits(
    __half value)
{
    uint16_t bits = 0;

    static_assert(
        sizeof(bits) == sizeof(value));

    std::memcpy(
        &bits,
        &value,
        sizeof(bits));

    return bits;
}

} // namespace


int main(
    int argc,
    char** argv)
{
    using namespace
        trt_edgellm::examples::whisper;

    if (argc < 3)
    {
        std::cerr
            << "Usage:\n"
            << "  "
            << argv[0]
            << " <audio.wav>"
            << " <python_reference_fp16.bin>"
            << " [cpp_output_fp16.bin]\n";

        return 1;
    }

    std::string const audioPath
        = argv[1];

    std::string const referencePath
        = argv[2];

    std::string const outputPath
        = argc >= 4
        ? argv[3]
        : "whisper_cpp_preprocessed_fp16.bin";

    try
    {
        std::cout
            << "================================\n"
            << " Whisper C++ preprocessing\n"
            << "================================\n";

        WhisperAudioProcessor processor;

        std::vector<__half> cppFeatures;

        if (!processor.processFile(
                audioPath,
                cppFeatures))
        {
            std::cerr
                << "Preprocessing failed\n";

            return 1;
        }

        auto reference
            = readHalfFile(
                referencePath,
                kWhisperFeatureElements);

        writeHalfFile(
            outputPath,
            cppFeatures);

        // ----------------------------------------------------
        // Compare C++ features against Python HF reference.
        // ----------------------------------------------------

        double absSum = 0.0;
        double squaredSum = 0.0;

        double dot = 0.0;
        double normCpp = 0.0;
        double normReference = 0.0;

        float maxDiff = 0.0f;

        std::size_t worstIndex = 0;
        std::size_t exactMismatch = 0;

        std::size_t above001 = 0;
        std::size_t above01 = 0;
        std::size_t above005 = 0;
        std::size_t above1 = 0;

        for (std::size_t i = 0;
             i < kWhisperFeatureElements;
             ++i)
        {
            float const cppValue
                = __half2float(
                    cppFeatures[i]);

            float const refValue
                = __half2float(
                    reference[i]);

            float const difference
                = std::fabs(
                    cppValue - refValue);

            absSum += difference;

            squaredSum
                += static_cast<double>(
                       difference)
                * static_cast<double>(
                       difference);

            dot
                += static_cast<double>(
                       cppValue)
                * static_cast<double>(
                       refValue);

            normCpp
                += static_cast<double>(
                       cppValue)
                * static_cast<double>(
                       cppValue);

            normReference
                += static_cast<double>(
                       refValue)
                * static_cast<double>(
                       refValue);

            if (difference > maxDiff)
            {
                maxDiff = difference;
                worstIndex = i;
            }

            if (halfBits(cppFeatures[i])
                != halfBits(reference[i]))
            {
                ++exactMismatch;
            }

            if (difference > 0.001f)
            {
                ++above001;
            }

            if (difference > 0.01f)
            {
                ++above01;
            }

            if (difference > 0.05f)
            {
                ++above005;
            }

            if (difference > 0.1f)
            {
                ++above1;
            }
        }

        double const count
            = static_cast<double>(
                kWhisperFeatureElements);

        double const meanAbs
            = absSum / count;

        double const rmse
            = std::sqrt(
                squaredSum / count);

        double const cosine
            = dot
            / (
                std::sqrt(normCpp)
                * std::sqrt(normReference)
            );

        std::size_t const worstMel
            = worstIndex
            / kWhisperMelFrames;

        std::size_t const worstFrame
            = worstIndex
            % kWhisperMelFrames;

        float const worstCpp
            = __half2float(
                cppFeatures[worstIndex]);

        float const worstReference
            = __half2float(
                reference[worstIndex]);

        std::cout
            << "\n================================\n"
            << " C++ vs Hugging Face preprocessing\n"
            << "================================\n";

        std::cout
            << "Elements              : "
            << kWhisperFeatureElements
            << '\n';

        std::cout
            << "Mean abs diff         : "
            << meanAbs
            << '\n';

        std::cout
            << "Max abs diff          : "
            << maxDiff
            << '\n';

        std::cout
            << "RMSE                  : "
            << rmse
            << '\n';

        std::cout
            << "Cosine similarity     : "
            << cosine
            << '\n';

        std::cout
            << "Exact FP16 mismatches : "
            << exactMismatch
            << " / "
            << kWhisperFeatureElements
            << '\n';

        std::cout
            << "\nDifference thresholds\n";

        std::cout
            << "> 0.001 : "
            << above001
            << '\n';

        std::cout
            << "> 0.01  : "
            << above01
            << '\n';

        std::cout
            << "> 0.05  : "
            << above005
            << '\n';

        std::cout
            << "> 0.1   : "
            << above1
            << '\n';

        std::cout
            << "\nWorst difference\n";

        std::cout
            << "index  : [0, "
            << worstMel
            << ", "
            << worstFrame
            << "]\n";

        std::cout
            << "C++    : "
            << worstCpp
            << '\n';

        std::cout
            << "Python : "
            << worstReference
            << '\n';

        std::cout
            << "diff   : "
            << maxDiff
            << '\n';

        std::cout
            << "\nSaved C++ features:\n"
            << outputPath
            << '\n';

        return 0;
    }
    catch (std::exception const& e)
    {
        std::cerr
            << "ERROR: "
            << e.what()
            << '\n';

        return 1;
    }
}