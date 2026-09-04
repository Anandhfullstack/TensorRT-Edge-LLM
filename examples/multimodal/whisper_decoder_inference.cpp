#include "common/logger.h"
#include "common/trtUtils.h"

#include <NvInfer.h>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace trt_edgellm;

namespace
{

constexpr char const* kInputIdsName = "input_ids";
constexpr char const* kEncoderHiddenName = "encoder_hidden_states";
constexpr char const* kLogitsName = "logits";

constexpr int64_t kBatch = 1;
constexpr int64_t kEncoderFrames = 1500;
constexpr int64_t kHiddenSize = 768;

// Whisper <|startoftranscript|>
constexpr int64_t kDecoderStartToken = 50258;

constexpr std::size_t kEncoderElements
    = kBatch
    * kEncoderFrames
    * kHiddenSize;

void checkCuda(
    cudaError_t status,
    char const* operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(operation)
            + ": "
            + cudaGetErrorString(status));
    }
}

std::vector<__half> readHalfFile(
    std::string const& path,
    std::size_t expectedElements)
{
    std::ifstream file(
        path,
        std::ios::binary
            | std::ios::ate);

    if (!file)
    {
        throw std::runtime_error(
            "Failed to open: " + path);
    }

    std::size_t const bytes
        = static_cast<std::size_t>(
            file.tellg());

    std::size_t const expectedBytes
        = expectedElements
        * sizeof(__half);

    if (bytes != expectedBytes)
    {
        throw std::runtime_error(
            "Unexpected encoder tensor size. Expected "
            + std::to_string(expectedBytes)
            + " bytes, got "
            + std::to_string(bytes));
    }

    file.seekg(0);

    std::vector<__half> data(
        expectedElements);

    file.read(
        reinterpret_cast<char*>(
            data.data()),
        expectedBytes);

    if (!file)
    {
        throw std::runtime_error(
            "Failed reading encoder tensor.");
    }

    return data;
}

std::size_t volume(
    nvinfer1::Dims const& dims)
{
    std::size_t result = 1;

    for (int32_t i = 0;
         i < dims.nbDims;
         ++i)
    {
        if (dims.d[i] <= 0)
        {
            throw std::runtime_error(
                "TensorRT returned unresolved dimension.");
        }

        result *= static_cast<std::size_t>(
            dims.d[i]);
    }

    return result;
}

} // namespace

int main(
    int argc,
    char* argv[])
{
    if (argc != 3)
    {
        std::cerr
            << "Usage:\n"
            << argv[0]
            << " <whisper_decoder.engine>"
            << " <encoder_hidden_states.bin>\n";

        return EXIT_FAILURE;
    }

    std::string const enginePath = argv[1];

    std::string const encoderPath = argv[2];

    try
    {
        gLogger.setLevel(
            nvinfer1::ILogger::Severity::kINFO);

        // ----------------------------------------------------
        // Load Edge-LLM plugins
        // ----------------------------------------------------

        auto pluginHandle
            = loadEdgellmPluginLib();

        // ----------------------------------------------------
        // Load encoder output
        //
        // [1, 1500, 768] FP16
        // ----------------------------------------------------

        std::vector<__half> encoderHidden
            = readHalfFile(
                encoderPath,
                kEncoderElements);

        std::cout
            << "Loaded encoder hidden states: "
            << encoderHidden.size()
            << " FP16 elements\n";

        // ----------------------------------------------------
        // Create TensorRT runtime
        // ----------------------------------------------------

        auto runtime
            = std::unique_ptr<nvinfer1::IRuntime>(
                nvinfer1::createInferRuntime(
                    gLogger));

        if (!runtime)
        {
            throw std::runtime_error(
                "Failed to create TensorRT runtime.");
        }

        // ----------------------------------------------------
        // Deserialize decoder engine
        // ----------------------------------------------------

        auto engine
            = deserializeCudaEngineFromFile(
                *runtime,
                enginePath);

        if (!engine)
        {
            throw std::runtime_error(
                "Failed to deserialize decoder engine.");
        }

        std::cout
            << "\n==============================\n"
            << " Decoder Engine\n"
            << "==============================\n";

        std::cout
            << printEngineInfo(
                   engine.get(),
                   0)
            << '\n';

        // ----------------------------------------------------
        // Validate expected I/O
        // ----------------------------------------------------

        if (engine->getTensorIOMode(
                kInputIdsName)
            != nvinfer1::TensorIOMode::kINPUT)
        {
            throw std::runtime_error(
                "input_ids input not found.");
        }

        if (engine->getTensorIOMode(
                kEncoderHiddenName)
            != nvinfer1::TensorIOMode::kINPUT)
        {
            throw std::runtime_error(
                "encoder_hidden_states input not found.");
        }

        if (engine->getTensorIOMode(
                kLogitsName)
            != nvinfer1::TensorIOMode::kOUTPUT)
        {
            throw std::runtime_error(
                "logits output not found.");
        }

        std::cout
            << "input_ids dtype: "
            << getDataTypeString(
                   engine->getTensorDataType(
                       kInputIdsName))
            << '\n';

        std::cout
            << "encoder_hidden_states dtype: "
            << getDataTypeString(
                   engine->getTensorDataType(
                       kEncoderHiddenName))
            << '\n';

        std::cout
            << "logits dtype: "
            << getDataTypeString(
                   engine->getTensorDataType(
                       kLogitsName))
            << '\n';

        // ----------------------------------------------------
        // First test:
        //
        // input_ids = [50258]
        //
        // Shape:
        // [1, 1]
        // ----------------------------------------------------

        std::vector<int64_t> inputIds{
            kDecoderStartToken};

        int64_t const decoderLength
            = static_cast<int64_t>(
                inputIds.size());

        // ----------------------------------------------------
        // Create execution context
        // ----------------------------------------------------

        auto context
            = std::unique_ptr<
                nvinfer1::IExecutionContext>(
                engine->createExecutionContext());

        if (!context)
        {
            throw std::runtime_error(
                "Failed to create execution context.");
        }

        cudaStream_t stream{nullptr};

        checkCuda(
            cudaStreamCreate(&stream),
            "cudaStreamCreate");

        // ----------------------------------------------------
        // Select profile 0
        // ----------------------------------------------------

        if (!context->setOptimizationProfileAsync(
                0,
                stream))
        {
            throw std::runtime_error(
                "Failed to set optimization profile.");
        }

        // ----------------------------------------------------
        // input_ids shape
        //
        // [1, decoderLength]
        // ----------------------------------------------------

        nvinfer1::Dims2 inputIdsShape{
            static_cast<int32_t>(kBatch),
            static_cast<int32_t>(
                decoderLength)};

        if (!context->setInputShape(
                kInputIdsName,
                inputIdsShape))
        {
            throw std::runtime_error(
                "Failed to set input_ids shape.");
        }

        // ----------------------------------------------------
        // encoder_hidden_states shape
        //
        // [1, 1500, 768]
        // ----------------------------------------------------

        nvinfer1::Dims3 encoderShape{
            static_cast<int32_t>(kBatch),
            static_cast<int32_t>(
                kEncoderFrames),
            static_cast<int32_t>(
                kHiddenSize)};

        if (!context->setInputShape(
                kEncoderHiddenName,
                encoderShape))
        {
            throw std::runtime_error(
                "Failed to set encoder_hidden_states shape.");
        }

        // ----------------------------------------------------
        // Get actual output shape
        //
        // Expected:
        // [1, decoderLength, vocab_size]
        // ----------------------------------------------------

        nvinfer1::Dims logitsShape
            = context->getTensorShape(
                kLogitsName);

        std::cout
            << "\nRuntime logits shape: "
            << dimsToString(
                   logitsShape)
            << '\n';

        if (logitsShape.nbDims != 3)
        {
            throw std::runtime_error(
                "Expected logits rank 3.");
        }

        int64_t const vocabSize
            = logitsShape.d[2];

        if (vocabSize <= 0)
        {
            throw std::runtime_error(
                "Invalid vocabulary size.");
        }

        std::size_t const logitsElements
            = volume(logitsShape);

        // ----------------------------------------------------
        // Allocate GPU buffers
        // ----------------------------------------------------

        void* inputIdsDevice{nullptr};
        void* encoderDevice{nullptr};
        void* logitsDevice{nullptr};

        std::size_t const inputIdsBytes
            = inputIds.size()
            * sizeof(int64_t);

        std::size_t const encoderBytes
            = encoderHidden.size()
            * sizeof(__half);

        nvinfer1::DataType const logitsType
            = engine->getTensorDataType(
                kLogitsName);

        std::size_t logitsElementBytes{0};

        if (logitsType
            == nvinfer1::DataType::kHALF)
        {
            logitsElementBytes
                = sizeof(__half);
        }
        else if (
            logitsType
            == nvinfer1::DataType::kFLOAT)
        {
            logitsElementBytes
                = sizeof(float);
        }
        else
        {
            throw std::runtime_error(
                "Unsupported logits datatype: "
                + std::string(
                    getDataTypeString(
                        logitsType)));
        }

        std::size_t const logitsBytes
            = logitsElements
            * logitsElementBytes;

        checkCuda(
            cudaMalloc(
                &inputIdsDevice,
                inputIdsBytes),
            "cudaMalloc input_ids");

        checkCuda(
            cudaMalloc(
                &encoderDevice,
                encoderBytes),
            "cudaMalloc encoder_hidden_states");

        checkCuda(
            cudaMalloc(
                &logitsDevice,
                logitsBytes),
            "cudaMalloc logits");

        // ----------------------------------------------------
        // Copy inputs CPU -> GPU
        // ----------------------------------------------------

        checkCuda(
            cudaMemcpyAsync(
                inputIdsDevice,
                inputIds.data(),
                inputIdsBytes,
                cudaMemcpyHostToDevice,
                stream),
            "copy input_ids");

        checkCuda(
            cudaMemcpyAsync(
                encoderDevice,
                encoderHidden.data(),
                encoderBytes,
                cudaMemcpyHostToDevice,
                stream),
            "copy encoder_hidden_states");

        // ----------------------------------------------------
        // Bind TensorRT addresses
        // ----------------------------------------------------

        if (!context->setTensorAddress(
                kInputIdsName,
                inputIdsDevice))
        {
            throw std::runtime_error(
                "Failed binding input_ids.");
        }

        if (!context->setTensorAddress(
                kEncoderHiddenName,
                encoderDevice))
        {
            throw std::runtime_error(
                "Failed binding encoder_hidden_states.");
        }

        if (!context->setTensorAddress(
                kLogitsName,
                logitsDevice))
        {
            throw std::runtime_error(
                "Failed binding logits.");
        }

        // ----------------------------------------------------
        // Run TensorRT decoder
        // ----------------------------------------------------

        std::cout
            << "\nRunning Whisper decoder...\n";

        if (!context->enqueueV3(
                stream))
        {
            throw std::runtime_error(
                "Whisper decoder enqueueV3 failed.");
        }

        // ----------------------------------------------------
        // Copy logits GPU -> CPU
        // ----------------------------------------------------

        std::vector<float> logitsFloat(
            logitsElements);

        if (logitsType
            == nvinfer1::DataType::kHALF)
        {
            std::vector<__half> logitsHalf(
                logitsElements);

            checkCuda(
                cudaMemcpyAsync(
                    logitsHalf.data(),
                    logitsDevice,
                    logitsBytes,
                    cudaMemcpyDeviceToHost,
                    stream),
                "copy logits");

            checkCuda(
                cudaStreamSynchronize(
                    stream),
                "cudaStreamSynchronize");

            for (std::size_t i = 0;
                 i < logitsElements;
                 ++i)
            {
                logitsFloat[i]
                    = __half2float(
                        logitsHalf[i]);
            }
        }
        else
        {
            checkCuda(
                cudaMemcpyAsync(
                    logitsFloat.data(),
                    logitsDevice,
                    logitsBytes,
                    cudaMemcpyDeviceToHost,
                    stream),
                "copy logits");

            checkCuda(
                cudaStreamSynchronize(
                    stream),
                "cudaStreamSynchronize");
        }

        // ----------------------------------------------------
        // Greedy token from LAST decoder position
        //
        // logits:
        //
        // [batch, sequence, vocab]
        // ----------------------------------------------------

        std::size_t const lastTokenOffset
            = static_cast<std::size_t>(
                  decoderLength - 1)
            * static_cast<std::size_t>(
                  vocabSize);

        auto begin
            = logitsFloat.begin()
            + lastTokenOffset;

        auto end
            = begin
            + vocabSize;

        auto best
            = std::max_element(
                begin,
                end);

        int64_t const nextToken
            = std::distance(
                begin,
                best);

        float const bestLogit
            = *best;

        std::cout
            << "\n==============================\n"
            << " Decoder Result\n"
            << "==============================\n";

        std::cout
            << "Input token : "
            << kDecoderStartToken
            << '\n';

        std::cout
            << "Vocab size  : "
            << vocabSize
            << '\n';

        std::cout
            << "Next token  : "
            << nextToken
            << '\n';

        std::cout
            << "Best logit  : "
            << bestLogit
            << '\n';

        // ----------------------------------------------------
        // Cleanup
        // ----------------------------------------------------

        cudaFree(inputIdsDevice);
        cudaFree(encoderDevice);
        cudaFree(logitsDevice);

        cudaStreamDestroy(stream);

        std::cout
            << "\nWhisper decoder inference succeeded.\n";

        return EXIT_SUCCESS;
    }
    catch (std::exception const& e)
    {
        std::cerr
            << "\nWhisper decoder inference failed:\n"
            << e.what()
            << '\n';

        return EXIT_FAILURE;
    }
}