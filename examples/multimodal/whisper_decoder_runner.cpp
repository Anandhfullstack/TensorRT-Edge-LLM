#include "whisper_decoder_runner.h"

#include "common/logger.h"
#include "common/trtUtils.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace trt_edgellm
{
namespace examples
{
namespace whisper
{

namespace
{

constexpr char const* kInputIds = "input_ids";
constexpr char const* kEncoderHidden = "encoder_hidden_states";
constexpr char const* kLogits = "logits";

constexpr int64_t kBatch = 1;
constexpr int64_t kEncoderFrames = 1500;
constexpr int64_t kHiddenSize = 768;

constexpr int64_t kDecoderStartToken = 50258;
constexpr int64_t kEosToken = 50257;

constexpr int64_t kMaxDecoderLength = 448;

constexpr std::size_t kEncoderElements
    = kBatch * kEncoderFrames * kHiddenSize;

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

} // namespace

WhisperDecoderRunner::WhisperDecoderRunner(
    std::string enginePath)
    : mEnginePath(std::move(enginePath))
{
}

WhisperDecoderRunner::~WhisperDecoderRunner()
{
    if (mInputIdsDevice)
        cudaFree(mInputIdsDevice);

    if (mEncoderDevice)
        cudaFree(mEncoderDevice);

    if (mLogitsDevice)
        cudaFree(mLogitsDevice);

    if (mStream)
        cudaStreamDestroy(mStream);
}

bool WhisperDecoderRunner::initialize()
{
    try
    {
        if (mContext)
            return true;

        // --------------------------------------------
        // TensorRT runtime
        // --------------------------------------------

        mRuntime.reset(
            nvinfer1::createInferRuntime(
                gLogger));

        if (!mRuntime)
        {
            throw std::runtime_error(
                "Failed to create TensorRT runtime.");
        }

        // --------------------------------------------
        // Load decoder engine
        // --------------------------------------------

        mEngine
            = deserializeCudaEngineFromFile(
                *mRuntime,
                mEnginePath);

        if (!mEngine)
        {
            throw std::runtime_error(
                "Failed to load Whisper decoder engine.");
        }

        std::cout
            << "\n================================\n"
            << " Whisper Decoder Engine\n"
            << "================================\n"
            << printEngineInfo(
                   mEngine.get(),
                   0)
            << '\n';

        // --------------------------------------------
        // Validate bindings
        // --------------------------------------------

        if (!isEngineInput(
                *mEngine,
                kInputIds))
        {
            throw std::runtime_error(
                "input_ids input missing.");
        }

        if (!isEngineInput(
                *mEngine,
                kEncoderHidden))
        {
            throw std::runtime_error(
                "encoder_hidden_states input missing.");
        }

        if (!engineHasOutputTensor(
                mEngine.get(),
                kLogits))
        {
            throw std::runtime_error(
                "logits output missing.");
        }

        mInputIdsType
            = mEngine->getTensorDataType(
                kInputIds);

        mLogitsType
            = mEngine->getTensorDataType(
                kLogits);

        if (mEngine->getTensorDataType(
                kEncoderHidden)
            != nvinfer1::DataType::kHALF)
        {
            throw std::runtime_error(
                "encoder_hidden_states must be FP16.");
        }

        if (mInputIdsType
                != nvinfer1::DataType::kINT64
            && mInputIdsType
                != nvinfer1::DataType::kINT32)
        {
            throw std::runtime_error(
                "input_ids must be INT32 or INT64.");
        }

        if (mLogitsType
                != nvinfer1::DataType::kHALF
            && mLogitsType
                != nvinfer1::DataType::kFLOAT)
        {
            throw std::runtime_error(
                "logits must be FP16 or FP32.");
        }

        // --------------------------------------------
        // Vocabulary size
        // --------------------------------------------

        auto logitsShape
            = mEngine->getTensorShape(
                kLogits);

        if (logitsShape.nbDims != 3
            || logitsShape.d[2] <= 0)
        {
            throw std::runtime_error(
                "Unexpected logits shape: "
                + dimsToString(
                    logitsShape));
        }

        mVocabSize
            = logitsShape.d[2];

        std::cout
            << "Decoder vocab size: "
            << mVocabSize
            << '\n';

        // --------------------------------------------
        // Execution context
        // --------------------------------------------

        mContext.reset(
            mEngine->createExecutionContext());

        if (!mContext)
        {
            throw std::runtime_error(
                "Failed to create decoder context.");
        }

        checkCuda(
            cudaStreamCreate(
                &mStream),
            "cudaStreamCreate");

        if (!mContext->setOptimizationProfileAsync(
                0,
                mStream))
        {
            throw std::runtime_error(
                "Failed selecting decoder profile 0.");
        }

        // --------------------------------------------
        // Allocate maximum buffers
        // --------------------------------------------

        std::size_t inputIdElementSize
            = mInputIdsType
                    == nvinfer1::DataType::kINT64
            ? sizeof(int64_t)
            : sizeof(int32_t);

        std::size_t logitsElementSize
            = mLogitsType
                    == nvinfer1::DataType::kHALF
            ? sizeof(__half)
            : sizeof(float);

        checkCuda(
            cudaMalloc(
                &mInputIdsDevice,
                kMaxDecoderLength
                    * inputIdElementSize),
            "cudaMalloc input_ids");

        checkCuda(
            cudaMalloc(
                &mEncoderDevice,
                kEncoderElements
                    * sizeof(__half)),
            "cudaMalloc encoder");

        checkCuda(
            cudaMalloc(
                &mLogitsDevice,
                kMaxDecoderLength
                    * mVocabSize
                    * logitsElementSize),
            "cudaMalloc logits");

        // Addresses don't change between decode steps.

        if (!mContext->setTensorAddress(
                kInputIds,
                mInputIdsDevice))
        {
            throw std::runtime_error(
                "Failed binding input_ids.");
        }

        if (!mContext->setTensorAddress(
                kEncoderHidden,
                mEncoderDevice))
        {
            throw std::runtime_error(
                "Failed binding encoder states.");
        }

        if (!mContext->setTensorAddress(
                kLogits,
                mLogitsDevice))
        {
            throw std::runtime_error(
                "Failed binding logits.");
        }

        std::cout
            << "Whisper decoder initialized.\n";

        return true;
    }
    catch (std::exception const& e)
    {
        std::cerr
            << "Decoder initialization failed: "
            << e.what()
            << '\n';

        return false;
    }
}

bool WhisperDecoderRunner::generate(
    std::vector<__half> const& encoderHiddenStates,
    std::vector<int64_t>& outputTokens,
    int32_t maxNewTokens)
{
    if (!initialize())
        return false;

    if (encoderHiddenStates.size()
        != kEncoderElements)
    {
        std::cerr
            << "Unexpected encoder output size: "
            << encoderHiddenStates.size()
            << '\n';

        return false;
    }

    try
    {
        // ====================================================
        // Encoder output CPU -> Decoder GPU input
        //
        // This replaces encoder_hidden_states.bin.
        // ====================================================

        checkCuda(
            cudaMemcpyAsync(
                mEncoderDevice,
                encoderHiddenStates.data(),
                encoderHiddenStates.size()
                    * sizeof(__half),
                cudaMemcpyHostToDevice,
                mStream),
            "copy encoder hidden states");

        nvinfer1::Dims3 encoderShape{
            1,
            static_cast<int32_t>(
                kEncoderFrames),
            static_cast<int32_t>(
                kHiddenSize)};

        if (!mContext->setInputShape(
                kEncoderHidden,
                encoderShape))
        {
            throw std::runtime_error(
                "Failed setting encoder shape.");
        }

        // ====================================================
        // Start autoregressive decoding
        // ====================================================

        std::vector<int64_t> decoderTokens{
            kDecoderStartToken};

        outputTokens.clear();

        for (int32_t step = 0;
             step < maxNewTokens;
             ++step)
        {
            int64_t const sequenceLength
                = decoderTokens.size();

            if (sequenceLength
                >= kMaxDecoderLength)
            {
                break;
            }

            // --------------------------------------------
            // Set current decoder sequence shape
            // --------------------------------------------

            nvinfer1::Dims2 idsShape{
                1,
                static_cast<int32_t>(
                    sequenceLength)};

            if (!mContext->setInputShape(
                    kInputIds,
                    idsShape))
            {
                throw std::runtime_error(
                    "Failed setting input_ids shape.");
            }

            // --------------------------------------------
            // Copy input IDs
            // --------------------------------------------

            if (mInputIdsType
                == nvinfer1::DataType::kINT64)
            {
                checkCuda(
                    cudaMemcpyAsync(
                        mInputIdsDevice,
                        decoderTokens.data(),
                        sequenceLength
                            * sizeof(int64_t),
                        cudaMemcpyHostToDevice,
                        mStream),
                    "copy INT64 input_ids");
            }
            else
            {
                std::vector<int32_t> ids32(
                    sequenceLength);

                for (int64_t i = 0;
                     i < sequenceLength;
                     ++i)
                {
                    ids32[i]
                        = static_cast<int32_t>(
                            decoderTokens[i]);
                }

                checkCuda(
                    cudaMemcpyAsync(
                        mInputIdsDevice,
                        ids32.data(),
                        sequenceLength
                            * sizeof(int32_t),
                        cudaMemcpyHostToDevice,
                        mStream),
                    "copy INT32 input_ids");
            }

            // --------------------------------------------
            // Resolve output shape
            // --------------------------------------------

            auto outputShape
                = mContext->getTensorShape(
                    kLogits);

            if (outputShape.nbDims != 3
                || outputShape.d[1]
                    != sequenceLength)
            {
                throw std::runtime_error(
                    "Unexpected runtime logits shape: "
                    + dimsToString(
                        outputShape));
            }

            // --------------------------------------------
            // Decoder inference
            // --------------------------------------------

            if (!mContext->enqueueV3(
                    mStream))
            {
                throw std::runtime_error(
                    "Decoder enqueueV3 failed.");
            }

            // We only need logits corresponding to
            // the LAST decoder token.

            std::size_t const offset
                = static_cast<std::size_t>(
                      sequenceLength - 1)
                * static_cast<std::size_t>(
                      mVocabSize);

            int64_t nextToken{-1};

            // --------------------------------------------
            // FP16 logits
            // --------------------------------------------

            if (mLogitsType
                == nvinfer1::DataType::kHALF)
            {
                std::vector<__half> hostLogits(
                    mVocabSize);

                auto* source
                    = static_cast<__half*>(
                          mLogitsDevice)
                    + offset;

                checkCuda(
                    cudaMemcpyAsync(
                        hostLogits.data(),
                        source,
                        mVocabSize
                            * sizeof(__half),
                        cudaMemcpyDeviceToHost,
                        mStream),
                    "copy logits");

                checkCuda(
                    cudaStreamSynchronize(
                        mStream),
                    "decoder sync");

                float bestValue
                    = -1e30F;

                for (int64_t token = 0;
                     token < mVocabSize;
                     ++token)
                {
                    float const value
                        = __half2float(
                            hostLogits[token]);

                    if (value > bestValue)
                    {
                        bestValue = value;
                        nextToken = token;
                    }
                }
            }

            // --------------------------------------------
            // FP32 logits
            // --------------------------------------------

            else
            {
                std::vector<float> hostLogits(
                    mVocabSize);

                auto* source
                    = static_cast<float*>(
                          mLogitsDevice)
                    + offset;

                checkCuda(
                    cudaMemcpyAsync(
                        hostLogits.data(),
                        source,
                        mVocabSize
                            * sizeof(float),
                        cudaMemcpyDeviceToHost,
                        mStream),
                    "copy logits");

                checkCuda(
                    cudaStreamSynchronize(
                        mStream),
                    "decoder sync");

                auto best
                    = std::max_element(
                        hostLogits.begin(),
                        hostLogits.end());

                nextToken
                    = std::distance(
                        hostLogits.begin(),
                        best);
            }

            // --------------------------------------------
            // Append generated token
            // --------------------------------------------

            decoderTokens.push_back(
                nextToken);

            outputTokens.push_back(
                nextToken);

            std::cout
                << "step "
                << step
                << " -> token "
                << nextToken
                << '\n';

            if (nextToken == kEosToken)
            {
                std::cout
                    << "EOS reached.\n";

                break;
            }
        }

        return true;
    }
    catch (std::exception const& e)
    {
        std::cerr
            << "Decoder generation failed: "
            << e.what()
            << '\n';

        return false;
    }
}

} // namespace whisper
} // namespace examples
} // namespace trt_edgellm