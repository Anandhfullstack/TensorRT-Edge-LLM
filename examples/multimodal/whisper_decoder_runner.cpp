#include "whisper_decoder_runner.h"

#include "common/logger.h"
#include "common/trtUtils.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <utility>


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
constexpr char const* kPositionIds = "position_ids";
constexpr char const* kPastKeyValues = "past_key_values";
constexpr char const* kPresentKeyValues = "present_key_values";

constexpr char const* kCrossKeyValues
    = "cross_key_values";

constexpr int64_t kBatch = 1;
constexpr int64_t kEncoderFrames = 1500;
constexpr int64_t kHiddenSize = 768;
constexpr int64_t kDecoderStartToken = 50258;
constexpr int64_t kEosToken = 50257;
constexpr int64_t kMaxDecoderLength = 448;

constexpr int64_t kDecoderLayers = 12;
constexpr int64_t kKeyValueCount = 2;
constexpr int64_t kAttentionHeads = 12;
constexpr int64_t kHeadDimension = 64;

constexpr std::size_t kMaxCacheElements
    = static_cast<std::size_t>(kDecoderLayers)
    * static_cast<std::size_t>(kKeyValueCount)
    * static_cast<std::size_t>(kBatch)
    * static_cast<std::size_t>(kAttentionHeads)
    * static_cast<std::size_t>(kMaxDecoderLength)
    * static_cast<std::size_t>(kHeadDimension);

// Cross-attention K/V are projected once from the 1500-frame encoder output and
// never grow with decode position: [12, 2, 1, 12, 1500, 64].
constexpr std::size_t kCrossCacheElements
    = static_cast<std::size_t>(kDecoderLayers)
    * static_cast<std::size_t>(kKeyValueCount)
    * static_cast<std::size_t>(kBatch)
    * static_cast<std::size_t>(kAttentionHeads)
    * static_cast<std::size_t>(kEncoderFrames)
    * static_cast<std::size_t>(kHeadDimension);

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

bool hasDynamicDims(
    nvinfer1::Dims const& dims) noexcept
{
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] < 0)
        {
            return true;
        }
    }

    return false;
}

//! ``setInputShape`` is only meaningful for inputs the engine declares
//! dynamic; a fully static input is already bound to its build-time shape.
void setInputShapeIfDynamic(
    nvinfer1::ICudaEngine const& engine,
    nvinfer1::IExecutionContext& context,
    char const* tensorName,
    nvinfer1::Dims const& shape)
{
    if (!hasDynamicDims(engine.getTensorShape(tensorName)))
    {
        return;
    }

    if (!context.setInputShape(tensorName, shape))
    {
        throw std::runtime_error(
            "Failed setting shape for "
            + std::string(tensorName)
            + ": "
            + dimsToString(shape));
    }
}

nvinfer1::Dims makeCrossCacheShape() noexcept
{
    nvinfer1::Dims shape{};
    shape.nbDims = 6;
    shape.d[0] = static_cast<int32_t>(kDecoderLayers);
    shape.d[1] = static_cast<int32_t>(kKeyValueCount);
    shape.d[2] = static_cast<int32_t>(kBatch);
    shape.d[3] = static_cast<int32_t>(kAttentionHeads);
    shape.d[4] = static_cast<int32_t>(kEncoderFrames);
    shape.d[5] = static_cast<int32_t>(kHeadDimension);
    return shape;
}

} // namespace

WhisperDecoderRunner::WhisperDecoderRunner(
    std::string crossKvEnginePath,
    std::string decoderEnginePath)
    : mCrossKvEnginePath(
          std::move(crossKvEnginePath))
    , mDecoderEnginePath(
          std::move(decoderEnginePath))
{
}

WhisperDecoderRunner::~WhisperDecoderRunner()
{
    if (mInputIdsDevice)
    {
        cudaFree(mInputIdsDevice);
    }

    if (mPositionIdsDevice)
    {
        cudaFree(mPositionIdsDevice);
    }

    if (mEncoderDevice)
    {
        cudaFree(mEncoderDevice);
    }

    if (mCrossKeyValuesDevice)
    {
        cudaFree(mCrossKeyValuesDevice);
    }

    if (mLogitsDevice)
    {
        cudaFree(mLogitsDevice);
    }

    if (mPastKeyValuesDevice)
    {
        cudaFree(mPastKeyValuesDevice);
    }

    if (mPresentKeyValuesDevice)
    {
        cudaFree(mPresentKeyValuesDevice);
    }

    if (mStream)
    {
        cudaStreamDestroy(mStream);
    }
}

bool WhisperDecoderRunner::initialize()
{
    try
    {
        if (mCrossKvContext && mDecoderContext)
        {
            return true;
        }

        // ------------------------------------------------------------
        // TensorRT runtime
        // ------------------------------------------------------------

        mRuntime.reset(
            nvinfer1::createInferRuntime(gLogger));

        if (!mRuntime)
        {
            throw std::runtime_error(
                "Failed to create TensorRT runtime");
        }

        // ------------------------------------------------------------
        // Load both engines
        // ------------------------------------------------------------

        mCrossKvEngine
            = deserializeCudaEngineFromFile(
                *mRuntime,
                mCrossKvEnginePath);

        if (!mCrossKvEngine)
        {
            throw std::runtime_error(
                "Failed to load cross-KV engine: "
                + mCrossKvEnginePath);
        }

        mDecoderEngine
            = deserializeCudaEngineFromFile(
                *mRuntime,
                mDecoderEnginePath);

        if (!mDecoderEngine)
        {
            throw std::runtime_error(
                "Failed to load decoder engine: "
                + mDecoderEnginePath);
        }

        // ------------------------------------------------------------
        // Validate cross-KV engine interface
        // ------------------------------------------------------------

        if (!isEngineInput(
                *mCrossKvEngine,
                kEncoderHidden))
        {
            throw std::runtime_error(
                "Cross-KV engine is missing input tensor: "
                + std::string(kEncoderHidden));
        }

        if (!engineHasOutputTensor(
                mCrossKvEngine.get(),
                kCrossKeyValues))
        {
            throw std::runtime_error(
                "Cross-KV engine is missing output tensor: "
                + std::string(kCrossKeyValues));
        }

        // ------------------------------------------------------------
        // Validate decoder engine interface
        // ------------------------------------------------------------

        if (!isEngineInput(
                *mDecoderEngine,
                kInputIds))
        {
            throw std::runtime_error(
                "Decoder engine is missing input tensor: "
                + std::string(kInputIds));
        }

        if (!isEngineInput(
                *mDecoderEngine,
                kPositionIds))
        {
            throw std::runtime_error(
                "Decoder engine is missing input tensor: "
                + std::string(kPositionIds));
        }

        if (!isEngineInput(
                *mDecoderEngine,
                kPastKeyValues))
        {
            throw std::runtime_error(
                "Decoder engine is missing input tensor: "
                + std::string(kPastKeyValues));
        }

        if (!isEngineInput(
                *mDecoderEngine,
                kCrossKeyValues))
        {
            throw std::runtime_error(
                "Decoder engine is missing input tensor: "
                + std::string(kCrossKeyValues));
        }

        if (!engineHasOutputTensor(
                mDecoderEngine.get(),
                kLogits))
        {
            throw std::runtime_error(
                "Decoder engine is missing output tensor: "
                + std::string(kLogits));
        }

        if (!engineHasOutputTensor(
                mDecoderEngine.get(),
                kPresentKeyValues))
        {
            throw std::runtime_error(
                "Decoder engine is missing output tensor: "
                + std::string(kPresentKeyValues));
        }

        // ------------------------------------------------------------
        // Read tensor data types
        // ------------------------------------------------------------

        auto const encoderInputType
            = mCrossKvEngine->getTensorDataType(
                kEncoderHidden);

        mCrossKeyValuesType
            = mCrossKvEngine->getTensorDataType(
                kCrossKeyValues);

        auto const decoderCrossKeyValuesType
            = mDecoderEngine->getTensorDataType(
                kCrossKeyValues);

        mInputIdsType
            = mDecoderEngine->getTensorDataType(
                kInputIds);

        mPositionIdsType
            = mDecoderEngine->getTensorDataType(
                kPositionIds);

        mLogitsType
            = mDecoderEngine->getTensorDataType(
                kLogits);

        mCacheType
            = mDecoderEngine->getTensorDataType(
                kPastKeyValues);

        auto const presentCacheType
            = mDecoderEngine->getTensorDataType(
                kPresentKeyValues);

        // ------------------------------------------------------------
        // Validate tensor data types
        // ------------------------------------------------------------

        if (encoderInputType
            != nvinfer1::DataType::kHALF)
        {
            throw std::runtime_error(
                "Cross-KV encoder input must be FP16");
        }

        if (mCrossKeyValuesType
            != nvinfer1::DataType::kHALF)
        {
            throw std::runtime_error(
                "Cross-KV engine output must be FP16");
        }

        if (decoderCrossKeyValuesType
            != mCrossKeyValuesType)
        {
            throw std::runtime_error(
                "Cross-KV output type does not match "
                "decoder cross_key_values input type");
        }

        if (mInputIdsType
                != nvinfer1::DataType::kINT32
            && mInputIdsType
                != nvinfer1::DataType::kINT64)
        {
            throw std::runtime_error(
                "input_ids must be INT32 or INT64");
        }

        if (mPositionIdsType
                != nvinfer1::DataType::kINT32
            && mPositionIdsType
                != nvinfer1::DataType::kINT64)
        {
            throw std::runtime_error(
                "position_ids must be INT32 or INT64");
        }

        if (mLogitsType
                != nvinfer1::DataType::kHALF
            && mLogitsType
                != nvinfer1::DataType::kFLOAT)
        {
            throw std::runtime_error(
                "logits must be FP16 or FP32");
        }

        if (mCacheType
                != nvinfer1::DataType::kHALF
            && mCacheType
                != nvinfer1::DataType::kFLOAT)
        {
            throw std::runtime_error(
                "Self-attention KV cache must be FP16 or FP32");
        }

        if (presentCacheType != mCacheType)
        {
            throw std::runtime_error(
                "past_key_values and present_key_values "
                "must have the same data type");
        }

        // ------------------------------------------------------------
        // Validate cross-attention cache shapes
        // ------------------------------------------------------------

        auto const projectorCrossShape
            = mCrossKvEngine->getTensorShape(
                kCrossKeyValues);

        auto const decoderCrossShape
            = mDecoderEngine->getTensorShape(
                kCrossKeyValues);

        auto validateCrossShape =
            [](nvinfer1::Dims const& shape,
               std::string const& tensorDescription)
            {
                if (shape.nbDims != 6)
                {
                    throw std::runtime_error(
                        tensorDescription
                        + " must have 6 dimensions, got "
                        + dimsToString(shape));
                }

                if (shape.d[0] != kDecoderLayers
                    || shape.d[1] != kKeyValueCount
                    || shape.d[3] != kAttentionHeads
                    || shape.d[4] != kEncoderFrames
                    || shape.d[5] != kHeadDimension)
                {
                    throw std::runtime_error(
                        "Unexpected "
                        + tensorDescription
                        + " shape: "
                        + dimsToString(shape));
                }

                // Batch dimension may be either dynamic (-1)
                // or statically fixed to one.
                if (shape.d[2] != -1
                    && shape.d[2] != kBatch)
                {
                    throw std::runtime_error(
                        "Unexpected batch dimension in "
                        + tensorDescription
                        + ": "
                        + dimsToString(shape));
                }
            };

        validateCrossShape(
            projectorCrossShape,
            "cross-KV engine output");

        validateCrossShape(
            decoderCrossShape,
            "decoder cross_key_values input");

        // ------------------------------------------------------------
        // Read vocabulary size
        // ------------------------------------------------------------

        auto const logitsShape
            = mDecoderEngine->getTensorShape(
                kLogits);

        if (logitsShape.nbDims != 3
            || logitsShape.d[2] <= 0)
        {
            throw std::runtime_error(
                "Unexpected logits shape: "
                + dimsToString(logitsShape));
        }

        mVocabSize = logitsShape.d[2];

        // ------------------------------------------------------------
        // Create execution contexts
        // ------------------------------------------------------------

        mCrossKvContext.reset(
            mCrossKvEngine->createExecutionContext());

        if (!mCrossKvContext)
        {
            throw std::runtime_error(
                "Failed to create cross-KV execution context");
        }

        mDecoderContext.reset(
            mDecoderEngine->createExecutionContext());

        if (!mDecoderContext)
        {
            throw std::runtime_error(
                "Failed to create decoder execution context");
        }

        // ------------------------------------------------------------
        // CUDA stream
        // ------------------------------------------------------------

        checkCuda(
            cudaStreamCreate(&mStream),
            "create decoder CUDA stream");

        if (!mCrossKvContext->setOptimizationProfileAsync(
                0,
                mStream))
        {
            throw std::runtime_error(
                "Failed to select cross-KV optimization profile");
        }

        if (!mDecoderContext->setOptimizationProfileAsync(
                0,
                mStream))
        {
            throw std::runtime_error(
                "Failed to select decoder optimization profile");
        }

        // ------------------------------------------------------------
        // Buffer sizes
        // ------------------------------------------------------------

        std::size_t const inputIdElementSize
            = mInputIdsType
                    == nvinfer1::DataType::kINT64
            ? sizeof(int64_t)
            : sizeof(int32_t);

        std::size_t const positionElementSize
            = mPositionIdsType
                    == nvinfer1::DataType::kINT64
            ? sizeof(int64_t)
            : sizeof(int32_t);

        std::size_t const logitsElementSize
            = mLogitsType
                    == nvinfer1::DataType::kHALF
            ? sizeof(__half)
            : sizeof(float);

        std::size_t const cacheElementSize
            = mCacheType
                    == nvinfer1::DataType::kHALF
            ? sizeof(__half)
            : sizeof(float);

        std::size_t const crossCacheElementSize
            = mCrossKeyValuesType
                    == nvinfer1::DataType::kHALF
            ? sizeof(__half)
            : sizeof(float);

        std::size_t const inputIdsBytes
            = static_cast<std::size_t>(
                  kMaxDecoderLength)
            * inputIdElementSize;

        std::size_t const positionIdsBytes
            = static_cast<std::size_t>(
                  kMaxDecoderLength)
            * positionElementSize;

        std::size_t const encoderBytes
            = kEncoderElements
            * sizeof(__half);

        std::size_t const crossCacheBytes
            = kCrossCacheElements
            * crossCacheElementSize;

        std::size_t const logitsBytes
            = static_cast<std::size_t>(
                  kMaxDecoderLength)
            * static_cast<std::size_t>(
                  mVocabSize)
            * logitsElementSize;

        std::size_t const selfCacheBytes
            = kMaxCacheElements
            * cacheElementSize;

        // ------------------------------------------------------------
        // Allocate device buffers
        // ------------------------------------------------------------

        checkCuda(
            cudaMalloc(
                &mInputIdsDevice,
                inputIdsBytes),
            "allocate input_ids");

        checkCuda(
            cudaMalloc(
                &mPositionIdsDevice,
                positionIdsBytes),
            "allocate position_ids");

        checkCuda(
            cudaMalloc(
                &mEncoderDevice,
                encoderBytes),
            "allocate encoder_hidden_states");

        checkCuda(
            cudaMalloc(
                &mCrossKeyValuesDevice,
                crossCacheBytes),
            "allocate cross_key_values");

        checkCuda(
            cudaMalloc(
                &mLogitsDevice,
                logitsBytes),
            "allocate logits");

        checkCuda(
            cudaMalloc(
                &mPastKeyValuesDevice,
                selfCacheBytes),
            "allocate past_key_values");

        checkCuda(
            cudaMalloc(
                &mPresentKeyValuesDevice,
                selfCacheBytes),
            "allocate present_key_values");

        checkCuda(
            cudaMemsetAsync(
                mPastKeyValuesDevice,
                0,
                selfCacheBytes,
                mStream),
            "clear past_key_values");

        checkCuda(
            cudaMemsetAsync(
                mPresentKeyValuesDevice,
                0,
                selfCacheBytes,
                mStream),
            "clear present_key_values");

        // ------------------------------------------------------------
        // Cross-KV engine bindings
        //
        // encoder_hidden_states -> cross_key_values
        // ------------------------------------------------------------

        if (!mCrossKvContext->setTensorAddress(
                kEncoderHidden,
                mEncoderDevice))
        {
            throw std::runtime_error(
                "Failed to bind cross-KV encoder input");
        }

        if (!mCrossKvContext->setTensorAddress(
                kCrossKeyValues,
                mCrossKeyValuesDevice))
        {
            throw std::runtime_error(
                "Failed to bind cross-KV output");
        }

        // ------------------------------------------------------------
        // Decoder engine bindings
        //
        // cross_key_values points directly at the projector output.
        // No GPU-to-CPU-to-GPU copy is required.
        // ------------------------------------------------------------

        if (!mDecoderContext->setTensorAddress(
                kInputIds,
                mInputIdsDevice))
        {
            throw std::runtime_error(
                "Failed to bind decoder input_ids");
        }

        if (!mDecoderContext->setTensorAddress(
                kPositionIds,
                mPositionIdsDevice))
        {
            throw std::runtime_error(
                "Failed to bind decoder position_ids");
        }

        if (!mDecoderContext->setTensorAddress(
                kPastKeyValues,
                mPastKeyValuesDevice))
        {
            throw std::runtime_error(
                "Failed to bind decoder past_key_values");
        }

        if (!mDecoderContext->setTensorAddress(
                kCrossKeyValues,
                mCrossKeyValuesDevice))
        {
            throw std::runtime_error(
                "Failed to bind decoder cross_key_values");
        }

        if (!mDecoderContext->setTensorAddress(
                kLogits,
                mLogitsDevice))
        {
            throw std::runtime_error(
                "Failed to bind decoder logits");
        }

        if (!mDecoderContext->setTensorAddress(
                kPresentKeyValues,
                mPresentKeyValuesDevice))
        {
            throw std::runtime_error(
                "Failed to bind decoder present_key_values");
        }

        checkCuda(
            cudaStreamSynchronize(mStream),
            "initialize decoder buffers");

        std::cout
            << "Whisper decoder initialized:\n"
            << "  Cross-KV engine: "
            << mCrossKvEnginePath
            << '\n'
            << "  Decoder engine: "
            << mDecoderEnginePath
            << '\n'
            << "  Vocabulary size: "
            << mVocabSize
            << '\n'
            << "  Cross-KV cache: "
            << crossCacheBytes
                / (1024.0 * 1024.0)
            << " MiB\n";

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
    {
        return false;
    }

    if (encoderHiddenStates.size() != kEncoderElements)
    {
        std::cerr
            << "Unexpected encoder output size: "
            << encoderHiddenStates.size()
            << '\n';

        return false;
    }

    try
    {
        // ----------------------------------------------------
        // Copy encoder output to the cross-KV engine input
        // ----------------------------------------------------

        checkCuda(
            cudaMemcpyAsync(
                mEncoderDevice,
                encoderHiddenStates.data(),
                encoderHiddenStates.size() * sizeof(__half),
                cudaMemcpyHostToDevice,
                mStream),
            "copy encoder hidden states");

        nvinfer1::Dims encoderShape{};
        encoderShape.nbDims = 3;
        encoderShape.d[0] = static_cast<int32_t>(kBatch);
        encoderShape.d[1] = static_cast<int32_t>(kEncoderFrames);
        encoderShape.d[2] = static_cast<int32_t>(kHiddenSize);

        // ----------------------------------------------------
        // Cross-attention projection, run once per utterance
        //
        // encoder_hidden_states -> cross_key_values
        //
        // The result stays on the device for the whole decode
        // loop: cross K/V do not depend on decode position.
        // ----------------------------------------------------

        setInputShapeIfDynamic(
            *mCrossKvEngine,
            *mCrossKvContext,
            kEncoderHidden,
            encoderShape);

        if (!mCrossKvContext->enqueueV3(mStream))
        {
            throw std::runtime_error(
                "Cross-KV enqueueV3 failed.");
        }

        // The decoder reads cross_key_values from the same
        // stream, so the projection above is already ordered
        // ahead of every decode step.
        nvinfer1::Dims const crossCacheShape
            = makeCrossCacheShape();

        setInputShapeIfDynamic(
            *mDecoderEngine,
            *mDecoderContext,
            kCrossKeyValues,
            crossCacheShape);

        outputTokens.clear();

        // First input is Whisper's decoder start token.
        int64_t currentToken = kDecoderStartToken;

        // Number of tokens already stored in the cache.
        int64_t pastLength = 0;

        for (int32_t step = 0;
             step < maxNewTokens;
             ++step)
        {
            if (pastLength >= kMaxDecoderLength)
            {
                break;
            }

            // Every cached decoding step processes one token.
            nvinfer1::Dims2 tokenShape{1, 1};
            nvinfer1::Dims2 positionShape{1, 1};

            // Cache:
            // [layers, K/V, batch, heads, past_length, head_dim]
            nvinfer1::Dims pastCacheShape{};
            pastCacheShape.nbDims = 6;
            pastCacheShape.d[0]
                = static_cast<int32_t>(kDecoderLayers);
            pastCacheShape.d[1]
                = static_cast<int32_t>(kKeyValueCount);
            pastCacheShape.d[2]
                = static_cast<int32_t>(kBatch);
            pastCacheShape.d[3]
                = static_cast<int32_t>(kAttentionHeads);
            pastCacheShape.d[4]
                = static_cast<int32_t>(pastLength);
            pastCacheShape.d[5]
                = static_cast<int32_t>(kHeadDimension);

            if (!mDecoderContext->setInputShape(
                    kInputIds,
                    tokenShape))
            {
                throw std::runtime_error(
                    "Failed setting input_ids shape.");
            }

            if (!mDecoderContext->setInputShape(
                    kPositionIds,
                    positionShape))
            {
                throw std::runtime_error(
                    "Failed setting position_ids shape.");
            }

            if (!mDecoderContext->setInputShape(
                    kPastKeyValues,
                    pastCacheShape))
            {
                throw std::runtime_error(
                    "Failed setting past cache shape.");
            }

            // ------------------------------------------------
            // Copy the current token
            // ------------------------------------------------

            if (mInputIdsType
                == nvinfer1::DataType::kINT64)
            {
                checkCuda(
                    cudaMemcpyAsync(
                        mInputIdsDevice,
                        &currentToken,
                        sizeof(int64_t),
                        cudaMemcpyHostToDevice,
                        mStream),
                    "copy INT64 input token");
            }
            else
            {
                int32_t const token32
                    = static_cast<int32_t>(currentToken);

                checkCuda(
                    cudaMemcpyAsync(
                        mInputIdsDevice,
                        &token32,
                        sizeof(int32_t),
                        cudaMemcpyHostToDevice,
                        mStream),
                    "copy INT32 input token");
            }

            // Position equals the number of cached tokens.
            if (mPositionIdsType
                == nvinfer1::DataType::kINT64)
            {
                int64_t const position = pastLength;

                checkCuda(
                    cudaMemcpyAsync(
                        mPositionIdsDevice,
                        &position,
                        sizeof(int64_t),
                        cudaMemcpyHostToDevice,
                        mStream),
                    "copy INT64 position_id");
            }
            else
            {
                int32_t const position
                    = static_cast<int32_t>(pastLength);

                checkCuda(
                    cudaMemcpyAsync(
                        mPositionIdsDevice,
                        &position,
                        sizeof(int32_t),
                        cudaMemcpyHostToDevice,
                        mStream),
                    "copy INT32 position_id");
            }

            // Cache addresses change whenever the two cache
            // buffers are swapped.
            if (!mDecoderContext->setTensorAddress(
                    kPastKeyValues,
                    mPastKeyValuesDevice))
            {
                throw std::runtime_error(
                    "Failed binding past cache.");
            }

            if (!mDecoderContext->setTensorAddress(
                    kPresentKeyValues,
                    mPresentKeyValuesDevice))
            {
                throw std::runtime_error(
                    "Failed binding present cache.");
            }

            // ------------------------------------------------
            // Validate runtime output shapes
            // ------------------------------------------------

            auto const logitsShape
                = mDecoderContext->getTensorShape(kLogits);

            if (logitsShape.nbDims != 3
                || logitsShape.d[0] != 1
                || logitsShape.d[1] != 1
                || logitsShape.d[2] != mVocabSize)
            {
                throw std::runtime_error(
                    "Unexpected logits shape: "
                    + dimsToString(logitsShape));
            }

            auto const presentCacheShape
                = mDecoderContext->getTensorShape(
                    kPresentKeyValues);

            if (presentCacheShape.nbDims != 6
                || presentCacheShape.d[4]
                    != pastLength + 1)
            {
                throw std::runtime_error(
                    "Unexpected present cache shape: "
                    + dimsToString(presentCacheShape));
            }

            // ------------------------------------------------
            // Run TensorRT decoder
            // ------------------------------------------------

            if (!mDecoderContext->enqueueV3(mStream))
            {
                throw std::runtime_error(
                    "Decoder enqueueV3 failed.");
            }

            int64_t nextToken{-1};

            // Only one token was processed, so logits begin
            // at offset zero.
            if (mLogitsType
                == nvinfer1::DataType::kHALF)
            {
                std::vector<__half> hostLogits(
                    static_cast<std::size_t>(mVocabSize));

                checkCuda(
                    cudaMemcpyAsync(
                        hostLogits.data(),
                        mLogitsDevice,
                        hostLogits.size() * sizeof(__half),
                        cudaMemcpyDeviceToHost,
                        mStream),
                    "copy FP16 logits");

                checkCuda(
                    cudaStreamSynchronize(mStream),
                    "synchronize decoder stream");

                auto const maximum
                    = std::max_element(
                        hostLogits.begin(),
                        hostLogits.end(),
                        [](__half left, __half right)
                        {
                            return __half2float(left)
                                < __half2float(right);
                        });

                nextToken = std::distance(
                    hostLogits.begin(),
                    maximum);
            }
            else
            {
                std::vector<float> hostLogits(
                    static_cast<std::size_t>(mVocabSize));

                checkCuda(
                    cudaMemcpyAsync(
                        hostLogits.data(),
                        mLogitsDevice,
                        hostLogits.size() * sizeof(float),
                        cudaMemcpyDeviceToHost,
                        mStream),
                    "copy FP32 logits");

                checkCuda(
                    cudaStreamSynchronize(mStream),
                    "synchronize decoder stream");

                auto const maximum
                    = std::max_element(
                        hostLogits.begin(),
                        hostLogits.end());

                nextToken = std::distance(
                    hostLogits.begin(),
                    maximum);
            }

            // present_key_values now contains:
            // old cache + current token's K/V.
            //
            // Swap it into the past-cache position so it
            // becomes the input on the next iteration.
            std::swap(
                mPastKeyValuesDevice,
                mPresentKeyValuesDevice);

            ++pastLength;

            

            if (nextToken == kEosToken)
            {
                std::cout << "EOS reached.\n";
                break;
            }
            outputTokens.push_back(nextToken);
            currentToken = nextToken;
        }

        return true;
    }
    catch (std::exception const& error)
    {
        std::cerr
            << "Decoder generation failed: "
            << error.what()
            << '\n';

        return false;
    }
}

} // namespace whisper
} // namespace examples
} // namespace trt_edgellm