#include "whisper_decoder_runner.h"

#include "common/logger.h"
#include "common/cudaUtils.h"
#include "common/trtUtils.h"

#include <algorithm>
#include <cstdlib>
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

constexpr char const* kCachePosition
    = "cache_position";

constexpr int64_t kBatch = 1;
constexpr int64_t kEncoderFrames = 1500;
constexpr int64_t kHiddenSize = 768;
constexpr int64_t kDecoderStartToken = 50258;
constexpr int64_t kEosToken = 50257;

//! Whisper's forced decoder prompt. Feeding all four in one prefill both pins
//! language/task (greedy decoding is free to drift otherwise) and collapses
//! three decode steps into one.
constexpr int64_t kLanguageEnToken = 50259;
constexpr int64_t kTranscribeToken = 50359;
constexpr int64_t kNoTimestampsToken = 50363;

constexpr int64_t kPromptTokens[]
    = {kDecoderStartToken, kLanguageEnToken, kTranscribeToken, kNoTimestampsToken};

constexpr int32_t kPromptLength
    = static_cast<int32_t>(sizeof(kPromptTokens) / sizeof(kPromptTokens[0]));
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

nvinfer1::Dims makeSelfCacheShape() noexcept
{
    nvinfer1::Dims shape{};
    shape.nbDims = 6;
    shape.d[0] = static_cast<int32_t>(kDecoderLayers);
    shape.d[1] = static_cast<int32_t>(kKeyValueCount);
    shape.d[2] = static_cast<int32_t>(kBatch);
    shape.d[3] = static_cast<int32_t>(kAttentionHeads);
    shape.d[4] = static_cast<int32_t>(kMaxDecoderLength);
    shape.d[5] = static_cast<int32_t>(kHeadDimension);
    return shape;
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

    if (mCachePositionDevice)
    {
        cudaFree(mCachePositionDevice);
    }

    if (mSelfKeyValuesDevice)
    {
        cudaFree(mSelfKeyValuesDevice);
    }

    if (mInputIdsHost)
    {
        cudaFreeHost(mInputIdsHost);
    }

    if (mPositionIdsHost)
    {
        cudaFreeHost(mPositionIdsHost);
    }

    if (mCachePositionHost)
    {
        cudaFreeHost(mCachePositionHost);
    }

    if (mLogitsHost)
    {
        cudaFreeHost(mLogitsHost);
    }

    if (mDecodeGraphExec)
    {
        cudaGraphExecDestroy(mDecodeGraphExec);
    }

    if (mDecodeGraph)
    {
        cudaGraphDestroy(mDecodeGraph);
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

        if (!isEngineInput(
                *mDecoderEngine,
                kCachePosition))
        {
            throw std::runtime_error(
                "Decoder engine is missing input tensor: "
                + std::string(kCachePosition)
                + ". This runner requires a fixed-capacity KV cache engine "
                  "(past_key_values statically shaped with a cache_position "
                  "scatter index); a growing-concat engine will not work.");
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

        // The self cache must be fully static: that is what lets past and
        // present share one allocation and keeps the decode step capturable.
        auto const selfCacheShape
            = mDecoderEngine->getTensorShape(kPastKeyValues);

        nvinfer1::Dims const expectedSelfShape
            = makeSelfCacheShape();

        if (selfCacheShape.nbDims != expectedSelfShape.nbDims)
        {
            throw std::runtime_error(
                "Unexpected past_key_values rank: "
                + dimsToString(selfCacheShape));
        }

        for (int32_t i = 0; i < expectedSelfShape.nbDims; ++i)
        {
            if (selfCacheShape.d[i] != expectedSelfShape.d[i])
            {
                throw std::runtime_error(
                    "past_key_values must be statically shaped "
                    + dimsToString(expectedSelfShape)
                    + ", got "
                    + dimsToString(selfCacheShape));
            }
        }

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

        std::size_t const cachePositionBytes
            = static_cast<std::size_t>(
                  kMaxDecoderLength)
            * sizeof(int64_t);

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
                &mCachePositionDevice,
                cachePositionBytes),
            "allocate cache_position");

        checkCuda(
            cudaMalloc(
                &mSelfKeyValuesDevice,
                selfCacheBytes),
            "allocate self-attention cache");

        checkCuda(
            cudaMemsetAsync(
                mSelfKeyValuesDevice,
                0,
                selfCacheBytes,
                mStream),
            "clear self-attention cache");

        checkCuda(
            cudaMallocHost(
                reinterpret_cast<void**>(&mInputIdsHost),
                cachePositionBytes),
            "allocate pinned input_ids staging");

        checkCuda(
            cudaMallocHost(
                reinterpret_cast<void**>(&mPositionIdsHost),
                cachePositionBytes),
            "allocate pinned position_ids staging");

        checkCuda(
            cudaMallocHost(
                reinterpret_cast<void**>(&mCachePositionHost),
                cachePositionBytes),
            "allocate pinned cache_position staging");

        checkCuda(
            cudaMallocHost(
                &mLogitsHost,
                static_cast<std::size_t>(mVocabSize)
                    * logitsElementSize),
            "allocate pinned logits staging");

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
                kCachePosition,
                mCachePositionDevice))
        {
            throw std::runtime_error(
                "Failed to bind decoder cache_position");
        }

        // Past and present alias one allocation: the scatter writes each step's
        // K/V into slot `cache_position` and leaves every other slot untouched,
        // so there is nothing to ping-pong.
        if (!mDecoderContext->setTensorAddress(
                kPastKeyValues,
                mSelfKeyValuesDevice))
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
                mSelfKeyValuesDevice))
        {
            throw std::runtime_error(
                "Failed to bind decoder present_key_values");
        }

        checkCuda(
            cudaStreamSynchronize(mStream),
            "initialize decoder buffers");

        // Capture the 1-token decode step now that every binding address is
        // final. Non-fatal: a failure just leaves every step on enqueueV3.
        mDecodeGraphReady = captureDecodeGraph();

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
            << " MiB\n"
            << "  Self-KV cache: "
            << selfCacheBytes
                / (1024.0 * 1024.0)
            << " MiB (fixed, in-place)\n"
            << "  Decode CUDA graph: "
            << (mDecodeGraphReady ? "captured" : "unavailable (enqueueV3)")
            << '\n';

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
bool WhisperDecoderRunner::captureDecodeGraph()
{
    // Escape hatch for bisecting graph-related issues.
    if (std::getenv("WHISPER_DISABLE_CUDA_GRAPH") != nullptr)
    {
        LOG_INFO("WHISPER_DISABLE_CUDA_GRAPH set; using enqueueV3");
        return false;
    }

    // Steps 2 + 3: settle the decode-time shapes against the already-bound
    // addresses. Every binding is fixed for the life of the runner, so one
    // capture serves every decode step.
    nvinfer1::Dims2 const tokenShape{static_cast<int32_t>(kBatch), 1};

    nvinfer1::Dims cachePositionShape{};
    cachePositionShape.nbDims = 1;
    cachePositionShape.d[0] = 1;

    if (!mDecoderContext->setInputShape(kInputIds, tokenShape)
        || !mDecoderContext->setInputShape(kPositionIds, tokenShape)
        || !mDecoderContext->setInputShape(kCachePosition, cachePositionShape))
    {
        LOG_WARNING("Could not set 1-token decode shapes; CUDA graph disabled");
        return false;
    }

    // Step 4: warm up so TRT finishes any lazy setup before capture.
    if (!mDecoderContext->enqueueV3(mStream))
    {
        LOG_WARNING("Warmup enqueueV3 failed; CUDA graph disabled");
        return false;
    }

    if (cudaStreamSynchronize(mStream) != cudaSuccess)
    {
        LOG_WARNING("Warmup synchronize failed; CUDA graph disabled");
        return false;
    }

    // Capture the whole 1-token step: the three scalar H2D copies, the
    // enqueue, and the logits readback. Replay then costs a single
    // cudaGraphLaunch instead of five separate submissions.
    //
    // This is wider than rt::EngineExecutor::captureGraph(), which captures
    // enqueueV3 alone. The failure handling below mirrors captureTRTCudaGraph():
    // end any in-flight capture and clear the error state so the caller can
    // fall back to enqueueV3 on a healthy stream.
    cudaGraph_t graph{nullptr};

    if (cudaStreamBeginCapture(mStream, cudaStreamCaptureModeThreadLocal) != cudaSuccess)
    {
        LOG_WARNING("cudaStreamBeginCapture failed; CUDA graph disabled");
        static_cast<void>(cudaGetLastError());
        return false;
    }

    bool captureOk = true;

    captureOk = captureOk
        && cudaMemcpyAsync(mInputIdsDevice, mInputIdsHost, sizeof(int64_t),
               cudaMemcpyHostToDevice, mStream)
            == cudaSuccess;

    captureOk = captureOk
        && cudaMemcpyAsync(mPositionIdsDevice, mPositionIdsHost, sizeof(int64_t),
               cudaMemcpyHostToDevice, mStream)
            == cudaSuccess;

    captureOk = captureOk
        && cudaMemcpyAsync(mCachePositionDevice, mCachePositionHost, sizeof(int64_t),
               cudaMemcpyHostToDevice, mStream)
            == cudaSuccess;

    captureOk = captureOk && mDecoderContext->enqueueV3(mStream);

    // Single-token step, so the only logits row is at offset zero.
    captureOk = captureOk
        && cudaMemcpyAsync(mLogitsHost, mLogitsDevice, logitsRowBytes(),
               cudaMemcpyDeviceToHost, mStream)
            == cudaSuccess;

    if (cudaStreamEndCapture(mStream, &graph) != cudaSuccess || !captureOk)
    {
        LOG_WARNING("CUDA graph capture failed; falling back to enqueueV3");
        static_cast<void>(cudaGetLastError());

        cudaStreamCaptureStatus streamStatus{};
        if (cudaStreamIsCapturing(mStream, &streamStatus) == cudaSuccess
            && streamStatus != cudaStreamCaptureStatusNone)
        {
            static_cast<void>(cudaStreamEndCapture(mStream, &graph));
            static_cast<void>(cudaGetLastError());
        }

        if (graph != nullptr)
        {
            static_cast<void>(cudaGraphDestroy(graph));
        }

        static_cast<void>(cudaGetLastError());
        return false;
    }

    if (instantiateCudaGraph(&mDecodeGraphExec, graph) != cudaSuccess)
    {
        LOG_WARNING("cudaGraphInstantiate failed; falling back to enqueueV3");
        static_cast<void>(cudaGraphDestroy(graph));
        static_cast<void>(cudaGetLastError());
        mDecodeGraphExec = nullptr;
        return false;
    }

    mDecodeGraph = graph;

    // The warmup wrote a garbage K/V entry into slot 0; generate() clears the
    // cache before every utterance, so nothing survives into real decoding.
    return true;
}

std::size_t WhisperDecoderRunner::logitsRowBytes() const noexcept
{
    std::size_t const elementSize
        = mLogitsType == nvinfer1::DataType::kHALF ? sizeof(__half) : sizeof(float);

    return static_cast<std::size_t>(mVocabSize) * elementSize;
}

std::size_t WhisperDecoderRunner::cacheElementSize() const noexcept
{
    return mCacheType == nvinfer1::DataType::kHALF
        ? sizeof(__half)
        : sizeof(float);
}

int64_t WhisperDecoderRunner::runDecoderStep(
    int64_t const* tokens,
    int32_t count)
{
    // The captured graph bakes in the 1-token shapes, so shapes only need
    // setting on the prefill path (or when no graph was captured).
    bool const useGraph = (count == 1) && mDecodeGraphReady;

    if (!useGraph)
    {
        // input_ids / position_ids / cache_position are the only dynamic
        // inputs. The self cache is statically shaped, so its binding never
        // changes.
        nvinfer1::Dims2 const tokenShape{
            static_cast<int32_t>(kBatch),
            count};

        nvinfer1::Dims cachePositionShape{};
        cachePositionShape.nbDims = 1;
        cachePositionShape.d[0] = count;

        if (!mDecoderContext->setInputShape(kInputIds, tokenShape)
            || !mDecoderContext->setInputShape(kPositionIds, tokenShape)
            || !mDecoderContext->setInputShape(kCachePosition, cachePositionShape))
        {
            throw std::runtime_error(
                "Failed setting decoder input shapes");
        }
    }

    for (int32_t i = 0; i < count; ++i)
    {
        mInputIdsHost[i] = tokens[i];
        mPositionIdsHost[i] = mCacheLength + i;
        mCachePositionHost[i] = mCacheLength + i;
    }

    if (useGraph)
    {
        // The graph already contains the scalar H2D copies, the enqueue and the
        // logits D2H; the pinned buffers written above are its inputs.
        checkCuda(
            cudaGraphLaunch(mDecodeGraphExec, mStream),
            "launch decode CUDA graph");

        checkCuda(cudaStreamSynchronize(mStream), "synchronize decoder stream");
    }
    else
    {
        std::size_t const idBytes
            = static_cast<std::size_t>(count) * sizeof(int64_t);

        checkCuda(cudaMemcpyAsync(mInputIdsDevice, mInputIdsHost, idBytes,
                      cudaMemcpyHostToDevice, mStream),
            "copy input_ids");
        checkCuda(cudaMemcpyAsync(mPositionIdsDevice, mPositionIdsHost, idBytes,
                      cudaMemcpyHostToDevice, mStream),
            "copy position_ids");
        checkCuda(cudaMemcpyAsync(mCachePositionDevice, mCachePositionHost, idBytes,
                      cudaMemcpyHostToDevice, mStream),
            "copy cache_position");

        if (!mDecoderContext->enqueueV3(mStream))
        {
            throw std::runtime_error("Decoder enqueueV3 failed");
        }

        // Only the last position's logits drive the next token.
        auto const* lastRow = static_cast<char const*>(mLogitsDevice)
            + static_cast<std::size_t>(count - 1) * logitsRowBytes();

        checkCuda(cudaMemcpyAsync(mLogitsHost, lastRow, logitsRowBytes(),
                      cudaMemcpyDeviceToHost, mStream),
            "copy logits");
        checkCuda(cudaStreamSynchronize(mStream), "synchronize decoder stream");
    }

    mCacheLength += count;

    if (mLogitsType == nvinfer1::DataType::kHALF)
    {
        auto const* logits = static_cast<__half const*>(mLogitsHost);
        auto const* best = std::max_element(logits, logits + mVocabSize,
            [](__half left, __half right)
            { return __half2float(left) < __half2float(right); });
        return std::distance(logits, best);
    }

    auto const* logits = static_cast<float const*>(mLogitsHost);
    auto const* best = std::max_element(logits, logits + mVocabSize);
    return std::distance(logits, best);
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
        mCacheLength = 0;

        checkCuda(
            cudaMemsetAsync(
                mSelfKeyValuesDevice,
                0,
                kMaxCacheElements * cacheElementSize(),
                mStream),
            "reset self-attention cache");

        // ----------------------------------------------------
        // Prefill: the whole forced prompt in one pass.
        //
        //   input_ids     [1, 4] = <|sot|> <|en|> <|transcribe|> <|notimestamps|>
        //   position_ids  [1, 4] = 0 1 2 3
        //   cache_position   [4] = 0 1 2 3
        //
        // Decode then continues one token at a time from slot 4.
        // ----------------------------------------------------

        int64_t nextToken = runDecoderStep(
            kPromptTokens,
            kPromptLength);

        outputTokens.push_back(nextToken);

        for (int32_t step = 1;
             step < maxNewTokens;
             ++step)
        {
            if (nextToken == kEosToken)
            {
                std::cout << "EOS reached.\n";
                break;
            }

            if (mCacheLength >= kMaxDecoderLength)
            {
                std::cout
                    << "Cache capacity reached.\n";
                break;
            }

            nextToken = runDecoderStep(
                &nextToken,
                1);

            outputTokens.push_back(nextToken);
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