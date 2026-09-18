#pragma once

#include <NvInfer.h>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace examples
{
namespace whisper
{

class WhisperDecoderRunner
{
public:
    WhisperDecoderRunner(
        std::string crossKvEnginePath,
        std::string decoderEnginePath);

    ~WhisperDecoderRunner();

    bool initialize();

    bool generate(
        std::vector<__half> const& encoderHiddenStates,
        std::vector<int64_t>& outputTokens,
        int32_t maxNewTokens = 32);

private:
    //! Run one decoder pass over ``count`` tokens starting at cache slot
    //! ``mCacheLength``, and return the argmax of the final position's logits.
    //! Advances ``mCacheLength`` by ``count``.
    int64_t runDecoderStep(
        int64_t const* tokens,
        int32_t count);

    //! Element size of the self-attention cache dtype.
    std::size_t cacheElementSize() const noexcept;

    //! Bytes in one row of logits (``mVocabSize`` elements).
    std::size_t logitsRowBytes() const noexcept;

    //! Capture the single-token decode step into a CUDA graph: the three
    //! scalar H2D copies, ``enqueueV3``, and the logits D2H. Only the 1-token
    //! path is captured; the prefill uses a different ``input_ids`` extent and
    //! stays on ``enqueueV3``. Failure is non-fatal — the runner falls back to
    //! ``enqueueV3`` for every step.
    bool captureDecodeGraph();

private:
    std::string mCrossKvEnginePath;
    std::string mDecoderEnginePath;

    // One TensorRT runtime can own both engines.
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;

    std::unique_ptr<nvinfer1::ICudaEngine>
        mCrossKvEngine;
    std::unique_ptr<nvinfer1::ICudaEngine>
        mDecoderEngine;

    std::unique_ptr<nvinfer1::IExecutionContext>
        mCrossKvContext;
    std::unique_ptr<nvinfer1::IExecutionContext>
        mDecoderContext;

    // Both engines execute sequentially on this stream.
    cudaStream_t mStream{nullptr};

    // Cross-KV engine buffers
    void* mEncoderDevice{nullptr};
    void* mCrossKeyValuesDevice{nullptr};

    // Decoder engine buffers
    void* mInputIdsDevice{nullptr};
    void* mPositionIdsDevice{nullptr};
    void* mCachePositionDevice{nullptr};
    void* mLogitsDevice{nullptr};

    //! Fixed-capacity self-attention cache, [12, 2, 1, 12, 448, 64]. The decoder
    //! scatters each step's K/V into slot `cache_position`, so past and present
    //! bind to this one allocation and the shape never changes.
    void* mSelfKeyValuesDevice{nullptr};

    //! Pinned staging for the per-step scalars. Pageable sources make
    //! cudaMemcpyAsync fall back to a synchronous copy.
    int64_t* mInputIdsHost{nullptr};
    int64_t* mPositionIdsHost{nullptr};
    int64_t* mCachePositionHost{nullptr};
    void* mLogitsHost{nullptr};

    nvinfer1::DataType mInputIdsType{};
    nvinfer1::DataType mPositionIdsType{};
    nvinfer1::DataType mLogitsType{};
    nvinfer1::DataType mCacheType{};
    nvinfer1::DataType mCrossKeyValuesType{};

    int64_t mVocabSize{0};

    //! Tokens written to the cache so far; also the next write slot.
    int64_t mCacheLength{0};

    //! Captured single-token decode step. Valid only while every binding
    //! address and the 1-token input shapes stay fixed, which the fixed-capacity
    //! cache guarantees.
    cudaGraph_t mDecodeGraph{nullptr};
    cudaGraphExec_t mDecodeGraphExec{nullptr};
    bool mDecodeGraphReady{false};
};

} // namespace whisper
} // namespace examples
} // namespace trt_edgellm