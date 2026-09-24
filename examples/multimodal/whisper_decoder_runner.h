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

//! Length of Whisper's forced decoder prompt with timestamps suppressed:
//! ``<|sot|> <|lang|> <|task|> <|notimestamps|>``.
constexpr int32_t kWhisperPromptLength = 4;

//! Same prompt with the final token dropped, letting the model emit timestamps.
constexpr int32_t kWhisperTimestampPromptLength = 3;

//! First timestamp token, ``<|0.00|>``. Timestamps occupy the top of the
//! vocabulary, one id per 0.02 s, through ``<|30.00|>``.
constexpr int64_t kWhisperTimestampBeginToken = 50364;

//! Seconds represented by one timestamp-token step.
constexpr double kWhisperTimestampPrecision = 0.02;

class WhisperDecoderRunner
{
public:
    WhisperDecoderRunner(
        std::string crossKvEnginePath,
        std::string decoderEnginePath);

    //! Share already-deserialized engines. Every slot in a server pool then gets
    //! its own execution contexts, caches and CUDA graph over one copy of the
    //! weights, instead of deserializing ~374 MiB per slot.
    WhisperDecoderRunner(
        std::shared_ptr<nvinfer1::IRuntime> runtime,
        std::shared_ptr<nvinfer1::ICudaEngine> crossKvEngine,
        std::shared_ptr<nvinfer1::ICudaEngine> decoderEngine);

    ~WhisperDecoderRunner();

    bool initialize();

    //! \param promptTokens Forced decoder prefix. ``nullptr`` uses the built-in
    //!        English transcribe prompt. Any length from 1 to the cache capacity
    //!        is accepted: the prefill runs through ``enqueueV3``, and only the
    //!        single-token decode step is captured in the CUDA graph.
    //! \param emitTimestamps Let the model predict timestamp tokens, applying
    //!        Whisper's timestamp logit rules before the argmax. The prompt must
    //!        then omit ``<|notimestamps|>``. Timestamp ids are returned in
    //!        \p outputTokens for the caller to seek on; the tokenizer drops
    //!        them from the text.
    bool generate(
        std::vector<__half> const& encoderHiddenStates,
        std::vector<int64_t>& outputTokens,
        int32_t maxNewTokens = 32,
        std::vector<int64_t> const* promptTokens = nullptr,
        bool emitTimestamps = false);

private:
    //! Run one decoder pass over ``count`` tokens starting at cache slot
    //! ``mCacheLength``, and return the argmax of the final position's logits.
    //! Advances ``mCacheLength`` by ``count``.
    int64_t runDecoderStep(
        int64_t const* tokens,
        int32_t count);

    //! Whisper's ``ApplyTimestampRules``: constrain the logits so the emitted
    //! timestamp tokens stay well-formed, then return the argmax.
    //!
    //! \param sampled Tokens generated so far this utterance, prompt excluded.
    //! \param atSampleBegin True on the very first generated token.
    int64_t selectTokenWithTimestampRules(
        std::vector<int64_t> const& sampled,
        bool atSampleBegin);

    //! Copy the current logits row into ``mLogitsScratch`` as float.
    void loadLogitsScratch();

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

    // One TensorRT runtime can own both engines. Both may be shared across
    // pipeline slots; everything below the contexts is per-instance. Declared
    // before the contexts so the contexts are destroyed first.
    std::shared_ptr<nvinfer1::IRuntime> mRuntime;

    std::shared_ptr<nvinfer1::ICudaEngine>
        mCrossKvEngine;
    std::shared_ptr<nvinfer1::ICudaEngine>
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

    //! Float view of one logits row, reused across steps. The timestamp rules
    //! need to mask and renormalize, which the packed FP16 row does not allow
    //! in place.
    std::vector<float> mLogitsScratch;

    //! Tokens generated for the current utterance, prompt excluded. The
    //! timestamp rules are stateful in this history.
    std::vector<int64_t> mSampled;

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