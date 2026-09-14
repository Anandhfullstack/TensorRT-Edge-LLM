#pragma once

#include <NvInfer.h>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

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
    void* mLogitsDevice{nullptr};

    // Growing self-attention cache
    void* mPastKeyValuesDevice{nullptr};
    void* mPresentKeyValuesDevice{nullptr};

    nvinfer1::DataType mInputIdsType{};
    nvinfer1::DataType mPositionIdsType{};
    nvinfer1::DataType mLogitsType{};
    nvinfer1::DataType mCacheType{};
    nvinfer1::DataType mCrossKeyValuesType{};

    int64_t mVocabSize{0};
};

} // namespace whisper
} // namespace examples
} // namespace trt_edgellm