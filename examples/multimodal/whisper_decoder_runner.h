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
    explicit WhisperDecoderRunner(
        std::string enginePath);

    ~WhisperDecoderRunner();

    bool initialize();

    bool generate(
        std::vector<__half> const& encoderHiddenStates,
        std::vector<int64_t>& outputTokens,
        int32_t maxNewTokens = 32);

private:
    std::string mEnginePath;

    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mContext;

    cudaStream_t mStream{nullptr};

    void* mInputIdsDevice{nullptr};
    void* mEncoderDevice{nullptr};
    void* mLogitsDevice{nullptr};

    nvinfer1::DataType mInputIdsType{};
    nvinfer1::DataType mLogitsType{};

    int64_t mVocabSize{0};
};

} // namespace whisper
} // namespace examples
} // namespace trt_edgellm