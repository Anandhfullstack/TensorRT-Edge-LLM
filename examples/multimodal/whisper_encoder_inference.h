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

// ============================================================
// Whisper Small encoder constants
// ============================================================

constexpr char const* kInputName = "input_features";
constexpr char const* kOutputName = "last_hidden_state";

constexpr int64_t kBatchSize = 1;
constexpr int64_t kMelBins = 80;
constexpr int64_t kInputFrames = 3000;

constexpr int64_t kOutputFrames = 1500;
constexpr int64_t kHiddenSize = 768;

constexpr std::size_t kInputElements
    = kBatchSize * kMelBins * kInputFrames;

constexpr std::size_t kOutputElements
    = kBatchSize * kOutputFrames * kHiddenSize;

constexpr std::size_t kInputBytes
    = kInputElements * sizeof(__half);

constexpr std::size_t kOutputBytes
    = kOutputElements * sizeof(__half);


// ============================================================
// TensorRT logger
// ============================================================

class TensorRTLogger : public nvinfer1::ILogger
{
public:
    void log(
        Severity severity,
        char const* message) noexcept override;
};


// ============================================================
// Whisper encoder TensorRT inference
// ============================================================

class WhisperEncoderInference
{
public:

    explicit WhisperEncoderInference(
        std::string enginePath);
    WhisperEncoderInference(
        std::string enginePath,
        std::string inputPath,
        std::string referencePath,
        std::string outputPath);

    ~WhisperEncoderInference();

    WhisperEncoderInference(
        WhisperEncoderInference const&) = delete;

    WhisperEncoderInference& operator=(
        WhisperEncoderInference const&) = delete;

    bool initialize();

    bool run();
      // Runtime path:
    // preprocessed [1,80,3000] FP16
    //              ↓
    // TensorRT encoder
    //              ↓
    // [1,1500,768] FP16
    bool run(
        std::vector<__half> const& inputFeatures,
        std::vector<__half>& outputFeatures);

private:
    // --------------------------------------------------------
    // Main stages
    // --------------------------------------------------------

    void loadValidationData();

    void loadEngine();

    void printEngineInfo() const;

    void validateEngineIO() const;

    void createExecutionResources();

    void execute();

    void saveOutput() const;

    void compareWithPythonOutput() const;


    // --------------------------------------------------------
    // File helpers
    // --------------------------------------------------------

    static std::vector<char> readBinaryFile(
        std::string const& path);

    static std::vector<__half> readHalfFile(
        std::string const& path,
        std::size_t expectedElements);

    static void writeHalfFile(
        std::string const& path,
        std::vector<__half> const& data);


    // --------------------------------------------------------
    // Utility helpers
    // --------------------------------------------------------

    static void checkCuda(
        cudaError_t error,
        char const* operation);

    static std::string dimsToString(
        nvinfer1::Dims const& dims);


private:
    // --------------------------------------------------------
    // Input paths
    // --------------------------------------------------------

    std::string mEnginePath;
    std::string mInputPath;
    std::string mReferencePath;
    std::string mOutputPath;


    // --------------------------------------------------------
    // Host tensors
    // --------------------------------------------------------

    std::vector<__half> mInputHost;
    std::vector<__half> mPythonOutput;
    std::vector<__half> mCppOutput;


    // --------------------------------------------------------
    // TensorRT objects
    // --------------------------------------------------------

    TensorRTLogger mLogger;

    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mContext;


    // --------------------------------------------------------
    // CUDA resources
    // --------------------------------------------------------

    cudaStream_t mStream{nullptr};

    void* mInputDevice{nullptr};
    void* mOutputDevice{nullptr};
};

} // namespace whisper
} // namespace examples
} // namespace trt_edgellm