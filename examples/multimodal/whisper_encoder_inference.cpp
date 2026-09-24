#include "whisper_encoder_inference.h"

#include "common/logger.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace trt_edgellm
{
namespace examples
{
namespace whisper
{

// ============================================================
// TensorRT logger
// ============================================================

void TensorRTLogger::log(
    Severity severity,
    char const* message) noexcept
{
    if (severity <= Severity::kWARNING)
    {
        std::cerr
            << "[TensorRT] "
            << message
            << '\n';
    }
}

// ============================================================
// Runtime constructor
// ============================================================

WhisperEncoderInference::WhisperEncoderInference(
    std::string enginePath)
    : mEnginePath(std::move(enginePath))
{
}


// ============================================================
// Constructor
// ============================================================

WhisperEncoderInference::WhisperEncoderInference(
    std::shared_ptr<nvinfer1::IRuntime> runtime,
    std::shared_ptr<nvinfer1::ICudaEngine> engine)
    : mRuntime(std::move(runtime))
    , mEngine(std::move(engine))
{
}

WhisperEncoderInference::WhisperEncoderInference(
    std::string enginePath,
    std::string inputPath,
    std::string referencePath,
    std::string outputPath)
    : mEnginePath(std::move(enginePath))
    , mInputPath(std::move(inputPath))
    , mReferencePath(std::move(referencePath))
    , mOutputPath(std::move(outputPath))
{
}


// ============================================================
// Destructor
// ============================================================

WhisperEncoderInference::~WhisperEncoderInference()
{
    if (mInputDevice != nullptr)
    {
        cudaFree(mInputDevice);
        mInputDevice = nullptr;
    }

    if (mOutputDevice != nullptr)
    {
        cudaFree(mOutputDevice);
        mOutputDevice = nullptr;
    }

    if (mStream != nullptr)
    {
        cudaStreamDestroy(mStream);
        mStream = nullptr;
    }
}


// ============================================================
// CUDA helper
// ============================================================

void WhisperEncoderInference::checkCuda(
    cudaError_t error,
    char const* operation)
{
    if (error != cudaSuccess)
    {
        std::ostringstream oss;

        oss
            << operation
            << " failed: "
            << cudaGetErrorString(error);

        throw std::runtime_error(
            oss.str());
    }
}


// ============================================================
// TensorRT dimensions -> string
// ============================================================

std::string WhisperEncoderInference::dimsToString(
    nvinfer1::Dims const& dims)
{
    std::ostringstream oss;

    oss << "[";

    for (int32_t i = 0;
         i < dims.nbDims;
         ++i)
    {
        if (i != 0)
        {
            oss << ", ";
        }

        oss << dims.d[i];
    }

    oss << "]";

    return oss.str();
}


// ============================================================
// Read generic binary file
// ============================================================

std::vector<char>
WhisperEncoderInference::readBinaryFile(
    std::string const& path)
{
    std::ifstream file(
        path,
        std::ios::binary
            | std::ios::ate);

    if (!file)
    {
        throw std::runtime_error(
            "Failed to open file: "
            + path);
    }

    std::streamsize const size
        = file.tellg();

    if (size <= 0)
    {
        throw std::runtime_error(
            "File is empty: "
            + path);
    }

    file.seekg(
        0,
        std::ios::beg);

    std::vector<char> data(
        static_cast<std::size_t>(
            size));

    if (!file.read(
            data.data(),
            size))
    {
        throw std::runtime_error(
            "Failed to read file: "
            + path);
    }

    return data;
}



// ============================================================
// Initialize TensorRT encoder
// ============================================================

bool WhisperEncoderInference::initialize()
{
    try
    {
        // Already initialized.
        if (mContext
            && mStream != nullptr
            && mInputDevice != nullptr
            && mOutputDevice != nullptr)
        {
            return true;
        }

        std::cout
            << "\n================================\n"
            << " Initializing Whisper encoder\n"
            << "================================\n";

        loadEngine();

        printEngineInfo();

        validateEngineIO();

        createExecutionResources();

        std::cout
            << "\nWhisper encoder initialized successfully.\n";

        return true;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("Whisper encoder initialization failed: %s", error.what());

        return false;
    }
}


bool WhisperEncoderInference::run(
    std::vector<__half> const& inputFeatures,
    std::vector<__half>& outputFeatures)
{
    // --------------------------------------------------------
    // 1. Validate TensorRT resources
    // --------------------------------------------------------

   // Initialize automatically on first inference.
    if (!initialize())
    {
        return false;
    }

    // --------------------------------------------------------
    // 2. Validate input
    // --------------------------------------------------------

    if (inputFeatures.size() != kInputElements)
    {
        LOG_ERROR("Invalid Whisper input size: expected %zu, got %zu", kInputElements, inputFeatures.size());

        return false;
    }

    // [1, 1500, 768]
    outputFeatures.resize(
        kOutputElements);

    // --------------------------------------------------------
    // 3. CPU Mel features -> GPU TensorRT input
    // --------------------------------------------------------

    cudaError_t status = cudaMemcpyAsync(
        mInputDevice,
        inputFeatures.data(),
        kInputBytes,
        cudaMemcpyHostToDevice,
        mStream);

    if (status != cudaSuccess)
    {
        LOG_ERROR("Failed to copy Whisper input to GPU: %s", cudaGetErrorString(status));

        return false;
    }

    // --------------------------------------------------------
    // 4. Run TensorRT encoder
    //
    // Tensor addresses were already bound when execution
    // resources were created.
    // --------------------------------------------------------

    if (!mContext->enqueueV3(
            mStream))
    {
        LOG_ERROR("TensorRT Whisper encoder execution failed");

        return false;
    }

    // --------------------------------------------------------
    // 5. GPU encoder output -> CPU vector
    //
    // Temporary validation/runtime path.
    // Later the decoder can consume this directly on GPU.
    // --------------------------------------------------------

    status = cudaMemcpyAsync(
        outputFeatures.data(),
        mOutputDevice,
        kOutputBytes,
        cudaMemcpyDeviceToHost,
        mStream);

    if (status != cudaSuccess)
    {
        LOG_ERROR("Failed to copy Whisper encoder output: %s", cudaGetErrorString(status));

        return false;
    }

    // --------------------------------------------------------
    // 6. Wait until inference + copy finish
    // --------------------------------------------------------

    status = cudaStreamSynchronize(
        mStream);

    if (status != cudaSuccess)
    {
        LOG_ERROR("CUDA stream synchronization failed: %s", cudaGetErrorString(status));

        return false;
    }

    return true;
}


// ============================================================
// Read FP16 tensor
// ============================================================

std::vector<__half>
WhisperEncoderInference::readHalfFile(
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
            "Failed to open FP16 file: "
            + path);
    }

    std::size_t const actualBytes
        = static_cast<std::size_t>(
            file.tellg());

    std::size_t const expectedBytes
        = expectedElements
        * sizeof(__half);

    if (actualBytes != expectedBytes)
    {
        std::ostringstream oss;

        oss
            << "Unexpected file size for "
            << path
            << ". Expected "
            << expectedBytes
            << " bytes, got "
            << actualBytes
            << " bytes.";

        throw std::runtime_error(
            oss.str());
    }

    file.seekg(
        0,
        std::ios::beg);

    std::vector<__half> data(
        expectedElements);

    if (!file.read(
            reinterpret_cast<char*>(
                data.data()),
            static_cast<std::streamsize>(
                expectedBytes)))
    {
        throw std::runtime_error(
            "Failed to read FP16 tensor: "
            + path);
    }

    return data;
}


// ============================================================
// Write FP16 tensor
// ============================================================

void WhisperEncoderInference::writeHalfFile(
    std::string const& path,
    std::vector<__half> const& data)
{
    std::ofstream file(
        path,
        std::ios::binary);

    if (!file)
    {
        throw std::runtime_error(
            "Failed to create output file: "
            + path);
    }

    file.write(
        reinterpret_cast<char const*>(
            data.data()),
        static_cast<std::streamsize>(
            data.size()
            * sizeof(__half)));

    if (!file)
    {
        throw std::runtime_error(
            "Failed to write output file: "
            + path);
    }
}


// ============================================================
// Load validation tensors
// ============================================================

void WhisperEncoderInference::loadValidationData()
{
    std::cout
        << "\n================================\n"
        << " Loading validation tensors\n"
        << "================================\n";

    mInputHost = readHalfFile(
        mInputPath,
        kInputElements);

    mPythonOutput = readHalfFile(
        mReferencePath,
        kOutputElements);

    mCppOutput.resize(
        kOutputElements);


    std::cout
        << "Input elements     : "
        << mInputHost.size()
        << '\n';

    std::cout
        << "Input bytes        : "
        << kInputBytes
        << '\n';

    std::cout
        << "Reference elements : "
        << mPythonOutput.size()
        << '\n';

    std::cout
        << "Output bytes       : "
        << kOutputBytes
        << '\n';
}


// ============================================================
// Load TensorRT engine
// ============================================================

void WhisperEncoderInference::loadEngine()
{
    // A shared engine was injected by the caller; only the per-instance
    // execution context is still missing.
    if (!mEngine)
    {
        std::cout
            << "\n================================\n"
            << " Loading TensorRT engine\n"
            << "================================\n";

        std::vector<char> engineData
            = readBinaryFile(
                mEnginePath);

        mRuntime.reset(
            nvinfer1::createInferRuntime(
                mLogger));

        if (!mRuntime)
        {
            throw std::runtime_error(
                "Failed to create TensorRT runtime.");
        }

        mEngine.reset(
            mRuntime->deserializeCudaEngine(
                engineData.data(),
                engineData.size()));

        if (!mEngine)
        {
            throw std::runtime_error(
                "Failed to deserialize TensorRT engine.");
        }
    }


    mContext.reset(
        mEngine->createExecutionContext());


    if (!mContext)
    {
        throw std::runtime_error(
            "Failed to create TensorRT execution context.");
    }


    std::cout
        << "TensorRT engine loaded successfully.\n";
}


// ============================================================
// Print TensorRT engine information
// ============================================================

void WhisperEncoderInference::printEngineInfo() const
{
    std::cout
        << "\n================================\n"
        << " TensorRT engine I/O\n"
        << "================================\n";


    int32_t const tensorCount
        = mEngine->getNbIOTensors();


    for (int32_t i = 0;
         i < tensorCount;
         ++i)
    {
        char const* name
            = mEngine->getIOTensorName(i);

        nvinfer1::TensorIOMode const mode
            = mEngine->getTensorIOMode(
                name);

        nvinfer1::DataType const dataType
            = mEngine->getTensorDataType(
                name);

        nvinfer1::Dims const shape
            = mEngine->getTensorShape(
                name);


        std::cout
            << name
            << " mode="
            << (
                   mode
                           == nvinfer1::
                               TensorIOMode::
                                   kINPUT
                   ? "INPUT"
                   : "OUTPUT"
               )
            << " dtype="
            << static_cast<int>(
                   dataType)
            << " shape="
            << dimsToString(
                   shape)
            << '\n';
    }
}


// ============================================================
// Validate TensorRT I/O contract
// ============================================================

void WhisperEncoderInference::validateEngineIO() const
{
    if (mEngine->getTensorDataType(
            kInputName)
        != nvinfer1::DataType::kHALF)
    {
        throw std::runtime_error(
            "TensorRT input_features is not FP16.");
    }


    if (mEngine->getTensorDataType(
            kOutputName)
        != nvinfer1::DataType::kHALF)
    {
        throw std::runtime_error(
            "TensorRT last_hidden_state is not FP16.");
    }


    nvinfer1::Dims const inputShape
        = mEngine->getTensorShape(
            kInputName);


    if (inputShape.nbDims != 3
        || inputShape.d[0] != kBatchSize
        || inputShape.d[1] != kMelBins
        || inputShape.d[2] != kInputFrames)
    {
        throw std::runtime_error(
            "Unexpected TensorRT input shape: "
            + dimsToString(
                inputShape));
    }


    nvinfer1::Dims const outputShape
        = mEngine->getTensorShape(
            kOutputName);


    if (outputShape.nbDims != 3
        || outputShape.d[0] != kBatchSize
        || outputShape.d[1] != kOutputFrames
        || outputShape.d[2] != kHiddenSize)
    {
        throw std::runtime_error(
            "Unexpected TensorRT output shape: "
            + dimsToString(
                outputShape));
    }


    std::cout
        << "\nTensorRT I/O validation passed.\n";
}


// ============================================================
// Create CUDA buffers and bind engine tensors
// ============================================================

void WhisperEncoderInference::createExecutionResources()
{
    std::cout
        << "\n================================\n"
        << " Creating execution resources\n"
        << "================================\n";


    checkCuda(
        cudaStreamCreate(
            &mStream),
        "cudaStreamCreate");


    // Explicitly select profile 0.
    if (!mContext->setOptimizationProfileAsync(
            0,
            mStream))
    {
        throw std::runtime_error(
            "Failed to select optimization profile 0.");
    }


    nvinfer1::Dims3 const inputDimensions{
        static_cast<int32_t>(
            kBatchSize),

        static_cast<int32_t>(
            kMelBins),

        static_cast<int32_t>(
            kInputFrames)};


    if (!mContext->setInputShape(
            kInputName,
            inputDimensions))
    {
        throw std::runtime_error(
            "Failed to set input_features shape.");
    }


    nvinfer1::Dims const outputShape
        = mContext->getTensorShape(
            kOutputName);


    std::cout
        << "Resolved output shape: "
        << dimsToString(
               outputShape)
        << '\n';


    checkCuda(
        cudaMalloc(
            &mInputDevice,
            kInputBytes),
        "cudaMalloc(input)");


    checkCuda(
        cudaMalloc(
            &mOutputDevice,
            kOutputBytes),
        "cudaMalloc(output)");


    if (!mContext->setTensorAddress(
            kInputName,
            mInputDevice))
    {
        throw std::runtime_error(
            "Failed to bind input_features.");
    }


    if (!mContext->setTensorAddress(
            kOutputName,
            mOutputDevice))
    {
        throw std::runtime_error(
            "Failed to bind last_hidden_state.");
    }


    std::cout
        << "CUDA buffers created successfully.\n";
}


// ============================================================
// Run TensorRT inference
// ============================================================

void WhisperEncoderInference::execute()
{
    std::cout
        << "\n================================\n"
        << " Running TensorRT encoder\n"
        << "================================\n";


    checkCuda(
        cudaMemcpyAsync(
            mInputDevice,
            mInputHost.data(),
            kInputBytes,
            cudaMemcpyHostToDevice,
            mStream),
        "cudaMemcpyAsync H2D");


    if (!mContext->enqueueV3(
            mStream))
    {
        throw std::runtime_error(
            "TensorRT enqueueV3 failed.");
    }


    checkCuda(
        cudaMemcpyAsync(
            mCppOutput.data(),
            mOutputDevice,
            kOutputBytes,
            cudaMemcpyDeviceToHost,
            mStream),
        "cudaMemcpyAsync D2H");


    checkCuda(
        cudaStreamSynchronize(
            mStream),
        "cudaStreamSynchronize");


    std::cout
        << "TensorRT inference completed successfully.\n";
}


// ============================================================
// Save C++ output
// ============================================================

void WhisperEncoderInference::saveOutput() const
{
    writeHalfFile(
        mOutputPath,
        mCppOutput);


    std::cout
        << "\nSaved C++ output:\n"
        << mOutputPath
        << '\n';
}


// ============================================================
// Compare against Python TensorRT output
// ============================================================

void WhisperEncoderInference::compareWithPythonOutput() const
{
    if (mCppOutput.size()
        != mPythonOutput.size())
    {
        throw std::runtime_error(
            "C++ and Python output sizes differ.");
    }


    double absoluteSum = 0.0;
    double squaredSum = 0.0;

    double dotProduct = 0.0;
    double cppNorm = 0.0;
    double pythonNorm = 0.0;

    float maximumDifference = 0.0F;
    std::size_t maximumIndex = 0;

    std::size_t exactMismatchCount = 0;

    std::size_t above001 = 0;
    std::size_t above01 = 0;
    std::size_t above1 = 0;


    for (std::size_t i = 0;
         i < mCppOutput.size();
         ++i)
    {
        float const cppValue
            = __half2float(
                mCppOutput[i]);

        float const pythonValue
            = __half2float(
                mPythonOutput[i]);


        float const difference
            = std::abs(
                cppValue
                - pythonValue);


        absoluteSum
            += difference;


        squaredSum
            += static_cast<double>(
                   difference)
            * difference;


        dotProduct
            += static_cast<double>(
                   cppValue)
            * pythonValue;


        cppNorm
            += static_cast<double>(
                   cppValue)
            * cppValue;


        pythonNorm
            += static_cast<double>(
                   pythonValue)
            * pythonValue;


        if (difference
            > maximumDifference)
        {
            maximumDifference
                = difference;

            maximumIndex
                = i;
        }


        uint16_t cppBits{};
        uint16_t pythonBits{};


        std::memcpy(
            &cppBits,
            &mCppOutput[i],
            sizeof(uint16_t));


        std::memcpy(
            &pythonBits,
            &mPythonOutput[i],
            sizeof(uint16_t));


        if (cppBits
            != pythonBits)
        {
            ++exactMismatchCount;
        }


        if (difference > 0.001F)
        {
            ++above001;
        }

        if (difference > 0.01F)
        {
            ++above01;
        }

        if (difference > 0.1F)
        {
            ++above1;
        }
    }


    double const meanDifference
        = absoluteSum
        / mCppOutput.size();


    double const rmse
        = std::sqrt(
            squaredSum
            / mCppOutput.size());


    double const cosineSimilarity
        = dotProduct
        / (
            std::sqrt(
                cppNorm)
            * std::sqrt(
                pythonNorm)
        );


    std::size_t const hiddenIndex
        = maximumIndex
        % kHiddenSize;


    std::size_t const encoderPosition
        = (
              maximumIndex
              / kHiddenSize
          )
        % kOutputFrames;


    std::size_t const batchIndex
        = maximumIndex
        / (
            kOutputFrames
            * kHiddenSize
        );


    std::cout
        << "\n================================\n"
        << " C++ TRT vs Python TRT\n"
        << "================================\n";


    std::cout
        << std::setprecision(10);


    std::cout
        << "Elements              : "
        << mCppOutput.size()
        << '\n';


    std::cout
        << "Mean abs diff         : "
        << meanDifference
        << '\n';


    std::cout
        << "Max abs diff          : "
        << maximumDifference
        << '\n';


    std::cout
        << "RMSE                  : "
        << rmse
        << '\n';


    std::cout
        << "Cosine similarity     : "
        << cosineSimilarity
        << '\n';


    std::cout
        << "Exact FP16 mismatches : "
        << exactMismatchCount
        << " / "
        << mCppOutput.size()
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
        << "> 0.1   : "
        << above1
        << '\n';


    std::cout
        << "\nWorst difference\n";


    std::cout
        << "index  : ["
        << batchIndex
        << ", "
        << encoderPosition
        << ", "
        << hiddenIndex
        << "]\n";


    std::cout
        << "C++    : "
        << __half2float(
               mCppOutput[
                   maximumIndex])
        << '\n';


    std::cout
        << "Python : "
        << __half2float(
               mPythonOutput[
                   maximumIndex])
        << '\n';


    std::cout
        << "diff   : "
        << maximumDifference
        << '\n';
}


// ============================================================
// Full execution pipeline
// ============================================================

bool WhisperEncoderInference::run()
{
    try
    {
        std::cout
            << "================================\n"
            << " Whisper TensorRT C++ Encoder\n"
            << "================================\n";


        std::cout
            << "Engine    : "
            << mEnginePath
            << '\n';


        std::cout
            << "Input     : "
            << mInputPath
            << '\n';


        std::cout
            << "Reference : "
            << mReferencePath
            << '\n';


        loadValidationData();

        if (!initialize())
        {
            return false;
        }
       

        execute();

        saveOutput();

        compareWithPythonOutput();


        std::cout
            << "\n================================\n"
            << " Validation completed\n"
            << "================================\n";


        return true;
    }
    catch (std::exception const& error)
    {
        std::cerr
            << "\nERROR: "
            << error.what()
            << '\n';

        return false;
    }
}

} // namespace whisper
} // namespace examples
} // namespace trt_edgellm


