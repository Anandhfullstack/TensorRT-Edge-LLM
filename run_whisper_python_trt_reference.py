import numpy as np
import tensorrt as trt
import torch


ENGINE_PATH = "whisper_small_native_engine/audio/audio_encoder.engine"
INPUT_PATH = "whisper_cpp_preprocessed_fp16.bin"
OUTPUT_PATH = "whisper_python_trt_output_fp16.bin"

INPUT_NAME = "input_features"
OUTPUT_NAME = "last_hidden_state"

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


# --------------------------------------------------
# 1. Load exact C++-generated FP16 input
# --------------------------------------------------

input_np = np.fromfile(
    INPUT_PATH,
    dtype=np.float16,
)

assert input_np.size == 1 * 80 * 3000

input_np = input_np.reshape(1, 80, 3000)

print("Input shape   :", input_np.shape)
print("Input elements:", input_np.size)


# --------------------------------------------------
# 2. Load TensorRT engine
# --------------------------------------------------

with open(ENGINE_PATH, "rb") as f:
    engine_bytes = f.read()

runtime = trt.Runtime(TRT_LOGGER)

engine = runtime.deserialize_cuda_engine(
    engine_bytes
)

if engine is None:
    raise RuntimeError(
        "Failed to deserialize TensorRT engine"
    )

context = engine.create_execution_context()

if context is None:
    raise RuntimeError(
        "Failed to create TensorRT execution context"
    )


# --------------------------------------------------
# 3. Set input shape
# --------------------------------------------------

success = context.set_input_shape(
    INPUT_NAME,
    input_np.shape,
)

if not success:
    raise RuntimeError(
        "Failed to set input shape"
    )

output_shape = tuple(
    context.get_tensor_shape(OUTPUT_NAME)
)

print("Output shape  :", output_shape)


# --------------------------------------------------
# 4. Allocate CUDA tensors using PyTorch
# --------------------------------------------------

input_gpu = torch.from_numpy(
    input_np.copy()
).cuda()

output_gpu = torch.empty(
    output_shape,
    dtype=torch.float16,
    device="cuda",
)

print("Input GPU dtype :", input_gpu.dtype)
print("Output GPU dtype:", output_gpu.dtype)


# --------------------------------------------------
# 5. Bind TensorRT addresses
# --------------------------------------------------

if not context.set_tensor_address(
    INPUT_NAME,
    input_gpu.data_ptr(),
):
    raise RuntimeError(
        "Failed to bind input_features"
    )

if not context.set_tensor_address(
    OUTPUT_NAME,
    output_gpu.data_ptr(),
):
    raise RuntimeError(
        "Failed to bind last_hidden_state"
    )


# --------------------------------------------------
# 6. Run TensorRT using a PyTorch CUDA stream
# --------------------------------------------------

stream = torch.cuda.Stream()

with torch.cuda.stream(stream):

    success = context.execute_async_v3(
        stream_handle=stream.cuda_stream
    )

    if not success:
        raise RuntimeError(
            "TensorRT execution failed"
        )


stream.synchronize()


# --------------------------------------------------
# 7. GPU -> CPU
# --------------------------------------------------

output_np = (
    output_gpu
    .cpu()
    .numpy()
)


# --------------------------------------------------
# 8. Save Python TRT reference
# --------------------------------------------------

output_np.tofile(
    OUTPUT_PATH
)

print()
print("Saved:", OUTPUT_PATH)
print(
    "Output elements:",
    output_np.size
)
print(
    "Output bytes   :",
    output_np.nbytes
)