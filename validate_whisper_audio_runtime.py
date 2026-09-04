import numpy as np
import torch
import tensorrt as trt

from transformers import WhisperModel


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "openai/whisper-small"

ENGINE_PATH = (
    "whisper_small_native_engine/audio/"
    "audio_encoder.engine"
)

HF_INPUT_PATH = (
    "whisper_runtime_input_fp16.bin"
)

CPP_PREPROCESSED_INPUT_PATH = (
    "whisper_cpp_preprocessed_fp16.bin"
)

CPP_OUTPUT_PATH = (
    "whisper_audio_encoder_output_fp16.bin"
)

DEVICE = "cuda"

INPUT_NAME = "input_features"
OUTPUT_NAME = "last_hidden_state"

INPUT_SHAPE = (
    1,
    80,
    3000,
)

OUTPUT_SHAPE = (
    1,
    1500,
    768,
)


# ============================================================
# Comparison helper
# ============================================================

def compare(
    name,
    reference,
    test,
):
    reference = reference.float()
    test = test.float()

    diff = torch.abs(
        reference - test
    )

    rmse = torch.sqrt(
        torch.mean(
            (reference - test) ** 2
        )
    )

    cosine = (
        torch.nn.functional
        .cosine_similarity(
            reference.flatten(),
            test.flatten(),
            dim=0,
        )
    )

    print("\n================================")
    print(name)
    print("================================")

    print(
        "Mean abs diff :",
        diff.mean().item(),
    )

    print(
        "Median diff   :",
        diff.median().item(),
    )

    print(
        "99% diff      :",
        torch.quantile(
            diff.flatten(),
            0.99,
        ).item(),
    )

    print(
        "99.9% diff    :",
        torch.quantile(
            diff.flatten(),
            0.999,
        ).item(),
    )

    print(
        "Max abs diff  :",
        diff.max().item(),
    )

    print(
        "RMSE          :",
        rmse.item(),
    )

    print(
        "Cosine        :",
        cosine.item(),
    )

    print(
        "> 0.05        :",
        (diff > 0.05).sum().item(),
    )

    print(
        "> 0.1         :",
        (diff > 0.1).sum().item(),
    )

    print(
        "> 0.5         :",
        (diff > 0.5).sum().item(),
    )


# ============================================================
# TensorRT helper
# ============================================================

def run_trt(
    engine,
    input_tensor,
):
    context = (
        engine.create_execution_context()
    )

    if context is None:
        raise RuntimeError(
            "Failed to create TensorRT context"
        )

    if not context.set_input_shape(
        INPUT_NAME,
        INPUT_SHAPE,
    ):
        raise RuntimeError(
            "Failed to set TensorRT input shape"
        )

    output_tensor = torch.empty(
        OUTPUT_SHAPE,
        dtype=torch.float16,
        device=DEVICE,
    )

    if not context.set_tensor_address(
        INPUT_NAME,
        input_tensor.data_ptr(),
    ):
        raise RuntimeError(
            "Failed to bind TensorRT input"
        )

    if not context.set_tensor_address(
        OUTPUT_NAME,
        output_tensor.data_ptr(),
    ):
        raise RuntimeError(
            "Failed to bind TensorRT output"
        )

    stream = torch.cuda.Stream()

    stream.wait_stream(
        torch.cuda.current_stream()
    )

    if not context.execute_async_v3(
        stream_handle=stream.cuda_stream
    ):
        raise RuntimeError(
            "TensorRT execution failed"
        )

    stream.synchronize()

    return output_tensor


# ============================================================
# 1. Load Hugging Face preprocessing reference
#
# This is:
#
# original audio
#    ↓
# HF WhisperFeatureExtractor
#    ↓
# [1,80,3000]
# ============================================================

print(
    "Loading Hugging Face preprocessing reference..."
)

hf_input_np = np.fromfile(
    HF_INPUT_PATH,
    dtype=np.float16,
)

expected_input_elements = (
    np.prod(INPUT_SHAPE)
)

if hf_input_np.size != expected_input_elements:
    raise RuntimeError(
        f"Unexpected HF input size: "
        f"{hf_input_np.size}, "
        f"expected {expected_input_elements}"
    )

hf_input_np = hf_input_np.reshape(
    INPUT_SHAPE
)

hf_input_features = (
    torch.from_numpy(
        hf_input_np.copy()
    )
    .to(
        DEVICE,
        dtype=torch.float16,
    )
)

print(
    "HF input shape:",
    tuple(
        hf_input_features.shape
    ),
)


# ============================================================
# 2. Load exact C++ preprocessing output
#
# This is:
#
# audio.wav
#    ↓
# C++ WhisperAudioProcessor
#    ↓
# [1,80,3000]
# ============================================================

print(
    "Loading C++ preprocessing output..."
)

cpp_input_np = np.fromfile(
    CPP_PREPROCESSED_INPUT_PATH,
    dtype=np.float16,
)

if cpp_input_np.size != expected_input_elements:
    raise RuntimeError(
        f"Unexpected C++ preprocessing input size: "
        f"{cpp_input_np.size}, "
        f"expected {expected_input_elements}"
    )

cpp_input_np = cpp_input_np.reshape(
    INPUT_SHAPE
)

cpp_input_features = (
    torch.from_numpy(
        cpp_input_np.copy()
    )
    .to(
        DEVICE,
        dtype=torch.float16,
    )
)

print(
    "C++ preprocessing shape:",
    tuple(
        cpp_input_features.shape
    ),
)


# ============================================================
# 3. Load C++ complete runtime encoder output
#
# audio.wav
#    ↓
# C++ preprocessing
#    ↓
# C++ TensorRT encoder
#    ↓
# [1,1500,768]
# ============================================================

print(
    "Loading C++ runtime encoder output..."
)

cpp_output_np = np.fromfile(
    CPP_OUTPUT_PATH,
    dtype=np.float16,
)

expected_output_elements = (
    np.prod(OUTPUT_SHAPE)
)

if (
    cpp_output_np.size
    != expected_output_elements
):
    raise RuntimeError(
        f"Unexpected C++ output size: "
        f"{cpp_output_np.size}, "
        f"expected {expected_output_elements}"
    )

cpp_output = (
    torch.from_numpy(
        cpp_output_np.copy().reshape(
            OUTPUT_SHAPE
        )
    )
    .to(
        DEVICE,
        dtype=torch.float16,
    )
)

print(
    "C++ runtime output shape:",
    tuple(
        cpp_output.shape
    ),
)


# ============================================================
# 4. Load TensorRT engine
# ============================================================

print(
    "Loading TensorRT engine..."
)

logger = trt.Logger(
    trt.Logger.WARNING
)

with open(
    ENGINE_PATH,
    "rb",
) as f:
    engine_data = f.read()

runtime = trt.Runtime(
    logger
)

engine = (
    runtime.deserialize_cuda_engine(
        engine_data
    )
)

if engine is None:
    raise RuntimeError(
        "Failed to deserialize TensorRT engine"
    )


# ============================================================
# 5. Python TensorRT with HF preprocessing input
#
# HF Mel
#    ↓
# Python TensorRT
# ============================================================

print(
    "Running Python TensorRT "
    "with HF preprocessing input..."
)

python_trt_hf_input_output = run_trt(
    engine,
    hf_input_features,
)


# ============================================================
# 6. Python TensorRT with EXACT C++ preprocessing input
#
# C++ Mel
#    ↓
# Python TensorRT
#
# This is the important isolation test.
# ============================================================

print(
    "Running Python TensorRT "
    "with exact C++ preprocessing input..."
)

python_trt_cpp_input_output = run_trt(
    engine,
    cpp_input_features,
)


# ============================================================
# 7. Hugging Face encoder reference
#
# HF Mel
#    ↓
# HF Whisper encoder
# ============================================================

print(
    "Loading Hugging Face Whisper..."
)

hf_model = (
    WhisperModel.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
    )
    .to(DEVICE)
)

hf_model.eval()

print(
    "Running Hugging Face encoder..."
)

with torch.inference_mode():

    hf_output = (
        hf_model.encoder(
            hf_input_features
        )
        .last_hidden_state
    )

torch.cuda.synchronize()

print(
    "HF encoder output shape:",
    tuple(
        hf_output.shape
    ),
)


# ============================================================
# 8. Compare preprocessing itself
# ============================================================

compare(
    "C++ preprocessing vs "
    "Hugging Face preprocessing",
    hf_input_features,
    cpp_input_features,
)


# ============================================================
# 9. Python TRT vs Hugging Face
#
# Same HF input.
# Difference here = TRT engine numerical difference.
# ============================================================

compare(
    "Python TRT vs Hugging Face "
    "(HF MEL INPUT)",
    hf_output,
    python_trt_hf_input_output,
)


# ============================================================
# 10. Critical isolation test
#
# Both executions receive EXACT SAME C++ MEL INPUT.
#
# Python TensorRT
#      vs
# C++ TensorRT
#
# Ideally this should be identical or extremely close.
# ============================================================

compare(
    "C++ TensorRT vs Python TensorRT "
    "(EXACT C++ MEL INPUT)",
    python_trt_cpp_input_output,
    cpp_output,
)


# ============================================================
# 11. Complete C++ pipeline vs Hugging Face
#
# Different preprocessing + different runtime path.
# ============================================================

compare(
    "C++ AUDIO PIPELINE vs Hugging Face",
    hf_output,
    cpp_output,
)


# ============================================================
# 12. Full pipeline vs Python TensorRT with HF preprocessing
# ============================================================

compare(
    "C++ AUDIO PIPELINE vs Python TRT "
    "(HF MEL INPUT)",
    python_trt_hf_input_output,
    cpp_output,
)


print(
    "\n================================"
)

print(
    " Validation complete"
)

print(
    "================================"
)