import math

import numpy as np
import soundfile as sf
import tensorrt as trt
import torch

from scipy.signal import resample_poly
from transformers import WhisperFeatureExtractor, WhisperModel


# ============================================================
# Configuration
# ============================================================

ENGINE_PATH = (
    "whisper_small_native_engine/audio/audio_encoder.engine"
)

MODEL_NAME = "openai/whisper-small"

AUDIO_PATH = (
    "/home/nvidia/AK/Projects/TensorRT-Edge-LLM/"
    "tensorrt-edgellm-workspace/audio.wav"
)

INPUT_NAME = "input_features"
OUTPUT_NAME = "last_hidden_state"

DEVICE = "cuda"

TARGET_SAMPLE_RATE = 16000

# True:
#   repeat/crop source audio to exactly 30 seconds.
#
# False:
#   use original audio and allow WhisperFeatureExtractor
#   to pad it to 30 seconds.
USE_FULL_30_SECOND_AUDIO = True


# ============================================================
# 1. Load audio
# ============================================================

print("Loading audio:", AUDIO_PATH)

waveform, sample_rate = sf.read(
    AUDIO_PATH,
    dtype="float32",
)

print("Original sample rate:", sample_rate)
print("Original waveform shape:", waveform.shape)


# ============================================================
# 2. Convert stereo -> mono
# ============================================================

if waveform.ndim == 2:
    waveform = waveform.mean(
        axis=1
    ).astype(np.float32)

waveform = np.asarray(
    waveform,
    dtype=np.float32,
)

print(
    "Mono waveform shape:",
    waveform.shape,
)


# ============================================================
# 3. Resample to 16 kHz
# ============================================================

if sample_rate != TARGET_SAMPLE_RATE:

    divisor = math.gcd(
        sample_rate,
        TARGET_SAMPLE_RATE,
    )

    up = TARGET_SAMPLE_RATE // divisor
    down = sample_rate // divisor

    print(
        f"Resampling: "
        f"{sample_rate} Hz -> "
        f"{TARGET_SAMPLE_RATE} Hz"
    )

    waveform = resample_poly(
        waveform,
        up,
        down,
    ).astype(np.float32)

    sample_rate = TARGET_SAMPLE_RATE


# ============================================================
# IMPORTANT:
# Save the ORIGINAL resampled waveform here.
#
# This is the waveform that the real C++ runtime will receive.
# ============================================================

runtime_waveform = waveform.copy()


RUNTIME_AUDIO_16K = "whisper_runtime_16k.wav"

sf.write(
    RUNTIME_AUDIO_16K,
    runtime_waveform,
    TARGET_SAMPLE_RATE,
    subtype="FLOAT",
)

print(
    "Saved 16-kHz runtime audio:",
    RUNTIME_AUDIO_16K,
)

original_num_samples = len(runtime_waveform)

original_duration = (
    original_num_samples
    / sample_rate
)

print(
    f"Original resampled duration: "
    f"{original_duration:.3f} seconds"
)

print(
    "Original resampled samples:",
    original_num_samples,
)


# ============================================================
# 4. Create waveform for encoder validation
# ============================================================

TARGET_SECONDS = 30

TARGET_SAMPLES = (
    TARGET_SECONDS
    * TARGET_SAMPLE_RATE
)  # 480000


if USE_FULL_30_SECOND_AUDIO:

    # --------------------------------------------------------
    # This is ONLY for our encoder numerical stress test.
    #
    # Repeat the 9.94-second recording until exactly 30 seconds.
    #
    # Do NOT modify runtime_waveform.
    # --------------------------------------------------------

    repeats = math.ceil(
        TARGET_SAMPLES
        / len(runtime_waveform)
    )

    validation_waveform = np.tile(
        runtime_waveform,
        repeats,
    )

    validation_waveform = validation_waveform[
        :TARGET_SAMPLES
    ].astype(np.float32)

    print(
        "30-second validation waveform:",
        len(validation_waveform)
        / sample_rate,
        "seconds",
    )

else:

    # Use original audio.
    # WhisperFeatureExtractor will zero-pad it to 30 seconds.
    validation_waveform = (
        runtime_waveform.copy()
    )

    print(
        "Using original waveform:",
        len(validation_waveform)
        / sample_rate,
        "seconds",
    )


# ============================================================
# 5. Create Whisper log-Mel features
#    FOR ENCODER VALIDATION
# ============================================================

print(
    "\nLoading WhisperFeatureExtractor..."
)

feature_extractor = (
    WhisperFeatureExtractor.from_pretrained(
        MODEL_NAME
    )
)


features = feature_extractor(
    validation_waveform,
    sampling_rate=TARGET_SAMPLE_RATE,

    # Make the Whisper 30-second contract explicit.
    padding="max_length",
    max_length=TARGET_SAMPLES,
    truncation=True,

    return_tensors="pt",
)


input_features = (
    features.input_features.to(
        device=DEVICE,
        dtype=torch.float16,
    )
)


print("\n================================")
print(" Encoder validation input")
print("================================")

print(
    "source duration:",
    len(validation_waveform)
    / TARGET_SAMPLE_RATE,
    "seconds",
)

print(
    "shape :",
    tuple(input_features.shape),
)

print(
    "dtype :",
    input_features.dtype,
)

print(
    "device:",
    input_features.device,
)

print(
    "min   :",
    input_features.min().item(),
)

print(
    "max   :",
    input_features.max().item(),
)

print(
    "mean  :",
    input_features.float().mean().item(),
)


expected_input_shape = (
    1,
    80,
    3000,
)

if (
    tuple(input_features.shape)
    != expected_input_shape
):
    raise RuntimeError(
        "Unexpected Whisper input shape. "
        f"Expected {expected_input_shape}, "
        f"got {tuple(input_features.shape)}"
    )

# ============================================================
# 6. Hugging Face reference
# ============================================================

print(
    "\nLoading Hugging Face Whisper..."
)

hf_model = (
    WhisperModel.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
    ).to(DEVICE)
)

hf_model.eval()


print(
    "Running Hugging Face encoder..."
)

with torch.inference_mode():

    hf_output = (
        hf_model.encoder(
            input_features
        ).last_hidden_state
    )


torch.cuda.synchronize()


print("\n================================")
print(" Hugging Face output")
print("================================")

print(
    "shape:",
    tuple(hf_output.shape),
)

print(
    "dtype:",
    hf_output.dtype,
)

print(
    "min/max:",
    hf_output.min().item(),
    hf_output.max().item(),
)


print(
    "\nValidation finished."
)

# ============================================================
# 7. Load TensorRT engine
# ============================================================

print(
    "\nLoading TensorRT engine..."
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
        "Failed to deserialize "
        "TensorRT engine"
    )


context = (
    engine.create_execution_context()
)


if context is None:

    raise RuntimeError(
        "Failed to create "
        "TensorRT execution context"
    )


# ============================================================
# 8. Print engine I/O
# ============================================================

print("\n================================")
print(" TensorRT engine I/O")
print("================================")


for i in range(
    engine.num_io_tensors
):

    name = (
        engine.get_tensor_name(i)
    )

    print(
        f"{name}: "
        f"mode="
        f"{engine.get_tensor_mode(name)}, "
        f"dtype="
        f"{engine.get_tensor_dtype(name)}, "
        f"shape="
        f"{engine.get_tensor_shape(name)}"
    )


# ============================================================
# 9. Set TensorRT input shape
# ============================================================

input_shape = tuple(
    input_features.shape
)


success = context.set_input_shape(
    INPUT_NAME,
    input_shape,
)


if not success:

    raise RuntimeError(
        "Failed to set input shape "
        f"{INPUT_NAME}: {input_shape}"
    )


output_shape = tuple(
    context.get_tensor_shape(
        OUTPUT_NAME
    )
)


print(
    "\nResolved TensorRT "
    "output shape:",
    output_shape,
)


expected_output_shape = (
    1,
    1500,
    768,
)


if (
    output_shape
    != expected_output_shape
):

    raise RuntimeError(
        "Unexpected TensorRT "
        "output shape. "
        f"Expected "
        f"{expected_output_shape}, "
        f"got {output_shape}"
    )


# ============================================================
# 10. Allocate TensorRT output
# ============================================================

trt_output = torch.empty(
    output_shape,
    dtype=torch.float16,
    device=DEVICE,
)


# ============================================================
# 11. Bind tensors
# ============================================================

if not context.set_tensor_address(
    INPUT_NAME,
    input_features.data_ptr(),
):

    raise RuntimeError(
        "Failed to bind "
        "TensorRT input tensor"
    )


if not context.set_tensor_address(
    OUTPUT_NAME,
    trt_output.data_ptr(),
):

    raise RuntimeError(
        "Failed to bind "
        "TensorRT output tensor"
    )


# ============================================================
# 12. Run TensorRT
# ============================================================

print(
    "\nRunning TensorRT encoder..."
)


current_stream = (
    torch.cuda.current_stream()
)

trt_stream = (
    torch.cuda.Stream()
)


trt_stream.wait_stream(
    current_stream
)


success = context.execute_async_v3(
    stream_handle=
    trt_stream.cuda_stream
)


if not success:

    raise RuntimeError(
        "TensorRT execution failed"
    )


trt_stream.synchronize()


print("\n================================")
print(" TensorRT output")
print("================================")

print(
    "shape:",
    tuple(trt_output.shape),
)

print(
    "dtype:",
    trt_output.dtype,
)

print(
    "min/max:",
    trt_output.min().item(),
    trt_output.max().item(),
)


# ============================================================
# 13. Numerical difference
# ============================================================

hf_fp32 = (
    hf_output.float()
)

trt_fp32 = (
    trt_output.float()
)


diff = torch.abs(
    hf_fp32
    - trt_fp32
)


# ============================================================
# Helper: region statistics
# ============================================================

def print_region_stats(
    name,
    region,
):

    print(
        f"\n{name}"
    )

    if region.numel() == 0:

        print(
            " No elements in this region."
        )

        return


    flat = region.flatten()

    print(
        " elements :",
        region.numel(),
    )

    print(
        " mean diff:",
        region.mean().item(),
    )

    print(
        " max diff :",
        region.max().item(),
    )

    print(
        " 99%      :",
        torch.quantile(
            flat,
            0.99,
        ).item(),
    )

    print(
        " 99.9%    :",
        torch.quantile(
            flat,
            0.999,
        ).item(),
    )


    for threshold in [
        0.05,
        0.1,
        0.5,
        1.0,
    ]:

        count = (
            region > threshold
        ).sum().item()

        percentage = (
            count
            / region.numel()
            * 100.0
        )

        print(
            f" > {threshold:<4}: "
            f"{count} "
            f"({percentage:.6f}%)"
        )


# ============================================================
# 14. Audio region analysis
# ============================================================

if USE_FULL_30_SECOND_AUDIO:

    # Entire 30-second input is populated
    # with repeated real waveform.
    valid_mel_frames = 3000

else:

    valid_mel_frames = math.ceil(
        original_num_samples
        / feature_extractor.hop_length
    )

    valid_mel_frames = min(
        valid_mel_frames,
        3000,
    )


valid_encoder_positions = (
    math.ceil(
        valid_mel_frames
        / 2
    )
)


valid_encoder_positions = min(
    valid_encoder_positions,
    diff.shape[1],
)


padded_encoder_positions = (
    diff.shape[1]
    - valid_encoder_positions
)


print("\n================================")
print(" Audio region analysis")
print("================================")

print(
    "Valid mel frames        :",
    valid_mel_frames,
)

print(
    "Valid encoder positions :",
    valid_encoder_positions,
)

print(
    "Padded encoder positions:",
    padded_encoder_positions,
)


valid_diff = diff[
    :,
    :valid_encoder_positions,
    :
]


padded_diff = diff[
    :,
    valid_encoder_positions:,
    :
]


print_region_stats(
    "REAL AUDIO REGION",
    valid_diff,
)


if padded_diff.numel() > 0:

    print_region_stats(
        "PADDED REGION",
        padded_diff,
    )

else:

    print(
        "\nPADDED REGION"
    )

    print(
        " No padded encoder "
        "positions in this test."
    )


# ============================================================
# 15. Large-error locations
# ============================================================

large_error_indices = (
    torch.nonzero(
        diff > 0.5
    )
)


print("\n================================")
print(" Large errors > 0.5")
print("================================")


if (
    large_error_indices.numel()
    == 0
):

    print(
        "No values exceed 0.5"
    )

else:

    positions = (
        large_error_indices[:, 1]
    )

    valid_large = (
        positions
        < valid_encoder_positions
    ).sum().item()

    padded_large = (
        positions
        >= valid_encoder_positions
    ).sum().item()


    print(
        "Real audio region:",
        valid_large,
    )

    print(
        "Padded region    :",
        padded_large,
    )

    print(
        "Earliest position:",
        positions.min().item(),
    )

    print(
        "Latest position  :",
        positions.max().item(),
    )


# ============================================================
# 16. Basic comparison
# ============================================================

print("\n================================")
print(" HF vs TensorRT comparison")
print("================================")


print(
    "HF shape       :",
    tuple(hf_output.shape),
)

print(
    "TRT shape      :",
    tuple(trt_output.shape),
)

print(
    "Max abs diff   :",
    diff.max().item(),
)

print(
    "Mean abs diff  :",
    diff.mean().item(),
)

print(
    "Median abs diff:",
    diff.median().item(),
)

print(
    "HF min/max     :",
    hf_fp32.min().item(),
    hf_fp32.max().item(),
)

print(
    "TRT min/max    :",
    trt_fp32.min().item(),
    trt_fp32.max().item(),
)


# ============================================================
# 17. Allclose
# ============================================================

print(
    "allclose 1e-2  :",
    torch.allclose(
        hf_fp32,
        trt_fp32,
        rtol=1e-2,
        atol=1e-2,
    ),
)

print(
    "allclose 5e-2  :",
    torch.allclose(
        hf_fp32,
        trt_fp32,
        rtol=5e-2,
        atol=5e-2,
    ),
)


# ============================================================
# 18. Error percentiles
# ============================================================

flat_diff = (
    diff.flatten()
)


print("\n================================")
print(" Error percentiles")
print("================================")


for q in [
    0.50,
    0.90,
    0.95,
    0.99,
    0.999,
]:

    value = torch.quantile(
        flat_diff,
        q,
    ).item()

    print(
        f"{q * 100:6.1f}% : "
        f"{value}"
    )


# ============================================================
# 19. Threshold counts
# ============================================================

total = (
    diff.numel()
)


print("\n================================")
print(" Elements exceeding thresholds")
print("================================")


for threshold in [
    0.01,
    0.02,
    0.05,
    0.1,
    0.5,
    1.0,
]:

    count = (
        diff > threshold
    ).sum().item()

    percentage = (
        count
        / total
        * 100.0
    )

    print(
        f"> {threshold:<4}: "
        f"{count:8d} / "
        f"{total} "
        f"({percentage:.6f}%)"
    )


# ============================================================
# 20. RMSE
# ============================================================

rmse = torch.sqrt(
    torch.mean(
        (
            hf_fp32
            - trt_fp32
        ) ** 2
    )
)


# ============================================================
# 21. Cosine similarity
# ============================================================

cosine = (
    torch.nn.functional
    .cosine_similarity(
        hf_fp32.flatten(),
        trt_fp32.flatten(),
        dim=0,
    )
)


print("\n================================")
print(" Global similarity")
print("================================")


print(
    "RMSE              :",
    rmse.item(),
)

print(
    "Cosine similarity :",
    cosine.item(),
)


# ============================================================
# 22. Worst difference
# ============================================================

max_index = (
    torch.argmax(diff)
)


index = torch.unravel_index(
    max_index,
    diff.shape,
)


index_tuple = tuple(
    value.item()
    for value in index
)


print("\n================================")
print(" Worst difference")
print("================================")


print(
    "index:",
    index_tuple,
)

print(
    "HF   :",
    hf_fp32[index].item(),
)

print(
    "TRT  :",
    trt_fp32[index].item(),
)

print(
    "diff :",
    diff[index].item(),
)


# ============================================================
# 23. Final validation summary
# ============================================================

print("\n================================")
print(" Validation summary")
print("================================")


print(
    "Input shape  :",
    tuple(input_features.shape),
)

print(
    "Output shape :",
    tuple(trt_output.shape),
)

print(
    "Mean diff    :",
    diff.mean().item(),
)

print(
    "Median diff  :",
    diff.median().item(),
)

print(
    "99% diff     :",
    torch.quantile(
        flat_diff,
        0.99,
    ).item(),
)

print(
    "99.9% diff   :",
    torch.quantile(
        flat_diff,
        0.999,
    ).item(),
)

print(
    "Max diff     :",
    diff.max().item(),
)

print(
    "RMSE         :",
    rmse.item(),
)

print(
    "Cosine       :",
    cosine.item(),
)


above_half = (
    diff > 0.5
).sum().item()

above_one = (
    diff > 1.0
).sum().item()


print(
    "> 0.5        :",
    above_half,
    f"({above_half / total * 100:.6f}%)",
)

print(
    "> 1.0        :",
    above_one,
    f"({above_one / total * 100:.6f}%)",
)


print(
    "\nValidation finished."
)

# ============================================================
# Save tensors for C++ runtime validation
# ============================================================

INPUT_BIN = "whisper_input_fp16.bin"
HF_OUTPUT_BIN = "whisper_hf_output_fp16.bin"
TRT_OUTPUT_BIN = "whisper_python_trt_output_fp16.bin"


input_features.detach().cpu().contiguous().numpy().astype(
    np.float16
).tofile(INPUT_BIN)

hf_output.detach().cpu().contiguous().numpy().astype(
    np.float16
).tofile(HF_OUTPUT_BIN)

trt_output.detach().cpu().contiguous().numpy().astype(
    np.float16
).tofile(TRT_OUTPUT_BIN)


print("\n================================")
print(" Saved C++ validation tensors")
print("================================")

print(
    INPUT_BIN,
    "shape =",
    tuple(input_features.shape),
)

print(
    HF_OUTPUT_BIN,
    "shape =",
    tuple(hf_output.shape),
)

print(
    TRT_OUTPUT_BIN,
    "shape =",
    tuple(trt_output.shape),
)


# ============================================================
# Runtime preprocessing reference
#
# This is DIFFERENT from whisper_input_fp16.bin.
#
# runtime_waveform:
#   original audio after resampling to 16 kHz (~9.94 sec)
#
# WhisperFeatureExtractor:
#   zero-pads it to 30 seconds / 480000 samples
#
# This is what our C++ MelExtractor must reproduce.
# ============================================================

RUNTIME_INPUT_BIN = (
    "whisper_runtime_input_fp16.bin"
)


print("\n================================")
print(" Creating runtime reference")
print("================================")

print(
    "Original runtime samples:",
    len(runtime_waveform),
)

print(
    "Original runtime duration:",
    len(runtime_waveform)
    / TARGET_SAMPLE_RATE,
    "seconds",
)


runtime_features = feature_extractor(
    runtime_waveform,

    sampling_rate=TARGET_SAMPLE_RATE,

    padding="max_length",

    max_length=TARGET_SAMPLES,

    truncation=True,

    return_tensors="pt",
)


runtime_input_features = (
    runtime_features.input_features
    .to(dtype=torch.float16)
)


# ============================================================
# Validate runtime feature shape
# ============================================================

if (
    tuple(runtime_input_features.shape)
    != (1, 80, 3000)
):

    raise RuntimeError(
        "Unexpected runtime "
        "Whisper feature shape: "
        f"{tuple(runtime_input_features.shape)}"
    )


print("\n================================")
print(" Runtime preprocessing reference")
print("================================")

print(
    "Original audio duration:",
    len(runtime_waveform)
    / TARGET_SAMPLE_RATE,
    "seconds",
)

print(
    "shape:",
    tuple(
        runtime_input_features.shape
    ),
)

print(
    "dtype:",
    runtime_input_features.dtype,
)

print(
    "min:",
    runtime_input_features.min().item(),
)

print(
    "max:",
    runtime_input_features.max().item(),
)

print(
    "mean:",
    runtime_input_features
    .float()
    .mean()
    .item(),
)


# ============================================================
# Save runtime preprocessing reference
# ============================================================

runtime_np = (
    runtime_input_features
    .cpu()
    .contiguous()
    .numpy()
    .astype(np.float16)
)


runtime_np.tofile(
    RUNTIME_INPUT_BIN
)


print("\n================================")
print(" Saved runtime reference")
print("================================")

print(
    "File:",
    RUNTIME_INPUT_BIN,
)

print(
    "Elements:",
    runtime_np.size,
)

print(
    "Bytes:",
    runtime_np.nbytes,
)


# ============================================================
# Sanity check:
#
# Runtime input SHOULD NOT be identical to the
# repeated 30-second encoder validation input.
# ============================================================

encoder_validation_np = (
    input_features
    .detach()
    .cpu()
    .contiguous()
    .numpy()
    .astype(np.float16)
)


same = np.array_equal(
    runtime_np,
    encoder_validation_np,
)


print("\n================================")
print(" Reference separation check")
print("================================")

print(
    "Encoder validation source :",
    len(validation_waveform)
    / TARGET_SAMPLE_RATE,
    "sec",
)

print(
    "Runtime source            :",
    len(runtime_waveform)
    / TARGET_SAMPLE_RATE,
    "sec",
)

print(
    "Feature tensors identical :",
    same,
)


if (
    USE_FULL_30_SECOND_AUDIO
    and same
):
    raise RuntimeError(
        "ERROR: runtime reference "
        "is identical to the repeated "
        "30-second validation reference."
    )


print(
    "\nRuntime preprocessing "
    "reference created successfully."
)