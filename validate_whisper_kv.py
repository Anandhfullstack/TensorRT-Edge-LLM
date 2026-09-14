import tensorrt as trt
import torch
import torch.nn.functional as F


ENGINE_PATH = (
    "whisper_small_onnx_with_cache/"
    "decoder/decoder_kv.engine"
)

LOGGER = trt.Logger(trt.Logger.WARNING)


def torch_dtype(trt_dtype):
    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT64: torch.int64,
        trt.DataType.BOOL: torch.bool,
    }

    if trt_dtype not in mapping:
        raise RuntimeError(
            f"Unsupported TensorRT datatype: {trt_dtype}"
        )

    return mapping[trt_dtype]


# ------------------------------------------------------------
# Load TensorRT engine
# ------------------------------------------------------------
with open(ENGINE_PATH, "rb") as file:
    engine_data = file.read()

runtime = trt.Runtime(LOGGER)
engine = runtime.deserialize_cuda_engine(engine_data)

if engine is None:
    raise RuntimeError("Failed to deserialize TensorRT engine")

context = engine.create_execution_context()

if context is None:
    raise RuntimeError("Failed to create execution context")


def binding_dtype(name):
    return torch_dtype(engine.get_tensor_dtype(name))


input_id_dtype = binding_dtype("input_ids")
position_dtype = binding_dtype("position_ids")
encoder_dtype = binding_dtype("encoder_hidden_states")
cache_dtype = binding_dtype("past_key_values")


# ------------------------------------------------------------
# Shared encoder output
#
# Random data is acceptable because both execution paths use
# exactly the same encoder_hidden_states.
# ------------------------------------------------------------
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

encoder_hidden_states = torch.randn(
    (1, 1500, 768),
    dtype=torch.float32,
    device="cuda",
).to(encoder_dtype)


# Five valid Whisper token IDs
tokens = torch.tensor(
    [[50258, 50259, 50359, 50363, 100]],
    dtype=input_id_dtype,
    device="cuda",
)

positions = torch.arange(
    5,
    dtype=position_dtype,
    device="cuda",
).unsqueeze(0)


# A non-null allocation is used as the address for the
# zero-volume initial cache.
empty_cache_storage = torch.zeros(
    1,
    dtype=cache_dtype,
    device="cuda",
)

empty_cache_shape = (12, 2, 1, 12, 0, 64)


def run_decoder(
    input_ids,
    position_ids,
    past_cache,
    past_cache_shape,
):
    inputs = {
        "input_ids": input_ids.contiguous(),
        "encoder_hidden_states": encoder_hidden_states,
        "position_ids": position_ids.contiguous(),
        "past_key_values": past_cache.contiguous(),
    }

    input_shapes = {
        "input_ids": tuple(input_ids.shape),
        "encoder_hidden_states": tuple(
            encoder_hidden_states.shape
        ),
        "position_ids": tuple(position_ids.shape),
        "past_key_values": tuple(past_cache_shape),
    }

    for name, tensor in inputs.items():
        success = context.set_input_shape(
            name,
            input_shapes[name],
        )

        if not success:
            raise RuntimeError(
                f"Failed to set {name} shape to "
                f"{input_shapes[name]}"
            )

        context.set_tensor_address(
            name,
            tensor.data_ptr(),
        )

    missing_tensors = context.infer_shapes()

    if missing_tensors:
        raise RuntimeError(
            f"TensorRT could not infer shapes: "
            f"{missing_tensors}"
        )

    outputs = {}

    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)

        if (
            engine.get_tensor_mode(name)
            != trt.TensorIOMode.OUTPUT
        ):
            continue

        shape = tuple(context.get_tensor_shape(name))

        if any(dimension < 0 for dimension in shape):
            raise RuntimeError(
                f"Unresolved output shape for {name}: {shape}"
            )

        output = torch.empty(
            shape,
            dtype=binding_dtype(name),
            device="cuda",
        )

        context.set_tensor_address(
            name,
            output.data_ptr(),
        )

        outputs[name] = output

    stream = torch.cuda.current_stream()

    success = context.execute_async_v3(
        stream_handle=stream.cuda_stream
    )

    if not success:
        raise RuntimeError("TensorRT execution failed")

    stream.synchronize()

    return outputs


# ------------------------------------------------------------
# Reference: process all five tokens without a cache
# ------------------------------------------------------------
full_outputs = run_decoder(
    input_ids=tokens,
    position_ids=positions,
    past_cache=empty_cache_storage,
    past_cache_shape=empty_cache_shape,
)


# ------------------------------------------------------------
# Prefill: process only the first four tokens
# ------------------------------------------------------------
prefill_outputs = run_decoder(
    input_ids=tokens[:, :4],
    position_ids=positions[:, :4],
    past_cache=empty_cache_storage,
    past_cache_shape=empty_cache_shape,
)

prefill_cache = prefill_outputs["present_key_values"]


# ------------------------------------------------------------
# Cached decode: process only token five
# ------------------------------------------------------------
cached_outputs = run_decoder(
    input_ids=tokens[:, 4:5],
    position_ids=positions[:, 4:5],
    past_cache=prefill_cache,
    past_cache_shape=tuple(prefill_cache.shape),
)


# ------------------------------------------------------------
# Compare the final-token logits
# ------------------------------------------------------------
full_logits = full_outputs["logits"][0, -1].float()
cached_logits = cached_outputs["logits"][0, -1].float()

difference = (full_logits - cached_logits).abs()

max_absolute_difference = difference.max().item()
mean_absolute_difference = difference.mean().item()

cosine_similarity = F.cosine_similarity(
    full_logits.unsqueeze(0),
    cached_logits.unsqueeze(0),
).item()

full_token = torch.argmax(full_logits).item()
cached_token = torch.argmax(cached_logits).item()


# Compare cache generated for the first four tokens
full_cache_first_four = full_outputs[
    "present_key_values"
][..., :4, :].float()

cache_difference = (
    full_cache_first_four
    - prefill_cache.float()
).abs().max().item()


print("\n=== KV Cache Validation ===")
print(
    "Prefill cache shape:",
    tuple(prefill_cache.shape),
)
print(
    "Cached output shape:",
    tuple(
        cached_outputs["present_key_values"].shape
    ),
)
print(
    "Cache max difference:",
    cache_difference,
)
print(
    "Logit max difference:",
    max_absolute_difference,
)
print(
    "Logit mean difference:",
    mean_absolute_difference,
)
print(
    "Logit cosine similarity:",
    cosine_similarity,
)
print("Full-run predicted token:", full_token)
print("Cached predicted token:", cached_token)


passed = (
    tuple(prefill_cache.shape)
    == (12, 2, 1, 12, 4, 64)
    and tuple(
        cached_outputs["present_key_values"].shape
    )
    == (12, 2, 1, 12, 5, 64)
    and cosine_similarity > 0.9999
    and full_token == cached_token
)

if passed:
    print("\nKV CACHE VALIDATION: PASSED")
else:
    print("\nKV CACHE VALIDATION: FAILED")
    raise SystemExit(1)