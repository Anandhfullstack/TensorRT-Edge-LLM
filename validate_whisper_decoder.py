from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


ONNX_PATH = Path(
    "whisper_small_onnx/decoder/model.onnx"
)

EXPECTED_INPUTS = [
    "input_ids",
    "position_ids",
    "past_key_values",
    "cross_key_values",
]

EXPECTED_OUTPUTS = [
    "logits",
    "present_key_values",
]


def print_session_contract(session):
    print("\nInputs:")

    for value in session.get_inputs():
        print(
            f"  {value.name}: "
            f"shape={value.shape}, type={value.type}"
        )

    print("\nOutputs:")

    for value in session.get_outputs():
        print(
            f"  {value.name}: "
            f"shape={value.shape}, type={value.type}"
        )


def main():
    if not ONNX_PATH.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {ONNX_PATH}"
        )

    # ---------------------------------------------------------
    # Structural validation
    # ---------------------------------------------------------

    print("Checking ONNX structure...")

    onnx.checker.check_model(
        str(ONNX_PATH),
        full_check=True,
    )

    print("ONNX structural check: PASS")

    # ---------------------------------------------------------
    # Select ONNX Runtime provider
    # ---------------------------------------------------------

    available = ort.get_available_providers()

    if "CUDAExecutionProvider" in available:
        providers = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    else:
        providers = ["CPUExecutionProvider"]

    print("Providers:", providers)

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=providers,
    )

    print_session_contract(session)

    actual_inputs = [
        value.name for value in session.get_inputs()
    ]

    actual_outputs = [
        value.name for value in session.get_outputs()
    ]

    assert actual_inputs == EXPECTED_INPUTS, (
        f"\nWrong inputs.\n"
        f"Expected: {EXPECTED_INPUTS}\n"
        f"Actual:   {actual_inputs}"
    )

    assert actual_outputs == EXPECTED_OUTPUTS, (
        f"\nWrong outputs.\n"
        f"Expected: {EXPECTED_OUTPUTS}\n"
        f"Actual:   {actual_outputs}"
    )

    # ---------------------------------------------------------
    # Cross-attention cache: produced once by the cross_kv engine and
    # bound read-only for every step. Zeros are enough here — this is a
    # contract/shape check, not a numerical one.
    # ---------------------------------------------------------

    cross_key_values = np.zeros(
        (12, 2, 1, 12, 1500, 64),
        dtype=np.float16,
    )

    # ---------------------------------------------------------
    # Prefill: four tokens, empty cache
    # ---------------------------------------------------------

    prefill_ids = np.array(
        [[50258, 50359, 50363, 2425]],
        dtype=np.int64,
    )

    prefill_positions = np.array(
        [[0, 1, 2, 3]],
        dtype=np.int64,
    )

    empty_cache = np.empty(
        (12, 2, 1, 12, 0, 64),
        dtype=np.float16,
    )

    prefill_logits, cache_4 = session.run(
        EXPECTED_OUTPUTS,
        {
            "input_ids": prefill_ids,
            "position_ids": prefill_positions,
            "past_key_values": empty_cache,
            "cross_key_values": cross_key_values,
        },
    )

    assert prefill_logits.shape == (
        1,
        4,
        51865,
    )

    assert cache_4.shape == (
        12,
        2,
        1,
        12,
        4,
        64,
    )

    assert np.isfinite(prefill_logits).all()
    assert np.isfinite(cache_4).all()

    next_token = int(
        np.argmax(prefill_logits[0, -1])
    )

    print("\nPrefill: PASS")
    print("  logits:", prefill_logits.shape)
    print("  cache:", cache_4.shape)
    print("  next token:", next_token)

    # ---------------------------------------------------------
    # Decode: one token, cache length four
    # ---------------------------------------------------------

    decode_ids = np.array(
        [[next_token]],
        dtype=np.int64,
    )

    decode_positions = np.array(
        [[4]],
        dtype=np.int64,
    )

    decode_logits, cache_5 = session.run(
        EXPECTED_OUTPUTS,
        {
            "input_ids": decode_ids,
            "position_ids": decode_positions,
            "past_key_values": cache_4,
            "cross_key_values": cross_key_values,
        },
    )

    assert decode_logits.shape == (
        1,
        1,
        51865,
    )

    assert cache_5.shape == (
        12,
        2,
        1,
        12,
        5,
        64,
    )

    assert np.isfinite(decode_logits).all()
    assert np.isfinite(cache_5).all()

    second_token = int(
        np.argmax(decode_logits[0, -1])
    )

    print("\nDecode: PASS")
    print("  logits:", decode_logits.shape)
    print("  cache:", cache_5.shape)
    print("  next token:", second_token)

    print("\n===================================")
    print("CACHED ONNX EXPORT VALIDATION: PASS")
    print("Cache growth: 0 -> 4 -> 5")
    print("===================================")


if __name__ == "__main__":
    main()