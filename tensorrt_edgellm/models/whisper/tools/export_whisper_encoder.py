import os

import torch
from transformers import WhisperModel

from tensorrt_edgellm import config as config_module
from tensorrt_edgellm.models.whisper.modeling_whisper_audio import (
    build_whisper_audio,
)
from tensorrt_edgellm.onnx.export_encoder import _run_dynamo_export


MODEL_ID = "openai/whisper-small"

OUTPUT_DIR = "whisper_small_onnx/audio"
OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "model.onnx",
)


print("Loading Hugging Face Whisper Small...")

hf_model = WhisperModel.from_pretrained(
    MODEL_ID,
    attn_implementation="eager",
)

hf_model.eval()


# ---------------------------------------------------------------
# Hugging Face config + weights
# ---------------------------------------------------------------

config = hf_model.config.to_dict()

weights = hf_model.state_dict()


# ---------------------------------------------------------------
# Edge-LLM ModelConfig
# ---------------------------------------------------------------

model_config = config_module.ModelConfig(
    model_type="whisper",
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    num_key_value_heads=12,
    intermediate_size=3072,
    head_dim=64,
    rms_norm_eps=1e-5,
    vocab_size=51865,
    rope_theta=10000.0,
    max_position_embeddings=1500,
    default_attention_scale=64 ** -0.5,
)


# ---------------------------------------------------------------
# Build our validated Whisper encoder
# ---------------------------------------------------------------

print("Building TensorRT-Edge-LLM Whisper encoder...")

model = build_whisper_audio(
    config=config,
    weights=weights,
    dtype=torch.float16,
    model_config=model_config,
)

model = model.to("cpu").eval()


# ---------------------------------------------------------------
# Obtain ONNX export contract
# ---------------------------------------------------------------

(
    dynamo_inputs,
    input_names,
    output_names,
    dynamic_shapes,
) = model.get_onnx_export_args(
    config,
    device="cpu",
)


print(
    "Input:",
    dynamo_inputs["input_features"].shape,
)

print(
    "Output names:",
    output_names,
)


# ---------------------------------------------------------------
# Export using TensorRT-Edge-LLM's own Dynamo exporter
# ---------------------------------------------------------------

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

print(
    "Exporting ONNX to:",
    OUTPUT_PATH,
)

_run_dynamo_export(
    model,
    dynamo_inputs,
    OUTPUT_PATH,
    input_names,
    output_names,
    dynamic_shapes,
)

print("\nSUCCESS")
print("ONNX:", OUTPUT_PATH)