import os

import torch
from transformers import WhisperForConditionalGeneration

from tensorrt_edgellm.config import ModelConfig
from tensorrt_edgellm.models.whisper.modeling_whisper_decoder import (
    build_whisper_decoder_export,
)
from tensorrt_edgellm.onnx.export_encoder import _run_dynamo_export


MODEL_NAME = "openai/whisper-small"

OUTPUT_DIR = "whisper_small_onnx/decoder"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "model.onnx")


print("Loading Hugging Face Whisper Small...")

hf_model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
)

hf_model.eval()


# ---------------------------------------------------------------
# Hugging Face config + weights
# ---------------------------------------------------------------

hf_config = hf_model.config

decoder_config = {
    "d_model": hf_config.d_model,
    "decoder_layers": hf_config.decoder_layers,
    "decoder_attention_heads": hf_config.decoder_attention_heads,
    "decoder_ffn_dim": hf_config.decoder_ffn_dim,
    "vocab_size": hf_config.vocab_size,
    "max_target_positions": hf_config.max_target_positions,
    "max_source_positions": hf_config.max_source_positions,
    "pad_token_id": hf_config.pad_token_id,
}

weights = hf_model.state_dict()


# ---------------------------------------------------------------
# Edge-LLM ModelConfig
# ---------------------------------------------------------------

model_config = ModelConfig(
    model_type="whisper",
    hidden_size=hf_config.d_model,
    num_hidden_layers=hf_config.decoder_layers,
    num_attention_heads=hf_config.decoder_attention_heads,
    num_key_value_heads=hf_config.decoder_attention_heads,
    intermediate_size=hf_config.decoder_ffn_dim,
    head_dim=hf_config.d_model // hf_config.decoder_attention_heads,
    rms_norm_eps=1e-5,
    vocab_size=hf_config.vocab_size,
    rope_theta=10000.0,
    max_position_embeddings=hf_config.max_target_positions,
    default_attention_scale=(hf_config.d_model //
                             hf_config.decoder_attention_heads)**-0.5,
    torch_dtype="float16",
    tie_word_embeddings=True,
)


# ---------------------------------------------------------------
# Build our validated Whisper decoder (export-wrapped)
# ---------------------------------------------------------------

print("Building TensorRT-Edge-LLM Whisper decoder...")

# ``with_cache=True`` matches what ``tensorrt-edgellm-export`` ships, so what is
# debugged here is the graph that actually gets built into an engine.
model = build_whisper_decoder_export(
    config=decoder_config,
    weights=weights,
    dtype=torch.float16,
    prefix="model.decoder.",
    model_config=model_config,
    with_cache=True,
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
    decoder_config,
    device="cpu",
)

for name, tensor in zip(input_names, dynamo_inputs):
    print(f"  {name}: {tuple(tensor.shape)}")
print("Output names:", output_names)


# ---------------------------------------------------------------
# Export using TensorRT-Edge-LLM's own Dynamo exporter
# ---------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Exporting ONNX to:", OUTPUT_PATH)

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
