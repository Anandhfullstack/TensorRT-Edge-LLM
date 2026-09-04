from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import config as config_module
from .. import ops
from ..linear import make_linear

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default architecture constants (Qwen3-ASR / Qwen3-Omni)
# ---------------------------------------------------------------------------

_D_MODEL = 768
_NUM_LAYERS = 12
_NUM_HEADS = 12
_FFN_DIM = 3072
_NUM_MEL_BINS = 80
_MAX_SOURCE_POSITIONS = 1500


class WhisperPositionalEmbedding(nn.Embedding):
    """Fixed positional embedding loaded from the Whisper checkpoint."""

    def __init__(
        self,
        length: int = _MAX_SOURCE_POSITIONS,
        channels: int = _D_MODEL,
    ) -> None:
        super().__init__(
            num_embeddings=length,
            embedding_dim=channels,
        )

        # Whisper encoder positional embeddings are fixed.
        self.weight.requires_grad_(False)

    def forward(
        self,
        seqlen: int,
    ) -> torch.Tensor:

        return self.weight[:seqlen, :]


class WhisperAudioEncoder(nn.Module):

    def __init__(
        self,
        model_config: config_module.ModelConfig,
        num_mel_bins: int = _NUM_MEL_BINS,
        d_model: int = _D_MODEL,
        num_layers: int = _NUM_LAYERS,
        num_heads: int = _NUM_HEADS,
        ffn_dim: int = _FFN_DIM,
        max_source_positions: int = _MAX_SOURCE_POSITIONS,
        name_prefix: str = "encoder",
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(
            num_mel_bins,
            d_model,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.conv2 = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.embed_positions = WhisperPositionalEmbedding(
            max_source_positions,
            d_model,
        )

        self.layers = nn.ModuleList([
            WhisperAudioEncoderLayer(
                model_config=model_config,
                d_model=d_model,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                name_prefix=f"{name_prefix}.layers.{i}",
            )
            for i in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_features: [B, 80, 3000]

        Returns:
            hidden_states: [B, 1500, 768]
        """

        # [B, 80, 3000] -> [B, 768, 3000]
        x = F.gelu(self.conv1(input_features))

        # [B, 768, 3000] -> [B, 768, 1500]
        x = F.gelu(self.conv2(x))

        # [B, 768, 1500] -> [B, 1500, 768]
        x = x.permute(0, 2, 1)

        # Positional embedding: [1500, 768]
        pos = self.embed_positions(x.shape[1])

        # [1500, 768] -> [1, 1500, 768]
        pos = pos.unsqueeze(0).to(x.dtype)

        # Add position information
        x = x + pos

        for layer in self.layers:
            x = layer(x)

        x = self.layer_norm(x)

        return x
    
    def get_onnx_export_args(
            self,
            config: dict,
            device: str,
        ):
            """Return ONNX export inputs for the Whisper audio encoder."""
    
            num_mel_bins = config.get(
                "num_mel_bins",
                _NUM_MEL_BINS,
            )
    
            # Standard Whisper input:
            # [batch, 80, 3000]
            input_features = torch.zeros(
                1,
                num_mel_bins,
                3000,
                dtype=torch.float16,
                device=device,
            )
    
            # Input tensor passed into forward()
            dynamo_inputs = {
                "input_features": input_features,
            }
    
            # ONNX input name
            onnx_input_names = [
                "input_features",
            ]
    
            # ONNX output name
            output_names = [
                "last_hidden_state",
            ]
    
            # Allow batch size to vary.
            batch_dim = torch.export.Dim("batch")
    
            dynamic_shapes = {
                "input_features": {
                    0: batch_dim,
                },
            }
    
            return (
                dynamo_inputs,
                onnx_input_names,
                output_names,
                dynamic_shapes,
            )
    




# ---------------------------------------------------------------------------
# Whisper Encoder Self-Attention
# ---------------------------------------------------------------------------


class WhisperAudioAttention(nn.Module):
    """Multi-head self-attention used by the Whisper audio encoder.

    Input:
        hidden_states: [B, T, d_model]

    Output:
        [B, T, d_model]

    For Whisper Small:
        d_model  = 768
        num_heads = 12
        head_dim  = 64
    """

    def __init__(
        self,
        model_config: config_module.ModelConfig,
        d_model: int = _D_MODEL,
        num_heads: int = _NUM_HEADS,
        name_prefix: str = "",
    ) -> None:
        super().__init__()

        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Equivalent to Whisper's attention scaling.
        self.attention_scale = self.head_dim ** -0.5

        self.q_proj = make_linear(
            model_config,
            d_model,
            d_model,
            bias=True,
            module_name=f"{name_prefix}.q_proj" if name_prefix else "",
        )

        # IMPORTANT:
        # Whisper key projection has NO bias.
        self.k_proj = make_linear(
            model_config,
            d_model,
            d_model,
            bias=False,
            module_name=f"{name_prefix}.k_proj" if name_prefix else "",
        )

        self.v_proj = make_linear(
            model_config,
            d_model,
            d_model,
            bias=True,
            module_name=f"{name_prefix}.v_proj" if name_prefix else "",
        )

        self.out_proj = make_linear(
            model_config,
            d_model,
            d_model,
            bias=True,
            module_name=f"{name_prefix}.out_proj" if name_prefix else "",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states:
                [B, T, d_model]

        Returns:
            attention_output:
                [B, T, d_model]
        """

        batch_size, seq_len, _ = hidden_states.shape

        # ---------------------------------------------------------------
        # 1. Q / K / V projections
        #
        # [B, T, 768]
        #       ↓
        # [B, T, 768]
        # ---------------------------------------------------------------

        q = self.q_proj(hidden_states) * self.attention_scale
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # ---------------------------------------------------------------
        # 2. Split into attention heads
        #
        # [B, T, 768]
        #
        #       ↓
        #
        # [B, T, 12, 64]
        #
        #       ↓ transpose
        #
        # [B, 12, T, 64]
        # ---------------------------------------------------------------

        q = q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        # ---------------------------------------------------------------
        # 3. Q @ K^T
        #
        # Q: [B, 12, T, 64]
        # K: [B, 12, T, 64]
        #
        #        ↓
        #
        # scores: [B, 12, T, T]
        # ---------------------------------------------------------------

        scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        )

        # scores = scores * self.attention_scale

        # ---------------------------------------------------------------
        # 4. Softmax
        # ---------------------------------------------------------------

        # attention_weights = torch.softmax(
        #     scores.float(),
        #     dim=-1,
        # ).to(q.dtype)

        attention_weights = torch.softmax(
            scores,
            dim=-1,
        )

        # ---------------------------------------------------------------
        # 5. Weighted Value
        #
        # [B, 12, T, T]
        #       ×
        # [B, 12, T, 64]
        #
        #       ↓
        #
        # [B, 12, T, 64]
        # ---------------------------------------------------------------

        attention_output = torch.matmul(
            attention_weights,
            v,
        )

        # ---------------------------------------------------------------
        # 6. Merge heads
        #
        # [B, 12, T, 64]
        #
        #       ↓
        #
        # [B, T, 12, 64]
        #
        #       ↓
        #
        # [B, T, 768]
        # ---------------------------------------------------------------

        attention_output = (
            attention_output
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, -1)
        )

        # ---------------------------------------------------------------
        # 7. Output projection
        # ---------------------------------------------------------------

        attention_output = self.out_proj(
            attention_output
        )

        return attention_output

    
# ---------------------------------------------------------------------------
# Whisper Encoder Layer
# ---------------------------------------------------------------------------


class WhisperAudioEncoderLayer(nn.Module):
    """Single Transformer block used by the Whisper audio encoder."""

    def __init__(
        self,
        model_config: config_module.ModelConfig,
        d_model: int = _D_MODEL,
        num_heads: int = _NUM_HEADS,
        ffn_dim: int = _FFN_DIM,
        name_prefix: str = "",
    ) -> None:
        super().__init__()

        # LayerNorm before self-attention
        self.self_attn_layer_norm = nn.LayerNorm(d_model)

        # Multi-head self-attention
        self.self_attn = WhisperAudioAttention(
            model_config=model_config,
            d_model=d_model,
            num_heads=num_heads,
            name_prefix=(
                f"{name_prefix}.self_attn"
                if name_prefix
                else ""
            ),
        )

        # LayerNorm before FFN
        self.final_layer_norm = nn.LayerNorm(d_model)

        # FFN: 768 -> 3072
        self.fc1 = make_linear(
            model_config,
            d_model,
            ffn_dim,
            bias=True,
            module_name=(
                f"{name_prefix}.fc1"
                if name_prefix
                else ""
            ),
        )

        # FFN: 3072 -> 768
        self.fc2 = make_linear(
            model_config,
            ffn_dim,
            d_model,
            bias=True,
            module_name=(
                f"{name_prefix}.fc2"
                if name_prefix
                else ""
            ),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:

        # ---------------------------------------------------------------
        # Self-attention block
        # ---------------------------------------------------------------

        residual = hidden_states

        hidden_states = self.self_attn_layer_norm(
            hidden_states
        )

        hidden_states = self.self_attn(
            hidden_states
        )

        hidden_states = residual + hidden_states

        # ---------------------------------------------------------------
        # Feed-forward block
        # ---------------------------------------------------------------

        residual = hidden_states

        hidden_states = self.final_layer_norm(
            hidden_states
        )

        hidden_states = self.fc1(
            hidden_states
        )

        hidden_states = F.gelu(
            hidden_states
        )

        hidden_states = self.fc2(
            hidden_states
        )

        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# Whisper Encoder Weight Loading
# ---------------------------------------------------------------------------


_CANDIDATE_PREFIXES = (
    "model.encoder.",
    "encoder.",
    "",
)


def _load_audio_weights(
    model: WhisperAudioEncoder,
    weights: Dict[str, torch.Tensor],
    dtype: torch.dtype = torch.float16,
    prefix: Optional[str] = None,
) -> None:
    """Load Hugging Face Whisper encoder weights."""

    from ...checkpoint.loader import load_submodule_weights
    # ---------------------------------------------------------------
    # 1. Detect checkpoint prefix
    # ---------------------------------------------------------------
    if prefix is None:
        for candidate in _CANDIDATE_PREFIXES:

            if candidate == "":
                prefix = candidate
                break

            if any(
                key.startswith(candidate)
                for key in weights.keys()
            ):
                prefix = candidate
                break

    if prefix is None:
        prefix = ""

    print("Whisper encoder weight prefix:", repr(prefix))

    # ---------------------------------------------------------------
    # 2. Select only encoder weights
    #    and strip "model.encoder."
    # ---------------------------------------------------------------

    stripped: Dict[str, torch.Tensor] = {}

    for key, value in weights.items():

        if prefix and not key.startswith(prefix):
            continue

        stripped_key = (
            key[len(prefix):]
            if prefix
            else key
        )

        stripped[stripped_key] = value

    # ---------------------------------------------------------------
    # 3. Key remapping
    # ---------------------------------------------------------------

    def _key_remap(key: str) -> Optional[str]:
        return key

    # ---------------------------------------------------------------
    # 4. Convert normal HF FP32 weights to Edge-LLM FP16
    # ---------------------------------------------------------------

    def _transform(
        key: str,
        tensor: torch.Tensor,
    ) -> torch.Tensor:

        if tensor.dtype in (
            torch.float32,
            torch.bfloat16,
        ):
            return tensor.to(dtype)

        return tensor

    # ---------------------------------------------------------------
    # 5. Load into our encoder
    # ---------------------------------------------------------------

    load_submodule_weights(
        model,
        stripped,
        key_remap=_key_remap,
        transform=_transform,
        label="WhisperAudioEncoder",
        log=logger,
    )


# ---------------------------------------------------------------------------
# Whisper Encoder Factory
# ---------------------------------------------------------------------------


def build_whisper_audio(
    config: Dict[str, Any],
    weights: Dict[str, torch.Tensor],
    dtype: torch.dtype = torch.float16,
    prefix: Optional[str] = None,
    *,
    model_config: config_module.ModelConfig,
) -> WhisperAudioEncoder:
    """Build Whisper audio encoder and load Hugging Face weights."""

    d_model = config.get(
        "d_model",
        _D_MODEL,
    )

    num_heads = config.get(
        "encoder_attention_heads",
        _NUM_HEADS,
    )

    model = WhisperAudioEncoder(
        model_config=model_config,

        num_mel_bins=config.get(
            "num_mel_bins",
            _NUM_MEL_BINS,
        ),

        d_model=d_model,

        num_layers=config.get(
            "encoder_layers",
            _NUM_LAYERS,
        ),

        num_heads=num_heads,

        ffn_dim=config.get(
            "encoder_ffn_dim",
            _FFN_DIM,
        ),

        max_source_positions=config.get(
            "max_source_positions",
            _MAX_SOURCE_POSITIONS,
        ),

        name_prefix="encoder",
    )

    # Conv1D, LayerNorm and FP16 linear components
    model = model.to(dtype=dtype)

    _load_audio_weights(
        model,
        weights,
        dtype=dtype,
        prefix=prefix,
    )

    model.eval()

    return model

if __name__ == "__main__":

    from transformers import WhisperModel

    print("Loading Hugging Face Whisper Small...")

    hf_model = WhisperModel.from_pretrained(
        "openai/whisper-small",
        attn_implementation="eager",
    )

    hf_model.eval()

    # ---------------------------------------------------------------
    # 1. Get HF config + weights
    # ---------------------------------------------------------------

    config = hf_model.config.to_dict()

    weights = hf_model.state_dict()

    print("HF tensors:", len(weights))

    # ---------------------------------------------------------------
    # 2. Create Edge-LLM ModelConfig
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
    # 3. Build our Whisper encoder using REAL HF weights
    # ---------------------------------------------------------------

    our_model = build_whisper_audio(
        config=config,
        weights=weights,
        dtype=torch.float16,
        model_config=model_config,
    )

    print("Whisper Edge-LLM encoder created successfully")

    # ---------------------------------------------------------------
    # 4. Verify some important weights
    # ---------------------------------------------------------------

    def compare(name, hf_tensor, our_tensor):

        hf_tensor = hf_tensor.detach().cpu().to(torch.float16)
        our_tensor = our_tensor.detach().cpu().to(torch.float16)

        diff = torch.max(
            torch.abs(hf_tensor - our_tensor)
        ).item()

        print(
            f"{name:<35}",
            "shape:",
            tuple(our_tensor.shape),
            "max_diff:",
            diff,
        )

    compare(
        "conv1.weight",
        hf_model.encoder.conv1.weight,
        our_model.conv1.weight,
    )

    compare(
        "conv2.weight",
        hf_model.encoder.conv2.weight,
        our_model.conv2.weight,
    )

    compare(
        "embed_positions.weight",
        hf_model.encoder.embed_positions.weight,
        our_model.embed_positions.weight,
    )

    compare(
        "layer0.q_proj.weight",
        hf_model.encoder.layers[0].self_attn.q_proj.weight,
        our_model.layers[0].self_attn.q_proj.weight,
    )

    compare(
        "layer0.fc1.weight",
        hf_model.encoder.layers[0].fc1.weight,
        our_model.layers[0].fc1.weight,
    )

    compare(
        "layer11.fc2.weight",
        hf_model.encoder.layers[11].fc2.weight,
        our_model.layers[11].fc2.weight,
    )

    compare(
        "final_layer_norm.weight",
        hf_model.encoder.layer_norm.weight,
        our_model.layer_norm.weight,
    )


    # ---------------------------------------------------------------
    # 5. Compare complete encoder output
    # ---------------------------------------------------------------

    print("\nTesting complete encoder output...")

    hf_model = hf_model.to(dtype=torch.float16)
    our_model = our_model.to(dtype=torch.float16)

    hf_model.eval()
    our_model.eval()

    # IMPORTANT: deterministic input
    torch.manual_seed(0)

    input_features = torch.randn(
        1,
        80,
        3000,
        dtype=torch.float16,
    )

    with torch.no_grad():

        hf_output = hf_model.encoder(
            input_features
        ).last_hidden_state

        our_output = our_model(
            input_features
        )

    print("HF output shape :", hf_output.shape)
    print("Our output shape:", our_output.shape)

    hf_output = hf_output.float()
    our_output = our_output.float()

    abs_diff = torch.abs(
        hf_output - our_output
    )

    print("Max difference :", abs_diff.max().item())
    print("Mean difference:", abs_diff.mean().item())

    print("HF mean :", hf_output.mean().item())
    print("Our mean:", our_output.mean().item())


    print("\nLayer-by-layer comparison")
    print("-" * 70)


    def report(name, hf_tensor, our_tensor):
        hf_tensor = hf_tensor.float()
        our_tensor = our_tensor.float()

        diff = torch.abs(hf_tensor - our_tensor)

        print(
            f"{name:<20}"
            f" max={diff.max().item():.8f}"
            f" mean={diff.mean().item():.8f}"
        )


    with torch.no_grad():

        # ===============================================================
        # Hugging Face frontend
        # ===============================================================

        hf_x = F.gelu(
            hf_model.encoder.conv1(input_features)
        )

        # Our frontend
        our_x = F.gelu(
            our_model.conv1(input_features)
        )

        report(
            "After Conv1",
            hf_x,
            our_x,
        )


        # ===============================================================
        # Conv2
        # ===============================================================

        hf_x = F.gelu(
            hf_model.encoder.conv2(hf_x)
        )

        our_x = F.gelu(
            our_model.conv2(our_x)
        )

        report(
            "After Conv2",
            hf_x,
            our_x,
        )


        # ===============================================================
        # Permute
        # ===============================================================

        hf_x = hf_x.permute(0, 2, 1)
        our_x = our_x.permute(0, 2, 1)


        # ===============================================================
        # Position embedding
        # ===============================================================

        hf_positions = torch.arange(
            hf_model.encoder.embed_positions.num_embeddings,
            device=hf_x.device,
        )

        hf_x = (
            hf_x
            + hf_model.encoder.embed_positions(
                hf_positions
            )
        )

        our_pos = our_model.embed_positions(
            our_x.shape[1]
        )

        our_x = (
            our_x
            + our_pos.unsqueeze(0).to(our_x.dtype)
        )

        report(
            "After Position",
            hf_x,
            our_x,
        )


        # ===============================================================
        # Encoder layers
        # ===============================================================

        for i in range(12):

            hf_x = hf_model.encoder.layers[i](
                hf_x,
                None,
            )

            our_x = our_model.layers[i](
                our_x
            )

            report(
                f"After Layer {i}",
                hf_x,
                our_x,
            )


        # ===============================================================
        # Final LayerNorm
        # ===============================================================

        hf_x = hf_model.encoder.layer_norm(hf_x)

        our_x = our_model.layer_norm(our_x)

        report(
            "Final output",
            hf_x,
            our_x,
        )

    print("\nTesting ONNX export arguments...")

    export_args = our_model.get_onnx_export_args(
        config,
        device="cpu",
    )

    dynamo_inputs, input_names, output_names, dynamic_shapes = export_args

    print(
        "Input shape:",
        dynamo_inputs["input_features"].shape
    )

    print(
        "Input dtype:",
        dynamo_inputs["input_features"].dtype
    )

    print(
        "Input names:",
        input_names
    )

    print(
        "Output names:",
        output_names
    )

    print(
        "Dynamic shapes:",
        dynamic_shapes
    )