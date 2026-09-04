from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import config as config_module
from ..linear import make_linear


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Whisper Small decoder defaults
# ---------------------------------------------------------------------------

_D_MODEL = 768
_NUM_LAYERS = 12
_NUM_HEADS = 12
_FFN_DIM = 3072
_VOCAB_SIZE = 51865
_MAX_TARGET_POSITIONS = 448
_PAD_TOKEN_ID = 50256


# ---------------------------------------------------------------------------
# Decoder Attention
# ---------------------------------------------------------------------------


class WhisperDecoderAttention(nn.Module):
    """
    Attention used by the Whisper decoder.

    Used for both:

        1. causal self-attention
        2. encoder-decoder cross-attention

    Whisper Small:

        hidden_size = 768
        num_heads   = 12
        head_dim    = 64
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

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        
        # Whisper scales Q before Q @ K^T.
        self.scaling = self.head_dim ** -0.5

       
        self.q_proj = make_linear(
            model_config,
            d_model,
            d_model,
            bias=True,
            module_name=(
                f"{name_prefix}.q_proj"
                if name_prefix
                else ""
            ),
        )

        # Whisper k_proj has no bias.
        self.k_proj = make_linear(
            model_config,
            d_model,
            d_model,
            bias=False,
            module_name=(
                f"{name_prefix}.k_proj"
                if name_prefix
                else ""
            ),
        )

        self.v_proj = make_linear(
            model_config,
            d_model,
            d_model,
            bias=True,
            module_name=(
                f"{name_prefix}.v_proj"
                if name_prefix
                else ""
            ),
        )

        self.out_proj = make_linear(
            model_config,
            d_model,
            d_model,
            bias=True,
            module_name=(
                f"{name_prefix}.out_proj"
                if name_prefix
                else ""
            ),
        )

    def _split_heads(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        [B, T, 768]
              ↓
        [B, 12, T, 64]
        """

        batch_size, seq_len, _ = x.shape

        return (
            x.view(
                batch_size,
                seq_len,
                self.num_heads,
                self.head_dim,
            )
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        causal: bool = False,):
        """
        Whisper Decoder Attention

        Self-attention:
            hidden_states:
                [B, T, 768]

            K/V:
                hidden_states

        Cross-attention:
            hidden_states:
                decoder states [B,T,768]

            K/V:
                encoder states [B,1500,768]
        """

        debug = {}

        # ---------------------------------------------------------------
        # Select source states
        # ---------------------------------------------------------------

        if key_value_states is None:
            source_states = hidden_states
        else:
            source_states = key_value_states


        # ---------------------------------------------------------------
        # QKV projection
        # ---------------------------------------------------------------

        q = self.q_proj(hidden_states)
        k = self.k_proj(source_states)
        v = self.v_proj(source_states)


        debug["q_before_split"] = q.detach()
        debug["k_before_split"] = k.detach()
        debug["v_before_split"] = v.detach()


        # ---------------------------------------------------------------
        # Split heads
        #
        # [B,T,768]
        #
        # ->
        #
        # [B,12,T,64]
        # ---------------------------------------------------------------

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)


        debug["q_before_scale"] = q.detach()
        debug["k"] = k.detach()
        debug["v"] = v.detach()


        # ---------------------------------------------------------------
        # Whisper scaling
        # q * (head_dim^-0.5)
        # ---------------------------------------------------------------

        q = q * self.scaling


        debug["q_after_scale"] = q.detach()


        # ---------------------------------------------------------------
        # Attention scores
        # ---------------------------------------------------------------

        scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        )


        debug["scores_before_mask"] = scores.detach()


        target_len = q.shape[-2]
        source_len = k.shape[-2]


        # ---------------------------------------------------------------
        # Causal mask
        # ---------------------------------------------------------------

        if causal:

            if source_len != target_len:
                raise ValueError(
                    "Full sequence self attention expects "
                    "source_len == target_len"
                )


            query_positions = torch.arange(
                target_len,
                device=scores.device,
            ).view(
                target_len,
                1,
            )


            key_positions = torch.arange(
                source_len,
                device=scores.device,
            ).view(
                1,
                source_len,
            )


            causal_mask = (
                key_positions > query_positions
            )


            causal_mask = (
                causal_mask
                .unsqueeze(0)
                .unsqueeze(0)
            )


            scores = scores.masked_fill(
                causal_mask,
                torch.finfo(scores.dtype).min,
            )


            debug["scores_after_mask"] = scores.detach()


        # ---------------------------------------------------------------
        # Softmax
        # ---------------------------------------------------------------

        attention_weights = torch.softmax(
            scores,
            dim=-1,
            dtype=torch.float32,
        ).to(
            scores.dtype
        )


        debug["attention_weights"] = (
            attention_weights.detach()
        )


        # ---------------------------------------------------------------
        # Attention × Value
        # ---------------------------------------------------------------

        attention_output = torch.matmul(
            attention_weights,
            v,
        )


        debug["attention_output_before_merge"] = (
            attention_output.detach()
        )


        # ---------------------------------------------------------------
        # Merge heads
        #
        # [B,heads,T,head_dim]
        #
        # ->
        #
        # [B,T,hidden]
        # ---------------------------------------------------------------

        batch_size = hidden_states.shape[0]


        attention_output = (
            attention_output
            .transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                target_len,
                self.d_model,
            )
        )


        debug["attention_output_after_merge"] = (
            attention_output.detach()
        )


        # ---------------------------------------------------------------
        # Output projection
        # ---------------------------------------------------------------

        attention_output = self.out_proj(
            attention_output
        )


        debug["attention_output_final"] = (
            attention_output.detach()
        )


        return attention_output, debug


# ---------------------------------------------------------------------------
# Whisper Decoder Layer
# ---------------------------------------------------------------------------


class WhisperDecoderLayer(nn.Module):
    """
    One Whisper decoder Transformer block.

    Flow:

        input
          ↓
        LayerNorm
          ↓
        causal self-attention
          ↓
        residual
          ↓
        LayerNorm
          ↓
        encoder cross-attention
          ↓
        residual
          ↓
        LayerNorm
          ↓
        FFN
          ↓
        residual
    """

    def __init__(
        self,
        model_config: config_module.ModelConfig,
        d_model: int = _D_MODEL,
        num_heads: int = _NUM_HEADS,
        ffn_dim: int = _FFN_DIM,
        name_prefix: str = "",
    ) -> None:
        super().__init__()

        # ---------------------------------------------------------------
        # Self-attention
        # ---------------------------------------------------------------
        
        self.self_attn_layer_norm = nn.LayerNorm(
            d_model
        )

        self.self_attn = WhisperDecoderAttention(
            model_config=model_config,
            d_model=d_model,
            num_heads=num_heads,
            name_prefix=(
                f"{name_prefix}.self_attn"
                if name_prefix
                else ""
            ),
        )

        # ---------------------------------------------------------------
        # Encoder cross-attention
        # ---------------------------------------------------------------

        self.encoder_attn_layer_norm = nn.LayerNorm(
            d_model
        )

        self.encoder_attn = WhisperDecoderAttention(
            model_config=model_config,
            d_model=d_model,
            num_heads=num_heads,
            name_prefix=(
                f"{name_prefix}.encoder_attn"
                if name_prefix
                else ""
            ),
        )

        # ---------------------------------------------------------------
        # Feed-forward
        # ---------------------------------------------------------------

        self.final_layer_norm = nn.LayerNorm(
            d_model
        )

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
        encoder_hidden_states: torch.Tensor,
    ):

        debug_outputs = {}

        # ---------------------------------------------------------------
        # Layer input
        # ---------------------------------------------------------------

        debug_outputs["input"] = hidden_states.detach()


        # ---------------------------------------------------------------
        # 1. Causal self-attention
        # ---------------------------------------------------------------

        residual = hidden_states


        hidden_states = F.layer_norm(
            hidden_states.float(),
            self.self_attn_layer_norm.normalized_shape,
            self.self_attn_layer_norm.weight.float(),
            self.self_attn_layer_norm.bias.float(),
            self.self_attn_layer_norm.eps,
        ).to(hidden_states.dtype)

        debug_outputs["after_self_attn_norm"] = (
            hidden_states.detach()
        )


        self_attn_output, self_attn_debug = self.self_attn(
            hidden_states,
            key_value_states=None,
            causal=True,
        )


        debug_outputs["self_attn_debug"] = (
            self_attn_debug
        )


        hidden_states = residual + self_attn_output


        debug_outputs["after_self_attn"] = (
            hidden_states.detach()
        )

        debug_outputs["self_attn_output"] = (
            self_attn_output.detach()
        )

        debug_outputs["self_residual"] = (
            residual.detach()
        )


        # ---------------------------------------------------------------
        # 2. Encoder cross-attention
        # ---------------------------------------------------------------

        residual = hidden_states


        hidden_states = F.layer_norm(
            hidden_states.float(),
            self.encoder_attn_layer_norm.normalized_shape,
            self.encoder_attn_layer_norm.weight.float(),
            self.encoder_attn_layer_norm.bias.float(),
            self.encoder_attn_layer_norm.eps,
        ).to(hidden_states.dtype)


        cross_attn_output, cross_attn_debug = self.encoder_attn(
            hidden_states,
            key_value_states=encoder_hidden_states,
            causal=False,
        )


        debug_outputs["cross_attn_debug"] = (
            cross_attn_debug
        )


        hidden_states = residual + cross_attn_output


        debug_outputs["after_cross_attn"] = (
            hidden_states.detach()
        )

        debug_outputs["cross_attn_output"] = (
            cross_attn_output.detach()
        )

        debug_outputs["cross_residual"] = (
            residual.detach()
        )


        # ---------------------------------------------------------------
        # 3. Feed-forward
        # ---------------------------------------------------------------

        residual = hidden_states


        hidden_states = F.layer_norm(
            hidden_states.float(),
            self.final_layer_norm.normalized_shape,
            self.final_layer_norm.weight.float(),
            self.final_layer_norm.bias.float(),
            self.final_layer_norm.eps,
        ).to(hidden_states.dtype)

        hidden_states = self.fc1(
            hidden_states
        )

        debug_outputs["ffn_fc1"] = hidden_states.detach()


        hidden_states = F.gelu(
            hidden_states
        )

        debug_outputs["ffn_gelu"] = hidden_states.detach()


        hidden_states = self.fc2(
            hidden_states
        )


        debug_outputs["ffn_fc2"] = hidden_states.detach()


        # ===============================================================
        # ADD DEBUG HERE
        # ===============================================================

        debug_outputs["ffn_output_before_residual"] = (
            hidden_states.detach()
        )


        debug_outputs["ffn_residual_input"] = (
            residual.detach()
        )


        # Residual connection

        hidden_states = residual + hidden_states


        debug_outputs["after_ffn"] = (
            hidden_states.detach()
        )


        return hidden_states, debug_outputs


class WhisperDecoderExportWrapper(
    torch.nn.Module
):

    def __init__(
        self,
        decoder,
    ):
        super().__init__()

        self.decoder = decoder


    def forward(
        self,
        input_ids,
        encoder_hidden_states,
    ):

        hidden_states, _, _ = self.decoder(
            input_ids,
            encoder_hidden_states,
        )

        # Whisper ties the output projection to the decoder's input token
        # embedding (HF: ``proj_out`` shares weights with
        # ``model.decoder.embed_tokens``). Project here rather than inside
        # ``WhisperTextDecoder`` so the un-wrapped module keeps returning raw
        # hidden states for validation, while the exported graph emits real
        # logits.
        logits = F.linear(
            hidden_states,
            self.decoder.embed_tokens.weight,
        )

        return logits

    def get_onnx_export_args(
        self,
        config,
        device,
    ):

        input_ids = torch.tensor(
            [
                [
                    50258,
                    50359,
                    50363,
                    2425,
                    1917,
                    13,
                    50257,
                    50256,
                ]
            ],
            dtype=torch.long,
            device=device,
        )


        encoder_hidden_states = torch.zeros(
            1,
            1500,
            config["d_model"],
            dtype=torch.float16,
            device=device,
        )


        inputs = {
            "input_ids": input_ids,
            "encoder_hidden_states": encoder_hidden_states,
        }


        input_names = [
            "input_ids",
            "encoder_hidden_states",
        ]


        output_names = [
            "logits",
        ]


        batch_dim = torch.export.Dim(
            "batch"
        )

        sequence_dim = torch.export.Dim(
            "sequence"
        )


        dynamic_shapes = {

            "input_ids": {
                0: batch_dim,
                1: sequence_dim,
            },

            "encoder_hidden_states": {
                0: batch_dim,
            },

        }


        return (inputs,
            input_names,
            output_names,
            dynamic_shapes,)

# ---------------------------------------------------------------------------
# Complete Whisper Decoder
# ---------------------------------------------------------------------------


class WhisperTextDecoder(nn.Module):

    def __init__(
        self,
        model_config: config_module.ModelConfig,
        vocab_size: int = _VOCAB_SIZE,
        d_model: int = _D_MODEL,
        num_layers: int = _NUM_LAYERS,
        num_heads: int = _NUM_HEADS,
        ffn_dim: int = _FFN_DIM,
        max_target_positions: int = _MAX_TARGET_POSITIONS,
        pad_token_id: int = _PAD_TOKEN_ID,
        name_prefix: str = "decoder",
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_target_positions = max_target_positions

        # ---------------------------------------------------------------
        # Token embedding
        # ---------------------------------------------------------------

        self.embed_tokens = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_token_id,
        )

        # ---------------------------------------------------------------
        # Learned positional embedding
        #
        # IMPORTANT:
        # Unlike the Whisper encoder positional representation,
        # these are learned checkpoint parameters.
        # ---------------------------------------------------------------

        self.embed_positions = nn.Embedding(
            max_target_positions,
            d_model,
        )

        # ---------------------------------------------------------------
        # Decoder Transformer blocks
        # ---------------------------------------------------------------

        self.layers = nn.ModuleList([
            WhisperDecoderLayer(
                model_config=model_config,
                d_model=d_model,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                name_prefix=f"{name_prefix}.layers.{i}",
            )
            for i in range(num_layers)
        ])

        # ---------------------------------------------------------------
        # Final decoder LayerNorm
        # ---------------------------------------------------------------

        self.layer_norm = nn.LayerNorm(
            d_model
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:

            input_ids:
                [B, T]

            encoder_hidden_states:
                [B, encoder_seq_len, hidden_size]

        Returns:

            hidden_states:
                [B, T, hidden_size]

            all_hidden_states:
                list of layer outputs

            layer_debug:
                debug information
        """

        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape [B, T]"
            )

        if encoder_hidden_states.ndim != 3:
            raise ValueError(
                "encoder_hidden_states must have shape "
                "[B, encoder_seq_len, hidden_size]"
            )


        batch_size, target_len = input_ids.shape


        if target_len > self.max_target_positions:
            raise ValueError(
                f"Decoder sequence length {target_len} exceeds "
                f"max_target_positions={self.max_target_positions}"
            )


        if encoder_hidden_states.shape[0] != batch_size:
            raise ValueError(
                "input_ids and encoder_hidden_states "
                "must have the same batch size"
            )


        # ---------------------------------------------------------------
        # 1. Token embedding
        #
        # [B,T]
        #    |
        #    v
        # [B,T,768]
        # ---------------------------------------------------------------

        hidden_states = self.embed_tokens(
            input_ids
        )


        # ---------------------------------------------------------------
        # 2. Learned positional embedding
        #
        # [T,768]
        #    |
        #    v
        # [1,T,768]
        # ---------------------------------------------------------------

        positional_embedding = (
            self.embed_positions.weight[:target_len]
            .unsqueeze(0)
            .to(hidden_states.dtype)
        )


        # ---------------------------------------------------------------
        # 3. Add position
        # ---------------------------------------------------------------

        hidden_states = (
            hidden_states
            + positional_embedding
        )


        # ---------------------------------------------------------------
        # 4. Decoder layers
        # ---------------------------------------------------------------

        all_hidden_states = []
        layer_debug = []


        for layer in self.layers:

            hidden_states, debug = layer(
                hidden_states,
                encoder_hidden_states,
            )


            all_hidden_states.append(
                hidden_states
            )


            layer_debug.append(
                debug
            )


        # ---------------------------------------------------------------
        # 5. Final decoder LayerNorm
        #
        # IMPORTANT:
        #
        # HF Whisper does this AFTER all decoder layers.
        #
        # NOT inside the loop.
        #
        # ---------------------------------------------------------------

        hidden_states = F.layer_norm(
                hidden_states.float(),
                self.layer_norm.normalized_shape,
                self.layer_norm.weight.float(),
                self.layer_norm.bias.float(),
                self.layer_norm.eps,
            ).to(hidden_states.dtype)


        # ---------------------------------------------------------------
        # 6. Return
        #
        # LM projection is done outside during validation/export.
        #
        # hidden_states:
        #     [B,T,768]
        #
        # ---------------------------------------------------------------

        return (
            hidden_states,
            all_hidden_states,
            layer_debug,
        )
# ---------------------------------------------------------------------------
# Whisper Decoder Weight Loading
# ---------------------------------------------------------------------------


_DECODER_CANDIDATE_PREFIXES = (
    "model.decoder.",
    "decoder.",
    "",
)


def _load_decoder_weights(
    model: WhisperTextDecoder,
    weights: Dict[str, torch.Tensor],
    dtype: torch.dtype = torch.float16,
    prefix: Optional[str] = None,
) -> None:
    """
    Load Hugging Face Whisper decoder weights.
    """

    from ...checkpoint.loader import load_submodule_weights

    # ---------------------------------------------------------------
    # 1. Detect checkpoint prefix
    # ---------------------------------------------------------------

    if prefix is None:

        for candidate in _DECODER_CANDIDATE_PREFIXES:

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

    logger.info(
        "Whisper decoder weight prefix: %r",
        prefix,
    )

    # ---------------------------------------------------------------
    # 2. Extract decoder weights
    # ---------------------------------------------------------------

    stripped: Dict[str, torch.Tensor] = {}

    local_decoder_prefixes = (
        "embed_tokens.",
        "embed_positions.",
        "layers.",
        "layer_norm.",
    )

    for key, value in weights.items():

        if prefix:

            if not key.startswith(prefix):
                continue

            stripped_key = key[len(prefix):]

        else:

            # Protect against accidentally loading the encoder,
            # proj_out, or other model tensors when the checkpoint
            # has no model.decoder. prefix.

            if not key.startswith(
                local_decoder_prefixes
            ):
                continue

            stripped_key = key

        stripped[stripped_key] = value

    # ---------------------------------------------------------------
    # 3. Key mapping
    #
    # Decoder checkpoint names already match our module names.
    #
    # IMPORTANT:
    # Do NOT discard embed_positions.weight.
    # ---------------------------------------------------------------

    def _key_remap(
        key: str,
    ) -> Optional[str]:

        return key

    # ---------------------------------------------------------------
    # 4. Convert normal HF parameters to target dtype
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
    # 5. Load
    # ---------------------------------------------------------------

    load_submodule_weights(
        model,
        stripped,
        key_remap=_key_remap,
        transform=_transform,
        label="WhisperTextDecoder",
        log=logger,
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_whisper_decoder(
    config: Dict[str, Any],
    weights: Dict[str, torch.Tensor],
    dtype: torch.dtype = torch.float16,
    prefix: Optional[str] = None,
    *,
    model_config: config_module.ModelConfig,
) -> WhisperTextDecoder:
    """
    Build the native Whisper decoder and load HF checkpoint weights.
    """

    d_model = config.get(
        "d_model",
        _D_MODEL,
    )

    num_heads = config.get(
        "decoder_attention_heads",
        _NUM_HEADS,
    )

    model = WhisperTextDecoder(
        model_config=model_config,

        vocab_size=config.get(
            "vocab_size",
            _VOCAB_SIZE,
        ),

        d_model=d_model,

        num_layers=config.get(
            "decoder_layers",
            _NUM_LAYERS,
        ),

        num_heads=num_heads,

        ffn_dim=config.get(
            "decoder_ffn_dim",
            _FFN_DIM,
        ),

        max_target_positions=config.get(
            "max_target_positions",
            _MAX_TARGET_POSITIONS,
        ),

        pad_token_id=config.get(
            "pad_token_id",
            _PAD_TOKEN_ID,
        ),

        name_prefix="decoder",
    )

    model = model.to(
        dtype=dtype
    )

    _load_decoder_weights(
        model,
        weights,
        dtype=dtype,
        prefix=prefix,
    )

    model.eval()

    return model

def build_whisper_decoder_export(
    config,
    weights,
    dtype,
    prefix,
    model_config,
):

    decoder = build_whisper_decoder(
        config=config,
        weights=weights,
        dtype=dtype,
        prefix=prefix,
        model_config=model_config,
    )

    decoder = WhisperDecoderExportWrapper(
        decoder
    )

    return decoder

