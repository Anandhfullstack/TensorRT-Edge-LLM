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
        self.scaling = self.head_dim**-0.5

        self.q_proj = make_linear(
            model_config,
            d_model,
            d_model,
            bias=True,
            module_name=(f"{name_prefix}.q_proj" if name_prefix else ""),
        )

        # Whisper k_proj has no bias.
        self.k_proj = make_linear(
            model_config,
            d_model,
            d_model,
            bias=False,
            module_name=(f"{name_prefix}.k_proj" if name_prefix else ""),
        )

        self.v_proj = make_linear(
            model_config,
            d_model,
            d_model,
            bias=True,
            module_name=(f"{name_prefix}.v_proj" if name_prefix else ""),
        )

        self.out_proj = make_linear(
            model_config,
            d_model,
            d_model,
            bias=True,
            module_name=(f"{name_prefix}.out_proj" if name_prefix else ""),
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
        causal: bool = False,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        cross_attention: bool = False,
    ):
        """
        Whisper Decoder Attention

        Self-attention:
            hidden_states:
                [B, T, 768]
            K/V:
                hidden_states, prepended with ``past_key_value`` when decoding
        Cross-attention:
            hidden_states:
                decoder states [B,T,768]
            K/V:
                encoder states [B,1500,768]

        Returns ``(attention_output, debug, present_key_value)``.
        ``present_key_value`` contains the projected K/V for both
        self-attention and cross-attention. Self-attention grows it by one step
        per call; cross-attention K/V depends only on the encoder output, which
        is fixed for the whole sequence, so it is projected once and reused
        verbatim afterwards.
        """
        debug = {}

        # Once cross-attention K/V can arrive already projected, the presence of
        # ``key_value_states`` no longer identifies it, so callers say so
        # explicitly.
        is_cross_attention = cross_attention or key_value_states is not None

        # ---------------------------------------------------------------
        # Query projection
        #
        # Q always comes from the current decoder hidden state.
        # ---------------------------------------------------------------

        q_before_split = self.q_proj(hidden_states)
        debug["q_before_split"] = q_before_split.detach()

        q = self._split_heads(q_before_split)

        past_length = 0

        # ---------------------------------------------------------------
        # Cross-attention KV
        # ---------------------------------------------------------------

        if is_cross_attention:
            if past_key_value is not None:
                # Reuse the encoder K/V projected during prefill.
                #
                # Importantly, k_proj and v_proj are not executed in
                # this cached path.
                k, v = past_key_value
            else:
                if key_value_states is None:
                    raise ValueError(
                        "Cross-attention needs either key_value_states to "
                        "project, or a pre-projected past_key_value.")

                # Prefill: project the encoder output once.
                k_before_split = self.k_proj(
                    key_value_states
                )
                v_before_split = self.v_proj(
                    key_value_states
                )

                debug["k_before_split"] = (
                    k_before_split.detach()
                )
                debug["v_before_split"] = (
                    v_before_split.detach()
                )

                k = self._split_heads(k_before_split)
                v = self._split_heads(v_before_split)

            # Cross-attention cache has fixed encoder length.
            present_key_value = (k, v)

        # ---------------------------------------------------------------
        # Causal self-attention KV
        # ---------------------------------------------------------------

        else:
            k_before_split = self.k_proj(hidden_states)
            v_before_split = self.v_proj(hidden_states)

            debug["k_before_split"] = (
                k_before_split.detach()
            )
            debug["v_before_split"] = (
                v_before_split.detach()
            )

            k = self._split_heads(k_before_split)
            v = self._split_heads(v_before_split)

            if past_key_value is not None:
                past_key, past_value = past_key_value

                past_length = past_key.shape[2]

                k = torch.cat(
                    [past_key, k],
                    dim=2,
                )
                v = torch.cat(
                    [past_value, v],
                    dim=2,
                )

            present_key_value = (k, v)

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
            # Query i sits at absolute position ``past_length + i``, so it may
            # attend to every cached key plus its own prefix. Without the
            # offset a cached step would mask out its entire history.
            query_positions = torch.arange(
                target_len,
                device=scores.device,
            ).view(
                target_len,
                1,
            ) + past_length
            key_positions = torch.arange(
                source_len,
                device=scores.device,
            ).view(
                1,
                source_len,
            )
            causal_mask = key_positions > query_positions
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
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
        ).to(scores.dtype)

        debug["attention_weights"] = attention_weights.detach()

        # ---------------------------------------------------------------
        # Attention × Value
        # ---------------------------------------------------------------

        attention_output = torch.matmul(
            attention_weights,
            v,
        )

        debug["attention_output_before_merge"] = attention_output.detach()

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
            attention_output.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                target_len,
                self.d_model,
            )
        )

        debug["attention_output_after_merge"] = attention_output.detach()

        # ---------------------------------------------------------------
        # Output projection
        # ---------------------------------------------------------------

        attention_output = self.out_proj(attention_output)

        debug["attention_output_final"] = attention_output.detach()

        return attention_output, debug, present_key_value


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

        self.self_attn_layer_norm = nn.LayerNorm(d_model)

        self.self_attn = WhisperDecoderAttention(
            model_config=model_config,
            d_model=d_model,
            num_heads=num_heads,
            name_prefix=(f"{name_prefix}.self_attn" if name_prefix else ""),
        )

        # ---------------------------------------------------------------
        # Encoder cross-attention
        # ---------------------------------------------------------------

        self.encoder_attn_layer_norm = nn.LayerNorm(d_model)

        self.encoder_attn = WhisperDecoderAttention(
            model_config=model_config,
            d_model=d_model,
            num_heads=num_heads,
            name_prefix=(f"{name_prefix}.encoder_attn" if name_prefix else ""),
        )

        # ---------------------------------------------------------------
        # Feed-forward
        # ---------------------------------------------------------------

        self.final_layer_norm = nn.LayerNorm(d_model)

        self.fc1 = make_linear(
            model_config,
            d_model,
            ffn_dim,
            bias=True,
            module_name=(f"{name_prefix}.fc1" if name_prefix else ""),
        )

        self.fc2 = make_linear(
            model_config,
            ffn_dim,
            d_model,
            bias=True,
            module_name=(f"{name_prefix}.fc2" if name_prefix else ""),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        cross_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
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

        debug_outputs["after_self_attn_norm"] = hidden_states.detach()

        self_attn_output, self_attn_debug, present_key_value = self.self_attn(
            hidden_states,
            key_value_states=None,
            causal=True,
            past_key_value=past_key_value,
        )

        debug_outputs["self_attn_debug"] = self_attn_debug

        hidden_states = residual + self_attn_output

        debug_outputs["after_self_attn"] = hidden_states.detach()

        debug_outputs["self_attn_output"] = self_attn_output.detach()

        debug_outputs["self_residual"] = residual.detach()

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

        cross_attn_output, cross_attn_debug, _ = self.encoder_attn(
            hidden_states,
            key_value_states=encoder_hidden_states,
            causal=False,
            past_key_value=cross_key_value,
            cross_attention=True,
        )

        debug_outputs["cross_attn_debug"] = cross_attn_debug

        hidden_states = residual + cross_attn_output

        debug_outputs["after_cross_attn"] = hidden_states.detach()

        debug_outputs["cross_attn_output"] = cross_attn_output.detach()

        debug_outputs["cross_residual"] = residual.detach()

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

        hidden_states = self.fc1(hidden_states)

        debug_outputs["ffn_fc1"] = hidden_states.detach()

        hidden_states = F.gelu(hidden_states)

        debug_outputs["ffn_gelu"] = hidden_states.detach()

        hidden_states = self.fc2(hidden_states)

        debug_outputs["ffn_fc2"] = hidden_states.detach()

        # ===============================================================
        # ADD DEBUG HERE
        # ===============================================================

        debug_outputs["ffn_output_before_residual"] = hidden_states.detach()

        debug_outputs["ffn_residual_input"] = residual.detach()

        # Residual connection

        hidden_states = residual + hidden_states

        debug_outputs["after_ffn"] = hidden_states.detach()

        return hidden_states, present_key_value, debug_outputs


class WhisperDecoderExportWrapper(torch.nn.Module):

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

        hidden_states, _, _, _ = self.decoder(
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

        batch_dim = torch.export.Dim("batch")

        sequence_dim = torch.export.Dim("sequence")

        dynamic_shapes = {
            "input_ids": {
                0: batch_dim,
                1: sequence_dim,
            },
            "encoder_hidden_states": {
                0: batch_dim,
            },
        }

        return (
            inputs,
            input_names,
            output_names,
            dynamic_shapes,
        )


class WhisperCrossKVProjector(torch.nn.Module):
    """Projects the encoder output into per-layer cross-attention K/V, once.

    The encoder output is fixed for the whole generation, so its K/V projection
    is loop-invariant. Running it as its own engine keeps that work out of the
    decode loop while carrying only the ``2 * num_layers`` projection weights
    (~28 MB FP16 for whisper-small) instead of duplicating the whole decoder.

    ``encoder_hidden_states [batch, encoder_positions, d_model]``
    -> ``cross_key_values [num_layers, 2, batch, num_heads, encoder_positions, head_dim]``
    """

    def __init__(
        self,
        decoder,
    ):
        super().__init__()

        # These are the decoder's own projection modules, shared by reference,
        # so the two graphs cannot drift apart.
        self.k_projs = nn.ModuleList(
            [layer.encoder_attn.k_proj for layer in decoder.layers])
        self.v_projs = nn.ModuleList(
            [layer.encoder_attn.v_proj for layer in decoder.layers])

        self.num_heads = decoder.layers[0].encoder_attn.num_heads
        self.head_dim = decoder.layers[0].encoder_attn.head_dim

    def _split_heads(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, seq_len, _ = x.shape

        return (x.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2).contiguous())

    def forward(
        self,
        encoder_hidden_states,
    ):

        per_layer = []

        for k_proj, v_proj in zip(self.k_projs, self.v_projs):
            k = self._split_heads(k_proj(encoder_hidden_states))
            v = self._split_heads(v_proj(encoder_hidden_states))
            per_layer.append(torch.stack([k, v], dim=0))

        return torch.stack(per_layer, dim=0)

    def get_onnx_export_args(
        self,
        config,
        device,
    ):

        d_model = config["d_model"]

        encoder_positions = config.get(
            "max_source_positions",
            1500,
        )

        encoder_hidden_states = torch.zeros(
            1,
            encoder_positions,
            d_model,
            dtype=torch.float16,
            device=device,
        )

        inputs = (encoder_hidden_states, )

        input_names = [
            "encoder_hidden_states",
        ]

        output_names = [
            "cross_key_values",
        ]

        batch_dim = torch.export.Dim("batch")

        dynamic_shapes = ({
            0: batch_dim,
        }, )

        return (
            inputs,
            input_names,
            output_names,
            dynamic_shapes,
        )


class WhisperDecoderWithCacheExportWrapper(torch.nn.Module):
    """Single decoder graph that serves both the prefill and the decode phase.

    I/O contract::

        input_ids          [batch, sequence]
        position_ids       [batch, sequence]
        past_key_values    [num_layers, 2, batch, num_heads, past_length, head_dim]
        cross_key_values   [num_layers, 2, batch, num_heads, encoder_positions, head_dim]

        logits             [batch, sequence, vocab_size]
        present_key_values [num_layers, 2, batch, num_heads, past_length + sequence, head_dim]

    Prefill is ``past_length == 0`` with the whole forced prefix in
    ``input_ids``; decode is ``sequence == 1`` with the accumulated cache. Both
    dimensions are dynamic, so one engine covers both phases instead of
    duplicating the decoder weights across two.

    ``cross_key_values`` comes from :class:`WhisperCrossKVProjector`, run once
    after the encoder. It is read-only and fixed-length, so it is *not* returned
    — only the self-attention cache grows and is handed back each step.

    The per-layer caches are stacked into one tensor rather than exposed as
    ``2 * num_layers`` separate bindings, so the runtime allocates and binds a
    single contiguous buffer.
    """

    def __init__(
        self,
        decoder,
    ):
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        input_ids,
        position_ids,
        past_key_values,
        cross_key_values,
    ):
        num_layers = len(self.decoder.layers)

        past_values = [(past_key_values[i, 0], past_key_values[i, 1])
                       for i in range(num_layers)]

        cross_values = [(cross_key_values[i, 0], cross_key_values[i, 1])
                        for i in range(num_layers)]

        hidden_states, present_values, _, _ = self.decoder(
            input_ids,
            past_values=past_values,
            position_ids=position_ids,
            cross_values=cross_values,
        )

        logits = F.linear(
            hidden_states,
            self.decoder.embed_tokens.weight,
        )

        present_key_values = torch.stack(
            [torch.stack(layer_kv, dim=0) for layer_kv in present_values],
            dim=0,
        )

        return logits, present_key_values


    def get_onnx_export_args(
        self,
        config,
        device,
    ):

        num_layers = config["decoder_layers"]
        num_heads = config["decoder_attention_heads"]
        d_model = config["d_model"]

        head_dim = d_model // num_heads

        max_target_positions = config.get(
            "max_target_positions",
            _MAX_TARGET_POSITIONS,
        )

        encoder_positions = config.get(
            "max_source_positions",
            1500,
        )

        # Trace with several new tokens on top of a non-empty cache so neither
        # length is constant-folded into the graph.
        sequence_length = 4
        past_length = 4

        input_ids = torch.zeros(
            1,
            sequence_length,
            dtype=torch.long,
            device=device,
        )

        position_ids = torch.arange(
            past_length,
            past_length + sequence_length,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

        past_key_values = torch.zeros(
            num_layers,
            2,
            1,
            num_heads,
            past_length,
            head_dim,
            dtype=torch.float16,
            device=device,
        )

        cross_key_values = torch.zeros(
            num_layers,
            2,
            1,
            num_heads,
            encoder_positions,
            head_dim,
            dtype=torch.float16,
            device=device,
        )

        # A flat tuple (not a dict) keeps the argument count, the name list and
        # the dynamic-shape spec aligned, which ``_run_dynamo_export`` asserts.
        inputs = (
            input_ids,
            position_ids,
            past_key_values,
            cross_key_values,
        )

        input_names = [
            "input_ids",
            "position_ids",
            "past_key_values",
            "cross_key_values",
        ]

        output_names = [
            "logits",
            "present_key_values",
        ]

        batch_dim = torch.export.Dim("batch")

        sequence_dim = torch.export.Dim(
            "sequence",
            max=max_target_positions,
        )

        # ``min=0`` is what lets one engine cover both phases: prefill binds an
        # empty cache, decode binds the accumulated one.
        past_dim = torch.export.Dim(
            "past_length",
            min=0,
            max=max_target_positions,
        )

        dynamic_shapes = (
            {
                0: batch_dim,
                1: sequence_dim,
            },
            {
                0: batch_dim,
                1: sequence_dim,
            },
            {
                2: batch_dim,
                4: past_dim,
            },
            {
                2: batch_dim,
            },
        )

        return (
            inputs,
            input_names,
            output_names,
            dynamic_shapes,
        )

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

        self.layers = nn.ModuleList(
            [
                WhisperDecoderLayer(
                    model_config=model_config,
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    name_prefix=f"{name_prefix}.layers.{i}",
                )
                for i in range(num_layers)
            ]
        )

        # ---------------------------------------------------------------
        # Final decoder LayerNorm
        # ---------------------------------------------------------------

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        past_values=None,
        position_ids=None,
        cross_values=None,
    ):
        """``cross_values`` supplies per-layer cross-attention K/V already
        projected from the encoder output, in which case
        ``encoder_hidden_states`` is not needed at all."""

        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape [B,T]"
            )


        if cross_values is None:
            if encoder_hidden_states is None:
                raise ValueError(
                    "Pass either encoder_hidden_states or cross_values"
                )

            if encoder_hidden_states.ndim != 3:
                raise ValueError(
                    "encoder_hidden_states must have shape [B,S,H]"
                )


        batch_size, target_len = input_ids.shape


        # ---------------------------------------------------------------
        # Position handling for KV cache
        # ---------------------------------------------------------------

        if position_ids is None:

            if past_values is not None:
                past_length = past_values[0][0].shape[2]

            else:
                past_length = 0


            position_ids = torch.arange(
                past_length,
                past_length + target_len,
                device=input_ids.device,
            )


            position_ids = position_ids.unsqueeze(0).expand(
                batch_size,
                -1,
            )


        # ---------------------------------------------------------------
        # Token embedding
        # ---------------------------------------------------------------

        hidden_states = self.embed_tokens(
            input_ids
        )


        # ---------------------------------------------------------------
        # Learned position embedding
        # ---------------------------------------------------------------

        positional_embedding = (
            self.embed_positions(position_ids)
            .to(hidden_states.dtype)
        )


        hidden_states = hidden_states + positional_embedding


        all_hidden_states = []

        layer_debug = []

        present_values = []


        # ---------------------------------------------------------------
        # Decoder layers
        # ---------------------------------------------------------------

        for i, layer in enumerate(self.layers):

            layer_past = None

            if past_values is not None:
                layer_past = past_values[i]

            layer_cross = None

            if cross_values is not None:
                layer_cross = cross_values[i]


            hidden_states, present_value, debug = layer(
                hidden_states,
                encoder_hidden_states,
                past_key_value=layer_past,
                cross_key_value=layer_cross,
            )


            all_hidden_states.append(
                hidden_states
            )

            layer_debug.append(
                debug
            )

            present_values.append(
                present_value
            )


        # ---------------------------------------------------------------
        # Final LayerNorm
        # ---------------------------------------------------------------

        hidden_states = F.layer_norm(
            hidden_states.float(),
            self.layer_norm.normalized_shape,
            self.layer_norm.weight.float(),
            self.layer_norm.bias.float(),
            self.layer_norm.eps,
        ).to(hidden_states.dtype)



        return (
            hidden_states,
            present_values,
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

            if any(key.startswith(candidate) for key in weights.keys()):
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

            stripped_key = key[len(prefix) :]

        else:

            # Protect against accidentally loading the encoder,
            # proj_out, or other model tensors when the checkpoint
            # has no model.decoder. prefix.

            if not key.startswith(local_decoder_prefixes):
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

    model = model.to(dtype=dtype)

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
    with_cache: bool = False,
):
    """Build the decoder wrapped for ONNX export.

    ``with_cache`` selects the prefill+decode graph (KV cache as engine I/O);
    the default is the stateless full-sequence graph, which recomputes the whole
    prefix on every call and is kept as the numerical reference.
    """

    decoder = build_whisper_decoder(
        config=config,
        weights=weights,
        dtype=dtype,
        prefix=prefix,
        model_config=model_config,
    )

    if with_cache:
        return WhisperDecoderWithCacheExportWrapper(decoder)

    return WhisperDecoderExportWrapper(decoder)


def build_whisper_cross_kv_export(
    config,
    weights,
    dtype,
    prefix,
    model_config,
):
    """Build the one-shot encoder K/V projector wrapped for ONNX export."""

    decoder = build_whisper_decoder(
        config=config,
        weights=weights,
        dtype=dtype,
        prefix=prefix,
        model_config=model_config,
    )

    return WhisperCrossKVProjector(decoder)


# 1. WhisperDecoderAttention: Implements the attention mechanism used in the Whisper decoder, supporting both causal self-attention and encoder-decoder cross-attention. It includes methods for projecting queries, keys, and values, splitting heads, applying scaling, and computing attention scores.
# 2. WhisperDecoderLayer: Represents a single layer of the Whisper decoder, consisting of causal self-attention, encoder cross-attention, and a feed-forward network (FFN). It applies layer normalization and residual connections at each stage.
# 3. WhisperTextDecoder: The complete Whisper decoder model, which includes token embeddings, learned positional embeddings, multiple decoder layers, and a final layer normalization. It processes input token IDs and encoder hidden states to produce output hidden states.
# 4. WhisperDecoderExportWrapper: A wrapper around the Whisper decoder that adds a linear projection to convert hidden states into logits for output. It also provides methods for preparing inputs and outputs for ONNX export.
    