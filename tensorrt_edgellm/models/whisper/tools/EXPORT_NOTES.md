# Whisper ONNX export — changes made, and the general export flow

## 1. What changed in this pass

### 1a. `tools/export_whisper_decoder.py` — rewritten (standalone script)
Previously built its own local wrapper class and called raw
`torch.onnx.export(..., opset_version=18)` (legacy exporter, no custom op translation).
Now imports `build_whisper_decoder_export()` and exports through the shared
`_run_dynamo_export()` helper, same as every other encoder/decoder in this repo (opset 24,
dynamo, custom translation table, `external_data=True`).

### 1b. `onnx/export_encoder.py` — fixed a real blocker
A stray trailing comma in the `whisper` branch's decoder import broke the whole module's
import. This wasn't Whisper-only: every model's audio/visual export routes through this
file, so it blocked the entire export CLI.

### 1c. Decoder now wired into the CLI, matching the encoder's shape
The decoder previously had **no CLI path at all** and **no config.json** — the only way to
get it was running the standalone tool script by hand. It's now a proper export component:

- **`modeling_whisper_decoder.py`** — `WhisperDecoderExportWrapper.forward()` now projects
  through the tied output embedding (`F.linear(hidden_states, embed_tokens.weight)`,
  matching HF's `proj_out`) and returns **logits**, not raw hidden states. The undecorated
  `WhisperTextDecoder` is untouched — the projection lives only in the export wrapper, so
  existing validation of the base module still holds. Output name changed
  `"hidden_states"` → `"logits"`.
- **`scripts/export.py`** — new `_export_whisper_decoder()` (modeled directly on
  `_export_rnnt_decoder`), a `_has_whisper_decoder()` predicate, a `"whisper_decoder"`
  entry in `_DEFAULT_LAYOUT` (→ `decoder/` output subdir), a new stage in `main()`'s
  dispatch table, and `"whisper_decoder"` added to `_VALID_COMPONENTS` so it's selectable
  via `--components`. It also copies `tokenizer.json`/`tokenizer_config.json` into the
  decoder dir (needed to detokenize whatever the C++ side emits).

Now:
```
tensorrt-edgellm-export openai/whisper-small out/ --components audio,whisper_decoder --dtype float16
```
produces:
```
out/audio/model.onnx + model.onnx.data + config.json     (unchanged, encoder)
out/decoder/model.onnx + model.onnx.data + config.json    (new)
out/decoder/tokenizer.json, tokenizer_config.json         (new)
```
`decoder/config.json`:
```json
{
  "model_type": "whisper_text_decoder",
  "whisper_decoder_config": {
    "d_model": 768, "num_decoder_layers": 12, "num_attention_heads": 12,
    "decoder_ffn_dim": 3072, "vocab_size": 51865, "max_target_positions": 448,
    "pad_token_id": 50257, "bos_token_id": 50257, "eos_token_id": 50257,
    "decoder_start_token_id": 50258
  }
}
```
Note `bos_token_id` (50257) vs. `decoder_start_token_id` (50258): Whisper's own config sets
`bos == eos == pad == 50257`, but the token that actually primes generation — Whisper's
"start of transcript" token — is the separate `decoder_start_token_id`. Feed
`decoder_start_token_id`, not `bos_token_id`, as the first input to the decode loop; both
are included explicitly (not merged into one field) specifically to avoid that trap.
Omitting `--components` entirely exports everything the checkpoint supports (both audio
and whisper_decoder, since `_allow()` has no restriction by default) — same behavior as
every other model type.

**Verified, not just asserted:**
- Full CLI run (`tensorrt-edgellm-export openai/whisper-small ... --components audio,whisper_decoder`)
  against the cached checkpoint produced all of the files above.
- Exported graph I/O: `input_ids [batch, sequence]` + `encoder_hidden_states [batch, 1500, 768]`
  → `logits [batch, sequence, 51865]` (51865 = Whisper's vocab size, confirming the lm_head
  fusion is wired to the right dimension, not just present).
- **Numerical correctness**, not just shape: ran the exported graph through onnxruntime
  and compared against the eager PyTorch wrapper on identical inputs. Mean abs logit
  magnitude ≈3.65, mean abs diff ≈0.0087 (~0.24%), max abs diff 0.033, argmax token
  identical at every position — consistent with ordinary FP16 backend-to-backend rounding
  noise, not a correctness bug. (Had to load the ONNX with a temporary opset downgrade to
  22 for this check only, since the installed onnxruntime doesn't yet support the
  in-development opset 24 the framework exports at by default — the shipped
  `decoder/model.onnx` is untouched, still opset 24.)
- Regression check: re-ran the *unmodified* `qwen3_asr` audio-encoder export path through
  the patched `export_encoder.py` to confirm the syntax fix didn't disturb any other
  model's export (it didn't — see prior turn).

## 2. Deliberately NOT done here — scope boundary agreed with the user

You said the C++ engine build + runtime happens on your Jetson device, and you'll
implement "the balance" there. Given that, this pass stopped at a **correct, self-describing
ONNX export** and intentionally did **not** guess at your C++ decode-loop design:

- **No KV cache exported.** The decoder still takes the *full* token prefix on every call
  and returns logits for every position (same full-recompute contract as before, just now
  with real logits at the end instead of raw hidden states). Whether your engine drives
  this as "re-run the whole prefix every step" (works as exported, O(n²) compute) or you
  want a single-token incremental step with a growing self-attention KV cache (more
  efficient, but requires modifying `WhisperDecoderAttention`/`WhisperDecoderLayer` to
  accept/return `past_key_values` per layer, plus a position-id input for the positional
  embedding lookup) is a decode-loop architecture decision that belongs with your Jetson
  implementation, not something to bake into the export unasked. Say the word if you want
  the incremental-KV-cache version exported instead/also — it's a real modeling change
  (new I/O per layer), not a one-line tweak, so it's worth doing as its own reviewed step.
- **No C++ builder/runtime code touched.** `cpp/builder/audioBuilder.cpp` still has no
  `whisper_audio_encoder` or `whisper_text_decoder` entry in its `model_type` dispatch (see
  prior analysis: `stringToModelType()` → `UNKNOWN` → `setupAudioEncoderProfile()`'s switch
  hits `default: LOG_ERROR(...); return false;`). That's expected — it's the part you're
  building on-device. The `decoder/config.json` fields above (`whisper_decoder_config`)
  are what a `parseWhisperDecoderConfig()`-style function on your side would read; naming
  follows the `rnnt_decoder_config` sidecar convention already in this codebase so it's
  recognizable, but nothing in C++ consumes it yet.
- **Cross-attention is plain ONNX ops, not a plugin.** Worth knowing for your builder work:
  the decoder's cross-attention was exported as ordinary MatMul/Softmax nodes (no
  `AttentionPlugin`), since it attends over a fixed, already-computed `encoder_hidden_states`
  rather than a paged KV cache. Self-attention is the same — plain causal masking, no
  plugin. So there's no AttentionPlugin optional-input wiring to worry about on this graph;
  it's a "normal" ONNX subgraph TRT should parse natively.

## 3. General flow: adding a new model's ONNX export in TensorRT Edge-LLM

This repo never traces HuggingFace's model code. Every model is reimplemented from
scratch against ONNX builtin + this framework's custom ops, reading HF checkpoint
tensors directly. Steps, in order:

1. **Classify the component.** Decide which of the existing shapes it fits:
   - LLM backbone (`CausalLM` subclass, autoregressive decode with KV cache) →
     `onnx/export.py`.
   - Visual/audio encoder (one shot, no KV cache, splices embeddings into an LLM) →
     `onnx/export_encoder.py`.
   - Something with its own decode loop but no LLM (RNN-T step, Whisper's
     cross-attention decoder) → needs a bespoke `_export_<component>()` in
     `scripts/export.py`, following `_export_rnnt_decoder` as the closest template.

2. **Implement the from-scratch model** under `tensorrt_edgellm/models/<family>/`.
   Build every `Linear` through `make_linear()` (`models/linear.py`) driven by a shared
   `ModelConfig` — this is what makes FP16/FP8/NVFP4/INT4 dispatch automatic instead of
   hardcoded per model.

3. **Give it an export contract.**
   - `CausalLM` subclasses implement `onnx_export_spec()` → wrapped module + example
     args + input/output names + `dynamic_shapes`, consumed by `onnx/export.py::export_onnx`.
   - Everything else implements `get_onnx_export_args(config, device)` →
     `(inputs, input_names, output_names, dynamic_shapes)`, consumed by
     `onnx/export_encoder.py::_run_dynamo_export`.

4. **Translate the HF config.** HF `config.json` field names rarely match this
   framework's `ModelConfig` field names (Whisper's `d_model`/`encoder_attention_heads`
   vs. the default `hidden_size`/`num_attention_heads`) — translate explicitly where the
   model is built (see the `whisper` branch in `export_encoder.py::export_audio_onnx`).

5. **Load weights.** Go from the flat HF safetensors dict to the module tree via
   `checkpoint/loader.py::load_submodule_weights` (or a local equivalent): detect the
   checkpoint's key prefix, remap key names, cast dtypes on the way in.

6. **Register the dispatch**, don't call the exporter ad hoc:
   - `scripts/export.py`: add the model type to the relevant classification frozenset
     (`_VLM_MODEL_TYPES` / `_AUDIO_MODEL_TYPES` / `_LLM_COMPONENTS` / ...) and, if the
     output layout differs from the default, to `_LAYOUT_OVERRIDES`.
   - `onnx/export_encoder.py`: add to `_VISUAL_REGISTRY`/`_VISUAL_FAMILY_MODULE`/
     `_VISUAL_FAMILY_BUILD_FN` for a visual encoder, or an `elif model_type == ...:`
     branch in `export_audio_onnx()` for audio.

7. **Always export through the shared helper**, never raw `torch.onnx.export`:
   `onnx.export.export_onnx(...)` for LLM backbones, `_run_dynamo_export(...)` for
   everything else. Both fix opset (24), enable the custom translation table (custom
   attention/MoE/Mamba/quant ops — see `onnx/dynamo_translations.py`), and use
   `external_data=True`. The LLM path additionally runs TRT-compatibility passes
   (NVFP4 dtype fixup, AttentionPlugin optional-input stripping, FP32/FP16 initializer
   normalization, zero-volume initializer cleanup) — if a new model uses quantization
   outside the LLM path, call the relevant fixups explicitly afterward (see how
   `export_visual_onnx` does it).

8. **Write the runtime config.** LLM backbones get this for free from
   `write_runtime_artifacts()`. A standalone component (encoder, RNN-T step, and
   presumably a future Whisper decoder) needs its own hand-written `config.json` with a
   `model_type` tag plus whatever fields the corresponding C++ builder expects — see
   `_export_rnnt_decoder`'s `rnnt_decoder_config` block as the naming pattern.

9. **Validate `export → build → inference`, in that order** (per `AGENTS.md`).
   Export succeeding is a smoke test, not proof the model works — build the TensorRT
   engine and run it through the C++ runtime before calling it done. For a genuinely new
   architecture (like Whisper's decode loop), this last step may not exist yet and is
   its own piece of work, not an afterthought of the ONNX export.
