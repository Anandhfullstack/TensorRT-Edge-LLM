import torch

from transformers import WhisperForConditionalGeneration

from tensorrt_edgellm.models.whisper.modeling_whisper_decoder import (
    build_whisper_decoder,
)
from tensorrt_edgellm.config import ModelConfig


DEVICE = "cuda"
MODEL_NAME = "openai/whisper-small"


def main():

    dtype = torch.float16

    print("=" * 80)
    print("Loading Hugging Face Whisper")
    print("=" * 80)

    hf_model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
    ).to(DEVICE)

    hf_model.eval()

    config = hf_model.config

    decoder_config = {
        "d_model": config.d_model,
        "decoder_layers": config.decoder_layers,
        "decoder_attention_heads": config.decoder_attention_heads,
        "decoder_ffn_dim": config.decoder_ffn_dim,
        "vocab_size": config.vocab_size,
        "max_target_positions": config.max_target_positions,
        "pad_token_id": config.pad_token_id,
    }

    model_config = ModelConfig(
        model_type="whisper",
        hidden_size=config.d_model,
        num_hidden_layers=config.decoder_layers,
        num_attention_heads=config.decoder_attention_heads,
        num_key_value_heads=config.decoder_attention_heads,
        intermediate_size=config.decoder_ffn_dim,
        head_dim=config.d_model // config.decoder_attention_heads,
        rms_norm_eps=1e-5,
        vocab_size=config.vocab_size,
        rope_theta=10000,
        max_position_embeddings=config.max_target_positions,
        default_attention_scale=1.0,
        torch_dtype="float16",
        tie_word_embeddings=True,
    )

    native_decoder = build_whisper_decoder(
        config=decoder_config,
        weights=hf_model.state_dict(),
        dtype=dtype,
        prefix="model.decoder.",
        model_config=model_config,
    ).to(DEVICE)

    native_decoder.eval()

    input_ids = torch.tensor(
        [[50258, 50359, 50363, 2425, 1917, 13, 50257, 50256]],
        dtype=torch.long,
        device=DEVICE,
    )

    encoder_hidden_states = torch.randn(
        1,
        1500,
        config.d_model,
        dtype=dtype,
        device=DEVICE,
    )

    hf_debug = {}

    layer = hf_model.model.decoder.layers[11]

    layer.fc1.register_forward_hook(
        lambda m, i, o: hf_debug.update({"fc1": o.detach()})
    )

    layer.fc2.register_forward_hook(
        lambda m, i, o: hf_debug.update({"fc2": o.detach()})
    )

    layer.self_attn.register_forward_hook(
        lambda m, i, o: hf_debug.update({"self_attn": o[0].detach()})
    )

    layer.encoder_attn.register_forward_hook(
        lambda m, i, o: hf_debug.update({"cross_attn": o[0].detach()})
    )

    layer.register_forward_hook(
        lambda m, i, o: hf_debug.update({"layer11_output": o.detach()})
    )


    with torch.no_grad():

        hf_out = hf_model.model.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            output_hidden_states=True,
        )

        hf_hidden = hf_out.last_hidden_state
        hf_layers = hf_out.hidden_states
        hf_logits = hf_model.proj_out(hf_hidden)


        native_hidden, native_layers, native_debug = native_decoder(
            input_ids,
            encoder_hidden_states,
        )


    print("=" * 80)
    print("Layer comparison")
    print("=" * 80)

    for i in range(config.decoder_layers):

        diff = (
            hf_layers[i]
            -
            (native_layers[i-1] if i > 0 else hf_layers[0])
        ).abs()

        print(
            f"Layer {i} input max:",
            diff.max().item()
        )


    print()
    print("=" * 80)
    print("Layer 11 pre-final-norm comparison")
    print("=" * 80)


    diff = (
        hf_debug["layer11_output"]
        -
        native_layers[11]
    ).abs()


    print(
        "max:",
        diff.max().item()
    )

    print(
        "mean:",
        diff.mean().item()
    )


    print("=" * 80)
    print("Layer 11 FFN comparison")
    print("=" * 80)

    native_debug11 = native_debug[11]

    for name, hf_tensor, native_tensor in [
        ("fc1", hf_debug["fc1"], native_debug11["ffn_fc1"]),
        ("fc2", hf_debug["fc2"], native_debug11["ffn_fc2"]),
        ("self_attn", hf_debug["self_attn"], native_debug11["self_attn_output"]),
        ("cross_attn", hf_debug["cross_attn"], native_debug11["cross_attn_output"]),
    ]:

        diff = (hf_tensor - native_tensor).abs()

        print(
            name,
            "max:",
            diff.max().item(),
            "mean:",
            diff.mean().item(),
        )


    print("=" * 80)
    print("Final comparison")
    print("=" * 80)

    hidden_diff = (hf_hidden - native_hidden).abs()

    print(
        "hidden max:",
        hidden_diff.max().item()
    )

    print(
        "hidden mean:",
        hidden_diff.mean().item()
    )


    native_logits = torch.nn.functional.linear(
        native_hidden,
        native_decoder.embed_tokens.weight,
    )

    logits_diff = (hf_logits - native_logits).abs()

    print(
        "logits max:",
        logits_diff.max().item()
    )

    print(
        "logits mean:",
        logits_diff.mean().item()
    )


if __name__ == "__main__":
    main()
