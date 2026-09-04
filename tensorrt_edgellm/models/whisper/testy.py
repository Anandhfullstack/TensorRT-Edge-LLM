import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


_D_MODEL = 768
_MAX_SOURCE_POSITIONS = 1500


class SinusoidsPositionEmbedding(nn.Module):

    def __init__(
        self,
        length: int = _MAX_SOURCE_POSITIONS,
        channels: int = _D_MODEL,
        max_timescale: int = 10000,
    ):
        super().__init__()

        assert channels % 2 == 0

        # --------------------------------------------------
        # Create different frequencies
        # --------------------------------------------------
        log_timescale_increment = (
            math.log(max_timescale) /
            (channels // 2 - 1)
        )

        inv_timescales = torch.exp(
            -log_timescale_increment *
            torch.arange(
                channels // 2,
                dtype=torch.float32
            )
        )

        # --------------------------------------------------
        # Position × frequency
        #
        # shape:
        # [1500, 1] * [1, 384]
        #       ↓
        # [1500, 384]
        # --------------------------------------------------
        scaled_time = (
            torch.arange(
                length,
                dtype=torch.float32
            ).unsqueeze(1)
            *
            inv_timescales.unsqueeze(0)
        )

        # --------------------------------------------------
        # 384 sine values
        # +
        # 384 cosine values
        #
        # = 768-dimensional positional vector
        # --------------------------------------------------
        pos_emb = torch.cat(
            [
                torch.sin(scaled_time),
                torch.cos(scaled_time),
            ],
            dim=1,
        )

        self.register_buffer(
            "positional_embedding",
            pos_emb,
            persistent=False,
        )

    def forward(self, seqlen: int):
        return self.positional_embedding[:seqlen, :]


# ==========================================================
# Generate positional encodings
# ==========================================================

encoder = SinusoidsPositionEmbedding(
    length=1500,
    channels=768,
)

pe = encoder(1500).detach().cpu()

print("Full positional embedding shape:")
print(pe.shape)

# Expected:
# torch.Size([1500, 768])


# ==========================================================
# Select TWO positions
# ==========================================================

position_1 = 37
position_2 = 108


pattern_1 = pe[position_1]
pattern_2 = pe[position_2]


print("\nPosition 37 shape:")
print(pattern_1.shape)

print("\nPosition 38 shape:")
print(pattern_2.shape)

# Each one:
# torch.Size([768])


# ==========================================================
# Print first few values
# ==========================================================

print("\nFirst 10 values of Position 37:")
print(pattern_1[:10])

print("\nFirst 10 values of Position 38:")
print(pattern_2[:10])


# ==========================================================
# Plot the TWO complete positional patterns
# ==========================================================

dimensions = torch.arange(_D_MODEL)

plt.figure(figsize=(15, 7))

plt.plot(
    dimensions,
    pattern_1.numpy(),
    label=f"Position {position_1}",
    linewidth=2,
)

plt.plot(
    dimensions,
    pattern_2.numpy(),
    label=f"Position {position_2}",
    linewidth=2,
)

plt.xlabel("Embedding Dimension (0 → 767)")
plt.ylabel("Sin / Cos Value")

plt.title(
    f"Complete Positional Encoding Patterns: "
    f"Position {position_1} vs Position {position_2}"
)

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()