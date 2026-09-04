import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# ============================================================
# Whisper configuration
# ============================================================

_D_MODEL = 768
_MAX_SOURCE_POSITIONS = 1500


# ============================================================
# Sinusoidal Positional Encoding
# ============================================================

class SinusoidsPositionEmbedding(nn.Module):

    def __init__(
        self,
        length: int = _MAX_SOURCE_POSITIONS,
        channels: int = _D_MODEL,
        max_timescale: int = 10000,
    ):
        super().__init__()

        assert channels % 2 == 0

        # Number of frequencies = 768 / 2 = 384
        log_timescale_increment = (
            math.log(max_timescale)
            / (channels // 2 - 1)
        )

        inv_timescales = torch.exp(
            -log_timescale_increment
            * torch.arange(
                channels // 2,
                dtype=torch.float32,
            )
        )

        # ----------------------------------------------------
        # Positions
        #
        # [1500, 1]
        # ----------------------------------------------------

        positions = torch.arange(
            length,
            dtype=torch.float32,
        ).unsqueeze(1)

        # ----------------------------------------------------
        # Position × frequency
        #
        # [1500, 1] × [1, 384]
        #
        # =
        #
        # [1500, 384]
        # ----------------------------------------------------

        scaled_time = (
            positions
            * inv_timescales.unsqueeze(0)
        )

        # ----------------------------------------------------
        # Create
        #
        # 384 sine values
        # +
        # 384 cosine values
        #
        # =
        #
        # 768 values per position
        # ----------------------------------------------------

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

        return self.positional_embedding[
            :seqlen,
            :
        ]


# ============================================================
# Create positional encoding
# ============================================================

position_encoder = SinusoidsPositionEmbedding(
    length=1500,
    channels=768,
)

pe = position_encoder(1500)

print("Full positional encoding shape:")
print(pe.shape)

# torch.Size([1500, 768])


# ============================================================
# Simulate audio features
# ============================================================
#
# Assume after:
#
# Mel spectrogram
#       ↓
# Conv1
#       ↓
# Conv2
#       ↓
#
# Whisper produced:
#
# [batch, positions, features]
#
# [1, 250, 768]
#
# Here 250 positions could correspond roughly to a
# 5-second audio after downsampling.
#
# ============================================================

torch.manual_seed(10)

audio_features = torch.randn(
    1,
    250,
    768,
) * 0.5


print("\nAudio feature shape:")
print(audio_features.shape)

# [1, 250, 768]


# ============================================================
# Choose ONE position
# ============================================================

position = 37


# ============================================================
# Get AUDIO information at position 37
# ============================================================

audio_vector = audio_features[
    0,
    position,
    :
]


# ============================================================
# Get POSITION information at position 37
# ============================================================

position_vector = pe[
    position,
    :
]


# ============================================================
# Whisper operation
#
# AUDIO
# +
# POSITION
#
# ============================================================

combined_vector = (
    audio_vector
    +
    position_vector
)


# ============================================================
# Print shapes
# ============================================================

print("\n================================")
print("POSITION:", position)
print("================================")

print(
    "Audio vector shape:",
    audio_vector.shape
)

print(
    "Position vector shape:",
    position_vector.shape
)

print(
    "Combined vector shape:",
    combined_vector.shape
)


# ============================================================
# Print first 10 values
# ============================================================

print("\nFirst 10 AUDIO values:")

print(
    audio_vector[:10]
)


print("\nFirst 10 POSITION values:")

print(
    position_vector[:10]
)


print("\nFirst 10 COMBINED values:")

print(
    combined_vector[:10]
)


# ============================================================
# Verify addition manually
# ============================================================

print("\nExample: Dimension 0")

print(
    f"Audio      : {audio_vector[0]:.4f}"
)

print(
    f"Position   : {position_vector[0]:.4f}"
)

print(
    f"Combined   : {combined_vector[0]:.4f}"
)

print(
    "\nCheck:"
)

print(
    f"{audio_vector[0]:.4f}"
    f" + "
    f"{position_vector[0]:.4f}"
    f" = "
    f"{combined_vector[0]:.4f}"
)


# ============================================================
# Plot
# ============================================================

dimensions = torch.arange(
    _D_MODEL
)


plt.figure(
    figsize=(16, 8)
)


# ------------------------------------------------------------
# Audio feature
# ------------------------------------------------------------

plt.plot(
    dimensions,
    audio_vector.numpy(),
    label="Audio Feature (WHAT)",
    linewidth=1.3,
)


# ------------------------------------------------------------
# Positional information
# ------------------------------------------------------------

plt.plot(
    dimensions,
    position_vector.numpy(),
    label=f"Position {position} Encoding (WHERE)",
    linewidth=2,
)


# ------------------------------------------------------------
# Combined
# ------------------------------------------------------------

plt.plot(
    dimensions,
    combined_vector.numpy(),
    label="Audio + Position",
    linewidth=1.5,
)


# ============================================================
# Plot labels
# ============================================================

plt.xlabel(
    "Embedding Dimension (0 → 767)"
)

plt.ylabel(
    "Feature Value"
)

plt.title(
    f"Whisper Input at Position {position}\n"
    "Audio Feature + Positional Encoding"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()