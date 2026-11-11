"""
ASLTransformer: Inception-CNN + Shallow Transformer Encoder

Architecture:
  Input Projection  : Linear(255 → 256) + LayerNorm + GELU
  Inception CNN     : 3 parallel Conv1d branches (k=3,5,7), element-wise sum,
                      residual add  (permuted before & after for Conv1d)
  Transformer       : 4× PreLN TransformerEncoderLayer (d=256, h=8, ffn=512)
  Global Avg Pool   : mean over time dimension
  Classifier Head   : LN → Dropout → Linear(256→512) → GELU → Dropout → Linear(512→100)

Total parameters: ~4.3M
"""

import math
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# POSITIONAL ENCODING (sinusoidal, fixed — no learnable parameters)
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding (Vaswani et al., 2017).
    Adds a unique timestamp to each frame so the Transformer knows
    whether it is looking at frame 1 or frame 60.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model)
        return self.dropout(x + self.pe[:, : x.size(1)])


# ─────────────────────────────────────────────────────────────────────────────
# INCEPTION-STYLE MULTI-SCALE 1D-CNN BLOCK
# ─────────────────────────────────────────────────────────────────────────────

class InceptionCNNBlock(nn.Module):
    """
    Three parallel Conv1d branches with kernels 3, 5, 7.
    Each branch independently processes the SAME input tensor,
    capturing motion at a different timescale:
      k=3  →  sharp wrist flicks      (3-frame window)
      k=5  →  medium-speed arm arcs   (5-frame window)
      k=7  →  slow shoulder rolls     (7-frame window)

    Outputs are element-wise SUMMED (not concatenated) so the channel
    dimension stays at `channels`, enabling a clean residual add.

    ⚠ IMPORTANT: expects input in Conv1d format (batch, channels, seq).
    The ASLTransformer.forward() handles permuting before and after this block.
    """

    def __init__(self, channels: int = 256, dropout: float = 0.1):
        super().__init__()

        def _branch(kernel_size: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(channels),
                nn.GELU(),
            )

        self.branch3 = _branch(3)
        self.branch5 = _branch(5)
        self.branch7 = _branch(7)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq)
        residual = x
        # All three branches see the SAME input — truly parallel
        a = self.branch3(x)
        b = self.branch5(x)
        c = self.branch7(x)
        out = a + b + c                      # element-wise sum of all timescales
        return self.dropout(out + residual)  # residual connection


# ─────────────────────────────────────────────────────────────────────────────
# MAIN MODEL
# ─────────────────────────────────────────────────────────────────────────────

class ASLTransformer(nn.Module):
    """
    Full hybrid model: Input Projection → Inception CNN → Transformer → Head.

    Args:
        input_dim   : features per frame (85 keypoints × 3 coords = 255)
        d_model     : internal model dimension
        nhead       : number of attention heads
        num_layers  : transformer encoder depth
        dim_ff      : feedforward dimension inside transformer
        num_classes : output classes
        dropout     : base dropout rate
    """

    def __init__(
        self,
        input_dim:   int = 255,
        d_model:     int = 256,
        nhead:       int = 8,
        num_layers:  int = 4,
        dim_ff:      int = 512,
        num_classes: int = 100,
        dropout:     float = 0.1,
    ):
        super().__init__()

        # Block 1 — Input Projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Block 2 — Inception CNN
        # NOTE: caller must permute (batch,seq,feat)→(batch,feat,seq) before,
        #       and permute back after. See forward().
        self.cnn = InceptionCNNBlock(channels=d_model, dropout=dropout)

        # Block 3 — Positional Encoding + Transformer
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,   # expects (batch, seq, features)
            norm_first=True,    # Pre-LN: more stable training than Post-LN
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classifier Head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.3),
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim_ff, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x      : (batch, seq=60, input_dim=255)
        returns: (batch, num_classes=100)  — raw logits
        """
        # Block 1 — project features
        x = self.input_proj(x)            # (batch, 60, 256)

        # Block 2 — Inception CNN
        # Conv1d requires (batch, channels, seq) — permute in, permute back out
        x = x.permute(0, 2, 1)            # (batch, 256, 60)
        x = self.cnn(x)                   # (batch, 256, 60)
        x = x.permute(0, 2, 1)            # (batch, 60, 256)

        # Block 3 — Transformer (expects batch_first → (batch, seq, features) ✅)
        x = self.pos_enc(x)               # (batch, 60, 256)
        x = self.transformer(x)           # (batch, 60, 256)

        # Global Average Pool over the time dimension
        x = x.mean(dim=1)                 # (batch, 256)

        # Classifier
        return self.head(x)               # (batch, 100)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SHAPE SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = ASLTransformer()
    dummy = torch.randn(4, 60, 255)     # batch=4, seq=60, features=255
    out   = model(dummy)
    assert out.shape == (4, 100), f"Expected (4, 100), got {out.shape}"
    print(f"✅ Forward pass OK — output shape: {out.shape}")
    print(f"   Total parameters: {model.count_parameters():,}")
