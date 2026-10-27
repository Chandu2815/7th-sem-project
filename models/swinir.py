"""Compact SwinIR-style residual network for thermal SR + denoise.

Implements a lightweight architecture inspired by SwinIR (Liang et al.):
shallow feature extraction → residual transformer-style blocks → reconstruction.
When pretrained weights are missing, falls back to bicubic upsample + Gaussian denoise
so the pipeline remains runnable for demos.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "checkpoints" / "thermal-swinir.pt"


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.act = nn.GELU()
        self.attn = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        # Channel attention proxy (self-attention substitute for compact deploy)
        w = torch.sigmoid(self.attn(F.adaptive_avg_pool2d(x, 1)))
        x = x * w
        return residual + x


class SwinIRLite(nn.Module):
    """Lightweight SwinIR-inspired SR network (scale 2/4)."""

    def __init__(self, in_ch: int = 1, embed_dim: int = 64, depth: int = 4, scale: int = 4):
        super().__init__()
        self.scale = scale
        self.shallow = nn.Conv2d(in_ch, embed_dim, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(embed_dim) for _ in range(depth)])
        self.conv_after = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)
        up_layers = []
        n_ups = int(np.log2(scale)) if scale in (2, 4, 8) else 2
        ch = embed_dim
        for _ in range(n_ups):
            up_layers += [
                nn.Conv2d(ch, ch * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.GELU(),
            ]
        self.upsample = nn.Sequential(*up_layers)
        self.tail = nn.Conv2d(embed_dim, in_ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shallow = self.shallow(x)
        deep = self.conv_after(self.body(shallow)) + shallow
        out = self.upsample(deep)
        return self.tail(out)


def load_swinir(
    checkpoint: Path | str | None = None,
    scale: int = 4,
    device: str | None = None,
) -> tuple[SwinIRLite, str]:
    """Load SwinIRLite weights if present; otherwise return untrained model."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinIRLite(in_ch=1, embed_dim=64, depth=4, scale=scale)
    ckpt = Path(checkpoint) if checkpoint else DEFAULT_CHECKPOINT
    status = "random_init"
    if ckpt.is_file():
        state = torch.load(ckpt, map_location=device, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        status = f"loaded:{ckpt}"
        logger.info("Loaded SwinIR checkpoint from %s", ckpt)
    else:
        logger.warning("No checkpoint at %s — using random init / classical fallback", ckpt)
    model.to(device)
    model.eval()
    return model, status


class SwinIREnhancer:
    """High-level thermal enhancement: SR + denoise via SwinIRLite or classical fallback."""

    def __init__(
        self,
        checkpoint: Path | str | None = None,
        scale: int = 4,
        device: str | None = None,
        use_classical_if_untrained: bool = True,
    ):
        self.scale = scale
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.load_status = load_swinir(checkpoint, scale, self.device)
        self.use_classical = use_classical_if_untrained and self.load_status == "random_init"

    @torch.inference_mode()
    def enhance_array(self, thermal: np.ndarray) -> np.ndarray:
        """Enhance 2D float thermal array → upscaled float array."""
        if thermal.ndim != 2:
            raise ValueError("thermal must be HxW")

        if self.use_classical:
            return self._classical_sr_denoise(thermal)

        x = torch.from_numpy(np.nan_to_num(thermal, nan=0.0).astype(np.float32))[None, None]
        x = x.to(self.device)
        # Pad to multiple of 8
        _, _, h, w = x.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        y = self.model(x)
        y = y[:, :, : h * self.scale, : w * self.scale]
        return y.squeeze().cpu().numpy()

    def enhance_patch_file(self, npz_path: Path | str) -> np.ndarray:
        data = np.load(npz_path)
        thermal = data["thermal"]
        return self.enhance_array(thermal)

    @staticmethod
    def _classical_sr_denoise(thermal: np.ndarray, scale: int = 4) -> np.ndarray:
        """Bicubic upsample + mild Gaussian denoise (demo fallback)."""
        from scipy.ndimage import gaussian_filter, zoom

        clean = gaussian_filter(np.nan_to_num(thermal, nan=0.0), sigma=0.8)
        return zoom(clean, scale, order=3).astype(np.float32)
