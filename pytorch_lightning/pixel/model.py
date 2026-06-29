"""ELFPixel — the ELF transformer with pixel-glyph I/O.

The flow runs over a sequence of glyph-strip *patches* instead of T5 token
embeddings. A glyph strip (B, 1, H, W) is patchified into (B, N, P) with
P = patch_h * patch_w (one token per patch_w-wide column block); the flow's
`x0` is exactly that tensor in [-1, 1]. So every modality-agnostic piece of the
upstream ELF code (interpolant, velocity target, ODE/SDE steppers, EMA,
optimizer, trainer) is reused verbatim — only the input projection, output head,
and decode head are pixel-specific.

Reused from `modules/`:
  ELFBlock, RMSNorm, TextRotaryEmbeddingFast, TimestepEmbedder,
  BottleneckTextProj, FinalLayer, _normal_002_.

Two heads (matching ELF's dual objective):
  * denoiser head : FinalLayer(hidden -> P)            -> (B, N, P) x-prediction
  * OCR decode head: hidden -> chars_per_token * vocab  -> (B, N*cpt, V) logits
The decode head is the pixel analog of ELF's token-unembed; alignment is exact
because rendering is monospaced at char_w and patch_w is an integer multiple.

`forward` keeps ELF's contract: (x, t, attention_mask, self_cond_cfg_scale,
decoder_step_active) -> (x_pred, decoder_logits|None), so `net_out_to_v_x`,
`generate_samples`, and `dlm_decode_batch` work unchanged.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.layers import (
    BottleneckTextProj, FinalLayer, RMSNorm, TextRotaryEmbeddingFast,
    TimestepEmbedder, _normal_002_,
)
from modules.model import ELFBlock


# --------------------------------------------------------------------------- #
# Patch <-> image helpers (single row of patches: grid_h == 1)
# --------------------------------------------------------------------------- #
def patchify(img: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
    """(B, C, H, W) -> (B, N, C*patch_h*patch_w), N = W // patch_w.

    H must equal patch_h. Flatten order within a patch is (C, h, w) row-major,
    matching `unpatchify`."""
    B, C, H, W = img.shape
    assert H == patch_h, f"H ({H}) must equal patch_h ({patch_h})"
    assert W % patch_w == 0, f"W ({W}) must be a multiple of patch_w ({patch_w})"
    N = W // patch_w
    x = img.reshape(B, C, H, N, patch_w)          # split width
    x = x.permute(0, 3, 1, 2, 4)                   # (B, N, C, H, patch_w)
    return x.reshape(B, N, C * H * patch_w)


def unpatchify(x: torch.Tensor, patch_h: int, patch_w: int, channels: int = 1) -> torch.Tensor:
    """(B, N, C*patch_h*patch_w) -> (B, C, patch_h, N*patch_w)."""
    B, N, _ = x.shape
    x = x.reshape(B, N, channels, patch_h, patch_w)
    x = x.permute(0, 2, 3, 1, 4)                   # (B, C, H, N, patch_w)
    return x.reshape(B, channels, patch_h, N * patch_w)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class ELFPixel(nn.Module):
    def __init__(
        self,
        patch_h: int = 16,
        patch_w: int = 64,
        char_w: int = 8,
        in_channels: int = 1,
        vocab_size: int = 1000,
        max_length: int = 20,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        bottleneck_dim: int = 128,
        decode_bottleneck: int = 256,
        num_time_tokens: int = 4,
        num_self_cond_cfg_tokens: int = 0,
        num_model_mode_tokens: int = 4,
        self_cond_input: bool = False,
        use_flash: bool = True,
    ):
        super().__init__()
        if num_time_tokens <= 0:
            raise ValueError("num_time_tokens must be positive")
        assert patch_w % char_w == 0, f"patch_w ({patch_w}) must be a multiple of char_w ({char_w})"

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.char_w = char_w
        self.in_channels = in_channels
        self.patch_pixels = in_channels * patch_h * patch_w   # P (per-patch flow dim)
        self.chars_per_token = patch_w // char_w
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.max_chars = max_length * self.chars_per_token
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_time_tokens = num_time_tokens
        self.num_self_cond_cfg_tokens = num_self_cond_cfg_tokens
        self.num_model_mode_tokens = num_model_mode_tokens
        self.self_cond_input = self_cond_input

        # Alias the per-patch flow dim under ELF's attribute name, so the eval /
        # sampling code (which reads `model.text_encoder_dim` for the noise shape)
        # works unchanged.
        self.text_encoder_dim = self.patch_pixels

        P = self.patch_pixels
        if self_cond_input:
            self.self_cond_proj = nn.Linear(2 * P, P, bias=True)
            nn.init.xavier_uniform_(self.self_cond_proj.weight)
            nn.init.zeros_(self.self_cond_proj.bias)

        # input "patch-embed": P -> hidden (Linear, ELF's bottleneck projector)
        self.text_proj = BottleneckTextProj(P, hidden_size, bottleneck_dim)

        # time / cfg / mode prefix conditioning (ELF-faithful in-context tokens)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_emb_tokens = nn.Parameter(torch.empty(1, num_time_tokens, hidden_size))
        _normal_002_(self.t_emb_tokens)
        if num_self_cond_cfg_tokens > 0:
            self.self_cond_cfg_embedder = TimestepEmbedder(hidden_size)
            self.self_cond_cfg_tokens = nn.Parameter(
                torch.empty(1, num_self_cond_cfg_tokens, hidden_size))
            _normal_002_(self.self_cond_cfg_tokens)
        if num_model_mode_tokens > 0:
            self.mode_tokens = nn.Parameter(torch.empty(1, num_model_mode_tokens, hidden_size))
            _normal_002_(self.mode_tokens)

        head_dim = hidden_size // num_heads
        prefix_len = num_time_tokens + (num_self_cond_cfg_tokens if num_self_cond_cfg_tokens > 0 else 0)
        empty_offset = prefix_len + (num_model_mode_tokens if num_model_mode_tokens > 0 else 0)
        self.feat_rope = TextRotaryEmbeddingFast(
            dim=head_dim, pt_seq_len=max_length, num_empty_token=empty_offset)

        q1, q3 = depth // 4, depth // 4 * 3
        self.blocks = nn.ModuleList([
            ELFBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                     attn_drop=(attn_drop if q3 > i >= q1 else 0.0),
                     proj_drop=(proj_drop if q3 > i >= q1 else 0.0),
                     use_flash=use_flash)
            for i in range(depth)
        ])

        # denoiser (pixel) head: hidden -> P  (zero-init, ELF FinalLayer)
        self.final_layer = FinalLayer(hidden_size, patch_size=1, out_channels=P)

        # OCR decode head: per patch token -> chars_per_token * vocab logits
        self.dec_norm = RMSNorm(hidden_size)
        self.dec_proj = nn.Linear(hidden_size, decode_bottleneck)
        self.dec_unemb = nn.Linear(decode_bottleneck, self.chars_per_token * vocab_size)
        nn.init.xavier_uniform_(self.dec_proj.weight); nn.init.zeros_(self.dec_proj.bias)
        nn.init.xavier_uniform_(self.dec_unemb.weight); nn.init.zeros_(self.dec_unemb.bias)

    def build_context(self, t, self_cond_cfg_scale=None):
        B = t.shape[0]
        out = [self.t_emb_tokens.expand(B, -1, -1) + self.t_embedder(t).unsqueeze(1)]
        if self_cond_cfg_scale is not None and self.num_self_cond_cfg_tokens > 0:
            sc_emb = self.self_cond_cfg_embedder(self_cond_cfg_scale)
            out.append(self.self_cond_cfg_tokens.expand(B, -1, -1) + sc_emb.unsqueeze(1))
        return out

    def forward(self, x, t, attention_mask=None, self_cond_cfg_scale=None,
                decoder_step_active: bool = False
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """x: (B, N, P) or (B, N, 2P) when self-cond.  t: (B,).
        -> (x_pred (B,N,P), decoder_logits (B, N*cpt, V) | None)."""
        B = x.shape[0]
        if self.self_cond_input and x.shape[-1] == 2 * self.patch_pixels:
            x = self.self_cond_proj(x)
        x = self.text_proj(x)                                   # (B, N, hidden)

        model_mode_offset = 0
        if self.num_model_mode_tokens > 0:
            mode_tokens = self.mode_tokens.expand(B, -1, -1)
            if not decoder_step_active:
                mode_tokens = torch.zeros_like(mode_tokens)     # silent on denoiser steps
            x = torch.cat([mode_tokens, x], dim=1)
            model_mode_offset = self.num_model_mode_tokens

        prefix_len = 0
        context = self.build_context(t, self_cond_cfg_scale=self_cond_cfg_scale)
        if context:
            prefix = torch.cat(context, dim=1)
            prefix_len = prefix.shape[1]
            x = torch.cat([prefix, x], dim=1)

        for block in self.blocks:
            x = block(x, rope_fn=self.feat_rope, attention_mask=attention_mask)
        x = x[:, prefix_len + model_mode_offset:]               # strip prefix + mode

        decoder_logits = None
        if decoder_step_active:
            h = F.gelu(self.dec_proj(self.dec_norm(x)))
            logits = self.dec_unemb(h)                          # (B, N, cpt*V)
            decoder_logits = logits.reshape(B, x.shape[1] * self.chars_per_token,
                                            self.vocab_size)     # (B, max_chars, V)

        return self.final_layer(x), decoder_logits


def ELFPixel_B(**kw): return ELFPixel(hidden_size=768, depth=12, num_heads=12, **kw)
def ELFPixel_M(**kw): return ELFPixel(hidden_size=1056, depth=24, num_heads=16, **kw)
def ELFPixel_L(**kw): return ELFPixel(hidden_size=1280, depth=32, num_heads=16, **kw)
ELFPixel_models = {"ELFPixel-B": ELFPixel_B, "ELFPixel-M": ELFPixel_M, "ELFPixel-L": ELFPixel_L}
