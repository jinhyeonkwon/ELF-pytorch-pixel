"""Glyph-vocab atlas: a pixel-defined vocabulary for the ELF pixel variant.

A "token" here is one Unicode character rendered to a fixed, deterministic glyph
image. The vocabulary is three small files produced offline (the builder lives in
the sibling `flm_ours` repo; we ship only the outputs under `pixel/assets/`):

  * `*_glyph_atlas.npy`         (V, char_h, char_w) float32 in [0,1]  — 1=white bg, 0=black ink
  * `*_codepoint_to_index.npy`  (max_codepoint,)     int64           — Unicode cp -> vocab index
  * `*_vocab_meta.json`         metadata (char_h/char_w, unk/eos index, normalizer, V)

A discriminator stamp at build time makes every glyph bit-distinct (look-alike
o/o, dashes, the whitespace family, tofu, null<->EOS are all disambiguated), so a
clean glyph image is a deterministic, lossless-up-to-the-tokenizer encoding of
text. That is what lets the flow model's OCR head read a generated image back to
characters exactly.

Self-contained: reads the npy/json directly and builds the HF normalizer from
`transformers` (the same policy used when the atlas was built, e.g.
bert-base-uncased lowercases / strips accents / collapses whitespace).

This is a trimmed port of `flm_ours/pixel_predict_mse/flm_based/glyph_render.py`
(+ the `glyph_data` index->char helpers), with no dependency on that repo.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

EOS_CHAR = "\0"          # document-boundary sentinel (one full-black EOS glyph)
IGNORE_INDEX = -100      # cross-entropy ignore for empty (white background) cells

_NORMALIZER_CACHE: dict = {}


def build_normalizer(name):
    """str->str normalizer for tokenizer ``name`` (None == identity).

    Reuses the tokenizer's own HF backend normalizer so the encode-time policy
    matches the one used when the vocab atlas was built.
    """
    if not name:
        return None
    if name in _NORMALIZER_CACHE:
        return _NORMALIZER_CACHE[name]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    backend = getattr(tok, "backend_tokenizer", None)
    norm = getattr(backend, "normalizer", None) if backend is not None else None
    fn = (lambda s: norm.normalize_str(s)) if norm is not None else None
    _NORMALIZER_CACHE[name] = fn
    return fn


class GlyphAtlasVocab:
    """Codepoint->index LUT + glyph atlas.

    ``vocab_dir`` is a directory holding the three files described above, e.g.
    ``pixel/assets/bert-base-uncased``.
    """

    def __init__(self, vocab_dir, load_atlas=True, apply_normalizer=True):
        self.vocab_dir = vocab_dir
        meta_path = self._one(vocab_dir, "*_vocab_meta.json")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        self.meta = meta
        self.tokenizer = meta["tokenizer"]
        self.char_h = int(meta["char_h"])
        self.char_w = int(meta["char_w"])
        self.unk_index = int(meta["unk_index"])
        self.eos_index = int(meta["eos_index"])
        self.vocab_size = int(meta["n_total"])

        self.lut = np.load(self._one(vocab_dir, "*_codepoint_to_index.npy")
                           ).astype(np.int64, copy=False)
        self.atlas = None
        if load_atlas:
            atlas = np.load(self._one(vocab_dir, "*_glyph_atlas.npy"))
            self.atlas = np.ascontiguousarray(atlas.astype(np.float32, copy=False))
            assert self.atlas.shape == (self.vocab_size, self.char_h, self.char_w), (
                f"atlas shape {self.atlas.shape} != "
                f"({self.vocab_size}, {self.char_h}, {self.char_w})")

        self.encode_normalizer = meta.get("encode_normalizer")
        self._normalize = (build_normalizer(self.encode_normalizer)
                           if apply_normalizer else None)

    @staticmethod
    def _one(d, pattern):
        hits = sorted(glob.glob(os.path.join(d, pattern)))
        if not hits:
            raise FileNotFoundError(f"no {pattern} under {d}")
        return hits[0]

    def normalize(self, text):
        if self._normalize is not None and text:
            return self._normalize(text)
        return text

    def encode(self, text):
        """str -> np.int64 array of per-character vocab indices (UNK fallback)."""
        text = self.normalize(text)
        if not text:
            return np.empty(0, dtype=np.int64)
        codes = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
        # Codepoints beyond the LUT fall back to UNK.
        codes = np.where(codes < len(self.lut), codes, 0).astype(np.int64)
        out = self.lut[codes]
        return out

    def encode_window_indices(self, text, max_chars, eos_char=EOS_CHAR):
        """Window text (with EOS_CHAR doc separators) -> list[int] of <= max_chars.

        Each EOS_CHAR becomes one EOS glyph; the text between separators is
        normalized then mapped through the LUT. Truncated to max_chars. May be
        shorter than max_chars (normalization collapses whitespace); the caller
        pads the remainder with the white background.
        """
        idxs: list[int] = []
        for i, part in enumerate(text.split(eos_char)):
            if i > 0:
                idxs.append(self.eos_index)
                if len(idxs) >= max_chars:
                    break
            if part:
                idxs.extend(int(x) for x in self.encode(part))
            if len(idxs) >= max_chars:
                break
        return idxs[:max_chars]


def build_index_to_char(vocab: GlyphAtlasVocab):
    """Approximate inverse of the codepoint->index LUT, for decoding readouts.

    For each vocab index pick the first codepoint that maps to it. UNK->'?',
    EOS->newline, the whitespace family forced to ' '. Good enough for
    human-readable / PPL readout (the rendering is deterministic per index).
    """
    inv = [""] * vocab.vocab_size
    seen = set()
    lut = vocab.lut
    for cp in range(len(lut)):
        idx = int(lut[cp])
        if 0 <= idx < vocab.vocab_size and idx not in seen:
            try:
                inv[idx] = chr(cp)
            except ValueError:
                inv[idx] = "?"
            seen.add(idx)
    if 0 <= vocab.unk_index < vocab.vocab_size:
        inv[vocab.unk_index] = "?"
    if 0 <= vocab.eos_index < vocab.vocab_size:
        inv[vocab.eos_index] = "\n"
    sp_idx = int(lut[ord(" ")]) if len(lut) > ord(" ") else -1
    if 0 <= sp_idx < vocab.vocab_size:
        inv[sp_idx] = " "
    return inv


def indices_to_text(idx_row, index_to_char):
    """(max_chars,) int array/tensor -> str."""
    try:
        import torch
        if torch.is_tensor(idx_row):
            idx_row = idx_row.tolist()
    except ImportError:
        pass
    return "".join(index_to_char[int(i)] for i in idx_row)
