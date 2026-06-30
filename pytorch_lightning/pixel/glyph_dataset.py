"""LM1B text -> fixed-width glyph-strip dataset for the ELF pixel variant.

Three pieces, all self-contained (no `flm_ours` / pygame dependency):

  1. `load_lm1b`  — download+extract the One Billion Word Benchmark once and
     expose it as an HF `Dataset` with a single `text` column.
  2. `ContinuousDocumentStreamDataset` — random-access fixed-character windows
     over `doc + EOS + next_doc + EOS + ...`. Each item is `{"text": <window>}`.
  3. `make_glyph_collate_with_labels` — render a batch of window texts to glyph
     strips via the atlas and also emit per-cell OCR labels.

Geometry (defaults, faithful to the flm_ours experiment):
  img 16x1280, char cell 16x8 -> 160 chars per window; patch 16x64 -> 20 tokens,
  8 chars/token. `pixel_values` is (B,1,16,1280) float32 in [0,1] (white bg).
"""

from __future__ import annotations

import bisect
import math
import os
import tarfile
import tempfile
import urllib.request

import numpy as np
import torch
from torch.utils.data import Dataset

from pixel.glyph_vocab import EOS_CHAR, IGNORE_INDEX, GlyphAtlasVocab


# ---------------------------------------------------------------------------
# LM1B loader (port of flm_ours/pixel/lm1b_local.py)
# ---------------------------------------------------------------------------
_LM1B_URL = ("https://www.statmt.org/lm-benchmark/"
             "1-billion-word-language-modeling-benchmark-r13output.tar.gz")
_LM1B_DIRNAME = "1-billion-word-language-modeling-benchmark-r13output"
_TRAIN_SUBDIR = "training-monolingual.tokenized.shuffled"
_TEST_SUBDIR = "heldout-monolingual.tokenized.shuffled"


def _default_root() -> str:
    data_dir = os.environ.get("DATA_DIR")
    if data_dir:
        return os.path.join(data_dir, "lm1b")
    cache = (os.environ.get("HF_DATASETS_CACHE")
             or os.path.expanduser("~/.cache/huggingface/datasets"))
    return os.path.join(cache, "lm1b_raw")


def _normalize_split(split: str) -> str:
    if split == "train":
        return "train"
    if split in ("test", "validation", "valid"):
        return "test"
    raise ValueError(f"lm1b split must be train/test/validation (got {split!r})")


def _ensure_downloaded(root: str) -> str:
    os.makedirs(root, exist_ok=True)
    extract_root = os.path.join(root, _LM1B_DIRNAME)
    if (os.path.isdir(os.path.join(extract_root, _TRAIN_SUBDIR))
            and os.path.isdir(os.path.join(extract_root, _TEST_SUBDIR))):
        return extract_root

    tar_path = os.path.join(root, "lm1b-r13output.tar.gz")
    if not os.path.exists(tar_path):
        tmp_fd, tmp_path = tempfile.mkstemp(dir=root, suffix=".part")
        os.close(tmp_fd)
        try:
            print(f"[lm1b] downloading LM1B (~1.8 GB) -> {tar_path}")
            urllib.request.urlretrieve(_LM1B_URL, tmp_path)
            os.replace(tmp_path, tar_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    print(f"[lm1b] extracting {tar_path} -> {root}")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(root)
    if not os.path.isdir(os.path.join(extract_root, _TRAIN_SUBDIR)):
        raise RuntimeError(f"LM1B extraction missing {extract_root}/{_TRAIN_SUBDIR}")
    return extract_root


def _split_files(extract_root: str, split: str):
    sub = _TRAIN_SUBDIR if split == "train" else _TEST_SUBDIR
    d = os.path.join(extract_root, sub)
    return sorted(os.path.join(d, f) for f in os.listdir(d) if not f.startswith("."))


def load_lm1b(split: str = "train", cache_dir: str = None, max_files: int = None):
    """Return an HF Dataset with one row per sentence (column ``text``)."""
    import datasets
    split = _normalize_split(split)
    root = cache_dir or _default_root()
    extract_root = _ensure_downloaded(root)
    files = _split_files(extract_root, split)
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise RuntimeError(f"No lm1b files in {extract_root} for split={split}")
    return datasets.load_dataset("text", data_files={split: files}, split=split,
                                 cache_dir=cache_dir)


def load_openwebtext(split: str = "train", cache_dir: str = None):
    """Return an HF Dataset with one document per row (column ``text``).

    Forward-compatible path for the larger setup (OWT @ 1024 patches). The
    windowing dataset + collate below are dataset-agnostic — they only need a
    ``text`` column — so OWT slots in with no other change. Untested at runtime
    here (the raw corpus is a large download); LM1B is the validated path.
    """
    import datasets
    hf_split = {"train": "train[:-100000]", "valid": "train[-100000:]",
                "validation": "train[-100000:]", "test": "train[-100000:]",
                "all": "train"}.get(split, split)
    return datasets.load_dataset("Skylion007/openwebtext", split=hf_split,
                                 cache_dir=cache_dir, trust_remote_code=True)


def load_text_dataset(name: str, split: str = "train", cache_dir: str = None):
    """Dispatch by dataset name -> HF Dataset with a ``text`` column."""
    name = (name or "lm1b").lower()
    if name in ("lm1b", "1b", "billion_word"):
        return load_lm1b(split=split, cache_dir=cache_dir)
    if name in ("openwebtext", "owt"):
        return load_openwebtext(split=split, cache_dir=cache_dir)
    raise ValueError(f"Unknown dataset {name!r} (expected 'lm1b' or 'openwebtext')")


# ---------------------------------------------------------------------------
# Continuous fixed-width windowing (pygame-free port)
# ---------------------------------------------------------------------------
class ContinuousDocumentStreamDataset(Dataset):
    """Random-access character windows over `doc + EOS + next_doc + EOS + ...`."""

    def __init__(self, source_dataset, chars_per_window: int,
                 limit_documents: int = None, eos_char: str = EOS_CHAR,
                 drop_last: bool = True, wrap: bool = False,
                 cache_path: str = None):
        self.source_dataset = source_dataset
        self.chars_per_window = int(chars_per_window)
        self.eos_char = eos_char
        self.drop_last = bool(drop_last)
        self.wrap = bool(wrap)
        n = len(source_dataset)
        self.num_documents = min(limit_documents, n) if limit_documents is not None else n
        if self.num_documents <= 0:
            raise ValueError("need at least one document")

        # Per-document char lengths are the only expensive part (a full scan of
        # the corpus). Cache them to disk keyed by (dataset, split, #docs) so
        # subsequent launches skip the ~minutes-long scan. Everything else
        # (prefix sums, #windows) is cheap and recomputed from doc_lengths.
        self.doc_lengths = self._load_or_build_lengths(cache_path)
        self.prefix = np.concatenate(
            [np.zeros(1, dtype=np.int64), np.cumsum(self.doc_lengths, dtype=np.int64)])
        self.total_chars = int(self.prefix[-1])
        if self.total_chars <= 0:
            raise ValueError("document stream is empty")

        windows = self.total_chars / self.chars_per_window
        self.num_windows = int(math.floor(windows) if self.drop_last else math.ceil(windows))
        if self.wrap:
            self.num_windows = max(1, self.num_windows)

    def _load_or_build_lengths(self, cache_path):
        if cache_path and os.path.exists(cache_path):
            try:
                d = np.load(cache_path)
                if int(d["num_documents"]) == self.num_documents:
                    print(f"[window-index] loaded cached lengths <- {cache_path} "
                          f"({self.num_documents} docs)")
                    return d["doc_lengths"].astype(np.int64, copy=False)
                print(f"[window-index] cache num_documents mismatch -> rebuilding")
            except Exception as e:
                print(f"[window-index] cache load failed ({e}) -> rebuilding")

        lengths = np.empty(self.num_documents, dtype=np.int64)
        for idx in range(self.num_documents):
            lengths[idx] = len(self._clean_text(self.source_dataset[idx]["text"])) + 1  # +EOS

        if cache_path:
            try:
                os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
                # tmp ends in .npz so np.savez doesn't append it; rename is atomic
                # (DDP-safe: ranks race, last writer wins; contents are identical).
                tmp = f"{cache_path}.{os.getpid()}.tmp.npz"
                np.savez(tmp, doc_lengths=lengths, num_documents=self.num_documents)
                os.replace(tmp, cache_path)
                print(f"[window-index] cached lengths -> {cache_path}")
            except Exception as e:
                print(f"[window-index] cache save skipped ({e})")
        return lengths

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> dict:
        idx = int(idx)
        if idx < 0 or idx >= self.num_windows:
            raise IndexError(idx)
        return {"text": self.window_text(idx * self.chars_per_window)}

    def _clean_text(self, text) -> str:
        return str(text).replace(self.eos_char, " ")

    def _doc_text(self, doc_idx: int) -> str:
        return self._clean_text(self.source_dataset[int(doc_idx)]["text"])

    def _locate(self, stream_pos: int):
        if self.wrap:
            stream_pos %= self.total_chars
        doc_idx = bisect.bisect_right(self.prefix, stream_pos) - 1
        doc_idx = min(max(0, doc_idx), self.num_documents - 1)
        return doc_idx, int(stream_pos - self.prefix[doc_idx])

    def window_text(self, stream_pos: int) -> str:
        remaining = self.chars_per_window
        pos = int(stream_pos)
        pieces: list[str] = []
        while remaining > 0:
            if pos >= self.total_chars:
                if not self.wrap:
                    break
                pos %= self.total_chars
            doc_idx, offset = self._locate(pos)
            text = self._doc_text(doc_idx)
            text_len = len(text)
            if offset < text_len:
                take = min(remaining, text_len - offset)
                pieces.append(text[offset: offset + take])
                pos += take
                remaining -= take
            else:                       # at the document boundary -> one EOS char
                pieces.append(self.eos_char)
                pos += 1
                remaining -= 1
        return "".join(pieces)


# ---------------------------------------------------------------------------
# Collate: render window text -> glyph strip + per-cell OCR labels
# ---------------------------------------------------------------------------
def make_glyph_collate_with_labels(vocab: GlyphAtlasVocab, max_chars: int,
                                   eos_char: str = EOS_CHAR):
    """collate -> {pixel_values (B,1,H,max_chars*W) float32 [0,1] white bg,
                   char_indices (B,max_chars) long, IGNORE_INDEX at empty cells}.

    Fully on-the-fly: gathers glyph cells straight from the in-memory atlas; no
    rendered image is ever cached to disk.
    """
    atlas_t = torch.from_numpy(np.ascontiguousarray(vocab.atlas))  # (V,H,W)
    H, W = vocab.char_h, vocab.char_w

    def _collate(batch):
        B = len(batch)
        out = torch.ones(B, 1, H, max_chars * W, dtype=torch.float32)        # white bg
        labels = torch.full((B, max_chars), IGNORE_INDEX, dtype=torch.long)
        for bi, item in enumerate(batch):
            idxs = vocab.encode_window_indices(item["text"], max_chars, eos_char)
            n = len(idxs)
            if n:
                t = torch.as_tensor(idxs, dtype=torch.long)
                cells = atlas_t[t]                                            # (n,H,W)
                out[bi, 0, :, : n * W] = cells.permute(1, 0, 2).reshape(H, n * W)
                labels[bi, :n] = t
        return {"pixel_values": out, "char_indices": labels}

    return _collate
