"""ELF pixel-vocab variant: flow-matching over glyph-strip pixels.

See `pixel/README.md` for the design. Public surface:
  GlyphAtlasVocab, build_index_to_char, indices_to_text  (pixel.glyph_vocab)
  load_lm1b, ContinuousDocumentStreamDataset, make_glyph_collate_with_labels  (pixel.glyph_dataset)
  ELFPixel, ELFPixel_models, patchify, unpatchify  (pixel.model)
  ELFPixelLitModule, PixelGlyphDataModule  (pixel.lit_module)
  PixelGenEvalCallback, generate_pixel_samples  (pixel.eval)
"""

from pixel.glyph_vocab import (
    EOS_CHAR, IGNORE_INDEX, GlyphAtlasVocab, build_index_to_char, indices_to_text,
)
from pixel.model import ELFPixel, ELFPixel_models, patchify, unpatchify

__all__ = [
    "EOS_CHAR", "IGNORE_INDEX", "GlyphAtlasVocab", "build_index_to_char",
    "indices_to_text", "ELFPixel", "ELFPixel_models", "patchify", "unpatchify",
]
