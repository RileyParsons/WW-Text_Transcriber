"""
segmentation.py — Page-to-line image segmentation for the CNN pipeline

The CRNN recogniser in ml_pipeline.py reads ONE text line at a time, but the
source data (data/pages/) is PAGE-level scans of WW1 diary pages. This module
bridges that gap: it detects the individual handwritten text lines on a full
page and returns them as ordered (top-to-bottom) image crops, ready to be
paired with the corresponding transcript lines.

Why this lives in its own module:
    Line segmentation is the single most error-prone stage of the pipeline —
    build_line_label_pairs() aligns the Nth crop to the Nth transcript line
    purely by index, so any over/under-splitting here cascades into label
    noise. Isolating it makes the segmentation approach easy to swap and
    benchmark (classic CV vs. a learned detector) without touching the model
    or training code.

Implementation options (the choice is still open — see CLAUDE.md):
    * Classic CV: binarise -> horizontal projection profile / connected
      components -> merge into line bands -> crop. Fast, no extra model and no
      weights, but brittle on slanted or densely spaced handwriting (exactly
      the conditions in these diaries), which inflates alignment label noise.
    * Learned segmentation: Kraken's `blla` baseline segmenter or docTR's
      detection models. These follow each line's actual baseline/polygon rather
      than an axis-aligned box, so they handle slanted, drifting and touching
      lines far better on historical layouts. They add a dependency and ship
      their own pretrained weights (Kraken's `blla` runs zero-shot, then can be
      fine-tuned on our own pages via eScriptorium if needed).

Whichever backend is chosen, the ordering contract below MUST hold.
"""

# Standard library imports for filesystem paths and type hints. Heavy vision
# dependencies (cv2/numpy, or kraken/doctr) are imported lazily inside the
# implementation so this module stays importable for inspecting the pipeline
# structure without the full stack installed.
from pathlib import Path
from typing import List


def segment_page_to_lines(page_image_path: Path) -> List["object"]:
    """Detect text lines on a full page and return them top-to-bottom.

    Returns a list of cropped line images (numpy arrays), ordered by vertical
    position so the ordering can be matched against transcript lines.

    The ordering of returned crops MUST be stable and in reading order (top to
    bottom) because build_line_label_pairs() in ml_pipeline.py aligns crops to
    transcript lines purely by index — a misordered crop silently mislabels a
    training example.
    """
    # TODO: load page_image_path, detect line bounding boxes/baselines, crop and
    #       return them sorted by top y-coordinate.
    raise NotImplementedError("Line segmentation not yet implemented")
