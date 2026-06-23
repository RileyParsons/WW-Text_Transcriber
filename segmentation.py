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
from typing import List, Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Path to the Kraken baseline segmentation model. Kraken ships a pretrained
# default ("blla.mlmodel") inside the installed package; leave this as None to
# use it, or point it at a model fine-tuned on our own diary pages (e.g. trained
# via eScriptorium) to override.
SEG_MODEL_PATH: Optional[Path] = None

# Process-wide cache for the loaded segmentation model. Loading the VGSL/torch
# model is expensive, and segmentation runs over thousands of pages, so we load
# it once and reuse it rather than reloading per page.
_SEG_MODEL = None


def _load_seg_model():
    """Load (and memoise) the Kraken baseline segmentation model.

    Uses SEG_MODEL_PATH when set, otherwise the default 'blla.mlmodel' bundled
    with the kraken package. The loaded model is cached so subsequent calls are
    free.
    """
    # Return the already-loaded model on every call after the first.
    global _SEG_MODEL
    if _SEG_MODEL is not None:
        return _SEG_MODEL

    # Imported lazily so importing this module never requires kraken installed.
    from kraken.lib import vgsl

    # Resolve the model path: explicit override, else kraken's packaged default.
    if SEG_MODEL_PATH is not None:
        model_path = str(SEG_MODEL_PATH)
    else:
        # The default baseline model ships as a data file inside the kraken pkg.
        from importlib.resources import files
        model_path = str(files("kraken") / "blla.mlmodel")

    # Load the VGSL-defined torch model once and cache it for reuse.
    _SEG_MODEL = vgsl.TorchVGSLModel.load_model(model_path)
    return _SEG_MODEL


def _boundary_top_y(record) -> float:
    """Return the top-most y-coordinate of a segmented line's boundary polygon.

    Used as a sort key to enforce strict top-to-bottom ordering. Handles both
    the kraken 4.x dict line records ({'boundary': [[x, y], ...]}) and the 5.x
    BaselineLine objects (record.boundary); falls back to +inf when no boundary
    is present so degenerate lines sort last rather than crash.
    """
    # Read the boundary polygon from either a dict record or an object record.
    if isinstance(record, dict):
        boundary = record.get("boundary")
    else:
        boundary = getattr(record, "boundary", None)

    # No usable boundary -> push this line to the end of the ordering.
    if not boundary:
        return float("inf")

    # The polygon is a list of [x, y] points; the line's top is the smallest y.
    return min(point[1] for point in boundary)


def segment_page_to_lines(page_image_path: Path) -> List["object"]:
    """Detect text lines on a full page and return them top-to-bottom.

    Returns a list of cropped line images (numpy arrays), ordered by vertical
    position so the ordering can be matched against transcript lines.

    The ordering of returned crops MUST be stable and in reading order (top to
    bottom) because build_line_label_pairs() in ml_pipeline.py aligns crops to
    transcript lines purely by index — a misordered crop silently mislabels a
    training example.
    """
    # Lazy heavy imports (kraken pulls in torch); keeps module import cheap so
    # ml_pipeline.py can import this module without the full vision stack.
    import numpy as np
    from PIL import Image
    from kraken import blla
    from kraken.lib.segmentation import extract_polygons

    # Load the page as a PIL image. blla expects PIL input; RGB is the safe mode.
    im = Image.open(page_image_path).convert("RGB")

    # Run baseline segmentation. The result's `lines` each carry a `baseline`
    # polyline and a `boundary` polygon, ordered by kraken's reading-order
    # heuristic (polygonal_reading_order).
    seg = blla.segment(im, model=_load_seg_model())

    # extract_polygons crops each line's boundary polygon AND dewarps it to a
    # horizontal strip, yielding (PIL line image, line record) in seg order.
    # We keep the record alongside each crop so we can sort by vertical position.
    crops_with_records = [
        (np.asarray(line_img), record)
        for line_img, record in extract_polygons(im, seg)
    ]

    # Guard the ordering contract: re-sort strictly top-to-bottom by each line's
    # boundary top-y. extract_polygons already preserves seg order, but this
    # protects the index-based alignment in build_line_label_pairs() on noisy or
    # multi-column pages where the reading-order heuristic may diverge.
    crops_with_records.sort(key=lambda pair: _boundary_top_y(pair[1]))

    # Return just the ordered crops (drop the records now ordering is applied).
    return [crop for crop, _record in crops_with_records]
