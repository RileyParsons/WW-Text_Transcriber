"""
segmentation.py — Page-to-line image segmentation for the CNN pipeline

The CRNN recogniser in ml_pipeline.py reads ONE text line at a time, but the
source data (data/pages/) is PAGE-level scans of WW1 diary pages. This module
bridges that gap: it detects the individual handwritten text lines on a full
page and returns them as ordered (top-to-bottom) image crops, ready to be
paired with the corresponding transcript lines.

Backend: classic OpenCV — binarise -> horizontal projection profile -> line
bands -> crop. Chosen for simplicity and portability: it needs only
opencv-python + numpy (no torch, no model weights, installs natively on
Windows), is fast, and is fully inspectable/deterministic. The trade-off is
that projection profiles assume roughly horizontal, separated lines; heavily
slanted or touching hands can merge or mis-split, which feeds label noise into
the index-based alignment in build_line_label_pairs(). A learned detector
(Kraken `blla` / docTR) remains the higher-accuracy alternative if this proves
too brittle across the ~959 hands.

Why this lives in its own module:
    Line segmentation is the single most error-prone stage of the pipeline —
    build_line_label_pairs() aligns the Nth crop to the Nth transcript line
    purely by index, so any over/under-splitting here cascades into label
    noise. Isolating it makes the segmentation approach easy to swap and
    benchmark without touching the model or training code.

Whichever backend is used, the ordering contract MUST hold: crops are returned
top-to-bottom in reading order, because build_line_label_pairs() matches them
to transcript lines purely by index.
"""

# Standard library imports for filesystem paths and type hints. The vision deps
# (cv2, numpy) are imported lazily inside segment_page_to_lines() so this module
# stays importable for inspecting the pipeline structure without them installed.
from pathlib import Path
from typing import List


# ---------------------------------------------------------------------------
# Configuration (tuning levers for the projection-profile segmenter)
# ---------------------------------------------------------------------------

# Adaptive-threshold neighbourhood size (must be odd) and subtracted constant.
# Adaptive thresholding (vs. a single global Otsu cut) copes with the uneven
# lighting, foxing and discolouration typical of century-old scanned pages.
BINARIZE_BLOCK_SIZE = 35
BINARIZE_C = 15

# A row counts as "text" when its ink-pixel count exceeds this fraction of the
# busiest row's ink. Scaling off the page's own max keeps the cut-off sensible
# whether the hand is sparse or dense. This separates line bands from gaps.
ROW_INK_THRESHOLD_FRAC = 0.10

# Discard detected bands shorter than this many pixels as speckle/noise rather
# than real text lines.
MIN_LINE_HEIGHT = 12

# Vertical padding (pixels) added above/below each band so tall ascenders and
# descenders are not clipped out of the crop.
LINE_PAD = 4


def _find_line_bands(profile, row_threshold: float, min_height: int):
    """Turn a horizontal ink profile into ordered (y_start, y_end) line bands.

    Walks the per-row ink counts top-to-bottom, opening a band when ink rises
    above row_threshold and closing it when ink drops back below. Bands shorter
    than min_height are dropped as noise. Returns bands in top-to-bottom order.
    """
    # Collected (y_start, y_end) bands and the start row of the band in progress.
    bands = []
    start = None

    # Scan every row's ink count, tracking the currently open band (if any).
    for y, ink in enumerate(profile):
        if ink > row_threshold and start is None:
            # Rising edge: a new text band begins at this row.
            start = y
        elif ink <= row_threshold and start is not None:
            # Falling edge: the band ends here; keep it only if tall enough.
            if y - start >= min_height:
                bands.append((start, y))
            start = None

    # Close a band still open when we reach the bottom edge of the page.
    if start is not None and len(profile) - start >= min_height:
        bands.append((start, len(profile)))

    return bands


def segment_page_to_lines(page_image_path: Path) -> List["object"]:
    """Detect text lines on a page via OpenCV projection profiling.

    Returns line crops (numpy arrays) in reading order (top to bottom),
    matching the contract build_line_label_pairs() relies on — a misordered
    crop silently mislabels a training example.
    """
    # Lazy heavy imports so importing this module never requires the vision
    # stack (ml_pipeline.py imports it at module load).
    import cv2
    import numpy as np

    # Load the page as greyscale; line segmentation works on intensity, not hue.
    img = cv2.imread(str(page_image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        # cv2.imread returns None for a missing/unreadable file (it does not
        # raise), so surface a clear error rather than crashing downstream.
        raise FileNotFoundError(f"Could not read page image: {page_image_path}")

    # Binarise to ink=255 / paper=0. Adaptive (Gaussian) thresholding handles
    # the uneven illumination and staining common in old diary scans.
    binary = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        BINARIZE_BLOCK_SIZE, BINARIZE_C,
    )

    # Horizontal projection profile: number of ink pixels per row. Peaks are
    # text lines, valleys are the gaps between them.
    profile = (binary > 0).sum(axis=1)

    # Derive this page's row threshold from its own busiest row so the cut-off
    # scales with how dense the handwriting is.
    row_threshold = profile.max() * ROW_INK_THRESHOLD_FRAC

    # Convert the profile into ordered (y_start, y_end) line bands.
    bands = _find_line_bands(profile, row_threshold, MIN_LINE_HEIGHT)

    # Crop each band from the ORIGINAL greyscale (not the binary mask) so the
    # CRNN sees real ink, padding vertically to preserve ascenders/descenders.
    height = img.shape[0]
    crops = []
    for y0, y1 in bands:
        # Clamp the padded band to the image bounds before slicing.
        top = max(0, y0 - LINE_PAD)
        bottom = min(height, y1 + LINE_PAD)
        crops.append(img[top:bottom, :])

    # Bands are produced top-to-bottom, so the list already satisfies the
    # reading-order contract; return the ordered crops.
    return crops
