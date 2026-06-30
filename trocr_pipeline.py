"""
trocr_pipeline.py — Page-level handwriting recognition via TrOCR (zero-shot)

WHY THIS EXISTS
    The CRNN path in ml_pipeline.py is stalled by a data-format reality (see the
    2026-06-30 finding in CLAUDE.md): the volunteer transcripts are page-level
    PROSE — one wrapped paragraph per diary entry — not one line per physical
    handwritten line. So the "Nth crop ↔ Nth transcript line" alignment a
    line-level CTC model needs cannot be built from this data.

    This pipeline avoids the alignment problem entirely by using a PRE-TRAINED
    transformer recogniser (Microsoft's TrOCR, trained on handwriting) zero-shot:

        page image
          └─ segment_page_to_lines()         # detect physical line crops (OpenCV)
               └─ TrOCR per line crop         # pretrained image→text, no training
                    └─ join with "\n"         # page-level prediction
                         └─ CER / WER vs the page transcript (jiwer)

    No training and no per-line labels are required, because TrOCR already knows
    how to read a line. We only ever score at the PAGE level, exactly like the
    Claude vision baseline (ai-transcription-pipeline.py), so the three
    approaches — CRNN, TrOCR, Claude — stay directly comparable.

    TrOCR is a LINE recogniser, so we still segment first; but segmentation
    errors now only cost recognition accuracy, they no longer corrupt training
    labels (there is no training). This is the key advantage over the CRNN path.

DEPENDENCIES (heavy; imported lazily so the module stays importable without them)
    pip install torch transformers pillow jiwer
    plus opencv-python + numpy (already used by segmentation.py)
    The first run downloads the model weights (~1.3 GB for trocr-base-handwritten).

USAGE
    python trocr_pipeline.py --sample 3                 # quick smoke test
    python trocr_pipeline.py --sample 50 --random --seed 0   # benchmark CER/WER
    python trocr_pipeline.py --model microsoft/trocr-small-handwritten  # faster
"""

# Standard library: CLI args, CSV parsing, randomness, paths, type hints.
import argparse
import csv
import random
from pathlib import Path
from typing import List, Tuple

# Page→line segmentation is shared with the CRNN path; reusing it keeps the two
# pipelines comparable on identical line crops.
from segmentation import segment_page_to_lines

# NOTE: torch / transformers / PIL / jiwer are imported lazily inside the
# functions that use them, so importing this module (e.g. to inspect structure)
# never requires the multi-GB ML stack to be installed.


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Canonical data locations, mirroring ml_pipeline.py so both pipelines agree.
DATA_DIR = Path("data")
PAGES_DIR = DATA_DIR / "pages"
TRANSCRIPT_DIR = DATA_DIR / "transcript"
PAIRS_CSV = DATA_DIR / "pairs.csv"

# Default pretrained checkpoint: TrOCR fine-tuned on handwriting. The "base"
# model is the accuracy/speed sweet spot; "small" is ~4x faster for smoke tests.
DEFAULT_MODEL = "microsoft/trocr-base-handwritten"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_pairs() -> List[Tuple[str, Path, str]]:
    """Read data/pairs.csv → list of (id, page_image_path, transcript_text).

    Identical contract to ml_pipeline.load_pairs(): resolves on-disk paths and
    skips rows whose (gitignored, possibly partial) files are missing locally.
    """
    pairs: List[Tuple[str, Path, str]] = []

    # Stream the index CSV keyed on the documented column headers.
    with PAIRS_CSV.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # Resolve the image and transcript locations for this page.
            image_path = PAGES_DIR / row["page_image_name"]
            txt_path = TRANSCRIPT_DIR / row["page_txt_name"]

            # Skip pages whose raw files are not present on this machine.
            if not image_path.exists() or not txt_path.exists():
                continue

            # The volunteer transcription is the page-level ground truth.
            pairs.append((row["id"], image_path, txt_path.read_text(encoding="utf-8")))

    return pairs


def normalise_transcript(text: str) -> str:
    """Strip editorial scaffolding so CER/WER score the HANDWRITING, not markup.

    Volunteer transcripts carry bracketed editorial notes that are not on the
    page as handwriting — e.g. "[Page 15]" headers or
    "[The following text is written along the left-hand margin]". Recognition
    cannot (and should not) reproduce these, so we drop bracketed segments and
    collapse whitespace before scoring. Applied identically to prediction and
    reference would be ideal, but predictions contain no brackets, so we only
    need it on the reference side.
    """
    import re

    # Remove any "[...]" editorial markers (non-greedy, across the whole text).
    no_brackets = re.sub(r"\[[^\]]*\]", " ", text)

    # Collapse runs of whitespace/newlines to single spaces for a stable compare.
    return re.sub(r"\s+", " ", no_brackets).strip()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def load_trocr(model_name: str):
    """Load the TrOCR processor + model once and move it to the best device.

    Returns (processor, model, device). Kept separate from recognition so the
    weights load a single time and are reused across every page/line.
    """
    # Heavy imports deferred until we actually run the model.
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    # Prefer CUDA if available; CPU works but is markedly slower per line.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # The processor handles image normalisation + tokenisation; the model is the
    # ViT encoder + autoregressive text decoder.
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
    model.eval()  # inference only — no dropout/gradient bookkeeping

    return processor, model, device


def recognise_lines(line_images: List["object"], processor, model, device) -> List[str]:
    """Run TrOCR on a batch of line crops and return the decoded text per line.

    line_images are greyscale numpy arrays from segment_page_to_lines(); TrOCR
    expects 3-channel RGB PIL images, so each crop is converted first.
    """
    # No lines detected on the page → nothing to recognise.
    if not line_images:
        return []

    # Heavy imports deferred to call time.
    import torch
    from PIL import Image

    # Convert each greyscale crop to an RGB PIL image (TrOCR's expected input).
    pil_lines = [Image.fromarray(crop).convert("RGB") for crop in line_images]

    # Batch all line crops through the processor into pixel tensors on-device.
    pixel_values = processor(images=pil_lines, return_tensors="pt").pixel_values.to(device)

    # Greedy decode each line; no_grad keeps memory/compute down at inference.
    # max_new_tokens is set explicitly because the model's default cap (21
    # tokens) silently truncates longer handwritten lines mid-sentence.
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_new_tokens=64)

    # Decode token ids back to strings, dropping special tokens.
    return processor.batch_decode(generated_ids, skip_special_tokens=True)


def predict_page(page_image_path: Path, processor, model, device) -> str:
    """Full page → predicted transcription, joining per-line TrOCR outputs.

    Mirrors ml_pipeline.predict_page()'s contract (lines joined with newlines)
    so page-level CER/WER are comparable across pipelines.
    """
    # Stage 0: detect ordered line crops on the page (shared OpenCV segmenter).
    crops = segment_page_to_lines(page_image_path)

    # Recognise each line, then rejoin in reading order as the page prediction.
    return "\n".join(recognise_lines(crops, processor, model, device))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def main():
    # Sampling / model knobs so the same script does smoke tests and benchmarks.
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=3, help="number of pages to transcribe")
    ap.add_argument("--random", action="store_true", help="random sample vs. head of CSV")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible sampling")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF TrOCR checkpoint to use")
    args = ap.parse_args()

    # jiwer provides the standard CER/WER metrics used by the Claude baseline too.
    from jiwer import cer, wer

    # Build the page sample: seeded-random or the first N indexed pages.
    pairs = load_pairs()
    if args.random:
        random.seed(args.seed)
        pairs = random.sample(pairs, min(args.sample, len(pairs)))
    else:
        pairs = pairs[: args.sample]

    # Load the model once up front (this triggers the weight download on run 1).
    print(f"Loading {args.model} ...")
    processor, model, device = load_trocr(args.model)
    print(f"Running on {device} over {len(pairs)} pages\n")

    # Accumulate per-page scores to report a mean at the end.
    cer_scores: List[float] = []
    wer_scores: List[float] = []

    for page_id, image_path, transcript in pairs:
        # Predict the page, then normalise the reference transcript for scoring.
        prediction = predict_page(image_path, processor, model, device)
        reference = normalise_transcript(transcript)
        pred_norm = " ".join(prediction.split())  # collapse whitespace to match

        # Guard against an empty reference (all-editorial page) which makes
        # CER/WER undefined; skip rather than divide by zero.
        if not reference:
            print(f"page {page_id:>5}  (empty reference after normalising — skipped)")
            continue

        # Score this page and stash the metrics.
        page_cer = cer(reference, pred_norm)
        page_wer = wer(reference, pred_norm)
        cer_scores.append(page_cer)
        wer_scores.append(page_wer)
        print(f"page {page_id:>5}  CER {page_cer:6.2%}  WER {page_wer:6.2%}")

    # Report the mean CER/WER — the headline comparison vs CRNN and Claude.
    if cer_scores:
        mean_cer = sum(cer_scores) / len(cer_scores)
        mean_wer = sum(wer_scores) / len(wer_scores)
        print(f"\nMEAN over {len(cer_scores)} pages:  CER {mean_cer:.2%}  WER {mean_wer:.2%}")
    else:
        print("\nNo pages scored.")


if __name__ == "__main__":
    main()
