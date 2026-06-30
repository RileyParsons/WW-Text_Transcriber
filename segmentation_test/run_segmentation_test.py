"""
run_segmentation_test.py — Benchmark segment_page_to_lines() on real pages

Samples pages from data/pairs.csv, runs the OpenCV projection-profile segmenter
on each, and reports — per page and in aggregate — how its detected line count
compares to the transcript's non-empty line count. This matters because
build_line_label_pairs() aligns the Nth crop to the Nth transcript line purely by
index and DROPS pages whose counts diverge, so the segmenter's count accuracy
directly sets how much of the 5008-page dataset survives into training.

Usage:
    python run_segmentation_test.py                      # default: 5 pages, head, save crops
    python run_segmentation_test.py --sample 100 --random --seed 0   # keep-rate benchmark
    python run_segmentation_test.py --sample 100 --random --no-crops # skip writing PNGs

By default crops are written under page_<id>/ for visual inspection; pass
--no-crops for large samples where only the aggregate keep-rate matters.
"""

# Standard library: CLI args, CSV parsing, randomness, paths.
import argparse
import csv
import random
import sys
from pathlib import Path

# Pull in the segmenter from the project root (this script lives one level down
# in segmentation_test/, so add the parent dir to the import path).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import cv2  # noqa: E402  (imported after sys.path tweak)
from segmentation import segment_page_to_lines  # noqa: E402

# Where the source pages and the index live, and where we write results.
PAGES_DIR = ROOT / "data" / "pages"
TRANSCRIPT_DIR = ROOT / "data" / "transcript"
PAIRS_CSV = ROOT / "data" / "pairs.csv"
OUT_DIR = Path(__file__).resolve().parent


def load_rows():
    """Return every (id, image, transcript) row from pairs.csv in order."""
    with open(PAIRS_CSV, newline="", encoding="utf-8") as f:
        return [
            (row["id"], row["page_image_name"], row["page_txt_name"])
            for row in csv.DictReader(f)
        ]


def count_text_lines(txt_name: str) -> int:
    """Count non-empty transcript lines — the same notion build_line_label_pairs()
    aligns crops against, so the comparison mirrors the real pipeline."""
    text = (TRANSCRIPT_DIR / txt_name).read_text(encoding="utf-8")
    return sum(1 for ln in text.splitlines() if ln.strip())


def classify(n_crops: int, n_text: int) -> str:
    """Bucket a page by how its crop count relates to its transcript line count,
    mirroring build_line_label_pairs()'s keep/drop rule (within 2 lines or 25%).

    Returns one of: 'keep', 'zero' (segmenter found nothing), 'under' (too few
    crops), 'over' (too many crops). The three non-keep buckets are all DROPPED
    by the real pipeline; splitting them out shows WHY pages are lost."""
    if abs(n_crops - n_text) <= max(2, 0.25 * n_text):
        return "keep"
    if n_crops == 0:
        return "zero"
    return "under" if n_crops < n_text else "over"


def main():
    # Parse the sampling knobs so the same script does both the 5-page eyeball
    # test and the 100-page keep-rate benchmark.
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=5, help="number of pages to test")
    ap.add_argument("--random", action="store_true", help="random sample vs. head of CSV")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible sampling")
    ap.add_argument("--no-crops", dest="save_crops", action="store_false",
                    help="skip writing per-line PNGs (for large samples)")
    args = ap.parse_args()

    # Build the sample: random (seeded, reproducible) or the first N rows.
    rows = load_rows()
    if args.random:
        random.seed(args.seed)
        rows = random.sample(rows, min(args.sample, len(rows)))
    else:
        rows = rows[: args.sample]

    # Per-page report rows and a tally of outcome buckets for the aggregate.
    header = f"{'page':>5}  {'crops':>5}  {'text':>4}  {'bucket':<6}  image"
    report = [header, "-" * len(header)]
    buckets = {"keep": 0, "zero": 0, "under": 0, "over": 0}

    for page_id, image_name, txt_name in rows:
        # Run the actual segmenter under test.
        crops = segment_page_to_lines(PAGES_DIR / image_name)

        # Optionally persist each ordered crop for visual inspection, preserving
        # the top-to-bottom reading-order contract the pipeline depends on.
        if args.save_crops:
            page_out = OUT_DIR / f"page_{page_id}"
            page_out.mkdir(parents=True, exist_ok=True)
            for idx, crop in enumerate(crops):
                cv2.imwrite(str(page_out / f"line_{idx:02d}.png"), crop)

        # Compare crop count to transcript line count and bucket the outcome.
        n_text = count_text_lines(txt_name)
        bucket = classify(len(crops), n_text)
        buckets[bucket] += 1
        report.append(f"{page_id:>5}  {len(crops):>5}  {n_text:>4}  {bucket:<6}  {image_name}")

    # Build the aggregate keep-rate summary — the headline number that decides
    # whether OpenCV segmentation is viable for the dataset.
    total = len(rows)
    kept = buckets["keep"]
    agg = [
        "",
        "=" * 48,
        f"AGGREGATE over {total} pages (seed={args.seed}, random={args.random})",
        "=" * 48,
        f"  keep  : {kept:>4}  ({kept / total:6.1%})  <- trainable",
        f"  DROP  : {total - kept:>4}  ({(total - kept) / total:6.1%})",
        f"    - under-split : {buckets['under']:>4}  (too few crops)",
        f"    - over-split  : {buckets['over']:>4}  (too many crops)",
        f"    - zero lines  : {buckets['zero']:>4}  (segmenter found nothing)",
    ]
    for line in agg:
        print(line)

    # Persist the full per-page report plus the aggregate for later reference.
    out_name = f"summary_{total}pages.txt" if args.random else "summary.txt"
    (OUT_DIR / out_name).write_text("\n".join(report + agg) + "\n", encoding="utf-8")
    print(f"\nWrote {out_name} under {OUT_DIR}")


if __name__ == "__main__":
    main()
