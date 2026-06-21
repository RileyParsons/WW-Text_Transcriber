"""
clean_dataset.py — remove outlier pages from the WW1 diary dataset

Some scraped pages are not usable training data: diary covers, blank pages,
pages whose text was transcribed elsewhere ("repeats"), and spreadsheet error
values ("#N/A"). Their transcription is only an editorial marker, so the image
and text do not correspond. This script finds those pages in pairs.csv and moves
them out of the dataset.

It reuses scraper.is_outlier_transcription so the cleanup rule is identical to
the skip rule the scraper now applies to future downloads — one rule, one place.

Outliers are MOVED to data/outliers/ (not deleted) so the action is reversible;
the surviving pairs.csv rows are then re-numbered 1..N so the id column stays
gap-free and scraper.next_row_id() keeps allocating correct ids on the next run.

USAGE:
    python clean_dataset.py            # dry run — list outliers, change nothing
    python clean_dataset.py --apply    # quarantine outliers and rewrite pairs.csv
"""

import argparse
import csv
import shutil
from pathlib import Path

# Reuse the scraper's outlier rule so cleanup and scraping never diverge
from scraper import is_outlier_transcription, PAGES_DIR, TRANS_DIR, PAIRS_CSV

# Quarantine destinations — outliers are moved here rather than deleted so the
# operation can be undone simply by moving the files back and re-scraping
OUT_DIR        = Path("data/outliers")
OUT_PAGES_DIR  = OUT_DIR / "pages"
OUT_TRANS_DIR  = OUT_DIR / "transcript"
OUT_CSV        = OUT_DIR / "outliers.csv"


def find_outliers(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Split pairs.csv rows into (keep, outliers) using the shared outlier rule.
    A missing transcript file is treated as keep here (nothing to inspect),
    leaving such rows untouched for manual review.
    """
    keep, outliers = [], []
    for row in rows:
        txt_path = TRANS_DIR / row["page_txt_name"]
        # Only inspect rows whose transcript exists; classify by its content
        if txt_path.exists() and is_outlier_transcription(
            txt_path.read_text(encoding="utf-8")
        ):
            outliers.append(row)
        else:
            keep.append(row)
    return keep, outliers


def quarantine(outliers: list[dict]) -> None:
    """Move each outlier's image and transcript into the quarantine folders."""
    # Make sure the quarantine folders exist before moving anything into them
    OUT_PAGES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TRANS_DIR.mkdir(parents=True, exist_ok=True)

    for row in outliers:
        # Move the scanned image out of the active dataset if it is still present
        img = PAGES_DIR / row["page_image_name"]
        if img.exists():
            shutil.move(str(img), str(OUT_PAGES_DIR / row["page_image_name"]))
        # Move the transcription text out of the active dataset if still present
        txt = TRANS_DIR / row["page_txt_name"]
        if txt.exists():
            shutil.move(str(txt), str(OUT_TRANS_DIR / row["page_txt_name"]))

    # Record exactly which pairs were quarantined (append, preserving any prior run)
    write_header = not OUT_CSV.exists()
    with OUT_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "page_image_name",
                                               "page_txt_name", "download_source"])
        if write_header:
            writer.writeheader()
        writer.writerows(outliers)


def rewrite_pairs(keep: list[dict]) -> None:
    """Rewrite pairs.csv with only the surviving rows, re-numbered 1..N."""
    with PAIRS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "page_image_name",
                                               "page_txt_name", "download_source"])
        writer.writeheader()
        # Re-number sequentially so ids stay gap-free and unique after removal
        for new_id, row in enumerate(keep, start=1):
            row["id"] = new_id
            writer.writerow(row)


def main() -> None:
    """Parse args, report the outliers, and (with --apply) quarantine them."""
    parser = argparse.ArgumentParser(description="Remove outlier pages from the dataset.")
    # Default is a safe dry run; --apply is required to actually move files
    parser.add_argument("--apply", action="store_true",
                        help="Quarantine outliers and rewrite pairs.csv (default: dry run).")
    args = parser.parse_args()

    # Load the current index of image/transcript pairs
    with PAIRS_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Classify every pair as keep or outlier using the shared rule
    keep, outliers = find_outliers(rows)

    # Report what was found before changing anything
    print(f"Total pairs:    {len(rows)}")
    print(f"Outliers found: {len(outliers)}")
    print(f"Will remain:    {len(keep)}\n")
    for row in outliers:
        # Show the first line of each outlier's transcript so the reason is visible
        txt_path = TRANS_DIR / row["page_txt_name"]
        first = ""
        if txt_path.exists():
            first = txt_path.read_text(encoding="utf-8").strip().replace("\n", " / ")
        print(f"  id={row['id']:>4}  {first[:45]:<45}  {row['page_txt_name']}")

    # In dry-run mode, stop here without touching any files
    if not args.apply:
        print("\nDry run — nothing changed. Re-run with --apply to quarantine these.")
        return

    # Apply: move outliers to quarantine and rewrite the index
    quarantine(outliers)
    rewrite_pairs(keep)
    print(f"\nMoved {len(outliers)} outlier pairs to {OUT_DIR}/ and re-numbered pairs.csv.")


if __name__ == "__main__":
    main()
