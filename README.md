# WW-Text-Transcriber

An exploration into AI-driven development for automated text transcription of historical handwritten documents.

## Purpose

This project investigates and compares several approaches to transcribing WW1-era handwritten diary pages, using existing volunteer transcriptions from the NSW State Library as labelled ground truth. It serves as a practical case study in both traditional machine learning and AI-assisted digitisation of historical records.

## Data Source

Scanned diary pages and volunteer transcriptions are sourced from the **NSW State Library WW1 Diaries Transcription Project**:
> https://transcripts.sl.nsw.gov.au/section/world-war-1-diaries

The volunteer transcriptions act as labelled ground truth for training and evaluating both pipelines.

## Running the Scraper

`scraper.py` collects paired page images and transcriptions from the archive. It
respects the site's `robots.txt` (a mandatory 10-second crawl delay), only saves
pages that have actually been transcribed, logs all activity to `scraper.log`,
and **resumes automatically** — re-running skips anything already in `data/pairs.csv`.

### Setup
```bash
pip install requests beautifulsoup4
```

### Commands
```bash
# Quick smoke test — 4 pages across 2 diaries (~2 min)
python scraper.py --sample 4 --max-diaries 2

# Collect a random sample of 5000 pages across all diaries (~30 hours)
python scraper.py --sample 5000

# Scrape a single diary (testing)
python scraper.py --diary <DIARY_URL>

# Crawl the entire archive
python scraper.py
```

### Options
| Flag | Description |
|------|-------------|
| `--sample N` | Randomly sample `N` transcribed pages spread across as many diaries as possible (maximum handwriting diversity) |
| `--max-diaries M` | Use at most `M` diaries — scopes a smaller run or speeds up testing |
| `--diary URL` | Scrape only one diary document |
| `--limit N` | Stop after processing `N` pages |

> **Note on runtime:** the 10-second crawl delay makes large runs slow — a full
> 5000-page sample takes roughly 30 hours. Because the scraper resumes cleanly,
> you can stop (`Ctrl + C`) and restart at any time without losing progress.

## Segmentation vs. transcription — an important distinction

Two very different jobs are easy to confuse:

- **Segmentation** finds *where* the text is — it splits a page image into
  individual line crops. It does **not** read anything.
- **Transcription** (recognition) turns the handwriting *into text*.

The trained recognition models in this project are **transcribers, not
segmenters**: they take a page image and output text. Whether segmentation is
needed *first* depends on the model — see below.

## Approach

Three transcription approaches are developed and benchmarked against each other
using page-level character/word error rate (CER/WER), so all results are
directly comparable.

### CRNN line-recognition pipeline (`ml_pipeline.py`) — stalled
A CRNN (CNN + BiLSTM + CTC) recognises **one text line at a time**, so a page
must first be **segmented** into line crops (`segmentation.py`, classic OpenCV).
Each crop is then paired with a transcript line and the model is trained on
(line image → line text).

**Why it stalled:** the model needs *line-level* labels, but the volunteer
transcripts are **page-level prose** — one wrapped paragraph per diary entry,
**not** one line per physical handwritten line. A 100-page benchmark
(`segmentation_test/`) showed only ~16% of pages have matching line counts, so
the "Nth crop ↔ Nth transcript line" alignment the CRNN depends on is broken by
construction. No segmenter can fix this — the labels simply aren't line-aligned.

### TrOCR page-level pipeline (`trocr_pipeline.py`, `trocr_finetune.py`) — current primary
To sidestep the alignment problem, this approach uses a transformer (Microsoft's
TrOCR) at the **page level**: a whole page image → the full transcript text,
end-to-end. The model learns reading order itself, so **no segmentation and no
line alignment are required** — the page-level prose labels are used directly.

- `trocr_pipeline.py` — inference + CER/WER evaluation (works with the
  pretrained model or a fine-tuned checkpoint).
- `trocr_finetune.py` — fine-tunes TrOCR on the paired data (page image → prose).

> **Why fine-tuning is required:** zero-shot pretrained TrOCR is unusable on
> these 1914 hands (CER ~118% — it hallucinates modern English, having been
> trained on the clean modern IAM corpus). Fine-tuning on this dataset is what
> adapts it to century-old handwriting.

### Claude Vision API pipeline (`ai-transcription-pipeline.py`) — zero-shot baseline
Passes page images directly to the Claude vision API (no training) and scores
CER/WER, providing a strong baseline to benchmark the trained models against.

### Running the TrOCR pipeline
```bash
# Install the ML stack (CUDA build of torch — see note below)
pip install transformers pillow jiwer opencv-python numpy
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Fine-tune on all paired pages (~1.25 h on an 8 GB GPU with eval subset)
python trocr_finetune.py --epochs 5 --batch-size 2 --grad-accum 8 --fp16 --eval-pages 100

# Evaluate the fine-tuned checkpoint on a random sample of pages
python trocr_pipeline.py --sample 20 --random --model models/trocr-ww1
```
> `pip install torch` from PyPI installs the **CPU-only** build on Windows; the
> CUDA index URL above is required for GPU training. The best-by-val-CER
> checkpoint is saved to `models/trocr-ww1/`, with the per-epoch training curve
> in `models/trocr-ww1/metrics.csv`.

## Project Structure

```
WW-Text-Transcriber/
├── scraper.py                    # Downloads page images and transcriptions from NSW State Library
├── clean_dataset.py              # Removes cover/blank/repeat outlier pages from an existing dataset
├── segmentation.py               # Page→line segmentation (OpenCV) — used by the CRNN path
├── ml_pipeline.py                # CRNN (CNN + BiLSTM + CTC) line-recognition pipeline (stalled)
├── trocr_pipeline.py             # TrOCR page-level transcription + CER/WER evaluation
├── trocr_finetune.py             # Fine-tunes TrOCR on the paired data (page image → prose)
├── ai-transcription-pipeline.py  # Zero-shot transcription via Claude vision API (baseline)
├── segmentation_test/            # Segmenter benchmark + the 16% keep-rate finding
├── data/
│   ├── pages/                    # Downloaded scanned diary page images
│   ├── transcript/               # Downloaded volunteer transcription text files
│   └── pairs.csv                 # Index linking each image to its transcription
└── models/                       # Saved trained model weights (e.g. trocr-ww1/)
```

## Status

Exploratory — in active development. Current focus: fine-tuning the page-level
TrOCR model after the CRNN path was blocked by the page-level (non-line-aligned)
transcript format.
