# CLAUDE.md — WW-Text-Transcriber

## Project Context

This project is an exploration into AI-driven development for text transcription of historical handwritten documents, targeting WW1 diary pages from the NSW State Library.

Two parallel transcription pipelines are maintained and benchmarked against each other:
- **Primary pipeline** (`ml_pipeline.py`): Traditional CNN-based handwritten text recognition (CRNN architecture — CNN + BiLSTM + CTC loss), trained on the NSW State Library paired data
- **Secondary pipeline** (`ai-transcription-pipeline.py`): Zero-shot transcription using the Claude vision API — no training required, used for comparison and benchmarking

## Data Source

**NSW State Library — WW1 Diaries Transcription Project**
URL: https://transcripts.sl.nsw.gov.au/section/world-war-1-diaries

The site hosts:
- Scanned images of handwritten WW1 diary pages
- Volunteer-created transcription text files (one per page)

Before scraping, always verify compliance with the site's `robots.txt` and Terms of Use. Prefer requesting a data dump from the State Library directly for bulk access.

## Data Structure

All sourced data lives under `data/`:

| Path | Contents |
|------|----------|
| `data/pages/` | Downloaded scanned page images (e.g. `page_001.jpg`) |
| `data/transcript/` | Downloaded volunteer transcription text files (e.g. `page_001.txt`) |
| `data/pairs.csv` | Index CSV linking each image to its transcription |

### pairs.csv columns
| Column | Description |
|--------|-------------|
| `id` | Unique identifier for the page pair |
| `page_image_name` | Filename of the scanned image in `data/pages/` |
| `page_txt_name` | Filename of the transcription text file in `data/transcript/` |
| `download_source` | URL the files were sourced from |

Raw data files (`pages/` and `transcript/`) are gitignored due to size. Only `pairs.csv` is tracked as it is the index.

## Data Collection

`scraper.py` gathers the training data from the NSW State Library archive.

**Target dataset:** ~5000 transcribed pages, randomly sampled for **maximum
handwriting diversity** — spread across as many of the ~959 diaries as possible
(~5–6 pages each, giving ~959 distinct hands). Diversity is favoured because the
CNN pipeline generalises better when trained on many different writing styles.

**Primary command:**
```bash
python scraper.py --sample 5000
```

**CLI options:**
| Flag | Description |
|------|-------------|
| `--sample N` | Randomly sample `N` transcribed pages across all diaries (max diversity) |
| `--max-diaries M` | Cap the number of diaries used (scoping / fast testing) |
| `--diary URL` | Scrape a single diary document |
| `--limit N` | Stop after processing `N` pages |

**Behaviour / constraints:**
- **Compliance:** checks `robots.txt` before every request and enforces its
  mandatory **10-second crawl delay**
- **Skip logic:** only saves pages whose transcription body has text and does
  **not** contain "not transcribed" — this is checked on the page's own
  `field-name-body` element, not the navigation (which lists every page's status)
- **Outlier filtering:** pages whose transcription is only an editorial marker —
  diary covers (`[Cover]`), blank pages (`[blank page]`), repeats
  (`[Transcribed on previous page]`, `[Duplicate of page N]`) or `#N/A` — carry no
  handwriting and are skipped via `is_outlier_transcription()`. The same function
  backs `clean_dataset.py`, which removes such outliers from an existing dataset
  (moving them to `data/outliers/`, gitignored) and re-numbers `pairs.csv`
- **Runtime:** the crawl delay makes a full 5000-page sample take **~30 hours**
- **Resume:** re-running skips pages already listed in `data/pairs.csv`, so runs
  can be interrupted (`Ctrl + C`) and restarted freely
- **Logging:** all progress, warnings, and errors are written to `scraper.log`
  (gitignored)

## Key Files

| File | Purpose |
|------|---------|
| `scraper.py` | Downloads page images and volunteer transcriptions from NSW State Library |
| `segmentation.py` | Page→line segmentation (`segment_page_to_lines`) — detects text lines on a page and returns ordered crops; consumed by `ml_pipeline.py` |
| `ml_pipeline.py` | Primary pipeline — trains a CRNN (CNN + BiLSTM + CTC) model on paired data; evaluates with CER/WER |
| `ai-transcription-pipeline.py` | Secondary pipeline — zero-shot transcription via Claude vision API; benchmarks against CNN results |
| `models/` | Saved trained CNN model weights (gitignored due to size) |

## ML Pipeline Architecture (`ml_pipeline.py`)

The CRNN (CNN + BiLSTM + CTC) recognises **one text line at a time**, but the
labels in `data/pairs.csv` are **page-level** transcriptions. The pipeline
therefore bridges page → line before recognition:

- **Stage 0 — Segment:** `segment_page_to_lines()` (in `segmentation.py`,
  imported by `ml_pipeline.py`) detects text lines on a page and returns ordered
  (top-to-bottom) line crops. **Implemented with classic OpenCV** (adaptive
  binarise → horizontal projection profile → line bands → crop): only
  opencv-python + numpy, no torch/weights, installs natively on Windows. Tunable
  via module constants (`BINARIZE_*`, `ROW_INK_THRESHOLD_FRAC`, `MIN_LINE_HEIGHT`,
  `LINE_PAD`). Brittle on heavily slanted/touching hands; a learned detector
  (Kraken `blla` / docTR) is the higher-accuracy fallback if needed.
- **Label alignment:** `build_line_label_pairs()` maps the *Nth crop to the Nth
  non-empty transcript line by order* (we don't know which text belongs to which
  line). Pages where line/text counts diverge badly are **dropped** rather than
  trained on scrambled labels. This order-alignment is the **single biggest
  source of label noise** in the pipeline.
- **Train/eval boundary:** split train/val **by page** (never leak lines from the
  same page across the split). Recognition is line-level; `predict_page()` joins
  per-line predictions back with newlines, and CER/WER are scored **at page
  level** so results stay comparable to the Claude vision baseline.

Candidate libraries: **PyTorch** (CRNN + `torch.nn.CTCLoss`), **OpenCV** (line
preprocessing), **Albumentations** (augmentation across the ~959 hands),
**jiwer** (CER/WER), optional **Kraken/docTR** (segmentation) or **TrOCR** (a
transformer alternative with a higher accuracy ceiling).

> Current state: `ml_pipeline.py` is an architecture **scaffold** — data-flow
> wiring (`load_pairs`, `build_charset`) is real; segmentation, the CRNN model,
> training, inference and evaluation are `NotImplementedError`/`TODO` stubs.

### Finding (2026-06-30): the transcripts are NOT line-aligned

A 100-page random benchmark of the OpenCV segmenter
(`segmentation_test/run_segmentation_test.py`, seed 0) gave only a **16%
keep-rate** under `build_line_label_pairs()`'s count rule — 68% of pages were
"over-split" (many more line crops than transcript lines). Inspecting those
transcripts showed the cause is **the label format, not the segmenter**: the
volunteer transcripts are **page-level prose** — one soft-wrapped paragraph per
diary entry/date — **not one line per physical handwritten line**. E.g. a page
with ~20 lines of handwriting has a 3-line transcript (3 paragraphs).

**Consequence:** the *Nth crop ↔ Nth transcript line* alignment that the CRNN
path depends on is **broken by construction**, not merely noisy. No segmenter
(OpenCV, Kraken, docTR) can raise the keep-rate, because there are no
per-physical-line labels to align crops to. The line-level CRNN design is
therefore stalled pending a label strategy that fits page-level prose.

**Chosen direction:** move to a **page-level model — TrOCR** (image of a
page/region → full text), which learns reading order itself and needs **no line
alignment**, fitting the prose labels directly. The CRNN + segmentation code is
retained for benchmarking but is no longer the primary path.

## GitHub Repository

- **Repo name:** `WW-Text_Transcriber`
- **Visibility:** Private
- **Remote:** `origin` → `https://github.com/RileyParsons/WW-Text_Transcriber.git`
- **Default branch:** `main`

> **Note on local path:** the project may be moved out of OneDrive (e.g. to
> `C:\Users\rrpar\Projects\WW-Text-Transcriber`) so the ~3 GB of downloaded
> training images are not synced to the cloud. Moving the whole folder (including
> `.git`) preserves the GitHub connection and history — only the local path
> changes. Do not assume the project lives under OneDrive.

## Workflow Rules

1. Every code change must be committed with a meaningful message describing *why* the change was made
2. Push to `origin main` immediately after every commit
3. All Python code must include inline comments explaining each code block

## Coding Conventions

- All Python files use inline comments to explain every code block
- Comments describe intent and reasoning, not just what the line does
- Comply with terms of use of any external data source before scraping
