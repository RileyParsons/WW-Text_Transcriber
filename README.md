# WW-Text-Transcriber

An exploration into AI-driven development for automated text transcription of historical handwritten documents.

## Purpose

This project investigates and compares two approaches to transcribing WW1-era handwritten diary pages, using existing volunteer transcriptions from the NSW State Library as labelled ground truth. It serves as a practical case study in both traditional machine learning and AI-assisted digitisation of historical records.

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

## Approach

Two transcription pipelines are developed and benchmarked against each other:

### Primary — CNN Pipeline (`ml_pipeline.py`)
A traditional Convolutional Neural Network trained on the NSW State Library paired data.

1. **Scrape** — Download paired scanned page images and volunteer transcription text files from the NSW State Library
2. **Preprocess** — Normalise and prepare images for model input
3. **Train** — Train a CRNN model (CNN + BiLSTM + CTC loss) on the paired data
4. **Evaluate** — Measure accuracy against volunteer ground truth using character error rate (CER) and word error rate (WER)

### Secondary — Claude Vision API Pipeline (`ai-transcription-pipeline.py`)
A zero-shot approach using the Claude vision API — no model training required.

1. **Transcribe** — Pass page images directly to the Claude vision API
2. **Evaluate** — Measure accuracy against volunteer ground truth (CER, WER)
3. **Compare** — Benchmark results against the CNN pipeline

## Project Structure

```
WW-Text-Transcriber/
├── scraper.py                    # Downloads page images and transcriptions from NSW State Library
├── ml_pipeline.py                # Primary pipeline: CRNN (CNN + BiLSTM + CTC) trained on paired data
├── ai-transcription-pipeline.py  # Secondary pipeline: zero-shot transcription via Claude vision API
├── data/
│   ├── pages/                    # Downloaded scanned diary page images
│   ├── transcript/               # Downloaded volunteer transcription text files
│   └── pairs.csv                 # Index linking each image to its transcription
└── models/                       # Saved trained CNN model weights
```

## Status

Exploratory — in active development.
