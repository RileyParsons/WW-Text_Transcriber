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

## Key Files

| File | Purpose |
|------|---------|
| `scraper.py` | Downloads page images and volunteer transcriptions from NSW State Library |
| `ml_pipeline.py` | Primary pipeline — trains a CRNN (CNN + BiLSTM + CTC) model on paired data; evaluates with CER/WER |
| `ai-transcription-pipeline.py` | Secondary pipeline — zero-shot transcription via Claude vision API; benchmarks against CNN results |
| `models/` | Saved trained CNN model weights (gitignored due to size) |

## GitHub Repository

- **Repo name:** `WW-Text-Transcriber`
- **Visibility:** Private
- **Remote:** `origin` → `https://github.com/<user>/WW-Text-Transcriber.git`
- **Default branch:** `main`

## Workflow Rules

1. Every code change must be committed with a meaningful message describing *why* the change was made
2. Push to `origin main` immediately after every commit
3. All Python code must include inline comments explaining each code block

## Coding Conventions

- All Python files use inline comments to explain every code block
- Comments describe intent and reasoning, not just what the line does
- Comply with terms of use of any external data source before scraping
