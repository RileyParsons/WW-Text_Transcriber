# WW-Text-Transcriber

An exploration into AI-driven development for automated text transcription of historical handwritten documents.

## Purpose

This project investigates whether modern AI vision models can accurately transcribe WW1-era handwritten diary pages, using existing volunteer transcriptions as ground truth to measure accuracy. It serves as a practical case study in AI-assisted digitisation of historical records.

## Data Source

Scanned diary pages and volunteer transcriptions are sourced from the **NSW State Library WW1 Diaries Transcription Project**:
> https://transcripts.sl.nsw.gov.au/section/world-war-1-diaries

The volunteer transcriptions act as labelled ground truth for evaluating and training the AI transcription pipeline.

## Approach

1. **Scrape** — Download paired scanned page images and volunteer transcription text files from the NSW State Library
2. **Transcribe** — Pass page images through the Claude vision API to generate AI transcriptions
3. **Evaluate** — Compare AI output against volunteer ground truth to measure accuracy
4. **Iterate** — Use findings to improve the pipeline or fine-tune a lightweight model

## Project Structure

```
WW-Text-Transcriber/
├── scraper.py        # Downloads page images and transcriptions from NSW State Library
├── ml_pipeline.py    # AI transcription pipeline using Claude vision API
├── data/
│   ├── pages/        # Downloaded scanned diary page images
│   ├── transcript/   # Downloaded volunteer transcription text files
│   └── pairs.csv     # Index linking each image to its transcription
└── models/           # Saved trained or fine-tuned transcription models
```

## Status

Exploratory — in active development.
