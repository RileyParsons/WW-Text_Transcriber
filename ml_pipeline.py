"""
ml_pipeline.py — AI-Driven Transcription Pipeline

Uses the Claude vision API (claude-sonnet-4-6 or claude-opus-4-8) to transcribe
scanned WW1 diary page images and evaluates the output against volunteer-created
ground truth transcriptions.

Pipeline stages:
    1. Load — Read pairs.csv to get the list of image/transcription pairs
    2. Transcribe — Send each page image to the Claude vision API
    3. Evaluate — Compare AI transcription against volunteer ground truth
                  using text similarity metrics (e.g. character error rate,
                  word error rate)
    4. Save — Write results and any fine-tuned models to models/

Data inputs:
    - data/pages/        Scanned diary page images
    - data/transcript/   Volunteer transcription text files
    - data/pairs.csv     Index linking images to their transcriptions

Output:
    - Accuracy metrics per page
    - Saved models in models/ (if fine-tuning is performed)
"""
