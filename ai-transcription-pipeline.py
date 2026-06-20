"""
ai-transcription-pipeline.py — Claude Vision API Transcription Pipeline

An alternative, zero-shot transcription pipeline that sends scanned WW1 diary
page images directly to the Claude vision API for transcription — no model
training required. Results are evaluated against volunteer ground truth and
benchmarked against the CNN pipeline in ml_pipeline.py.

Model options:
    - claude-sonnet-4-6  (balanced speed and accuracy)
    - claude-opus-4-8    (highest accuracy, slower and more expensive)

Pipeline stages:
    1. Load      — Read data/pairs.csv to get the list of image/transcription
                   pairs to process
    2. Transcribe — Encode each page image and send it to the Claude vision API
                   with a prompt instructing it to transcribe the handwritten text
    3. Evaluate  — Compare the API transcription against the volunteer ground
                   truth using character error rate (CER) and word error rate (WER)
    4. Compare   — Benchmark Claude vision API accuracy against the CNN pipeline
                   results from ml_pipeline.py

Data inputs:
    - data/pages/        Scanned diary page images
    - data/transcript/   Volunteer transcription text files (ground truth labels)
    - data/pairs.csv     Index linking each image to its transcription

Output:
    - Per-page accuracy metrics (CER, WER)
    - Comparison report against CNN pipeline results

Note: Requires an Anthropic API key set as the ANTHROPIC_API_KEY environment
variable. See .env.example (to be created) for setup instructions.

See ml_pipeline.py for the primary CNN-based training approach.
"""
