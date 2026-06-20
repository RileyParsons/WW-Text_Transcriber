"""
ml_pipeline.py — CNN-Based Handwritten Text Recognition Pipeline

Trains and evaluates a traditional Convolutional Neural Network (CNN) model to
transcribe scanned WW1 diary page images. Volunteer transcriptions from the NSW
State Library are used as labelled ground truth for training and evaluation.

Architecture: CRNN (CNN + BiLSTM + CTC loss)
    - CNN layers extract visual features from each page image
    - Bidirectional LSTM layers model sequential character dependencies
    - CTC (Connectionist Temporal Classification) loss enables alignment-free
      training between image features and text sequences

Pipeline stages:
    1. Preprocess — Load page images from data/pages/, convert to greyscale,
                    resize to a consistent height, and normalise pixel values
    2. Train      — Train the CRNN model on image/transcription pairs defined
                    in data/pairs.csv, using volunteer text as ground truth labels
    3. Predict    — Run the trained model on new page images to produce
                    transcription text
    4. Evaluate   — Compare model predictions against volunteer ground truth
                    using character error rate (CER) and word error rate (WER)
    5. Save       — Persist trained model weights to models/ for later use

Data inputs:
    - data/pages/        Scanned diary page images
    - data/transcript/   Volunteer transcription text files (ground truth labels)
    - data/pairs.csv     Index linking each image to its transcription

Output:
    - Trained model weights saved to models/
    - Per-page accuracy metrics (CER, WER)

See ai-transcription-pipeline.py for an alternative zero-shot approach using
the Claude vision API without any model training.
"""
