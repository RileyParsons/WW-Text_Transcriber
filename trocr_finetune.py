"""
trocr_finetune.py — Fine-tune TrOCR on WW1 diary pages (page-level, no alignment)

WHY THIS EXISTS
    Zero-shot TrOCR is unusable on these 1914 hands (it hallucinates modern
    English — see CLAUDE.md / trocr_pipeline.py: CER ~118%). The fix is to
    fine-tune it on this dataset. Crucially we fine-tune at the PAGE level:

        resized whole-page image  ->  full prose transcript

    The decoder learns to emit the entire page transcription (and its reading
    order) directly, so we never need to know which text belongs to which line.
    This sidesteps the label-alignment problem that stalled the CRNN path (the
    transcripts are page-level prose, not line-aligned — see CLAUDE.md finding).

    Trade-off: TrOCR's encoder ingests a 384x384 image, so a full page is heavily
    downsampled and fine handwriting may be unreadable — this caps achievable
    accuracy and is the main thing to watch. If it plateaus poorly, the fallback
    is region/strip-level fine-tuning (still no per-line alignment needed).

HARDWARE
    Targets a single ~8 GB GPU (user's RTX 3060 Ti). Defaults use a small batch
    size + gradient accumulation + fp16 to fit. Install torch from the CUDA index
    (see memory: cuda-torch-setup), not plain PyPI, or it runs on CPU.

USAGE
    # tiny sanity check (a few pages, 1 epoch) — proves the loop runs end-to-end
    python trocr_finetune.py --limit 8 --epochs 1 --batch-size 2

    # real run
    python trocr_finetune.py --epochs 5 --batch-size 2 --grad-accum 8 --fp16

OUTPUT
    Best-by-val-CER checkpoint saved under models/trocr-ww1/.
"""

# Standard library: CLI args, randomness, paths, type hints.
import argparse
import random
from pathlib import Path
from typing import List, Tuple

# Reuse the dataset wiring + transcript normalisation from the inference
# pipeline so training and evaluation see identically-prepared labels.
from trocr_pipeline import load_pairs, normalise_transcript

# Where to persist fine-tuned weights (models/ is gitignored for size).
OUTPUT_DIR = Path("models") / "trocr-ww1"
DEFAULT_MODEL = "microsoft/trocr-base-handwritten"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def build_dataset_cls():
    """Define the torch Dataset inside a factory so importing this module never
    requires torch (kept consistent with the lazy-import style elsewhere)."""
    import torch
    from PIL import Image

    class PageDataset(torch.utils.data.Dataset):
        """One example per page: (whole-page pixel tensor, tokenised transcript).

        Images are run through the TrOCR processor (resize+normalise to 384x384,
        3-channel). Labels are the normalised prose transcript, tokenised and
        truncated to max_target_length; pad positions are set to -100 so they are
        ignored by the cross-entropy loss.
        """

        def __init__(self, rows, processor, max_target_length):
            # rows: list of (page_id, image_path, raw_transcript)
            self.rows = rows
            self.processor = processor
            self.max_target_length = max_target_length

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            _page_id, image_path, transcript = self.rows[idx]

            # Whole page -> 3-channel RGB -> processor pixel tensor (squeeze the
            # batch dim the processor adds so the DataLoader can collate).
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values[0]

            # Normalise the transcript (strip editorial brackets, collapse
            # whitespace) so the model learns handwriting, not markup.
            text = normalise_transcript(transcript)

            # Tokenise to a fixed length; pad/truncate so the batch stacks.
            labels = self.processor.tokenizer(
                text,
                padding="max_length",
                max_length=self.max_target_length,
                truncation=True,
            ).input_ids

            # Mask pad tokens with -100 so they do not contribute to the loss.
            pad_id = self.processor.tokenizer.pad_token_id
            labels = [(t if t != pad_id else -100) for t in labels]

            return {
                "pixel_values": pixel_values,
                "labels": torch.tensor(labels),
            }

    return PageDataset


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------


def load_model(model_name: str):
    """Load the processor + model and wire the decoder special-token config.

    The VisionEncoderDecoder needs its decoder_start / pad / vocab ids set
    explicitly for training, otherwise loss/generation use wrong defaults.
    """
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)

    # Decoder generation/loss config — derive ids from the tokenizer so they
    # match the labels we feed in.
    tok = processor.tokenizer
    model.config.decoder_start_token_id = tok.cls_token_id
    model.config.pad_token_id = tok.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = tok.sep_token_id

    return processor, model, device


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(model, processor, loader, device, max_new_tokens):
    """Run generation over a val loader and return mean (CER, WER).

    Scored at page level via jiwer, identical to trocr_pipeline.py, so numbers
    are comparable to the zero-shot and Claude baselines.
    """
    import torch
    from jiwer import cer, wer

    model.eval()
    cers, wers = [], []

    # No grad during eval to save memory; greedy-decode each batch.
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            generated_ids = model.generate(pixel_values, max_new_tokens=max_new_tokens)
            preds = processor.batch_decode(generated_ids, skip_special_tokens=True)

            # Recover the reference text from the masked label ids: swap -100
            # back to pad so the tokenizer can decode, then skip specials.
            labels = batch["labels"].clone()
            labels[labels == -100] = processor.tokenizer.pad_token_id
            refs = processor.batch_decode(labels, skip_special_tokens=True)

            # Accumulate per-page metrics, guarding against empty references.
            for pred, ref in zip(preds, refs):
                ref = ref.strip()
                if not ref:
                    continue
                cers.append(cer(ref, " ".join(pred.split())))
                wers.append(wer(ref, " ".join(pred.split())))

    # Mean over scored pages (or 1.0 sentinel if nothing was scorable).
    mean_cer = sum(cers) / len(cers) if cers else 1.0
    mean_wer = sum(wers) / len(wers) if wers else 1.0
    return mean_cer, mean_wer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def split_by_page(rows, val_frac, seed) -> Tuple[List, List]:
    """Shuffle and split rows into (train, val) BY PAGE.

    Each row is already a distinct page, so a plain shuffle+slice keeps all of a
    page's content on one side of the split (no leakage) — matching the
    train/eval boundary rule in CLAUDE.md.
    """
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_frac))
    return shuffled[n_val:], shuffled[:n_val]


def main():
    # Training knobs, defaulted to fit an ~8 GB GPU.
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, help="base TrOCR checkpoint")
    ap.add_argument("--limit", type=int, default=0, help="cap total pages (0 = all) for quick tests")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8, help="steps to accumulate before optimiser step")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--max-target-length", type=int, default=256, help="max label tokens (page prose can be long)")
    ap.add_argument("--max-new-tokens", type=int, default=256, help="generation cap during eval")
    ap.add_argument("--fp16", action="store_true", help="mixed-precision training (saves VRAM)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimiser steps (0 = full epochs); for sanity checks")
    args = ap.parse_args()

    # Heavy imports now that we are actually training.
    import torch
    from torch.utils.data import DataLoader

    # Load and (optionally) cap the dataset, then split by page.
    rows = load_pairs()
    if args.limit:
        rows = rows[: args.limit]
    train_rows, val_rows = split_by_page(rows, args.val_frac, args.seed)
    print(f"pages: {len(rows)} total -> {len(train_rows)} train / {len(val_rows)} val")

    # Load model + processor and build the datasets/loaders.
    processor, model, device = load_model(args.model)
    PageDataset = build_dataset_cls()
    train_ds = PageDataset(train_rows, processor, args.max_target_length)
    val_ds = PageDataset(val_rows, processor, args.max_target_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    print(f"device: {device} | fp16: {args.fp16}")

    # AdamW optimiser + optional AMP scaler for fp16 mixed precision.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and device == "cuda")

    # Track the best validation CER so we only checkpoint genuine improvements.
    best_cer = float("inf")
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0

        for i, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # Forward under autocast for fp16; the model returns the CE loss
            # directly when given labels.
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                 enabled=args.fp16 and device == "cuda"):
                outputs = model(pixel_values=pixel_values, labels=labels)
                # Scale loss by accumulation steps so the effective batch is
                # batch_size * grad_accum.
                loss = outputs.loss / args.grad_accum

            # Backward with grad scaling (no-op when fp16 disabled).
            scaler.scale(loss).backward()
            running_loss += outputs.loss.item()

            # Optimiser step every grad_accum micro-batches.
            if (i + 1) % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

                # Periodic progress line so long runs are observable.
                if global_step % 10 == 0:
                    print(f"epoch {epoch} step {global_step} loss {outputs.loss.item():.4f}")

                # Early stop for sanity checks (--max-steps).
                if args.max_steps and global_step >= args.max_steps:
                    break

        # End-of-epoch validation + checkpoint on improvement.
        mean_cer, mean_wer = evaluate(model, processor, val_loader, device, args.max_new_tokens)
        avg_loss = running_loss / max(1, len(train_loader))
        print(f"[epoch {epoch}] train_loss {avg_loss:.4f}  val_CER {mean_cer:.2%}  val_WER {mean_wer:.2%}")

        if mean_cer < best_cer:
            best_cer = mean_cer
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(OUTPUT_DIR)
            processor.save_pretrained(OUTPUT_DIR)
            print(f"  saved new best (CER {mean_cer:.2%}) -> {OUTPUT_DIR}")

        # Honour --max-steps across epochs too.
        if args.max_steps and global_step >= args.max_steps:
            break

    print(f"done. best val CER: {best_cer:.2%}")


if __name__ == "__main__":
    main()
