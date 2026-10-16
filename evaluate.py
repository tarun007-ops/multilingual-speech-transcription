"""Evaluate speech transcripts against human-verified reference text."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from jiwer import cer, wer

from transcribe import LANGUAGES, transcribe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="CSV with audio_file, language, reference columns")
    parser.add_argument("--model", choices=("whisper", "mms"), default="whisper")
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    base_dir = args.manifest.resolve().parent
    rows: list[dict[str, str | float]] = []
    with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"audio_file", "language", "reference"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Manifest must include: audio_file, language, reference")
        for line_number, row in enumerate(reader, start=2):
            if not row["reference"].strip():
                raise ValueError(f"Add a verified reference transcript on CSV line {line_number} before scoring.")
            language = row["language"].strip().lower()
            if language not in LANGUAGES:
                raise ValueError(f"Unsupported language '{language}' on CSV line {line_number}; use en, hi, or te.")
            audio_path = Path(row["audio_file"].strip())
            if not audio_path.is_absolute():
                audio_path = base_dir / audio_path
            prediction = transcribe(audio_path, language, args.model)
            reference = row["reference"].strip()
            rows.append({
                "audio_file": str(audio_path),
                "language": language,
                "reference": reference,
                "prediction": prediction,
                "wer": wer(reference, prediction),
                "cer": cer(reference, prediction),
            })
            print(f"{audio_path.name}: WER={rows[-1]['wer']:.3f}, CER={rows[-1]['cer']:.3f}")

    if not rows:
        raise ValueError("Manifest has no audio rows to evaluate.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    mean_wer = sum(float(row["wer"]) for row in rows) / len(rows)
    mean_cer = sum(float(row["cer"]) for row in rows) / len(rows)
    print(f"Macro-average across {len(rows)} clips: WER={mean_wer:.3f}, CER={mean_cer:.3f}")
    print(f"Saved per-clip results: {args.output}")


if __name__ == "__main__":
    main()
