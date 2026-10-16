"""Transcribe one audio file with multilingual Whisper or MMS/Wav2Vec2."""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch


LANGUAGES = {
    "en": {"whisper": "english", "mms": "eng", "label": "English"},
    "hi": {"whisper": "hindi", "mms": "hin", "label": "Hindi"},
    "te": {"whisper": "telugu", "mms": "tel", "label": "Telugu"},
}
WHISPER_ID = "openai/whisper-tiny"
MMS_ID = "facebook/mms-1b-all"


def load_audio(path: Path) -> np.ndarray:
    """Read audio, convert to mono, and resample to 16 kHz."""
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sample_rate != 16_000:
        mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=16_000)
    return np.asarray(mono, dtype=np.float32)


def transcribe(audio_path: Path, language: str, model_name: str = "whisper") -> str:
    """Return a transcript. The model downloads from Hugging Face on first use."""
    if language not in LANGUAGES:
        raise ValueError(f"Language must be one of: {', '.join(LANGUAGES)}")
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio = load_audio(audio_path)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if model_name == "whisper":
        from transformers import pipeline

        recognizer = pipeline(
            "automatic-speech-recognition",
            model=WHISPER_ID,
            device=0 if device.startswith("cuda") else -1,
        )
        result = recognizer(
            {"raw": audio, "sampling_rate": 16_000},
            generate_kwargs={"language": LANGUAGES[language]["whisper"], "task": "transcribe"},
        )
        return result["text"].strip()

    if model_name == "mms":
        from transformers import AutoModelForCTC, AutoProcessor

        language_code = LANGUAGES[language]["mms"]
        processor = AutoProcessor.from_pretrained(MMS_ID)
        processor.tokenizer.set_target_lang(language_code)
        model = AutoModelForCTC.from_pretrained(MMS_ID).to(device)
        model.load_adapter(language_code)
        inputs = processor(audio, sampling_rate=16_000, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits
        token_ids = torch.argmax(logits, dim=-1)
        return processor.batch_decode(token_ids)[0].strip()

    raise ValueError("Model must be 'whisper' or 'mms'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True, help="Path to a WAV or FLAC audio file")
    parser.add_argument("--language", choices=LANGUAGES, required=True, help="Spoken language code: en, hi, or te")
    parser.add_argument("--model", choices=("whisper", "mms"), default="whisper")
    parser.add_argument("--output", type=Path, help="Optional UTF-8 text file for the transcript")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = transcribe(args.audio, args.language, args.model)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"Saved transcript: {args.output}")


if __name__ == "__main__":
    main()
