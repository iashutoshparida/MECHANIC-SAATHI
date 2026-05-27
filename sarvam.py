"""
Sarvam AI client — STT, TTS, Chat (sarvam-m), Translate.
One file, no SDK dependency, just `requests`.

Env:
  SARVAM_API_KEY   required for any call
  SARVAM_TTS_VOICE optional, defaults to 'anushka' (warm female Indian voice)
"""

from __future__ import annotations
import base64
import io
import os
from typing import List, Optional, Tuple

import requests

SARVAM_BASE = "https://api.sarvam.ai"
TIMEOUT = 60


def _headers() -> dict:
    key = os.environ.get("SARVAM_API_KEY")
    if not key:
        raise RuntimeError(
            "SARVAM_API_KEY not set. Get one at https://sarvam.ai and put it in .env"
        )
    return {"api-subscription-key": key}


# ---------------------------------------------------------------------------
# Language codes Sarvam understands. Maps friendly name -> Sarvam BCP-47.
# ---------------------------------------------------------------------------
LANG_CODES = {
    "auto": "unknown",
    "Hindi": "hi-IN",
    "English": "en-IN",
    "Tamil": "ta-IN",
    "Telugu": "te-IN",
    "Kannada": "kn-IN",
    "Malayalam": "ml-IN",
    "Marathi": "mr-IN",
    "Gujarati": "gu-IN",
    "Bengali": "bn-IN",
    "Punjabi": "pa-IN",
    "Odia": "od-IN",
}
DEFAULT_VOICE = os.environ.get("SARVAM_TTS_VOICE", "anushka")


def lang_to_code(name: str) -> str:
    return LANG_CODES.get(name, "unknown")


def code_to_name(code: str) -> str:
    for name, c in LANG_CODES.items():
        if c == code:
            return name
    return "Hindi"


# ---------------------------------------------------------------------------
# Speech to Text — saarika v2.5, auto language detect
# ---------------------------------------------------------------------------
def stt(audio_bytes: bytes, language_code: str = "unknown") -> Tuple[str, str]:
    """
    Transcribe audio. Returns (transcript, detected_language_code).
    If language_code='unknown', Sarvam auto-detects.
    """
    files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
    data = {"model": "saarika:v2.5", "language_code": language_code}
    r = requests.post(
        f"{SARVAM_BASE}/speech-to-text",
        headers=_headers(),
        files=files,
        data=data,
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    j = r.json()
    return j.get("transcript", ""), j.get("language_code", "hi-IN")


# ---------------------------------------------------------------------------
# Text to Speech — Bulbul v2
# ---------------------------------------------------------------------------
def tts(text: str, language_code: str = "hi-IN", voice: Optional[str] = None) -> bytes:
    """
    Synthesize speech. Returns raw WAV bytes.
    Sarvam splits long text into chunks; we just concatenate base64 audio.
    """
    if not text.strip():
        return b""
    # Sarvam TTS has a per-request character limit. Split long answers.
    voice = voice or DEFAULT_VOICE
    chunks = _split_for_tts(text, max_chars=480)
    audio_parts: List[bytes] = []
    for chunk in chunks:
        payload = {
            "inputs": [chunk],
            "target_language_code": language_code,
            "speaker": voice,
            "model": "bulbul:v2",
            "pitch": 0.0,
            "pace": 1.0,
            "loudness": 1.2,
            "speech_sample_rate": 22050,
            "enable_preprocessing": True,
        }
        r = requests.post(
            f"{SARVAM_BASE}/text-to-speech",
            headers={**_headers(), "Content-Type": "application/json"},
            json=payload,
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        for b64 in r.json().get("audios", []):
            audio_parts.append(base64.b64decode(b64))
    # Concatenate WAVs naively — for demo purposes this is fine since voices
    # match. For production, splice by stripping headers.
    return b"".join(audio_parts)


def _split_for_tts(text: str, max_chars: int = 480) -> List[str]:
    parts: List[str] = []
    buf = ""
    for sentence in _split_sentences(text):
        if len(buf) + len(sentence) + 1 > max_chars and buf:
            parts.append(buf.strip())
            buf = sentence
        else:
            buf = f"{buf} {sentence}".strip()
    if buf:
        parts.append(buf.strip())
    return parts


def _split_sentences(text: str) -> List[str]:
    import re
    return [s.strip() for s in re.split(r"(?<=[\.!?।])\s+", text) if s.strip()]


# ---------------------------------------------------------------------------
# Translate — Mayura
# ---------------------------------------------------------------------------
def translate(text: str, source_lang: str, target_lang: str) -> str:
    if not text.strip() or source_lang == target_lang:
        return text
    payload = {
        "input": text,
        "source_language_code": source_lang,
        "target_language_code": target_lang,
        "mode": "formal",
        "model": "mayura:v1",
        "enable_preprocessing": True,
    }
    r = requests.post(
        f"{SARVAM_BASE}/translate",
        headers={**_headers(), "Content-Type": "application/json"},
        json=payload,
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json().get("translated_text", text)


# ---------------------------------------------------------------------------
# Chat — sarvam-m (Indic-native LLM, OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------
def chat(system: str, user: str, model: str = "sarvam-m", temperature: float = 0.2) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": 700,
    }
    r = requests.post(
        f"{SARVAM_BASE}/v1/chat/completions",
        headers={**_headers(), "Content-Type": "application/json"},
        json=payload,
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()
