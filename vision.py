"""
Image → bike-symptom text. The job here is small but important: convert what
the user photographed (smoke, leak, dashboard light, broken chain) into a
concise English symptom description that we feed into the RAG layer.

Architecture:
- Primary: OpenAI vision (gpt-4o-mini) if OPENAI_API_KEY is set.
- Secondary: Anthropic Claude vision if ANTHROPIC_API_KEY is set.
- Fallback: dumb mode — return a generic "user uploaded an image" placeholder
  so the app still runs without a vision key (useful for offline demo).
"""

from __future__ import annotations
import base64
import os
from typing import Optional

import requests

VISION_PROMPT = (
    "You are helping a bike troubleshooting assistant. Look at this image of a "
    "motorcycle (or motorcycle part) and produce a ONE-LINE English description "
    "of the visible problem or symptom (e.g. 'white smoke from exhaust', "
    "'engine oil leaking near drain plug', 'low brake fluid in front reservoir', "
    "'chain visibly loose'). If nothing concerning is visible, reply 'no clear "
    "problem visible'. Reply with the description ONLY, no preamble."
)


def describe_bike_image(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return _openai_vision(image_bytes, mime)
        except Exception as e:
            print(f"[vision] OpenAI failed: {e}")
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return _anthropic_vision(image_bytes, mime)
        except Exception as e:
            print(f"[vision] Anthropic failed: {e}")
    return "user uploaded a bike image but no vision model is configured"


def _openai_vision(image_bytes: bytes, mime: str) -> str:
    b64 = base64.b64encode(image_bytes).decode()
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": VISION_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        }],
        "max_tokens": 80,
        "temperature": 0,
    }
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        },
        json=payload, timeout=45,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _anthropic_vision(image_bytes: bytes, mime: str) -> str:
    b64 = base64.b64encode(image_bytes).decode()
    payload = {
        "model": "claude-3-5-haiku-20241022",
        "max_tokens": 80,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
                {"type": "text", "text": VISION_PROMPT},
            ],
        }],
    }
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json=payload, timeout=45,
    )
    r.raise_for_status()
    return r.json()["content"][0]["text"].strip()
