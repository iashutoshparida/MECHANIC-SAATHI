"""
RAG layer for Mechanic Saathi.
Chunks the bike manual by section, embeds with a multilingual MiniLM,
persists to disk, retrieves top-k by cosine similarity.

Why local embeddings:
- Manual is ~10 KB; in-memory cosine over numpy is faster than spinning up FAISS.
- paraphrase-multilingual-MiniLM-L12-v2 (~120 MB) handles 50+ languages including
  Hindi, Tamil, Bengali, Marathi — matches our user base.
- Zero API dependency for retrieval keeps the demo running even offline.
"""

from __future__ import annotations
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
MANUAL_PATH = ROOT / "data" / "ktm_duke_manual.txt"
INDEX_DIR = ROOT / "index"
INDEX_DIR.mkdir(exist_ok=True)
EMB_PATH = INDEX_DIR / "embeddings.npy"
CHUNKS_PATH = INDEX_DIR / "chunks.json"

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def chunk_manual(text: str) -> List[Dict]:
    """
    Split by numbered subsections (e.g. "1.1 WHITE SMOKE..."). Each chunk keeps
    the parent SECTION heading as context so retrieval surfaces structured citations.
    """
    chunks: List[Dict] = []
    current_section = ""
    # Match either SECTION N or subsection N.N
    section_re = re.compile(r"^SECTION\s+\d+\s+—\s+.+$", re.MULTILINE)
    subsec_re = re.compile(r"^(\d+\.\d+)\s+([A-Z][A-Z0-9 ,\-/()]+)$", re.MULTILINE)

    # Build a flat list of (start_idx, kind, header) markers
    markers = []
    for m in section_re.finditer(text):
        markers.append((m.start(), "section", m.group(0).strip()))
    for m in subsec_re.finditer(text):
        markers.append((m.start(), "sub", f"{m.group(1)} {m.group(2).strip()}"))
    markers.sort()

    # Walk markers and slice content between subsections
    section_label = ""
    for i, (pos, kind, header) in enumerate(markers):
        end = markers[i + 1][0] if i + 1 < len(markers) else len(text)
        if kind == "section":
            section_label = header
            continue
        body = text[pos:end].strip()
        chunks.append({
            "id": header.split()[0],          # e.g. "1.1"
            "title": header,                  # e.g. "1.1 WHITE SMOKE FROM EXHAUST"
            "section": section_label,         # parent section
            "text": body,
        })
    return chunks


def build_index() -> None:
    text = MANUAL_PATH.read_text(encoding="utf-8")
    chunks = chunk_manual(text)
    if not chunks:
        raise RuntimeError("No chunks parsed from manual. Check formatting.")
    model = _get_model()
    embeddings = model.encode(
        [f"{c['title']}\n{c['text']}" for c in chunks],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    np.save(EMB_PATH, embeddings)
    CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2))
    print(f"[rag] Indexed {len(chunks)} chunks → {INDEX_DIR}")


def _ensure_index() -> Tuple[np.ndarray, List[Dict]]:
    if not EMB_PATH.exists() or not CHUNKS_PATH.exists():
        build_index()
    emb = np.load(EMB_PATH)
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    return emb, chunks


def retrieve(query: str, k: int = 3, min_score: float = 0.30) -> List[Dict]:
    """
    Return up to k chunks above min_score. If nothing passes min_score,
    return empty list — caller MUST refuse to answer.
    """
    if not query.strip():
        return []
    emb, chunks = _ensure_index()
    model = _get_model()
    q = model.encode([query], normalize_embeddings=True)[0]
    scores = emb @ q  # cosine, embeddings are pre-normalized
    top_idx = np.argsort(-scores)[:k]
    results = []
    for i in top_idx:
        s = float(scores[i])
        if s < min_score:
            continue
        c = dict(chunks[i])
        c["score"] = round(s, 3)
        results.append(c)
    return results


if __name__ == "__main__":
    build_index()
    for q in [
        "white smoke is coming from my exhaust",
        "मेरी bike start nahi ho rahi",
        "what is the capital of France",  # should refuse
    ]:
        print(f"\n>> {q}")
        for r in retrieve(q):
            print(f"  [{r['id']}] score={r['score']} — {r['title']}")
