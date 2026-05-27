# Mechanic Saathi

A voice-first, multilingual, image-aware bike troubleshooting assistant grounded strictly in owner's manuals. Riders ask questions (or upload a photo of a problem) and get answers sourced only from the indexed bike manuals.

> Submitted by ASHUTOSH PARIDA · Sarvam STT + TTS + sarvam-m + Mayura translate

\---

## Who it's for

The rider on the side of a Ballia,UP service road at 9:47pm with white smoke pouring from their exhaust, who doesn't speak fluent English, can't read a 200-page PDF, and whose nearest mechanic is closed. They photograph the smoke, hit the mic button, and say *"yeh kya ho raha hai bhai?"* — and get an answer in Hindi in 4 seconds, grounded in the actual manual, with the option to listen to the reply being read out.

Bharat doesn't lack motorcycle owners. It lacks 24×7 mechanic uncles who happen to also speak Marathi.

## Why it matters

Three things:

1. **Strictly grounded** — every answer cites a chunk ID from the manual. If retrieval scores below threshold, the bot refuses (`"yeh manual mein nahi mila"`) instead of hallucinating.
2. **Fully Sarvam-powered voice stack** — Sarvam STT (saarika) detects language, Sarvam Bulbul speaks the answer back, Sarvam-m generates the answer natively in the user's language, Mayura translates on demand.
3. **Multimodal input, on one row** — mic, image, and text input live in a single bar in the UI. The image flows through a vision model into an English symptom description (`"white smoke from exhaust"`) that joins the user's question before retrieval.


## Where things live

- `artifacts/mechanic-saathi/app.py` — Streamlit UI and chat logic
- `artifacts/mechanic-saathi/src/ingest.py` — PDF → chunks → TF-IDF index
- `artifacts/mechanic-saathi/src/rag.py` — Retriever class
- `artifacts/mechanic-saathi/src/bot.py` — MechanicSaathiBot (retrieval → prompt → LLM)
- `artifacts/mechanic-saathi/src/llm.py` — Sarvam / OpenAI / Mock providers
- `artifacts/mechanic-saathi/data/` — Bike PDF manuals (source of truth)
- `artifacts/mechanic-saathi/vectorstore/store.pkl` — Pickled TF-IDF index

## Bikes indexed

- Honda Activa 6G (504 chunks across both bikes combined)
- Royal Enfield Classic 350

## How it works

- Python + Streamlit UI
- PDF ingestion: `pypdf` → 600-char overlapping chunks
🎤 Voice ──► Sarvam STT (saarika v2.5, auto language detect)
📷 Photo ──► Vision model ──► English symptom description
✍ Text  ──► passthrough
                │
                ▼
        Multilingual MiniLM embedding
                │
                ▼
        Top-3 chunks (cosine ≥ 0.30)
                │
                ▼
        Sarvam-m chat
        ─ "answer in user's language"
        ─ "cite at least one chunk ID"
                │
                ▼
        Citation guardrail
        ─ if no chunk ID in answer → refuse
                │
                ▼
        🔊 Sarvam Bulbul TTS  /  🌐 Mayura translate (on-demand)
```

## How to run

```bash
git clone <this>
cd mechanic\\\_saathi
python -m venv .venv \\\&\\\& source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # fill in SARVAM\\\_API\\\_KEY (required)
                           # OPENAI\\\_API\\\_KEY or ANTHROPIC\\\_API\\\_KEY (optional, for image input)

# Build the manual index (\\\~30s first time, downloads \\\~120MB embedding model)
python -m core.rag

# Launch the UI
streamlit run app.py
```

Open `http://localhost:8501`. Pick `auto` for language detection, hit the mic, speak in any of: Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia, English, or Hinglish.

## Eval

The `evals/` folder has a 15-question test set:

* 10 in-manual questions, each tagged with the chunk IDs that MUST appear in retrieval
* 5 out-of-manual questions where the bot MUST refuse

```bash
python -m evals.run --retrieval-only    # lightweight check, no Sarvam key needed
python -m evals.run                      # full pipeline incl. LLM + grounding
```

The retrieval-only mode is what you'd want to run in CI — it answers the question "does our index still surface the right chunk?" without spending an API call.

## File layout

```
mechanic\\\_saathi/
├── app.py                       Streamlit UI
├── core/
│   ├── pipeline.py              ask() entrypoint, used by UI + n8n
│   ├── rag.py                   Chunking + multilingual embed + retrieve
│   ├── sarvam.py                STT, TTS, Chat, Translate wrappers
│   ├── vision.py                Image → symptom (OpenAI/Anthropic, pluggable)
│   └── prompts.py               Grounded-answer system prompt + citation guard
├── data/
│   └── ktm\\\_duke\\\_manual.pdf      Source manual (10 sections, 24 subsections)
├── assets/bg.jpg                Background image
├── evals/
│   ├── test\\\_set.json            15-question eval (10 in / 5 out)
│   └── run.py                   Retrieval-only + full-pipeline modes
├── n8n/workflow.json            WhatsApp ← Sarvam STT/TTS ← Mechanic Saathi
└── index/                       Built embeddings + chunk metadata (on first run)
```

## Deploy

**Streamlit Community Cloud:** push this folder to a public GitHub repo, point Streamlit Cloud at `app.py`, add `SARVAM\\\_API\\\_KEY` (and one of `OPENAI\\\_API\\\_KEY` / `ANTHROPIC\\\_API\\\_KEY`) as secrets, deploy. The first cold start runs `python -m core.rag` automatically because it's called lazily by `app.py`.

**WhatsApp (via n8n):** import `n8n/workflow.json` into a self-hosted n8n instance. Configure WhatsApp Cloud API credentials (n8n has a first-class integration) and three env vars: `SARVAM\\\_API\\\_KEY`, `MECHANIC\\\_SAATHI\\\_API\\\_URL` (the URL of a thin FastAPI wrapper around `core.pipeline.ask` - left as a 10-line exercise), and `WHATSAPP\\\_PHONE\\\_ID`. The workflow handles audio + text messages and replies with both voice and text.

## Product

Riders select their bike, type a question or upload a photo of the problem, and get a manual-grounded answer with page citations. Out-of-scope questions are refused cleanly.

## Secrets required

- `SARVAM_API_KEY` — Sarvam AI for text answers (primary)
- `OPENAI_API_KEY` — fallback + image vision (gpt-4o-mini)


