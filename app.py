"""
Mechanic Saathi — Streamlit UI
Voice-first, multilingual, image-aware bike troubleshooting grounded in the manual.

Run:
    streamlit run app.py
"""

from __future__ import annotations
import base64
import io
import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Lazy imports — heavier modules load on first interaction so the UI paints fast
from core.pipeline import ask, translate_answer, refusal_message
from core.sarvam import stt, tts, LANG_CODES, lang_to_code, code_to_name, DEFAULT_VOICE

ROOT = Path(__file__).resolve().parent
BG_PATH = ROOT / "assets" / "bg.jpg"

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Mechanic Saathi · Sarvam AI",
    page_icon="🛵",
    layout="centered",
    initial_sidebar_state="collapsed",
)


def inject_css():
    bg_b64 = base64.b64encode(BG_PATH.read_bytes()).decode() if BG_PATH.exists() else ""
    st.markdown(f"""
    <style>
    /* ----- Background ----- */
    .stApp {{
        background-image:
            linear-gradient(rgba(0,0,0,0.35), rgba(0,0,0,0.55)),
            url("data:image/jpeg;base64,{bg_b64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    /* ----- Header / branding ----- */
    .ms-hero {{
        text-align: center;
        padding: 30px 0 10px 0;
    }}
    .ms-hero h1 {{
        color: #fff;
        font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
        font-size: 54px;
        font-weight: 800;
        margin: 0;
        letter-spacing: 2px;
        text-shadow: 0 2px 12px rgba(0,0,0,0.6);
    }}
    .ms-hero .accent {{ color: #FF7A1A; }}
    .ms-hero .tagline {{
        color: #FFD9B8;
        font-size: 17px;
        font-weight: 400;
        margin-top: 4px;
        letter-spacing: 0.5px;
    }}
    .ms-hero .powered {{
        color: rgba(255,255,255,0.55);
        font-size: 12px;
        margin-top: 6px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }}

    /* ----- Chat bubbles ----- */
    [data-testid="stChatMessage"] {{
        background: rgba(20, 15, 12, 0.75) !important;
        border-radius: 18px;
        border: 1px solid rgba(255, 122, 26, 0.25);
        backdrop-filter: blur(8px);
        padding: 16px 20px !important;
    }}
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li {{
        color: #FFF1E2 !important;
        font-size: 16px;
        line-height: 1.55;
    }}

    /* ----- Input row container ----- */
    .ms-input-bar {{
        background: rgba(15, 10, 8, 0.85);
        border: 1px solid rgba(255, 122, 26, 0.35);
        border-radius: 22px;
        padding: 10px 14px;
        margin-top: 10px;
        backdrop-filter: blur(10px);
    }}

    /* Streamlit button polish */
    .stButton > button {{
        background: linear-gradient(135deg, #FF7A1A 0%, #FF5A00 100%);
        color: #fff;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        padding: 8px 14px;
        transition: transform 0.05s ease, box-shadow 0.2s ease;
    }}
    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(255, 122, 26, 0.45);
    }}

    /* Secondary "ghost" buttons */
    .stButton.ghost > button,
    button[kind="secondary"] {{
        background: transparent !important;
        border: 1px solid rgba(255, 122, 26, 0.5) !important;
        color: #FFD9B8 !important;
    }}

    /* Selectbox / inputs */
    .stSelectbox label, .stTextInput label {{ color: #FFD9B8 !important; }}
    .stTextInput input {{
        background: rgba(255,255,255,0.07) !important;
        color: #fff !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
    }}

    /* Hide footer + menu for cleanliness */
    #MainMenu, footer {{ visibility: hidden; }}
    header[data-testid="stHeader"] {{ background: transparent; }}

    /* Citation chips */
    .ms-citation {{
        display: inline-block;
        background: rgba(255, 122, 26, 0.18);
        border: 1px solid rgba(255, 122, 26, 0.55);
        color: #FFD9B8;
        padding: 2px 10px;
        margin: 2px 4px 2px 0;
        border-radius: 999px;
        font-size: 12px;
        font-family: monospace;
    }}
    .ms-image-symptom {{
        background: rgba(255, 122, 26, 0.12);
        border-left: 3px solid #FF7A1A;
        padding: 8px 12px;
        margin: 6px 0;
        border-radius: 8px;
        color: #FFD9B8;
        font-style: italic;
        font-size: 14px;
    }}
    </style>
    """, unsafe_allow_html=True)


inject_css()


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown("""
<div class="ms-hero">
    <h1>MECHANIC <span class="accent">SAATHI</span></h1>
    <div class="tagline">Apni bike ki problem bataiye — kisi bhi bhasha mein</div>
    <div class="powered">Powered by Sarvam AI · saarika · bulbul · sarvam-m</div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of {role, content, meta}
if "current_lang_code" not in st.session_state:
    st.session_state.current_lang_code = "en-IN"
if "audio_counter" not in st.session_state:
    st.session_state.audio_counter = 0


# ---------------------------------------------------------------------------
# Sidebar — language picker, voice picker
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Settings")
    lang_name = st.selectbox(
        "Answer language",
        list(LANG_CODES.keys()),
        index=list(LANG_CODES.keys()).index("auto"),
        help="'auto' uses the language Sarvam STT detects from your voice. "
             "For text input we default to English unless you pick one here.",
    )
    voice = st.selectbox(
        "TTS voice (Bulbul)",
        ["anushka", "manisha", "vidya", "arya", "abhilash", "karun", "hitesh"],
        index=0,
    )
    os.environ["SARVAM_TTS_VOICE"] = voice
    st.markdown("---")
    st.caption("Manual loaded: **KTM Duke 250** — Owner's Manual excerpts (10 sections).")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()


# ---------------------------------------------------------------------------
# Render history
# ---------------------------------------------------------------------------
def render_history():
    for i, msg in enumerate(st.session_state.history):
        with st.chat_message(msg["role"]):
            if msg.get("image_symptom"):
                st.markdown(
                    f'<div class="ms-image-symptom">📷 Saw in photo: '
                    f'<b>{msg["image_symptom"]}</b></div>',
                    unsafe_allow_html=True,
                )
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                # Show citations as chips
                if msg.get("chunks"):
                    chips = "".join(
                        f'<span class="ms-citation">📖 {c["title"]}</span>'
                        for c in msg["chunks"]
                    )
                    st.markdown(f'<div>{chips}</div>', unsafe_allow_html=True)
                # Per-answer action buttons
                c1, c2, c3 = st.columns([1, 1, 4])
                with c1:
                    if st.button("🔊 Listen", key=f"tts_{i}"):
                        try:
                            audio = tts(msg["content"], language_code=msg.get("lang", "en-IN"))
                            if audio:
                                st.audio(audio, format="audio/wav")
                        except Exception as e:
                            st.error(f"TTS failed: {e}")
                with c2:
                    other_langs = [n for n in LANG_CODES if n not in ("auto",)]
                    target = st.selectbox(
                        "🌐 Translate to",
                        other_langs,
                        index=other_langs.index("English"),
                        key=f"tx_pick_{i}",
                        label_visibility="collapsed",
                    )
                    if st.button("Translate", key=f"tx_btn_{i}"):
                        try:
                            t = translate_answer(
                                msg["content"],
                                msg.get("lang", "en-IN"),
                                lang_to_code(target),
                            )
                            st.info(t)
                        except Exception as e:
                            st.error(f"Translate failed: {e}")


# ---------------------------------------------------------------------------
# Input row — mic, image, text on one line
# ---------------------------------------------------------------------------
def handle_input(text: str, lang_code: str, image_bytes: bytes | None, image_mime: str):
    user_display = text if text else "(photo only)"
    st.session_state.history.append({
        "role": "user",
        "content": user_display,
        "lang": lang_code,
    })
    with st.spinner("Mechanic Saathi soch rahe hain..."):
        result = ask(
            query_text=text,
            user_lang_code=lang_code,
            image_bytes=image_bytes,
            image_mime=image_mime,
        )
    st.session_state.history.append({
        "role": "assistant",
        "content": result["answer"],
        "lang": result["lang"],
        "chunks": result.get("chunks", []),
        "image_symptom": result.get("image_symptom"),
        "grounded": result.get("grounded", False),
    })


# Show chat history first
render_history()

# Input bar at bottom
st.markdown('<div class="ms-input-bar">', unsafe_allow_html=True)

mic_col, img_col, txt_col, send_col = st.columns([1.3, 1.3, 5, 1.2])

with mic_col:
    st.caption("🎤 Voice")
    audio_input = st.audio_input("Speak", label_visibility="collapsed", key="mic_input")

with img_col:
    st.caption("📷 Photo")
    image_input = st.file_uploader(
        "Photo",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key="img_input",
    )

with txt_col:
    st.caption("✍ Text")
    text_input = st.text_input(
        "Describe your bike problem...",
        placeholder="e.g. white smoke from exhaust / मेरी bike start nahi ho rahi",
        label_visibility="collapsed",
        key="text_input",
    )

with send_col:
    st.caption("&nbsp;", unsafe_allow_html=True)
    send_clicked = st.button("Send →", use_container_width=True, type="primary")

st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Submit handling
# ---------------------------------------------------------------------------
if send_clicked:
    user_text = text_input or ""
    detected_lang = lang_to_code(lang_name) if lang_name != "auto" else "en-IN"

    # If mic recording exists, transcribe with Sarvam STT
    if audio_input is not None:
        try:
            audio_bytes = audio_input.getvalue()
            with st.spinner("Sun raha hoon..."):
                transcript, det = stt(
                    audio_bytes,
                    language_code="unknown" if lang_name == "auto" else lang_to_code(lang_name),
                )
            if transcript:
                user_text = f"{user_text} {transcript}".strip() if user_text else transcript
                detected_lang = det or detected_lang
        except Exception as e:
            st.error(f"STT failed: {e}")

    # Image bytes
    img_bytes = None
    img_mime = "image/jpeg"
    if image_input is not None:
        img_bytes = image_input.getvalue()
        img_mime = image_input.type or "image/jpeg"

    if not user_text and not img_bytes:
        st.warning("Please speak, type, or upload a photo first.")
    else:
        handle_input(user_text, detected_lang, img_bytes, img_mime)
        st.rerun()
