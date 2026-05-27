"""
The end-to-end ask() function. Used by both Streamlit UI and the n8n webhook.
"""

from __future__ import annotations
from typing import Optional

from .rag import retrieve
from .sarvam import chat, translate, lang_to_code, code_to_name
from .prompts import SYSTEM_PROMPT, build_context, has_citation
from .vision import describe_bike_image


REFUSAL = {
    "hi-IN": "यह जानकारी manual में नहीं मिली। कृपया authorised service centre से संपर्क करें।",
    "en-IN": "I couldn't find that in the manual. Please contact an authorised service centre.",
    "ta-IN": "இது manual-இல் இல்லை. தயவு செய்து authorised service centre-ஐ தொடர்பு கொள்ளுங்கள்.",
    "te-IN": "ఇది manual లో లేదు. దయచేసి authorised service centre ను సంప్రదించండి.",
    "mr-IN": "ही माहिती manual मध्ये नाही. कृपया authorised service centre शी संपर्क करा.",
    "gu-IN": "આ માહિતી manual માં નથી. કૃપા કરી authorised service centre નો સંપર્ક કરો.",
    "bn-IN": "এটি manual-এ পাওয়া যায়নি। দয়া করে authorised service centre এর সাথে যোগাযোগ করুন।",
    "kn-IN": "ಇದು manual ನಲ್ಲಿ ಸಿಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು authorised service centre ಸಂಪರ್ಕಿಸಿ.",
    "ml-IN": "ഇത് manual-ൽ ഇല്ല. ദയവായി authorised service centre-ലുമായി ബന്ധപ്പെടുക.",
    "pa-IN": "ਇਹ manual ਵਿੱਚ ਨਹੀਂ ਮਿਲਿਆ। ਕਿਰਪਾ ਕਰਕੇ authorised service centre ਨਾਲ ਸੰਪਰਕ ਕਰੋ।",
}


def refusal_message(lang_code: str) -> str:
    return REFUSAL.get(lang_code, REFUSAL["en-IN"])


def ask(
    query_text: str,
    user_lang_code: str = "en-IN",
    image_bytes: Optional[bytes] = None,
    image_mime: str = "image/jpeg",
) -> dict:
    """
    Single entrypoint: takes a user query (already text — STT done upstream)
    + optional image, returns a structured answer.

    Returns:
      {
        "answer": str,            # the answer in user_lang_code
        "lang": str,              # the language code we answered in
        "chunks": [..],           # retrieved chunks (for transparency)
        "image_symptom": str|None,# what the vision model saw, if image given
        "grounded": bool,         # whether the answer cites the manual
        "refused": bool,          # whether we refused (no match found)
      }
    """
    # Step 1: enrich query with image symptom if provided
    image_symptom = None
    full_query = query_text or ""
    if image_bytes:
        image_symptom = describe_bike_image(image_bytes, image_mime)
        full_query = f"{full_query}\n[Visible symptom from photo: {image_symptom}]".strip()

    if not full_query.strip():
        return {
            "answer": "Please describe your bike problem or upload a photo.",
            "lang": user_lang_code,
            "chunks": [], "image_symptom": None,
            "grounded": False, "refused": True,
        }

    # Step 2: retrieve from the manual (English-side index)
    # Sarvam-m can read Indic queries, but our manual is English, so retrieval
    # works best on English. We retrieve directly using the multilingual model —
    # it handles cross-lingual retrieval well, no need to translate first.
    chunks = retrieve(full_query, k=3, min_score=0.30)

    # Step 3: refuse cleanly if nothing matches
    if not chunks:
        return {
            "answer": refusal_message(user_lang_code),
            "lang": user_lang_code,
            "chunks": [], "image_symptom": image_symptom,
            "grounded": False, "refused": True,
        }

    # Step 4: ask Sarvam-m to compose the answer
    system = SYSTEM_PROMPT.format(context=build_context(chunks))
    user_msg = (
        f"USER QUESTION (reply in language code {user_lang_code}, matching their script):\n"
        f"{full_query}"
    )
    answer = chat(system=system, user=user_msg, temperature=0.2)

    # Step 5: guardrail — ensure the answer cites a chunk; if not, refuse
    grounded = has_citation(answer, chunks)
    if not grounded:
        return {
            "answer": refusal_message(user_lang_code),
            "lang": user_lang_code,
            "chunks": chunks, "image_symptom": image_symptom,
            "grounded": False, "refused": True,
        }

    return {
        "answer": answer,
        "lang": user_lang_code,
        "chunks": chunks,
        "image_symptom": image_symptom,
        "grounded": True,
        "refused": False,
    }


def translate_answer(answer: str, source_lang: str, target_lang: str) -> str:
    """User clicked the Translate button on a previous answer."""
    if source_lang == target_lang:
        return answer
    try:
        return translate(answer, source_lang, target_lang)
    except Exception as e:
        return f"[Translation failed: {e}]\n\n{answer}"
