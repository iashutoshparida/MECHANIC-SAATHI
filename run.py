"""
Eval harness — runs the test set and scores grounding + refusal accuracy.

Usage:
    python -m evals.run                  # full eval (needs SARVAM_API_KEY)
    python -m evals.run --retrieval-only # skip the LLM call, only check retrieval
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.rag import retrieve
from core.pipeline import ask


def cite_ids(chunks):
    return [c["id"] for c in chunks]


def run_retrieval_only(test_set):
    print("RETRIEVAL-ONLY MODE (no LLM call)\n" + "=" * 50)
    passed = 0
    total = 0

    for item in test_set["in_manual"]:
        total += 1
        chunks = retrieve(item["q"], k=3)
        retrieved = cite_ids(chunks)
        hit = any(cid in retrieved for cid in item["must_cite"])
        flag = "✅" if hit else "❌"
        print(f"{flag} [{', '.join(item['must_cite']):>10}] retrieved={retrieved}  Q: {item['q'][:60]}")
        if hit:
            passed += 1

    for item in test_set["out_of_manual"]:
        total += 1
        chunks = retrieve(item["q"], k=3, min_score=0.30)
        should_refuse = len(chunks) == 0
        flag = "✅" if should_refuse else "❌"
        retrieved = cite_ids(chunks)
        print(f"{flag} REFUSE-EXPECTED  retrieved={retrieved or 'none'}  Q: {item['q'][:60]}")
        if should_refuse:
            passed += 1

    print(f"\nScore: {passed}/{total} ({passed/total*100:.0f}%)")


def run_full(test_set):
    print("FULL PIPELINE (Sarvam LLM)\n" + "=" * 50)
    passed = 0
    total = 0

    for item in test_set["in_manual"]:
        total += 1
        r = ask(item["q"], user_lang_code=item.get("lang", "en-IN"))
        grounded = r["grounded"]
        retrieved = cite_ids(r["chunks"])
        cite_hit = any(cid in retrieved for cid in item["must_cite"])
        ok = grounded and cite_hit
        flag = "✅" if ok else "❌"
        print(f"{flag} grounded={grounded} cite_ok={cite_hit} | {item['q'][:60]}")
        print(f"   answer: {r['answer'][:120]}...")
        if ok:
            passed += 1

    for item in test_set["out_of_manual"]:
        total += 1
        r = ask(item["q"], user_lang_code=item.get("lang", "en-IN"))
        refused = r["refused"]
        flag = "✅" if refused else "❌"
        print(f"{flag} refused={refused} | {item['q'][:60]}")
        print(f"   answer: {r['answer'][:120]}...")
        if refused:
            passed += 1

    print(f"\nScore: {passed}/{total} ({passed/total*100:.0f}%)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--retrieval-only", action="store_true",
                   help="Skip the LLM call; only check retrieval quality.")
    args = p.parse_args()

    test_set = json.loads((Path(__file__).parent / "test_set.json").read_text())
    if args.retrieval_only:
        run_retrieval_only(test_set)
    else:
        run_full(test_set)
