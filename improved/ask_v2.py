"""Hands-on 2 v2: answer the teacher's 5 questions with the improved pipeline (C3)
and save to results/answers_v2.json (same format as teacher's authentic_rag_search_results.json)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, V2_BM25_THAI, V2_VECTOR
from rag_v2 import AuthenticRAGv2

QUESTIONS = [  # teacher's questions, unchanged
    'โรคหัดและโรคหัดเยอรมันแตกต่างกันอย่างไร?',
    'อธิบายสาเหตุของโรคหัดเยอรมันและการป้องกัน',
    'ทำไมโรคหัดเยอรมันจึงมีอันตรายกับหญิงตั้งครรภ์?',
    'ถ้าคนที่ฉีดวัคซีนป้องกันโรคหัดเยอรมันแล้ว จะมีโอกาสติดเชื้อหรือไม่?',
    'โรคหัดเยอรมันมีผลกระทบอย่างไรต่อระบบสาธารณสุขและเศรษฐกิจของประเทศ?',
]


def main():
    rag = AuthenticRAGv2(V2_VECTOR, V2_BM25_THAI, rerank=True)
    out = []
    for i, q in enumerate(QUESTIONS, 1):
        print(f"[{i}/{len(QUESTIONS)}] {q}")
        out.append(rag.search_for_question(q))
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "answers_v2.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved results/answers_v2.json")


if __name__ == "__main__":
    main()
