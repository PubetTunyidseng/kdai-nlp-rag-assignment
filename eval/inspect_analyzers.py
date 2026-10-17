"""Evidence for the analyzer change: how OpenSearch tokenizes Thai with `standard` vs `thai`,
and what BM25-only search returns for the same query on each index.
Output: results/tokenization.json"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "improved"))
from common import RESULTS_DIR, V2_BM25_STD, V2_BM25_THAI, heading_of, opensearch_client

SAMPLES = [
    "และวิธีการรักษา",  # the token shown on teacher's slide p.24
    "อาการของโรคกรดไหลย้อน",
    "ผ่าตัดต้อกระจกต้องทำอย่างไร",
]
QUERIES = ["กรดไหลย้อนรักษายังไง", "ต้อกระจกเกิดจากอะไร"]


def main():
    osc = opensearch_client()
    out = {"tokens": [], "bm25_top3": []}
    for text in SAMPLES:
        row = {"text": text}
        for name, idx in (("standard", V2_BM25_STD), ("thai", V2_BM25_THAI)):
            r = osc.indices.analyze(index=idx, body={"field": "content", "text": text})
            row[name] = [t["token"] for t in r["tokens"]]
        out["tokens"].append(row)
    for q in QUERIES:
        row = {"query": q}
        for name, idx in (("standard", V2_BM25_STD), ("thai", V2_BM25_THAI)):
            r = osc.search(index=idx, body={"query": {"multi_match": {
                "query": q, "fields": ["content", "contextualized_content"], "type": "best_fields"}}, "size": 3})
            row[name] = [f"{heading_of(h['_source']['content'])} ({h['_score']:.2f})" for h in r["hits"]["hits"]]
        out["bm25_top3"].append(row)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "tokenization.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
