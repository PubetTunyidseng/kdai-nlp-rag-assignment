"""Ablation study C0 -> C3 on the synthetic test set (retrieval-level metrics).

C0 Baseline        : teacher's indices (chunk-only context, standard analyzer), no rerank
C1 +Doc context    : v2 vector + v2 BM25(standard)   -> isolates the context-generation fix
C2 +Thai analyzer  : v2 vector + v2 BM25(thai)       -> isolates the analyzer change
C3 +Reranker       : C2 + bge-reranker-v2-m3          -> isolates reranking

Gold is matched by chunk-content hash, so every index is scored on the same labels.
Outputs: results/per_query.jsonl, results/metrics.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "improved"))
from common import (BASE_BM25, BASE_VECTOR, RESULTS_DIR, ROOT, V2_BM25_STD, V2_BM25_THAI, V2_VECTOR,
                    chash, heading_of, read_jsonl, write_jsonl)
from rag_v2 import AuthenticRAGv2, get_embedder

CONFIGS = [
    {"id": "C0", "name": "Baseline (teacher)", "vector": BASE_VECTOR, "bm25": BASE_BM25, "rerank": False},
    {"id": "C1", "name": "+ Doc-aware context", "vector": V2_VECTOR, "bm25": V2_BM25_STD, "rerank": False},
    {"id": "C2", "name": "+ Thai analyzer", "vector": V2_VECTOR, "bm25": V2_BM25_THAI, "rerank": False},
    {"id": "C3", "name": "+ Reranker", "vector": V2_VECTOR, "bm25": V2_BM25_THAI, "rerank": True},
]
K = 5  # chunks sent to the LLM (teacher's k)


def rank_of(results, gold):
    """1-based rank of gold chunk in a result list, None if absent."""
    for i, (_d, _s, src) in enumerate(results, 1):
        if chash(src.get("content", "")) == gold:
            return i
    return None


def summarize(rows, key):
    n = len(rows)
    ranks = [r[key] for r in rows]
    hit = lambda k: sum(1 for x in ranks if x is not None and x <= k) / n
    return {"n": n, "hit@1": hit(1), "hit@3": hit(3), "hit@5": hit(5), "fail@5": 1 - hit(5),
            "mrr": sum(1 / x for x in ranks if x) / n, "in_pool": sum(1 for x in ranks if x) / n}


def dump_baseline_contexts():
    """Save the teacher's generated contexts (stored only in OpenSearch) for the README comparison."""
    from common import CACHE_DIR, opensearch_client
    p = CACHE_DIR / "contexts_chunk.jsonl"
    if p.exists():
        return
    hits = opensearch_client().search(index=BASE_BM25, body={"query": {"match_all": {}}, "size": 10000})["hits"]["hits"]
    write_jsonl(p, [{"hash": chash(h["_source"]["content"]), "doc_id": h["_id"],
                     "heading": heading_of(h["_source"]["content"]),
                     "context": h["_source"].get("contextualized_content", "")} for h in hits])


def main():
    dump_baseline_contexts()
    qs = read_jsonl(ROOT / "eval" / "testset.jsonl")
    if not qs:
        raise SystemExit("eval/testset.jsonl not found - run eval/make_testset.py first")
    print(f"{len(qs)} questions")
    emb = get_embedder()
    qvecs = {q["qid"]: v.tolist() for q, v in zip(qs, emb.encode([q["question"] for q in qs], batch_size=8))}

    rows, cache = [], {}
    for cfg in CONFIGS:
        rag = AuthenticRAGv2(cfg["vector"], cfg["bm25"], rerank=False, k=K)
        print(f"== {cfg['id']} {cfg['name']}")
        for q in qs:
            key = (cfg["vector"], cfg["bm25"], q["qid"])
            if key not in cache:
                cache[key] = rag.retrieve(q["question"], qvec=qvecs[q["qid"]])
            out = cache[key]
            final, rerank_ms = out["fused"], 0.0
            if cfg["rerank"]:
                import time
                t = time.perf_counter()
                final = rag.rerank_candidates(q["question"], out["fused"])
                rerank_ms = (time.perf_counter() - t) * 1000
            g = q["gold_hash"]
            rows.append({
                "config": cfg["id"], "qid": q["qid"], "type": q["type"], "question": q["question"],
                "gold_heading": q["gold_heading"], "source": q["source"],
                "rank_final": rank_of(final, g), "rank_sparse": rank_of(out["sparse"], g),
                "rank_dense": rank_of(out["dense"], g),
                "top1_heading": heading_of(final[0][2].get("content", "")) if final else "",
                "top5_headings": [heading_of(s.get("content", "")) for _d, _s, s in final[:K]],
                "retrieval_ms": out["retrieval_ms"], "rerank_ms": rerank_ms,
            })

    metrics = {"k": K, "n_questions": len(qs), "configs": []}
    for cfg in CONFIGS:
        r = [x for x in rows if x["config"] == cfg["id"]]
        m = {"id": cfg["id"], "name": cfg["name"], "overall": summarize(r, "rank_final"),
             "by_type": {t: summarize([x for x in r if x["type"] == t], "rank_final")
                         for t in ("exact", "paraphrase")},
             "sparse_only": summarize(r, "rank_sparse"), "dense_only": summarize(r, "rank_dense"),
             "latency_ms": sum(x["retrieval_ms"] + x["rerank_ms"] for x in r) / len(r)}
        metrics["configs"].append(m)
        o = m["overall"]
        print(f"{cfg['id']}: hit@1={o['hit@1']:.3f} hit@5={o['hit@5']:.3f} mrr={o['mrr']:.3f} "
              f"| sparse hit@5={m['sparse_only']['hit@5']:.3f} dense hit@5={m['dense_only']['hit@5']:.3f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    write_jsonl(RESULTS_DIR / "per_query.jsonl", rows)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved results/metrics.json, results/per_query.jsonl")


if __name__ == "__main__":
    main()
