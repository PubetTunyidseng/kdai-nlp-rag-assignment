"""Improved search side (Hands-on 2 v2).

Same hybrid retrieval as teacher's onlysearchAuthenticRAG.py (BM25 multi_match + kNN, RRF k=60,
k*2 candidates per retriever), plus an optional cross-encoder reranker (BAAI/bge-reranker-v2-m3)
that re-scores the fused candidate pool before the top-k go to the LLM.
The generator (Qwen via OpenRouter), its prompt and temperature are unchanged.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EMBED_MODEL, RERANK_MAX_LEN, RERANK_MODEL, call_qwen, opensearch_client

_embedder = None
_reranker = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANK_MODEL, max_length=RERANK_MAX_LEN)
    return _reranker


class AuthenticRAGv2:
    def __init__(self, vector_index, bm25_index, rerank=False, k=5, rrf_k=60):
        self.os = opensearch_client()
        self.vector_index, self.bm25_index = vector_index, bm25_index
        self.rerank, self.k, self.rrf_k = rerank, k, rrf_k
        # id -> generated context (vector index doesn't store it; needed for reranker + answer)
        hits = self.os.search(index=bm25_index, body={"query": {"match_all": {}}, "size": 10000})["hits"]["hits"]
        self.ctx = {h["_id"]: h["_source"].get("contextualized_content", "") for h in hits}

    # ---- retrievers (same queries as teacher) ----
    def sparse_search(self, query, k):
        r = self.os.search(index=self.bm25_index, body={
            "query": {"multi_match": {"query": query, "fields": ["content", "contextualized_content"],
                                      "type": "best_fields"}}, "size": k})
        return [(h["_id"], h["_score"], h["_source"]) for h in r["hits"]["hits"]]

    def dense_search(self, query, k, qvec=None):
        qvec = qvec if qvec is not None else get_embedder().encode(query).tolist()
        r = self.os.search(index=self.vector_index, body={
            "query": {"knn": {"embedding": {"vector": qvec, "k": k}}}, "size": k})
        return [(h["_id"], h["_score"], h["_source"]) for h in r["hits"]["hits"]]

    @staticmethod
    def rrf_fusion(rankings, k=60):
        fused, src = {}, {}
        for results in rankings:
            for rank, (doc_id, _s, source) in enumerate(results):
                fused.setdefault(doc_id, 0.0)
                src.setdefault(doc_id, source)
                fused[doc_id] += 1.0 / (k + rank)
        return [(d, s, src[d]) for d, s in sorted(fused.items(), key=lambda x: x[1], reverse=True)]

    def rerank_candidates(self, query, candidates):
        pairs = [(query, f"{self.ctx.get(d, '')}\n{src.get('content', '')}") for d, _s, src in candidates]
        scores = get_reranker().predict(pairs, batch_size=8)
        order = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)
        return [(d, float(s), src) for (d, _o, src), s in order]

    def retrieve(self, query, qvec=None):
        """Returns dict with every stage so the evaluator can trace where a miss happens."""
        t0 = time.perf_counter()
        sparse = self.sparse_search(query, self.k * 2)
        dense = self.dense_search(query, self.k * 2, qvec)
        fused = self.rrf_fusion([sparse, dense], self.rrf_k)
        t1 = time.perf_counter()
        final = self.rerank_candidates(query, fused) if self.rerank else fused
        t2 = time.perf_counter()
        return {"sparse": sparse, "dense": dense, "fused": fused, "final": final,
                "retrieval_ms": (t1 - t0) * 1000, "rerank_ms": (t2 - t1) * 1000}

    # ---- generation: identical prompt/model/temperature to teacher ----
    def generate_response(self, question, context):
        prompt = f"""You are a question answering assistant. Answer the question as truthful and helpful as possible.

Context information:
{context}

Question: {question}"""
        return call_qwen(prompt, max_tokens=1000, temperature=0.6)

    def search_for_question(self, question):
        out = self.retrieve(question)
        top = out["final"][: self.k]
        context = ""
        for d, _s, src in top:
            context += f"Content: {src.get('content', '')}\nContext: {self.ctx.get(d, '')}\n\n"
        words = context.split()  # teacher's truncate_context
        if len(words) * 1.3 > 4000:
            context = " ".join(words[: int(4000 / 1.3)]) + " ...(truncated)"
        answer = self.generate_response(question, context)
        return {"question": question, "answer": answer,
                "results": [{"doc_id": d, "score": s, "content": src.get("content", ""),
                             "context": self.ctx.get(d, "")} for d, s, src in top]}
