"""Improved ingestion (Hands-on 1 v2).

Changes vs teacher's authenticRAG.py:
  1) Doc-aware contextual embedding: the teacher's code passes the *chunk itself* as <document>
     (full_doc_content = doc.page_content), so the LLM never sees the whole document and the
     "context" is just a paraphrase of the chunk. Here the LLM gets the document title + outline
     (all headings) + a text window around the chunk, so it can situate the chunk properly.
  2) BM25 index with OpenSearch's built-in `thai` analyzer (dictionary word segmentation)
     next to the `standard` one, so both can be compared in the ablation.
Chunking, embedding model, prompt wording and the vector-index schema are kept identical.

Usage:
  python improved/ingest_v2.py                      # build v2 indices (fulldoc context)
  python improved/ingest_v2.py --reproduce-baseline # rebuild teacher's anthropic-* indices
                                                      (fallback if teacher's script won't run)
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BASE_BM25, BASE_VECTOR, CACHE_DIR, CONTEXT_WINDOW_CHARS, EMBED_MODEL, LLM_WORKERS,
                    V2_BM25_STD, V2_BM25_THAI, V2_VECTOR, call_qwen, load_chunks, opensearch_client,
                    read_jsonl, write_jsonl)

# Teacher's (=Anthropic's) prompt, unchanged
CONTEXT_PROMPT = """<document>
{document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval. Answer only with the succinct context and nothing else.
"""


def doc_window(full: str, chunk: str, max_chars: int = CONTEXT_WINDOW_CHARS) -> str:
    """Document representation that fits the LLM context: whole doc if small, else
    title + outline + window around the chunk."""
    if len(full) <= max_chars:
        return full
    headings = [l.strip() for l in full.splitlines() if l.lstrip().startswith("#")]
    outline = "\n".join(headings)
    title = headings[0] if headings else ""
    pos = full.find(chunk[:200])
    pos = max(pos, 0)
    budget = max(max_chars - len(outline) - len(title) - 100, len(chunk) + 1000)
    pad = max((budget - len(chunk)) // 2, 500)
    window = full[max(0, pos - pad): min(len(full), pos + len(chunk) + pad)]
    return f"{title}\n\n[Document outline]\n{outline}\n\n[Text around the chunk]\n...{window}..."


def build_contexts(chunks, full_docs, mode):
    """mode='chunk' reproduces the teacher (document = chunk); mode='fulldoc' is the fix.
    Cached in cache/contexts_<mode>.jsonl keyed by chunk hash."""
    cache_path = CACHE_DIR / f"contexts_{mode}.jsonl"
    cached = {r["hash"]: r["context"] for r in read_jsonl(cache_path)}
    todo = [c for c in chunks if c["hash"] not in cached]
    print(f"[context:{mode}] cached={len(cached)} to_generate={len(todo)}")

    def gen(c):
        document = c["content"] if mode == "chunk" else doc_window(full_docs[c["source"]], c["content"])
        ctx = call_qwen(CONTEXT_PROMPT.format(document=document, chunk=c["content"]),
                        max_tokens=100, temperature=0.1)  # same params as teacher
        return c["hash"], ctx

    with ThreadPoolExecutor(LLM_WORKERS) as ex:
        for n, (h, ctx) in enumerate(ex.map(gen, todo), 1):
            cached[h] = ctx
            if n % 10 == 0 or n == len(todo):
                print(f"  generated {n}/{len(todo)}")
    write_jsonl(cache_path, [{"hash": c["hash"], "doc_id": c["doc_id"], "heading": c["heading"],
                              "context": cached[c["hash"]]} for c in chunks])
    return {c["hash"]: cached[c["hash"]] for c in chunks}


def recreate(os_client, name, body):
    if os_client.indices.exists(index=name):
        os_client.indices.delete(index=name)
    os_client.indices.create(index=name, body=body)


def vector_body(dim):
    # identical schema to teacher
    return {"settings": {"index.knn": True, "index.knn.space_type": "cosinesimil"},
            "mappings": {"properties": {"embedding": {"type": "knn_vector", "dimension": dim},
                                        "doc_id": {"type": "keyword"}, "content": {"type": "text"},
                                        "chunk_id": {"type": "keyword"}}}}


def bm25_body(analyzer):
    return {"settings": {"analysis": {"analyzer": {"default": {"type": analyzer}}},
                         "similarity": {"default": {"type": "BM25"}},
                         "index.queries.cache.enabled": False},
            "mappings": {"properties": {"content": {"type": "text", "analyzer": analyzer},
                                        "contextualized_content": {"type": "text", "analyzer": analyzer},
                                        "doc_id": {"type": "keyword"}, "chunk_id": {"type": "keyword"},
                                        "original_index": {"type": "integer", "index": False}}}}


def bulk(os_client, index, rows):
    body = []
    for _id, doc in rows:
        body += [{"index": {"_index": index, "_id": _id}}, doc]
    r = os_client.bulk(body=body)
    if r.get("errors"):
        raise RuntimeError(f"bulk errors in {index}")
    os_client.indices.refresh(index=index)
    print(f"  indexed {len(rows)} docs -> {index}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reproduce-baseline", action="store_true")
    args = ap.parse_args()

    chunks, full_docs = load_chunks()
    print(f"Loaded {len(chunks)} chunks (teacher's MarkdownNodeParser)")
    mode = "chunk" if args.reproduce_baseline else "fulldoc"
    contexts = build_contexts(chunks, full_docs, mode)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)
    texts = [f"{c['content']}\n\n{contexts[c['hash']]}" for c in chunks]  # same as teacher
    embs = model.encode(texts, batch_size=4, show_progress_bar=True)

    osc = opensearch_client()
    if args.reproduce_baseline:
        vec_name, bm25_targets = BASE_VECTOR, [(BASE_BM25, "standard")]
    else:
        vec_name, bm25_targets = V2_VECTOR, [(V2_BM25_STD, "standard"), (V2_BM25_THAI, "thai")]

    recreate(osc, vec_name, vector_body(len(embs[0])))
    bulk(osc, vec_name, [(c["doc_id"], {"embedding": e.tolist(), "doc_id": c["doc_id"],
                                       "content": c["content"], "chunk_id": c["idx"]})
                         for c, e in zip(chunks, embs)])
    for name, analyzer in bm25_targets:
        recreate(osc, name, bm25_body(analyzer))
        bulk(osc, name, [(c["doc_id"], {"content": c["content"], "contextualized_content": contexts[c["hash"]],
                                       "doc_id": c["doc_id"], "chunk_id": c["idx"], "original_index": c["idx"]})
                         for c in chunks])
    print("Done.")


if __name__ == "__main__":
    main()
