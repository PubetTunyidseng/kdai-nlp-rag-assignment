"""Shared helpers: config, chunk loading (identical to teacher's), LLM client, OpenSearch client."""
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_FILES = ["1.md", "2.md", "44.md", "5555.md"]  # same 4 files as teacher's main()
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"


def load_dotenv(path=ROOT / ".env"):
    """Minimal .env loader (no extra dependency)."""
    if not Path(path).exists():
        return
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


load_dotenv()

# ---- Tunable parameters (override via .env / environment) ----
OPENSEARCH_HOST = os.environ.get("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.environ.get("OPENSEARCH_PORT", "9200"))
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen/qwen-2.5-72b-instruct")  # same as teacher
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")               # same as teacher
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_MAX_LEN = int(os.environ.get("RERANK_MAX_LEN", "1024"))
CONTEXT_WINDOW_CHARS = int(os.environ.get("CONTEXT_WINDOW_CHARS", "12000"))
LLM_WORKERS = int(os.environ.get("LLM_WORKERS", "4"))

# Index names
BASE_VECTOR = "anthropic-vector-index"   # created by teacher's authenticRAG.py
BASE_BM25 = "anthropic-bm25-index"
V2_VECTOR = "v2-vector-index"
V2_BM25_STD = "v2-bm25-standard"
V2_BM25_THAI = "v2-bm25-thai"


def chash(text: str) -> str:
    """Stable id of a chunk by its content (index-independent gold label)."""
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:12]


def heading_of(text: str) -> str:
    first = text.strip().splitlines()[0] if text.strip() else ""
    return first.lstrip("#").strip()[:80]


def load_chunks():
    """Exactly the teacher's chunking: MarkdownNodeParser(chunk_size=256) over the 4 files, same order.
    Returns (chunks, full_docs). chunk['doc_id'] == teacher's f"doc_{i}"."""
    from llama_index.core import Document
    from llama_index.core.node_parser import MarkdownNodeParser

    parser = MarkdownNodeParser(chunk_size=256)
    full_docs, llama_docs = {}, []
    for name in CORPUS_FILES:
        p = ROOT / "corpus_input" / name
        text = p.read_text(encoding="utf-8")
        full_docs[name] = text
        llama_docs.append(Document(text=text, metadata={"source": f"./corpus_input/{name}"}))
    nodes = parser.get_nodes_from_documents(llama_docs)
    chunks = []
    for i, n in enumerate(nodes):
        chunks.append({
            "idx": i,
            "doc_id": f"doc_{i}",
            "source": Path(n.metadata.get("source", "")).name,
            "content": n.text,
            "hash": chash(n.text),
            "heading": heading_of(n.text),
        })
    return chunks, full_docs


def opensearch_client():
    from opensearchpy import OpenSearch
    return OpenSearch(hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}], use_ssl=False, timeout=60)


_llm = None


def call_qwen(prompt, max_tokens=500, temperature=0.2, retries=4):
    """Same call as teacher's call_qwen_api (OpenRouter + Qwen), with retry."""
    global _llm
    if _llm is None:
        from openai import OpenAI
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("OPENROUTER_API_KEY not set (put it in .env)")
        _llm = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
    last = None
    for attempt in range(retries):
        try:
            r = _llm.chat.completions.create(
                model=QWEN_MODEL,
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content": prompt}],
                max_tokens=max_tokens, temperature=temperature)
            return (r.choices[0].message.content or "").strip()
        except Exception as e:  # rate limit / transient
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Qwen call failed: {last}")


def read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
