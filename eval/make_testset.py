"""Build a synthetic retrieval test set from the corpus chunks.

For every content chunk, Qwen writes 2 Thai questions whose answer is in that chunk:
  - exact      : reuses the chunk's own wording        (tests lexical / BM25 matching)
  - paraphrase : everyday wording, avoids chunk's words (tests semantic / dense matching)
Gold label = the chunk's content hash, so it works for any index built from the same chunks.
Output: eval/testset.jsonl (commit this file; it is reused by run_eval.py)
"""
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "improved"))
from common import LLM_WORKERS, ROOT, call_qwen, load_chunks, write_jsonl

OUT = ROOT / "eval" / "testset.jsonl"
SKIP_HEADINGS = ("เอกสารอ้างอิง", "เรื่องที่เกี่ยวข้อง", "เมดไทย")
MIN_CHARS = 200

PROMPT = """คุณกำลังสร้างชุดคำถามสำหรับทดสอบระบบค้นคืนเอกสาร (retrieval test set)

เอกสารเรื่อง: {title}
ข้อความ (chunk):
<chunk>
{chunk}
</chunk>

สร้างคำถามภาษาไทย 2 ข้อ ที่คำตอบอยู่ใน chunk นี้อย่างชัดเจน และต้องระบุชื่อโรคในคำถามเพื่อไม่ให้กำกวม
1) "exact": ถามโดยใช้คำศัพท์/วลีเดียวกับที่ปรากฏใน chunk
2) "paraphrase": ถามแบบภาษาพูดของคนทั่วไป หลีกเลี่ยงคำที่ซ้ำกับ chunk (ใช้คำพ้องความหมาย หรืออธิบายอาการแทนศัพท์แพทย์) แต่ยังต้องหมายถึงโรคเดียวกัน

ตอบเป็น JSON บรรทัดเดียวเท่านั้น: {{"exact": "...", "paraphrase": "..."}}"""


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return d if d.get("exact") and d.get("paraphrase") else None
    except json.JSONDecodeError:
        return None


def main():
    if OUT.exists() and "--force" not in sys.argv:
        print(f"{OUT.name} exists, skip (use --force to regenerate)")
        return
    chunks, full_docs = load_chunks()
    titles = {name: next((l.lstrip('#').strip() for l in t.splitlines() if l.startswith('#')), name)
              for name, t in full_docs.items()}
    targets = [c for c in chunks if len(c["content"]) >= MIN_CHARS
               and not any(s in c["heading"] for s in SKIP_HEADINGS)]
    print(f"{len(targets)}/{len(chunks)} chunks eligible")

    def gen(c):
        for _ in range(3):
            d = parse(call_qwen(PROMPT.format(title=titles[c["source"]], chunk=c["content"][:4000]),
                                max_tokens=300, temperature=0.3))
            if d:
                return c, d
        return c, None

    rows = []
    with ThreadPoolExecutor(LLM_WORKERS) as ex:
        for c, d in ex.map(gen, targets):
            if not d:
                print(f"  skip {c['doc_id']} (unparseable)")
                continue
            for qtype in ("exact", "paraphrase"):
                rows.append({"qid": f"{c['doc_id']}-{qtype}", "type": qtype, "question": d[qtype].strip(),
                             "gold_hash": c["hash"], "gold_doc_id": c["doc_id"],
                             "source": c["source"], "gold_heading": c["heading"]})
    write_jsonl(OUT, rows)
    print(f"saved {len(rows)} questions -> {OUT}")


if __name__ == "__main__":
    main()
