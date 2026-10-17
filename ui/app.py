"""RAG Lab UI: ask one question, compare Baseline (teacher) vs v2 side by side.

Run:  python ui/app.py            -> http://localhost:7860
      python ui/app.py --share    -> also prints a temporary public link (gradio share)

Uses the indices already built by run_all.sh. Retrieval is free (local models);
"generate answer" calls Qwen via OpenRouter (costs a little per question).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "improved"))
from common import (BASE_BM25, BASE_VECTOR, RESULTS_DIR, V2_BM25_STD, V2_BM25_THAI, V2_VECTOR,
                    heading_of, opensearch_client, read_jsonl)
from rag_v2 import AuthenticRAGv2, get_embedder

K = 5
SYSTEMS = {
    "baseline": {"label": "Baseline (โค้ดอาจารย์)", "vector": BASE_VECTOR, "bm25": BASE_BM25, "rerank": False},
    "v2": {"label": "v2 (doc context + Thai analyzer + reranker)", "vector": V2_VECTOR, "bm25": V2_BM25_THAI,
           "rerank": True},
}
EXAMPLES = [
    "กรดไหลย้อนรักษายังไง",
    "ต้อกระจกเกิดจากอะไร",
    "ทำไมโรคหัดเยอรมันจึงมีอันตรายกับหญิงตั้งครรภ์?",
    "ท้องเสียหนักมากถ่ายเป็นน้ำซาวข้าว เป็นโรคอะไร ป้องกันยังไง",
    "ผ่าตัดต้อกระจกแล้วต้องดูแลตัวเองอย่างไร",
]

_rags = {}


def rag(name):
    if name not in _rags:
        s = SYSTEMS[name]
        _rags[name] = AuthenticRAGv2(s["vector"], s["bm25"], rerank=s["rerank"], k=K)
    return _rags[name]


def rank_map(results):
    return {d: i for i, (d, _s, _src) in enumerate(results, 1)}


def retrieval_table(out):
    """Top-k after the final stage, with each chunk's rank in every earlier stage (traceable)."""
    sp, de, fu = rank_map(out["sparse"]), rank_map(out["dense"]), rank_map(out["fused"])
    rows = []
    for i, (d, score, src) in enumerate(out["final"][:K], 1):
        rows.append([i, heading_of(src.get("content", "")), d,
                     sp.get(d, "–"), de.get(d, "–"), fu.get(d, "–"), round(float(score), 4)])
    return rows


def stage_summary(name, out):
    bm25_n = len(out["sparse"])
    warn = " ⚠️ BM25 ไม่เจอเอกสารเลย ผลมาจาก dense ขาเดียว" if bm25_n == 0 else ""
    rr = f" · rerank {out['rerank_ms']:.0f} ms" if SYSTEMS[name]["rerank"] else ""
    return (f"BM25 เจอ **{bm25_n}** ชิ้น · Dense เจอ **{len(out['dense'])}** ชิ้น · "
            f"หลัง RRF **{len(out['fused'])}** ชิ้น · retrieval {out['retrieval_ms']:.0f} ms{rr}{warn}")


def answer_from(r, question, out):
    context = ""
    for d, _s, src in out["final"][:K]:
        context += f"Content: {src.get('content', '')}\nContext: {r.ctx.get(d, '')}\n\n"
    words = context.split()
    if len(words) * 1.3 > 4000:
        context = " ".join(words[: int(4000 / 1.3)]) + " ...(truncated)"
    return r.generate_response(question, context)


HEADERS = ["#", "หัวข้อ chunk", "doc_id", "อันดับใน BM25", "อันดับใน Dense", "อันดับหลัง RRF", "คะแนนสุดท้าย"]


def compare(question, gen_answer):
    question = (question or "").strip()
    if not question:
        raise gr.Error("พิมพ์คำถามก่อน")
    qvec = get_embedder().encode(question).tolist()
    outputs = []
    for name in ("baseline", "v2"):
        r = rag(name)
        t0 = time.perf_counter()
        out = r.retrieve(question, qvec=qvec)
        ans = ""
        if gen_answer:
            try:
                ans = answer_from(r, question, out)
            except Exception as e:  # noqa: BLE001
                ans = f"_สร้างคำตอบไม่สำเร็จ: {e}_"
        total = (time.perf_counter() - t0) * 1000
        outputs += [stage_summary(name, out) + f" · รวม {total:.0f} ms", retrieval_table(out),
                    ans or "_ไม่ได้ติ๊กให้สร้างคำตอบ (ดูแค่ผลค้นหา ไม่เสียค่า API)_"]
    return outputs


def tokenize(text):
    text = (text or "").strip()
    if not text:
        return "", ""
    osc = opensearch_client()
    res = []
    for idx in (V2_BM25_STD, V2_BM25_THAI):
        toks = [t["token"] for t in osc.indices.analyze(index=idx, body={"field": "content", "text": text})["tokens"]]
        res.append(" · ".join(toks) + f"\n\n({len(toks)} tokens)")
    return res[0], res[1]


def index_status():
    osc = opensearch_client()
    lines = ["| Index | จำนวน chunks |", "|---|---|"]
    for idx in (BASE_VECTOR, BASE_BM25, V2_VECTOR, V2_BM25_STD, V2_BM25_THAI):
        try:
            n = osc.count(index=idx)["count"]
        except Exception:  # noqa: BLE001
            n = "ยังไม่ได้สร้าง"
        lines.append(f"| `{idx}` | {n} |")
    return "\n".join(lines)


def eval_tab_data():
    m_path = RESULTS_DIR / "metrics.json"
    if not m_path.exists():
        return "_ยังไม่มีผล รัน ./run_all.sh ก่อน_", []
    m = json.loads(m_path.read_text(encoding="utf-8"))
    rows = []
    by = {}
    for r in read_jsonl(RESULTS_DIR / "per_query.jsonl"):
        by.setdefault(r["qid"], {})[r["config"]] = r
    for qid, v in by.items():
        c0, c3 = v.get("C0"), v.get("C3")
        if not (c0 and c3):
            continue
        rows.append([c0["type"], c0["question"], c0["gold_heading"],
                     c0["rank_final"] or "ไม่เจอ", v["C1"]["rank_final"] or "ไม่เจอ",
                     v["C2"]["rank_final"] or "ไม่เจอ", c3["rank_final"] or "ไม่เจอ"])
    summary = ["| Config | Hit@1 | Hit@5 | Fail@5 | MRR |", "|---|---|---|---|---|"]
    for c in m["configs"]:
        o = c["overall"]
        summary.append(f"| {c['id']} {c['name']} | {o['hit@1']*100:.1f}% | {o['hit@5']*100:.1f}% | "
                       f"{o['fail@5']*100:.1f}% | {o['mrr']:.3f} |")
    return f"ทดสอบบน {m['n_questions']} คำถาม\n\n" + "\n".join(summary), rows


def build():
    with gr.Blocks(title="RAG Lab: Baseline vs v2") as demo:
        gr.Markdown("# 🔬 RAG Lab: Baseline vs v2\nพิมพ์คำถามเดียว แล้วดูว่าระบบเดิมของอาจารย์กับระบบที่ปรับปรุง "
                    "ดึง chunk อะไรมาให้ LLM บ้าง ตารางบอกด้วยว่าแต่ละ chunk ได้อันดับเท่าไรในแต่ละขั้น")
        with gr.Tab("เปรียบเทียบ"):
            q = gr.Textbox(label="คำถาม", placeholder="เช่น กรดไหลย้อนรักษายังไง", lines=1)
            gr.Examples(EXAMPLES, inputs=q)
            with gr.Row():
                gen = gr.Checkbox(label="สร้างคำตอบด้วย Qwen ด้วย (เสียค่า API เล็กน้อย)", value=False)
                btn = gr.Button("ค้นหาและเปรียบเทียบ", variant="primary")
            outs = []
            with gr.Row():
                for name in ("baseline", "v2"):
                    with gr.Column():
                        gr.Markdown(f"### {SYSTEMS[name]['label']}")
                        s = gr.Markdown()
                        t = gr.Dataframe(headers=HEADERS, wrap=True, interactive=False)
                        a = gr.Markdown()
                        outs += [s, t, a]
            btn.click(compare, [q, gen], outs)
            q.submit(compare, [q, gen], outs)
        with gr.Tab("ตัดคำ: standard vs thai"):
            gr.Markdown("ดูว่า OpenSearch แบ่งข้อความไทยเป็น token อย่างไร BM25 จับคู่ได้เฉพาะ token ที่ตรงกันทุกตัวอักษร")
            tx = gr.Textbox(label="ข้อความ", value="อาการของโรคกรดไหลย้อน")
            tbtn = gr.Button("ตัดคำ")
            with gr.Row():
                std = gr.Textbox(label="standard analyzer (baseline)", lines=4)
                thai = gr.Textbox(label="thai analyzer (v2)", lines=4)
            tbtn.click(tokenize, tx, [std, thai])
        with gr.Tab("ผลการทดลอง"):
            summary, rows = eval_tab_data()
            gr.Markdown(summary)
            chart = RESULTS_DIR / "ablation_chart.png"
            if chart.exists():
                gr.Image(str(chart), show_label=False)
            gr.Markdown("อันดับของ chunk ที่ถูก (gold) ในแต่ละ config ต่อคำถาม: 1 = ดีที่สุด, ไม่เจอ = หลุดจาก candidate pool")
            gr.Dataframe(rows, headers=["ประเภท", "คำถาม", "gold chunk", "C0", "C1", "C2", "C3"], wrap=True,
                         interactive=False)
        with gr.Tab("สถานะระบบ"):
            st = gr.Markdown()
            gr.Button("รีเฟรช").click(index_status, None, st)
            demo.load(index_status, None, st)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a temporary public gradio link")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    print("Loading models (first time may take a minute)...")
    get_embedder()
    build().launch(server_port=args.port, share=args.share)
