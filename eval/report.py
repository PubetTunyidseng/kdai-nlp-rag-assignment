"""Turn results/*.json into a chart + markdown tables and inject them into README.md
between <!-- NAME:START --> / <!-- NAME:END --> markers. Safe to re-run."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "improved"))
from common import CACHE_DIR, RESULTS_DIR, ROOT, read_jsonl

README = ROOT / "README.md"
DISEASES = ("หัดเยอรมัน", "อหิวาต", "ต้อกระจก", "กรดไหลย้อน")


def pct(x):
    return f"{x * 100:.1f}%"


def chart(metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfgs = metrics["configs"]
    labels = [f"{c['id']}\n{c['name']}" for c in cfgs]
    fail = [c["overall"]["fail@5"] * 100 for c in cfgs]
    hit1 = [c["overall"]["hit@1"] * 100 for c in cfgs]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    colors = ["#c9c6bd", "#e8dcc4", "#d9b99b", "#c8875f"]
    for ax, vals, title in ((axes[0], fail, "Failed retrievals @5 (lower is better)"),
                            (axes[1], hit1, "Hit@1 (higher is better)")):
        bars = ax.bar(labels, vals, color=colors[: len(vals)])
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("%")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.suptitle(f"Ablation on {metrics['n_questions']} synthetic Thai questions (k={metrics['k']})", fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "ablation_chart.png", dpi=150)


def ablation_md(m):
    lines = ["| Config | Hit@1 | Hit@3 | Hit@5 | Fail@5 | MRR | Gold in candidate pool | Latency/query |",
             "|---|---|---|---|---|---|---|---|"]
    for c in m["configs"]:
        o = c["overall"]
        lines.append(f"| **{c['id']}** {c['name']} | {pct(o['hit@1'])} | {pct(o['hit@3'])} | {pct(o['hit@5'])} | "
                     f"{pct(o['fail@5'])} | {o['mrr']:.3f} | {pct(o['in_pool'])} | {c['latency_ms']:.0f} ms |")
    return "\n".join(lines) + "\n\n![ablation](results/ablation_chart.png)"


def component_md(m):
    lines = ["| Config | BM25-only Hit@5 | BM25-only MRR | Dense-only Hit@5 | Dense-only MRR |", "|---|---|---|---|---|"]
    seen = set()
    for c in m["configs"]:
        key = (c["sparse_only"]["hit@5"], c["dense_only"]["hit@5"], c["sparse_only"]["mrr"])
        if c["id"] == "C3" and key in seen:
            continue  # same retrieval as C2
        seen.add(key)
        s, d = c["sparse_only"], c["dense_only"]
        lines.append(f"| {c['id']} {c['name']} | {pct(s['hit@5'])} | {s['mrr']:.3f} | {pct(d['hit@5'])} | {d['mrr']:.3f} |")
    return "\n".join(lines)


def bytype_md(m):
    lines = ["| Config | exact Hit@5 | exact MRR | paraphrase Hit@5 | paraphrase MRR |", "|---|---|---|---|---|"]
    for c in m["configs"]:
        e, p = c["by_type"]["exact"], c["by_type"]["paraphrase"]
        lines.append(f"| {c['id']} {c['name']} | {pct(e['hit@5'])} | {e['mrr']:.3f} | {pct(p['hit@5'])} | {p['mrr']:.3f} |")
    n_e, n_p = m["configs"][0]["by_type"]["exact"]["n"], m["configs"][0]["by_type"]["paraphrase"]["n"]
    return "\n".join(lines) + f"\n\n(exact = {n_e} questions, paraphrase = {n_p} questions)"


def tokens_md():
    p = RESULTS_DIR / "tokenization.json"
    if not p.exists():
        return "_run eval/inspect_analyzers.py_"
    t = json.loads(p.read_text(encoding="utf-8"))
    lines = ["| Text | `standard` analyzer | `thai` analyzer |", "|---|---|---|"]
    for r in t["tokens"]:
        lines.append(f"| {r['text']} | {' · '.join(r['standard'])} | {' · '.join(r['thai'])} |")
    lines += ["", "BM25-only top-3 for the same query:", "",
              "| Query | `standard` index | `thai` index |", "|---|---|---|"]
    for r in t["bm25_top3"]:
        lines.append(f"| {r['query']} | {'<br>'.join(r['standard']) or '(no hit)'} | {'<br>'.join(r['thai']) or '(no hit)'} |")
    return "\n".join(lines)


def context_md():
    a = {r["hash"]: r for r in read_jsonl(CACHE_DIR / "contexts_chunk.jsonl")}
    b = {r["hash"]: r for r in read_jsonl(CACHE_DIR / "contexts_fulldoc.jsonl")}
    common = [h for h in b if h in a]
    if not common:
        return ("_Baseline contexts are stored inside OpenSearch (`anthropic-bm25-index`); "
                "see results/context_compare.md after running `python eval/report.py`._")
    picks = [h for h in common if not any(d in b[h]["heading"] for d in DISEASES)][:3] or common[:3]
    lines = ["| Chunk heading | Baseline context (document = chunk) | v2 context (document-aware) |", "|---|---|---|"]
    for h in picks:
        clean = lambda s: s.replace("\n", " ").replace("|", "/")
        lines.append(f"| {a[h]['heading']} | {clean(a[h]['context'])} | {clean(b[h]['context'])} |")
    return "\n".join(lines)


def examples_md():
    rows = read_jsonl(RESULTS_DIR / "per_query.jsonl")
    by = {}
    for r in rows:
        by.setdefault(r["qid"], {})[r["config"]] = r
    hit = lambda r: r["rank_final"] is not None and r["rank_final"] <= 5
    wins = [v for v in by.values() if not hit(v["C0"]) and hit(v["C3"])]
    losses = [v for v in by.values() if hit(v["C0"]) and not hit(v["C3"])]
    lines = [f"- Questions fixed by v2 (C0 miss@5 → C3 hit@5): **{len(wins)}**",
             f"- Regressions (C0 hit@5 → C3 miss@5): **{len(losses)}**", ""]
    for v in sorted(wins, key=lambda v: v["C3"]["rank_final"])[:3]:
        c0, c3 = v["C0"], v["C3"]
        lines += [f"**Q ({c0['type']}):** {c0['question']}  ",
                  f"Gold chunk: `{c0['gold_heading']}` ({c0['source']})  ",
                  f"C0 top-1: `{c0['top1_heading']}` → gold rank = {c0['rank_final'] or 'not retrieved'}  ",
                  f"C3 top-1: `{c3['top1_heading']}` → gold rank = {c3['rank_final']}", ""]
    if losses:
        v = losses[0]
        lines += ["Example regression (reported for honesty):  ",
                  f"**Q ({v['C0']['type']}):** {v['C0']['question']} — gold `{v['C0']['gold_heading']}`, "
                  f"C3 top-1 `{v['C3']['top1_heading']}`", ""]
    return "\n".join(lines)


def answers_compare():
    base_p, v2_p = RESULTS_DIR / "baseline_answers.json", RESULTS_DIR / "answers_v2.json"
    if not (base_p.exists() and v2_p.exists()):
        return
    base, v2 = json.loads(base_p.read_text(encoding="utf-8")), json.loads(v2_p.read_text(encoding="utf-8"))
    head = lambda r: r["content"].strip().splitlines()[0].lstrip("#").strip()[:60] if r.get("content") else ""
    md = ["# Answers: baseline vs v2 (same generator, prompt, temperature)\n"]
    for b, n in zip(base, v2):
        md += [f"## {b['question']}\n",
               "| # | Baseline top-5 chunk | v2 top-5 chunk |", "|---|---|---|"]
        for i in range(max(len(b["results"]), len(n["results"]))):
            bh = head(b["results"][i]) if i < len(b["results"]) else ""
            nh = head(n["results"][i]) if i < len(n["results"]) else ""
            md.append(f"| {i + 1} | {bh} | {nh} |")
        md += ["", "**Baseline answer**\n", b["answer"], "", "**v2 answer**\n", n["answer"], "\n---\n"]
    (RESULTS_DIR / "answers_compare.md").write_text("\n".join(md), encoding="utf-8")


def inject(text, name, body):
    pat = re.compile(rf"(<!-- {name}:START -->)(.*?)(<!-- {name}:END -->)", re.S)
    return pat.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", text)


def main():
    m = json.loads((RESULTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    chart(m)
    answers_compare()
    readme = README.read_text(encoding="utf-8")
    for name, body in (("ABLATION", ablation_md(m)), ("COMPONENT", component_md(m)), ("BYTYPE", bytype_md(m)),
                       ("TOKENS", tokens_md()), ("CONTEXT", context_md()), ("EXAMPLES", examples_md())):
        readme = inject(readme, name, body)
    README.write_text(readme, encoding="utf-8")
    print("README.md updated, results/ablation_chart.png written")


if __name__ == "__main__":
    main()
