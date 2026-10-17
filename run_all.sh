#!/usr/bin/env bash
# One-shot pipeline: baseline (teacher) -> improved ingest -> test set -> ablation -> report
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] && set -a && . ./.env && set +a
: "${OPENROUTER_API_KEY:?Put OPENROUTER_API_KEY in .env}"
mkdir -p results/logs

curl -s "http://${OPENSEARCH_HOST:-localhost}:${OPENSEARCH_PORT:-9200}" >/dev/null \
  || { echo "OpenSearch is not running on :9200 (see README step 2)"; exit 1; }

echo "== [1/6] Baseline ingest (teacher's authenticRAG.py, unmodified)"
if ! python baseline/authenticRAG.py 2>&1 | tee results/logs/1_baseline_ingest.log; then
  echo "teacher's script failed -> rebuilding identical baseline indices with --reproduce-baseline"
  python improved/ingest_v2.py --reproduce-baseline 2>&1 | tee -a results/logs/1_baseline_ingest.log
fi

echo "== [2/6] Baseline Q&A (teacher's onlysearchAuthenticRAG.py, unmodified)"
python baseline/onlysearchAuthenticRAG.py 2>&1 | tee results/logs/2_baseline_search.log || true
[ -f authentic_rag_search_results.json ] && mv authentic_rag_search_results.json results/baseline_answers.json

echo "== [3/6] Improved ingest (doc-aware context + standard/thai BM25 indices)"
python improved/ingest_v2.py 2>&1 | tee results/logs/3_v2_ingest.log

echo "== [4/6] Synthetic test set"
python eval/make_testset.py 2>&1 | tee results/logs/4_testset.log

echo "== [5/6] Ablation C0..C3 + analyzer inspection"
python eval/run_eval.py 2>&1 | tee results/logs/5_eval.log
python eval/inspect_analyzers.py > results/logs/5b_analyzers.log 2>&1

echo "== [6/6] v2 answers for teacher's questions + report"
python improved/ask_v2.py 2>&1 | tee results/logs/6_ask_v2.log
python eval/report.py
echo "DONE -> README.md updated"
