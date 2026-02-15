#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash eval.sh [MODEL] [TEST_CATEGORY]
#
# Examples:
#   bash eval.sh qwen3-4b-instruct simple_python
#   SCORE_DIR=score_tmp bash eval.sh qwen3-4b-instruct simple_python
#   PARTIAL_EVAL=1 bash eval.sh qwen3-4b-instruct simple_python

MODEL="${1:-qwen3-4b-instruct}"
TEST_CATEGORY="${2:-simple_python}"

# These paths are interpreted relative to the BFCL repo root
# (`.../gorilla/berkeley-function-call-leaderboard`) by the `bfcl` CLI.
RESULT_DIR="${RESULT_DIR:-result}"
SCORE_DIR="${SCORE_DIR:-score}"

export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export BFCL_EXECUTE_TOOLS="${BFCL_EXECUTE_TOOLS:-1}"

EVAL_ARGS=()
if [[ "${PARTIAL_EVAL:-0}" == "1" ]]; then
  EVAL_ARGS+=(--partial-eval)
fi

bfcl evaluate \
  --model "${MODEL}" \
  --test-category "${TEST_CATEGORY}" \
  --result-dir "${RESULT_DIR}" \
  --score-dir "${SCORE_DIR}" \
  "${EVAL_ARGS[@]}"

bfcl scores --score-dir "${SCORE_DIR}"

echo
echo "Score artifacts written under: ${SCORE_DIR}/"
echo "Main table CSV: ${SCORE_DIR}/data_overall.csv"
