#!/usr/bin/env bash
#
# Multi-Model Multi-Category Evaluation Orchestrator
#
# This script evaluates multiple model checkpoints across different environment categories.
# Results are organized by model and category for easy analysis.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load environment categories
source "$SCRIPT_DIR/env_categories.sh"

# Configuration (can be overridden via environment variables)
MODEL_CHECKPOINTS_FILE="${MODEL_CHECKPOINTS_FILE:-$SCRIPT_DIR/model_checkpoints.txt}"
RESULTS_BASE_DIR="${RESULTS_BASE_DIR:-$SCRIPT_DIR/results}"
CATEGORIES=("tool-use" "code" "logic" "game")

# Evaluation parameters (can be overridden via environment variables)
START_DIFFICULTY="${START_DIFFICULTY:-0}"
END_DIFFICULTY="${END_DIFFICULTY:-1}"
NUM_DIFFICULTIES="${NUM_DIFFICULTIES:-1}"
RUNS_PER_DIFFICULTY="${RUNS_PER_DIFFICULTY:-16}"
ATTEMPTS_PER_INSTANCE="${ATTEMPTS_PER_INSTANCE:-1}"
GPUS="${GPUS:-0,1,2,3}"

# Check if model checkpoints file exists
if [[ ! -f "$MODEL_CHECKPOINTS_FILE" ]]; then
    echo "ERROR: Model checkpoints file not found: $MODEL_CHECKPOINTS_FILE" >&2
    exit 1
fi

# Read model checkpoints (skip comments and empty lines)
MODELS=()
while IFS= read -r line; do
    # Skip comments and empty lines
    [[ "$line" =~ ^#.*$ ]] && continue
    [[ -z "${line// }" ]] && continue
    MODELS+=("$line")
done < "$MODEL_CHECKPOINTS_FILE"

if [[ "${#MODELS[@]}" -eq 0 ]]; then
    echo "ERROR: No models found in $MODEL_CHECKPOINTS_FILE" >&2
    exit 1
fi

echo "========================================================================"
echo "Multi-Model Multi-Category Evaluation"
echo "========================================================================"
echo "Models to evaluate: ${#MODELS[@]}"
for model in "${MODELS[@]}"; do
    echo "  - $model"
done
echo ""
echo "Categories: ${CATEGORIES[*]}"
echo "Results directory: $RESULTS_BASE_DIR"
echo "========================================================================"
echo ""

# Create results base directory
mkdir -p "$RESULTS_BASE_DIR"

# Main evaluation loop
total_runs=$((${#MODELS[@]} * ${#CATEGORIES[@]}))
current_run=0

for MODEL_PATH in "${MODELS[@]}"; do
    # Extract model name from path and add timestamp
    BASE_MODEL_NAME=$(basename "$MODEL_PATH")

    # Support resume: use existing timestamp if provided, otherwise generate new one
    if [[ -n "${RESUME_TIMESTAMP:-}" ]]; then
        TIMESTAMP="$RESUME_TIMESTAMP"
        echo "📌 Resuming with timestamp: $TIMESTAMP"
    else
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        echo "🆕 New evaluation with timestamp: $TIMESTAMP"
    fi

    MODEL_NAME="${BASE_MODEL_NAME}_${TIMESTAMP}"

    echo ""
    echo "========================================================================"
    echo "Evaluating model: $BASE_MODEL_NAME"
    echo "Results directory: $MODEL_NAME"
    echo "Path: $MODEL_PATH"
    echo "========================================================================"

    for CATEGORY in "${CATEGORIES[@]}"; do
        current_run=$((current_run + 1))

        echo ""
        echo "--------------------------------------------------------------------"
        echo "[$current_run/$total_runs] Model: $MODEL_NAME | Category: $CATEGORY"
        echo "--------------------------------------------------------------------"

        # Get environments for this category
        ENVS=$(get_envs_for_category "$CATEGORY")
        if [[ -z "$ENVS" ]]; then
            echo "WARNING: No environments found for category: $CATEGORY" >&2
            continue
        fi

        # Convert array to comma-separated list
        ENV_LIST=$(echo "$ENVS" | tr ' ' ',')

        # Set output directory
        OUTPUT_DIR="$RESULTS_BASE_DIR/${MODEL_NAME}/${CATEGORY}"

        echo "Environments: $ENV_LIST"
        echo "Output: $OUTPUT_DIR"
        echo ""

        # Run evaluation
        START_DIFFICULTY="$START_DIFFICULTY" \
        END_DIFFICULTY="$END_DIFFICULTY" \
        NUM_DIFFICULTIES="$NUM_DIFFICULTIES" \
        RUNS_PER_DIFFICULTY="$RUNS_PER_DIFFICULTY" \
        ATTEMPTS_PER_INSTANCE="$ATTEMPTS_PER_INSTANCE" \
        MODEL_PATH="$MODEL_PATH" \
        GPUS="$GPUS" \
        ENV_LIST="$ENV_LIST" \
        OUTPUT_DIR="$OUTPUT_DIR" \
        bash "$SCRIPT_DIR/run_example_progressive_report_multi_gpu.sh"

        if [[ $? -eq 0 ]]; then
            echo "✓ Completed: $MODEL_NAME / $CATEGORY"
        else
            echo "✗ Failed: $MODEL_NAME / $CATEGORY" >&2
        fi
    done

    echo ""
    echo "========================================================================"
    echo "Completed model: $MODEL_NAME"
    echo "========================================================================"
done

echo ""
echo "========================================================================"
echo "All evaluations completed!"
echo "========================================================================"
echo "Results saved to: $RESULTS_BASE_DIR"
echo ""
echo "Next steps:"
echo "  1. Review results in: $RESULTS_BASE_DIR"
echo "  2. Run aggregation script: python aggregate_results.py"
echo "========================================================================"
