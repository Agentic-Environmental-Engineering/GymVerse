#!/usr/bin/env bash
#
# Clean multi-GPU runner for evaluating GEM example environments with:
#   EnvEval/difficulty/report_progressive.py
#
# - No env directory traversal: use explicit ENVS list.
# - One python process per GPU (model loaded once per process).
# - Output defaults to: ./results/... (relative to where you run the script).

# MODEL_PATH=xxx START_DIFFICULTY=0 END_DIFFICULTY=1 NUM_DIFFICULTIES=1 RUNS_PER_DIFFICULTY=32 ATTEMPTS_PER_INSTANCE=1 GPUS=0,1 bash ./run_codegym_eval.sh


set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_example_progressive_report_multi_gpu.sh [options]

Options:
  --env, --env-name, --env_name  Run only specified env(s). Repeatable or comma/pipe-separated.
                                 Accepts "example:AbacusCascade".
  --attempts-per-instance K      Number of attempts per instance (Pass@K uses K). Default: $ATTEMPTS_PER_INSTANCE
  -h, --help                     Show this help.

Examples:
  ./run_example_progressive_report_multi_gpu.sh
  ./run_example_progressive_report_multi_gpu.sh --env example:AbacusCascade
  ./run_example_progressive_report_multi_gpu.sh --env example:AbacusCascade,example:FormulaFoundry
  ENV_NAME=example:AbacusCascade GPUS=0,1 ./run_example_progressive_report_multi_gpu.sh
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Point to parent directory for report_progressive.py
REPORT_SCRIPT="$SCRIPT_DIR/report_progressive.py"

# ===========================
# Config (override via env)
# ===========================

GPUS="${GPUS:-4}"
MODEL_PATH="${MODEL_PATH:-/home/myy/Qwen3-8b/qwen3-4b-instruct-2507}"

SEED="${SEED:-42}"
NUM_DIFFICULTIES="${NUM_DIFFICULTIES:-10}"
RUNS_PER_DIFFICULTY="${RUNS_PER_DIFFICULTY:-8}"
ATTEMPTS_PER_INSTANCE="${ATTEMPTS_PER_INSTANCE:-8}"

TEMPERATURE="${TEMPERATURE:-0.6}"
PROMPT_TEMPLATE="${PROMPT_TEMPLATE:-qwen3_game}"
WRAPPERS="${WRAPPERS:-concat}"
VEC_BATCH_SIZE="${VEC_BATCH_SIZE:-256}"
RECORD_TRAJECTORIES="${RECORD_TRAJECTORIES:-1}"

MAX_TOKENS="${MAX_TOKENS:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.9}"
START_DIFFICULTY="${START_DIFFICULTY:-}"
END_DIFFICULTY="${END_DIFFICULTY:-}"
APPEND_OUTPUT="${APPEND_OUTPUT:-0}"

# Default env list.
ENVS=(
codegym:AnagramTransformationEnv
codegym:ArithmeticSequenceCheckEnv
codegym:BalloonBurstingEnv
codegym:BipartiteCheckEnv
codegym:CargoDeliveryEnv
codegym:CircleOverlapEnv
codegym:CoordinateTransformationEnv
codegym:DiagonalCountingEnv
codegym:DistinctElementsCountEnv
codegym:EnergyDifferenceMinimizingEnv
codegym:GoldCollectionEnv
codegym:GridPathCountEnv
codegym:GridShortestPathEnv
codegym:GridSumEnv
codegym:HamiltonianCycleEnv
codegym:HeapSortEnv
codegym:HistogramMaxAreaEnv
codegym:HouseRobberEnv
codegym:KnapsackEnv
codegym:LargestEmptySquareEnv
codegym:LargestHarmonicSubsetEnv
codegym:LargestRectangleEnv
codegym:LargestSquareEnv
codegym:LongestCommonSubsequenceEnv
codegym:LongestConsecutiveOnesEnv
codegym:LongestConsecutiveSubsequenceEnv
codegym:LongestFibSubseqEnv
codegym:LongestIncreasingSubarrayEnv
codegym:LongestSubsequenceWordEnv
codegym:LongestTwoColorSubarrayEnv
codegym:MajorityElementEnv
codegym:MarathonStationsEnv
codegym:MatrixCreationEnv
codegym:MaxApplesEnv
codegym:MaxCutTreesEnv
codegym:MaxFlowersEnv
codegym:MaxGoldCoinsEnv
codegym:MaxIncreasingSubarraySumEnv
codegym:MaxNonBlockingTowersEnv
codegym:MaxNonOverlappingProjectsEnv
codegym:MaxWaterContainerEnv
codegym:MaximumSpanningTreeEnv
codegym:MaximumSumSubgridEnv
codegym:MinContiguousSubarrayEnv
codegym:MinEnergyCombiningEnv
codegym:MinProductSegmentationEnv
codegym:MinSubarrayLenEnv
codegym:MinSubsetSumDiffEnv
codegym:MinSwapsToSortEnv
codegym:MinimizeMaxSubarraySumEnv
codegym:MinimizeMaxTimeEnv
codegym:MinimumPossibleSumEnv
codegym:MissingRangesEnv
codegym:MostFrequentBirdEnv
codegym:NextPalindromeEnv
codegym:OddOccurrenceFinderEnv
codegym:PalindromeVerificationEnv
codegym:ParitySortingEnv
codegym:PathCountingEnv
codegym:PerfectSquareSequenceEnv
codegym:PrefixPalindromeEnv
codegym:PrimeFilteringEnv
codegym:ProblemCountingEnv
codegym:RainwaterCollectionEnv
codegym:RamanujanNumberEnv
codegym:RemoveDuplicatesEnv
codegym:ResourceCombiningEnv
codegym:RotatedArrayMinEnv
codegym:SharedProblemPairsEnv
codegym:ShortestPathEnv
codegym:SmallestRangeEnv
codegym:SmallestRectangleEnv
codegym:SmallestSubarrayEnv
codegym:StringReorderEnv
codegym:StringSwapEnv
codegym:SubgridBeautyEnv
codegym:SudokuValidationEnv
codegym:SumArrayConstructionEnv
codegym:SymmetricGridEnv
codegym:TaskManagerEnv
codegym:TeamScoreBalancingEnv
codegym:TicketPriorityEnv
codegym:ToeplitzMatrixEnv
codegym:TreasureHuntExpectationEnv
codegym:TreeCheckEnv
codegym:TriangularTripletEnv
codegym:TwoSumEnv
codegym:UniquePathsEnv
codegym:UniqueSubstringCounterEnv
codegym:UniqueSubstringsWithOddEnv



)

ENV_NAME="${ENV_NAME:-}"
# Support ENV_LIST for batch evaluation (comma-separated)
ENV_LIST="${ENV_LIST:-}"

# ===========================
# Args
# ===========================

ENV_OVERRIDES=()
# Priority: ENV_LIST > ENV_NAME > command line args
if [[ -n "$ENV_LIST" ]]; then
  IFS=',' read -r -a ENV_OVERRIDES <<< "$ENV_LIST"
elif [[ -n "$ENV_NAME" ]]; then
  ENV_OVERRIDES+=("$ENV_NAME")
fi
seen_env_flag=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env|--env-name|--env_name)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: missing value for $1" >&2
        usage >&2
        exit 2
      fi
      if [[ "$seen_env_flag" -eq 0 ]]; then
        ENV_OVERRIDES=()
        seen_env_flag=1
      fi
      ENV_OVERRIDES+=("$2")
      shift 2
      ;;
    --attempts-per-instance|--attempts_per_instance)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: missing value for $1" >&2
        usage >&2
        exit 2
      fi
      ATTEMPTS_PER_INSTANCE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

# ===========================
# Resolve env ids
# ===========================

RAW_ENVS=()
if [[ "${#ENV_OVERRIDES[@]}" -gt 0 ]]; then
  for env_item in "${ENV_OVERRIDES[@]}"; do
    env_item="${env_item//|/,}"
    IFS=',' read -r -a parts <<< "$env_item"
    for part in "${parts[@]}"; do
      part="$(echo "$part" | xargs)"
      [[ -z "${part:-}" ]] && continue
      RAW_ENVS+=("$part")
    done
  done
else
  RAW_ENVS=("${ENVS[@]}")
fi

if [[ "${#RAW_ENVS[@]}" -eq 0 ]]; then
  echo "ERROR: no envs specified." >&2
  exit 1
fi

ENV_IDS=()
for e in "${RAW_ENVS[@]}"; do
  if [[ "$e" != *:* ]]; then
    echo "ERROR: env must be a full env id (missing ':'): $e" >&2
    echo "  Use e.g. --env example:$e" >&2
    exit 2
  fi
  ENV_IDS+=("$e")
done

if [[ ! -f "$REPORT_SCRIPT" ]]; then
  echo "ERROR: report script not found: $REPORT_SCRIPT" >&2
  exit 1
fi

IFS=', ' read -r -a GPU_ARR <<< "$GPUS"
GPU_LIST=()
for g in "${GPU_ARR[@]}"; do
  [[ -n "${g:-}" ]] && GPU_LIST+=("$g")
done
if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
  echo "ERROR: GPUS is empty (e.g. GPUS=0,1,2,3)" >&2
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$(pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/results/codegym_eval/${TIMESTAMP}}"

mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/logs"

echo "========================================================================"
echo "EnvEval_v0 Multi-GPU Progressive Report (example:*)"
echo "========================================================================"
echo "Start time: $(date)"
echo "GPUs: ${GPU_LIST[*]}"
echo "Model: $MODEL_PATH"
echo "Envs: ${#ENV_IDS[@]}"
echo "Difficulties: $NUM_DIFFICULTIES (complexity=1..10)"
echo "Runs per difficulty: $RUNS_PER_DIFFICULTY (attempts=$ATTEMPTS_PER_INSTANCE)"
echo "Seed: $SEED"
echo "Temperature: $TEMPERATURE"
echo "Wrappers: $WRAPPERS"
echo "Prompt template: $PROMPT_TEMPLATE"
echo "Vec batch size: $VEC_BATCH_SIZE"
echo "Record trajectories: $RECORD_TRAJECTORIES"
echo "Max model len: $MAX_MODEL_LEN"
echo "Start difficulty: ${START_DIFFICULTY:-<default>}"
echo "End difficulty: ${END_DIFFICULTY:-<default>}"
echo "Append output: $APPEND_OUTPUT"
echo "Output: $OUTPUT_DIR"
echo "========================================================================"

num_gpus="${#GPU_LIST[@]}"
pids=()
names=()

for ((i=0; i<num_gpus; i++)); do
  gpu="${GPU_LIST[$i]}"

  envs_for_gpu=()
  for ((j=0; j<${#ENV_IDS[@]}; j++)); do
    if (( j % num_gpus == i )); then
      envs_for_gpu+=("${ENV_IDS[$j]}")
    fi
  done
  if [[ "${#envs_for_gpu[@]}" -eq 0 ]]; then
    echo "Skipping gpu=$gpu (no envs assigned)"
    continue
  fi

  out_gpu="$OUTPUT_DIR/gpu_${gpu}"
  log="$OUTPUT_DIR/logs/gpu_${gpu}.log"
  mkdir -p "$out_gpu"

  cmd=(python -u "$REPORT_SCRIPT"
    --output "$out_gpu"
    --model "$MODEL_PATH"
    --seed "$SEED"
    --num-difficulties "$NUM_DIFFICULTIES"
    --runs-per-difficulty "$RUNS_PER_DIFFICULTY"
    --attempts-per-instance "$ATTEMPTS_PER_INSTANCE"
    --temperature "$TEMPERATURE"
    --prompt-template "$PROMPT_TEMPLATE"
    --wrappers "$WRAPPERS"
    --vec-batch-size "$VEC_BATCH_SIZE"
    --max-tokens "$MAX_TOKENS"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTIL"
  )
  if [[ "$RECORD_TRAJECTORIES" == "1" || "$RECORD_TRAJECTORIES" == "true" || "$RECORD_TRAJECTORIES" == "True" ]]; then
    cmd+=(--record-trajectories)
  fi
  if [[ -n "${START_DIFFICULTY:-}" ]]; then
    cmd+=(--start-difficulty "$START_DIFFICULTY")
  fi
  if [[ -n "${END_DIFFICULTY:-}" ]]; then
    cmd+=(--end-difficulty "$END_DIFFICULTY")
  fi
  if [[ "$APPEND_OUTPUT" == "1" || "$APPEND_OUTPUT" == "true" || "$APPEND_OUTPUT" == "True" ]]; then
    cmd+=(--append-output)
  fi
  for env_id in "${envs_for_gpu[@]}"; do
    cmd+=(--env "$env_id")
  done

  echo "[gpu=$gpu] Launching ${#envs_for_gpu[@]} env(s) -> $out_gpu"
  (
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}"
  ) >"$log" 2>&1 &

  pids+=("$!")
  names+=("gpu=$gpu")
done

failed=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  name="${names[$idx]}"
  if wait "$pid"; then
    echo "[$name] ✓ done"
  else
    echo "[$name] ✗ failed (see $OUTPUT_DIR/logs)" >&2
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

echo "Done: $(date)"
