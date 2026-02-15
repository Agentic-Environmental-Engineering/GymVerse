# Evaluation (Checkpoint Evaluation)

This directory contains the **checkpoint evaluation** workflow for trained models.
Only the difficulty-based, progressive evaluation pipeline is preserved.

## Structure

```
Evaluation/
├── run_multi_model_eval.sh
├── run_example_progressive_report_multi_gpu.sh
├── run_codegym_eval.sh
├── run_evolve_test.sh
├── env_categories.sh
├── model_ckpt.txt
├── aggregate_results.py
├── report_progressive.py         # Main evaluator (progressive difficulty)
├── progressive_runner.py         # Difficulty scheduler + metrics
├── pass_at_k.py                  # Instance-based evaluator
├── vllm_policy.py                # vLLM policy wrapper
├── env_registry.py               # Env registration (GEM / EnvSyn / CodeGym)
└── configs/           # Difficulty parameter configs
```

## Quick Start

### 1. Configure env categories
Edit `Evaluation/env_categories.sh` and set the environment lists.

### 2. Configure model checkpoints
Edit `Evaluation/model_ckpt.txt` and add your checkpoint paths (one per line).

### 3. Run multi-model evaluation
```bash
cd /home/wanglongxiang/project_code/GymVerse/Evaluation

MODEL_CHECKPOINTS_FILE=./model_ckpt.txt \
GPUS=0,1,2,3 \
./run_multi_model_eval.sh
```

### 4. Aggregate results
```bash
python3 aggregate_results.py
```

## Full Usage (from eval.md)

### 1. Prepare checkpoint list
Put the checkpoints to evaluate into `Evaluation/model_ckpt.txt`, one path per line:

```
# Add your model paths below:
/path/to/your/checkpoint
```

### 2. Run evaluation (examples)
```bash
cd /home/wanglongxiang/project_code/GymVerse/Evaluation

# quick test
MODEL_CHECKPOINTS_FILE=model_ckpt.txt \
RESULTS_BASE_DIR=results_evolve_1 \
START_DIFFICULTY=0 END_DIFFICULTY=1 NUM_DIFFICULTIES=1 \
RUNS_PER_DIFFICULTY=32 ATTEMPTS_PER_INSTANCE=1 \
GPUS=0,1 \
bash ./run_multi_model_eval.sh
```

Note: `GPUS=0,1` should match the GPU ids you want to use. `START_DIFFICULTY=0 END_DIFFICULTY=1 NUM_DIFFICULTIES=1` controls the evaluated environment complexity range.

### 3. Aggregate results (custom path)
```bash
python aggregate_results.py \
  --results-dir /path/to/your/results_dir \
  --output-dir /path/to/your/results_dir
```

Note: Just replace the path with the actual results directory you generated.

## Notes

- The evaluation uses **vLLM** for inference. Make sure the environment is set up to run vLLM.
- For CodeGym evaluation, ensure `CODEGYM__DIR` is set if using custom converted environments.
- For EnvSyn evaluation, set `ENVSYN_SAVED_DIR` if your saved environments are outside defaults.
