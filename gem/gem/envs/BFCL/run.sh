export LD_LIBRARY_PATH=/home/myy/Qwen3-8b/miniconda3/envs/openenv/lib:$LD_LIBRARY_PATH
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export BFCL_EXECUTE_TOOLS=1
export SERPAPI_API_KEY=d7cd93b94d67013deff48b53cb6ab9070fbff377
CUDA_VISIBLE_DEVICES=1 bfcl generate \
  --model qwen3-4b-instruct \
  --test-category multi_turn \
  --backend vllm \
  --max-model-len 32768 \
  --num-gpus 1 \
  --gpu-memory-utilization 0.7 \
  --local-model-path /home/myy/Qwen3-8b/qwen3-4b-instruct-2507   # ← optional