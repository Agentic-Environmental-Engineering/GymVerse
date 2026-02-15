#!/usr/bin/env bash
cd /mnt/usercache/wanglongxiang/GymVerse/gem
pip install -e .

train_envs () {
  local gpu="$1"; shift
  local envs=("$@")

  export CUDA_VISIBLE_DEVICES="$gpu"

  echo "========================================="
  echo "使用 GPU: ${CUDA_VISIBLE_DEVICES}"
  echo "环境列表: ${envs[*]}"
  echo "========================================="

  for env in "${envs[@]}"; do
    echo ">>> 开始训练: example:${env}"

    python examples/train_oat/train_oat_me_evolve_ctlTrainComp_length.py \
      --env_id "difficulty:${env}" \
      --evolve \
      --prompt_template qwen3_game \
      --wrappers concat \
      --gamma 0.9 \
      --norm_return \
      --gpus 4 \
      --gradient-checkpointing \
      --num_samples 1 \
      --num_env 16 \
      --rollout_batch_size 128 \
      --rollout_batch_size_per_device 16 \
      --pi_buffer_maxlen_per_device 16 \
      --pretrain /ssd/jinzhuoran/gem/Qwen3-4B-Instruct-2507/ \
      --enable_prefix_caching \
      --collocate \
      --vllm_sleep \
      --vllm_gpu_ratio 0.45 \
      --rnd-seed \
      --learning_rate 0.000001 \
      --lr_scheduler constant \
      --lr_warmup_ratio 0 \
      --num_ppo_epochs 2 \
      --train_batch_size 128 \
      --train_batch_size_per_device 1 \
      --beta 0 \
      --max_model_len 30000 \
      --generate_max_length 8192 \
      --temperature 1.0 \
      --top_p 1 \
      --eval_steps -1 \
      --save_steps -1 \
      --eval_temperature 0.6 \
      --eval_top_p 0.95 \
      --eval_generate_max_length 4096 \
      --max_train 6500 \
      --max_save_num 30 

    echo "✓ 完成: ${env}"
    echo ""
  done
}


# 两组环境
########################################
ENVS_0123=(
AutomationLab
ChoreCanvas
CoursePlanner
EmergencyOps
EnergyGrid
EventPlanner
FinanceOps
FoodDelivery
HospitalOps
RenoPlanner
ShoppingMall
SupplyChain
TerminalOps
TransitOps
TravelDesk
WebShop
)

ENVS_4567=(
AutomationLabWideGap
ChoreCanvasWideGap
CoursePlannerWideGap
EmergencyOpsWideGap
EnergyGridWideGap
EventPlannerWideGap
FinanceOpsWideGap
FoodDeliveryWideGap
HospitalOpsWideGap
RenoPlannerWideGap
ShoppingMallWideGap
SupplyChainWideGap
TerminalOpsWideGap
TransitOpsWideGap
TravelDeskWideGap
WebShopWideGap
)


# 并行启动两组
########################################
mkdir -p logs

train_envs "0,1,2,3" "${ENVS_0123[@]}" > logs/gpu0123.log 2>&1 &
PID1=$!

train_envs "4,5,6,7" "${ENVS_4567[@]}" > logs/gpu4567.log 2>&1 &
PID2=$!

wait $PID1 $PID2

echo "========================================="
echo "所有训练任务完成 🎉"
echo "========================================="


  # HugeGap 版本（16个）

  # AutomationLabHugeGap
  # ChoreCanvasHugeGap
  # CoursePlannerHugeGap
  # EmergencyOpsHugeGap
  # EnergyGridHugeGap
  # EventPlannerHugeGap
  # FinanceOpsHugeGap
  # FoodDeliveryHugeGap
  # HospitalOpsHugeGap
  # RenoPlannerHugeGap
  # ShoppingMallHugeGap
  # SupplyChainHugeGap
  # TerminalOpsHugeGap
  # TransitOpsHugeGap
  # TravelDeskHugeGap
  # WebShopHugeGap