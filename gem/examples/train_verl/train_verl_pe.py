# Copyright 2025 AxonRL Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Entry script of using VeRL to train LLM agents on GEM environments.

This file is largely borrowed from verl/trainer/main_ppo.py, with
critical modifications labeled with "[GEM Notes]" (Ctrl+F to navigate).
"""

import json
import logging
import os
import re
import socket
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from pprint import pprint
from typing import List, Optional, Sequence, Tuple

import hydra
import numpy as np
import ray
import torch
import torch.utils
import torch.utils.data
import tree
from omegaconf import OmegaConf
from tensordict import TensorDict
from tqdm import tqdm
from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.constants_ppo import PPO_RAY_RUNTIME_ENV
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    Dataset,
    RayPPOTrainer,
    ResourcePoolManager,
    Role,
    compute_response_mask,
)
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.device import is_cuda_available
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.fsdp_workers import ActorRolloutRefWorker

import gem
from gem.utils.parsing import extract_action_parameters, extract_last_boxed_answer
from gem.wrappers.wrapper_factory import get_wrapper_fns

WorkerType = type[Worker]

logger = logging.getLogger(__file__)


@hydra.main(config_path="./", config_name="config", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config_dict: Hydra configuration dictionary containing training parameters.
    """
    run_ppo(config)


# Define a function to run the PPO-like training process
def run_ppo(config) -> None:
    """Initialize Ray cluster and run distributed PPO training process.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed PPO training including Ray initialization settings,
                model paths, and training hyperparameters.
    """
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        ray.init(
            runtime_env=PPO_RAY_RUNTIME_ENV,
            num_cpus=config.ray_init.num_cpus,
        )

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if (
        is_cuda_available
        and OmegaConf.select(config.trainer, "profile_steps") is not None
        and len(OmegaConf.select(config.trainer, "profile_steps")) > 0
    ):
        nsight_options = OmegaConf.to_container(
            config.trainer.controller_nsight_options
        )
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


# Invalid action to be sent to the env to trigger format error penalty.
INVALID_ACTION = "<｜INVALID_ACTION｜>"


def _append_jsonl(path: str, record: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def apply_qwen3_game_template(observation: str) -> str:
    return (
        f"<|im_start|>user\nYou are playing language games. Make valid actions to win.\nObservation: {observation}"
        "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def apply_no_template(observation: str) -> str:
    return observation


def apply_qwen3_tool_template(question: str) -> str:
    return (
        "<|im_start|>user\n"
        "First think step by step inside <think>...</think>.\n"
        "Then execute exactly one tool call inside <action>...</action>.\n"
        "Tool call format inside <action>: [func_name1(params_name1=params_value1, params_name2=params_value2 ...)].\n"
        "If the function has no parameters, call it as [func_name1()].\n"
        "Example:\n"
        "<action>[func_name1(params_name1=params_value1, params_name2=params_value2)]</action>\n"
        "After you issue the function call, wait for the tool result.\n"
        "If the tool description is unclear, you may invoke it once to inspect the output and adjust your next call.\n\n"
        f"Question: {question}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def apply_qwen3_general_template(question: str) -> str:
    return (
        f"<|im_start|>user\nQuestion: {question}"
        "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def apply_code_template(question: str) -> str:
    return (
        "You are an expert Python programmer. "
        "You will be given a question (problem specification) and will generate a correct "
        "Python program that matches the specification and passes all tests."
        f"\nQuestion: {question}"
        "\nPlease reason step by step, and write your code in markdown format, e.g., ```python\n# YOUR CODE HERE\n```."
    )


TEMPLATE_FACTORY = {
    "qwen3_game": apply_qwen3_game_template,
    "qwen3_tool": apply_qwen3_tool_template,
    "no": apply_no_template,
    "na": apply_no_template,
    "qwen3_general": apply_qwen3_general_template,
    "code": apply_code_template,
}


@dataclass
class Transition:
    obs: str
    action: str
    reward: float
    done: bool

    prompt: str
    prompt_ids: list
    response: str
    response_ids: list

    attention_mask: list
    position_ids: list

    response_is_truncated: bool
    action_is_formatted: bool

    def format(self):
        return {
            "obs": self.obs,
            "action": self.action,
            "reward": self.reward,
            "done": int(self.done),
            "prompt": self.prompt,
            "response": self.response,
        }


class GEMActorRolloutRefWorker(ActorRolloutRefWorker):
    # @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    # def init_model(self):
    #     super().init_model()
    pass


class DummyPromptDataset(Dataset):
    """Empty dataset to satisfy VeRL's requirements without actually loading data."""

    def __init__(self, size=1):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        del idx
        return ""


class ReinforceGEMTrainer(RayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        device_name="cuda",
    ):
        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert (
                Role.ActorRollout in role_worker_mapping
            ), f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(
                self.config.algorithm.kl_ctrl
            )

        # [GEM Notes] We only support multi-turn REINFORCE now, not Actor-Critic.
        self.use_critic = False

        self._validate_config()
        # [GEM Notes] We do not need dataset since our experience comes from interacting with GEM environments,
        # [GEM Notes] just like humans learn from experience interacting with the world.
        # [GEM Notes] However, we build a dummy dataloader to facilitate the training loop.
        # self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
        self.train_dataloader = torch.utils.data.DataLoader(
            DummyPromptDataset(int(1e9)),
        )
        self.total_training_steps = (
            len(self.train_dataloader) * self.config.trainer.total_epochs
        )
        if self.config.trainer.total_training_steps is not None:
            self.total_training_steps = self.config.trainer.total_training_steps

        self.save_path = self.config.trainer.get(
            "save_dir", self.config.trainer.get("log_dir", os.getcwd())
        )
        self.game_state_save_path = os.path.join(self.save_path, "game_state")
        os.makedirs(self.game_state_save_path, exist_ok=True)
        self.step_count = 0
        self.actor_id = 0

        seed = int(time.time_ns())
        # [GEM Notes] Init environment.
        # [GEM Notes] Get environment wrappers.
        wrappers = get_wrapper_fns(
            self.config.actor_rollout_ref.env.wrappers, tokenizer=self.tokenizer
        )
        if "concat_chat" in self.config.actor_rollout_ref.env.wrappers:
            assert self.config.actor_rollout_ref.prompt_template in ["no", "na"], (
                "chat template is applied on env side already"
            )
        # [GEM Notes] Instantiate vectorized environments (supports multi-env training).
        env_id_raw = self.config.actor_rollout_ref.env.env_id
        env_id_list = (
            env_id_raw.split("|") if isinstance(env_id_raw, str) else list(env_id_raw)
        )
        self.env_id_list = env_id_list
        self.train_envs = {}
        for env_idx, env_id in enumerate(env_id_list):
            vec_env_ids = [env_id] * self.config.actor_rollout_ref.env.num_env
            vec_kwargs = [
                {
                    "seed": seed
                    + env_idx * self.config.actor_rollout_ref.env.num_env
                    + j
                }
                for j in range(self.config.actor_rollout_ref.env.num_env)
            ]
            self.train_envs[env_id] = gem.make_vec(
                vec_env_ids,
                vec_kwargs=vec_kwargs,
                wrappers=wrappers,
                async_mode=self.config.actor_rollout_ref.env.async_env,
            )
        self.cur_env_id, self.env = self._sample_env_for_batch()
        self.env_success_history = {env_id: [] for env_id in self.env_id_list}

        # [GEM Notes] Instantiate eval environments when configured.
        self.eval_envs = {}
        self.eval_env_ids = []
        self.eval_prompt_templates = []
        self.eval_n = 0
        self.eval_batch_size = self.config.actor_rollout_ref.env.num_env
        self.eval_instance_size = 16
        self.eval_difficulty = 1
        self.eval_async_env = False
        eval_envs = OmegaConf.select(self.config, "actor_rollout_ref.eval_envs")
        if eval_envs:
            self.eval_env_ids = (
                eval_envs.split("|")
                if isinstance(eval_envs, str)
                else list(eval_envs)
            )
            eval_wrappers = (
                OmegaConf.select(self.config, "actor_rollout_ref.eval_wrappers") or ""
            )
            eval_prompt_templates = (
                OmegaConf.select(
                    self.config, "actor_rollout_ref.eval_prompt_templates"
                )
                or "na"
            )
            self.eval_n = int(
                OmegaConf.select(self.config, "actor_rollout_ref.eval_n") or 1
            )
            self.eval_batch_size = int(
                OmegaConf.select(self.config, "actor_rollout_ref.eval_batch_size")
                or self.eval_batch_size
            )
            self.eval_instance_size = int(
                OmegaConf.select(self.config, "actor_rollout_ref.eval_instance_size")
                or self.eval_instance_size
            )
            self.eval_difficulty = int(
                OmegaConf.select(self.config, "actor_rollout_ref.eval_difficulty")
                or self.eval_difficulty
            )
            self.eval_async_env = bool(
                OmegaConf.select(self.config, "actor_rollout_ref.eval_async_env")
                or self.eval_async_env
            )

            def _normalize_eval_hp(hp):
                hp = hp.split("|") if isinstance(hp, str) else list(hp)
                if len(hp) == 1:
                    return hp * len(self.eval_env_ids)
                assert len(hp) == len(self.eval_env_ids), (
                    "eval_wrappers/eval_prompt_templates should be either a string "
                    "or a list of the same length as eval_envs"
                )
                return hp

            eval_wrappers = _normalize_eval_hp(eval_wrappers)
            self.eval_prompt_templates = _normalize_eval_hp(eval_prompt_templates)
            for eval_env_id, eval_wrapper in zip(self.eval_env_ids, eval_wrappers):
                eval_wrapper_fns = get_wrapper_fns(
                    eval_wrapper, tokenizer=self.tokenizer
                )
                vec_env = gem.make_vec(
                    [eval_env_id] * self.eval_batch_size,
                    wrappers=eval_wrapper_fns,
                    async_mode=self.eval_async_env,
                )
                for env in vec_env.envs:
                    try:
                        if hasattr(env, "set_complexity"):
                            env.set_complexity(self.eval_difficulty)
                    except Exception:
                        pass
                self.eval_envs[eval_env_id] = vec_env

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get(
            "val_before_train", True
        ):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Training Progress",
        )

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        for epoch in range(self.config.trainer.total_epochs):
            for _ in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                do_profile = (
                    self.global_steps in self.config.trainer.profile_steps
                    if self.config.trainer.profile_steps is not None
                    else False
                )
                with marked_timer("start_profile", timing_raw):
                    if do_profile:
                        self.actor_rollout_wg.start_profile(
                            role="e2e", profile_step=self.global_steps
                        )
                        if self.use_reference_policy:
                            self.ref_policy_wg.start_profile()
                        if self.use_critic:
                            self.critic_wg.start_profile()
                        if self.use_rm:
                            self.rm_wg.start_profile()

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        # [GEM Notes] We only support sync mode for now.
                        # [GEM Notes] We pass a dummy data proto to the rollout wg,
                        # [GEM Notes] because we do not rely on dataset for rollout -
                        # [GEM Notes] the agent directly interacts with GEM envs to
                        # [GEM Notes] generate data (both prompts and responses).
                        gen_batch_output = self.run_agent_env_loop()
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    gen_batch_output.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(gen_batch_output.batch))],
                        dtype=object,
                    )
                    batch = gen_batch_output

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    print("Trainer batch size:", len(gen_batch_output.batch))

                    # [GEM Notes] We do not compute reward here because the GEM envs
                    # [GEM Notes] return rewards by design!
                    # with marked_timer("reward", timing_raw, color="yellow"):
                    #     # compute reward model score
                    #     if self.use_rm:
                    #         reward_tensor = self.rm_wg.compute_rm_score(batch)
                    #         batch = batch.union(reward_tensor)

                    #     if self.config.reward_model.launch_reward_fn_async:
                    #         future_reward = compute_reward_async.remote(
                    #             batch, self.config, self.tokenizer
                    #         )
                    #     else:
                    #         reward_tensor, reward_extra_infos_dict = compute_reward(
                    #             batch, self.reward_fn
                    #         )

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = (
                            self.config.actor_rollout_ref.actor.loss_agg_mode
                        )
                        entropy_agg = agg_loss(
                            loss_mat=entropys,
                            loss_mask=response_masks,
                            loss_agg_mode=loss_agg_mode,
                        )
                        old_log_prob_metrics = {
                            "actor/entropy": entropy_agg.detach().item()
                        }
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(
                                rollout_probs_diff, response_mask.bool()
                            )
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.config.actor_rollout_ref.norm_return:
                        advantages = batch.batch["advantages"]
                        local_sum = advantages.sum()
                        local_square_sum = (advantages**2).sum()
                        local_num = torch.tensor(
                            [advantages.numel()],
                            dtype=torch.float32,
                            device=advantages.device,
                        )
                        mean_adv = local_sum / local_num
                        std_adv = torch.sqrt(
                            local_square_sum / local_num - mean_adv**2
                        )
                        batch.batch["advantages"] = (advantages - mean_adv) / (
                            std_adv + 1e-9
                        )

                    # Dummy metric logging to satisfy verl's `compute_data_metrics`
                    batch.batch["token_level_scores"] = batch.batch["advantages"]
                    batch.batch["token_level_rewards"] = batch.batch["advantages"]
                    batch.batch["returns"] = batch.batch["advantages"]

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(
                            critic_output.meta_info["metrics"]
                        )
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = (
                                self.config.actor_rollout_ref.rollout.multi_turn.enable
                            )
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(
                            actor_output.meta_info["metrics"]
                        )
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer(
                            "dump_rollout_generations", timing_raw, color="green"
                        ):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(
                                batch.batch["prompts"], skip_special_tokens=True
                            )
                            outputs = self.tokenizer.batch_decode(
                                batch.batch["responses"], skip_special_tokens=True
                            )
                            scores = batch.batch["advantages"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict={},
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (
                            is_last_step
                            or self.global_steps % self.config.trainer.test_freq == 0
                        )
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)
                    if (
                        self.eval_env_ids
                        and self.config.trainer.test_freq > 0
                        and (
                            is_last_step
                            or self.global_steps % self.config.trainer.test_freq == 0
                        )
                    ):
                        with marked_timer("eval_envs", timing_raw, color="green"):
                            eval_metrics = self.evaluate_envs(self.global_steps)
                        metrics.update(eval_metrics)

                    # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    # Check if the conditions for saving a checkpoint are met.
                    # The conditions include a mandatory condition (1) and
                    # one of the following optional conditions (2/3/4):
                    # 1. The save frequency is set to a positive value.
                    # 2. It's the last training step.
                    # 3. The current step number is a multiple of the save frequency.
                    # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print(
                                "Force saving checkpoint: ESI instance expiration approaching."
                            )
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    if do_profile:
                        self.actor_rollout_wg.stop_profile()
                        if self.use_reference_policy:
                            self.ref_policy_wg.stop_profile()
                        if self.use_critic:
                            self.critic_wg.stop_profile()
                        if self.use_rm:
                            self.rm_wg.stop_profile()

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(batch.meta_info["metrics"])
                metrics.update(
                    compute_data_metrics(batch=batch, use_critic=self.use_critic)
                )
                metrics.update(
                    compute_timing_metrics(batch=batch, timing_raw=timing_raw)
                )
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(
                    compute_throughout_metrics(
                        batch=batch, timing_raw=timing_raw, n_gpus=n_gpus
                    )
                )

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                # if hasattr(self.train_dataset, "on_batch_end"):
                #     # The dataset may be changed after each training batch
                #     self.train_dataset.on_batch_end(batch=batch)

    def _sample_env_for_batch(self) -> Tuple[str, object]:
        """Round-robin environment sampling for reproducibility."""
        idx = self.step_count % len(self.env_id_list)
        env_id = self.env_id_list[idx]
        env = self.train_envs[env_id]
        self.cur_env_id = env_id
        self.env = env
        return env_id, env

    def run_eval_episode(
        self, eval_env_id: str, eval_prompt_template: str, batch_size: int
    ) -> List[dict]:
        def get_attr_from_wrapper(env, attr):
            if hasattr(env, attr):
                return getattr(env, attr)
            if hasattr(env, "env"):
                return get_attr_from_wrapper(env.env, attr)
            raise ValueError(f"Cannot find {attr} in env.")

        vec_env = self.eval_envs[eval_env_id]
        try:
            dataset = get_attr_from_wrapper(vec_env.envs[0], "dataset")
        except ValueError:
            dataset = DummyPromptDataset(size=self.eval_instance_size)

        finished_episodes = []
        if batch_size > len(dataset):
            logging.info(
                "eval batch size %s is larger than dataset size %s, set batch size to %s",
                batch_size,
                len(dataset),
                len(dataset),
            )
            batch_size = len(dataset)

        for i in range(0, len(dataset), batch_size):
            n_parallel = min(batch_size, len(dataset) - i)
            _kwargs_map = {
                f"env{j}_kwargs": {"idx": i + j}
                for j in range(min(n_parallel, len(dataset) - i))
            }
            try:
                obs, info = vec_env.reset(**_kwargs_map)
            except TypeError:
                obs, info = vec_env.reset()
            obs, info = obs[:n_parallel], info[:n_parallel]
            episodes = [[] for _ in range(len(obs))]
            done = np.array([False] * n_parallel)
            while not all(done):
                action, _ = self.agent_act(obs, eval_prompt_template)
                action_iter = iter(action)
                next_obs, reward, terminated, truncated, info = vec_env.step(
                    {i: next(action_iter) for i, d in enumerate(done) if not d}
                )
                obs_len = [len(self.tokenizer.encode(o)) for o in next_obs]
                obs_exceeds_max_len = np.array(
                    [
                        l >= self.config.actor_rollout_ref.rollout.max_model_len
                        for l in obs_len
                    ]
                )
                cur_done = terminated | truncated | obs_exceeds_max_len
                iter_idx = 0
                for i in range(len(done)):
                    if not done[i]:
                        episodes[i].append(
                            {
                                "obs": obs[iter_idx],
                                "action": action[iter_idx],
                                "reward": reward[iter_idx],
                                "next_obs": next_obs[iter_idx],
                                "done": cur_done[iter_idx],
                                "info": info[iter_idx],
                            }
                        )
                        iter_idx += 1
                done[~done] = cur_done
                obs = [o for o, d in zip(next_obs, cur_done) if not d]
            finished_episodes.extend(episodes)
        return finished_episodes

    def evaluate_envs(self, steps: int) -> dict:
        if not self.eval_env_ids:
            return {}
        metrics = {
            f"eval/{env_id}/{metric}": 0.0
            for env_id in self.eval_env_ids
            for metric in [
                "accuracy",
                "last_step_acc",
                "elapse",
                "response_tok_len",
                "mean_episode_len",
                "num_tool_success",
            ]
        }
        t0 = time.time()
        for eval_env_id, eval_prompt_template in zip(
            self.eval_env_ids, self.eval_prompt_templates
        ):
            episodes = []
            for _ in range(self.eval_n):
                episodes.extend(
                    self.run_eval_episode(
                        eval_env_id, eval_prompt_template, self.eval_batch_size
                    )
                )
            run_elapse = time.time() - t0
            t0 = time.time()
            metrics.update(
                {
                    f"eval/{eval_env_id}/elapse": run_elapse,
                    f"eval/{eval_env_id}/response_tok_len": np.mean(
                        [
                            sum([len(self.tokenizer.encode(t["action"])) for t in ep])
                            for ep in episodes
                        ]
                    ),
                    f"eval/{eval_env_id}/accuracy": np.mean(
                        [sum([t["reward"] for t in ep]) for ep in episodes]
                    ),
                    f"eval/{eval_env_id}/last_step_acc": np.mean(
                        [ep[-1]["reward"] == 1 for ep in episodes]
                    ),
                    f"eval/{eval_env_id}/mean_episode_len": np.mean(
                        [len(ep) for ep in episodes]
                    ),
                    f"eval/{eval_env_id}/num_tool_success": np.mean(
                        [
                            ep[-1]["info"].get("tool_success_counter", 0)
                            + ep[-1]["info"].get("prev_ep_tool_success_counter", 0)
                            for ep in episodes
                        ]
                    ),
                }
            )
            transitions = [t for ep in episodes for t in ep]
            eval_res_path = os.path.join(self.save_path, "eval_results")
            os.makedirs(eval_res_path, exist_ok=True)
            with open(
                os.path.join(eval_res_path, f"{steps}_{eval_env_id}.json"), "w"
            ) as f:
                json.dump(transitions, f, indent=4, ensure_ascii=False)
            eval_metrics_save_path = os.path.join(
                self.save_path, "eval_env_metrics.jsonl"
            )
            _append_jsonl(
                eval_metrics_save_path,
                {
                    "step": steps,
                    "env_id": eval_env_id,
                    "last_step_acc": metrics[f"eval/{eval_env_id}/last_step_acc"],
                },
            )

        metrics["eval/average/accuracy"] = np.mean(
            [metrics[f"eval/{env_id}/accuracy"] for env_id in self.eval_env_ids]
        )
        metrics["eval/average/last_step_acc"] = np.mean(
            [metrics[f"eval/{env_id}/last_step_acc"] for env_id in self.eval_env_ids]
        )
        metrics["eval/average/mean_episode_len"] = np.mean(
            [metrics[f"eval/{env_id}/mean_episode_len"] for env_id in self.eval_env_ids]
        )
        metrics["eval/average/response_tok_len"] = np.mean(
            [metrics[f"eval/{env_id}/response_tok_len"] for env_id in self.eval_env_ids]
        )
        metrics["eval/average/elapse"] = np.mean(
            [metrics[f"eval/{env_id}/elapse"] for env_id in self.eval_env_ids]
        )
        return metrics

    def run_agent_env_loop(self):
        """
        [GEM Notes] This method is heavily modified to generate experiences
        for learning by making the agent interact with GEM environments.
        """
        generate_st = time.time()

        # Play multiple episodes to generate transitions (trajectories in language MDP)
        all_trajectories = []

        finished_episodes, collection_info, env_id = self.collect_experience(
            self.config.actor_rollout_ref.rollout.rollout_batch_size
        )
        for ep in finished_episodes:
            all_trajectories.extend(self.prepare_trajectories(ep))

        # Logging infos
        mean_episode_len = np.mean([len(ep) for ep in finished_episodes])
        mean_episode_return = np.mean(
            [
                sum(transition.reward for transition in episode)
                for episode in finished_episodes
            ]
        )
        mean_episode_success = np.mean(
            [episode[-1].reward == 1 for episode in finished_episodes]
        )  # NOTE: assuming success reward is always 1
        num_trajectories = len(all_trajectories)
        metrics = {
            "actor/mean_episode_len": mean_episode_len,
            "actor/mean_episode_return": mean_episode_return,
            "actor/mean_episode_success": mean_episode_success,
            "actor/num_trajectories": num_trajectories,
        }
        metrics[f"{env_id}/mean_episode_len"] = mean_episode_len
        metrics[f"{env_id}/mean_episode_return"] = mean_episode_return
        metrics[f"{env_id}/mean_episode_success"] = mean_episode_success
        metrics[f"{env_id}/num_trajectories"] = num_trajectories
        history = self.env_success_history[env_id]
        history.append(mean_episode_success)
        metrics.update(collection_info)

        if self.config.actor_rollout_ref.get("evolve", False) and hasattr(
            self.env, "evolve"
        ):
            metrics["actor/complexity"] = self.env.evolve(mean_episode_success)
            metrics[f"{env_id}/complexity"] = metrics["actor/complexity"]

        step_idx = self.step_count - 1
        metrics_save_path = os.path.join(self.save_path, "actor_env_metrics.jsonl")
        _append_jsonl(
            metrics_save_path,
            {
                "step": step_idx,
                "env_id": env_id,
                "mean_episode_success": mean_episode_success,
                "complexity": metrics.get("actor/complexity"),
            },
        )

        # Subsample trajectories if they exceed the batch size
        if (
            len(all_trajectories)
            > self.config.actor_rollout_ref.rollout.rollout_batch_size
        ):
            subsample_indices = np.random.choice(
                len(all_trajectories),
                self.config.actor_rollout_ref.rollout.rollout_batch_size,
                replace=False,
            )
            all_trajectories = [all_trajectories[si] for si in subsample_indices]

        ids = []
        attention_mask = []
        position_ids = []
        prompts = []
        responses = []
        adv = []
        max_prompt_len = max([len(x["prompt_ids"]) for x in all_trajectories])
        for transition in all_trajectories:
            # Padding to align the length
            num_to_pad = max_prompt_len - len(transition["prompt_ids"])
            transition["prompt_ids"] = [
                self.tokenizer.pad_token_id
            ] * num_to_pad + transition["prompt_ids"]
            transition["attention_mask"] = [0] * num_to_pad + transition[
                "attention_mask"
            ]
            transition["position_ids"] = [0] * num_to_pad + transition["position_ids"]
            ids.append(transition["prompt_ids"] + transition["response_ids"])
            attention_mask.append(transition["attention_mask"])
            position_ids.append(transition["position_ids"])
            responses.append(transition["response_ids"])
            prompts.append(transition["prompt_ids"])
            adv.append(transition["adv"])

        batch = TensorDict(
            {
                "input_ids": torch.tensor(ids),
                "responses": torch.tensor(responses),
                "attention_mask": torch.tensor(attention_mask),
                "position_ids": torch.tensor(position_ids),
                "advantages": torch.tensor(adv)[:, None],
            },
            batch_size=len(ids),
        )
        out = DataProto(batch=batch)
        out.meta_info["timing"] = {"actor_time": time.time() - generate_st}
        out.meta_info["metrics"] = metrics
        return out

    def collect_experience(self, min_steps: int):
        env_id, env = self._sample_env_for_batch()
        obs, _ = env.reset()
        done = False
        episodes = [[] for _ in range(env.num_envs)]
        finished_episodes = []
        finished_episodes_tool_uses = []
        finished_episodes_tool_success = []
        num_generation_failed = 0
        while True:
            action, extra = self.agent_act(obs)  # type: ignore
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated

            for i in range(env.num_envs):
                if extra[i]["generation_failed"]:
                    num_generation_failed += 1
                    if self.config.actor_rollout_ref.keep_generation_failed:
                        episodes[i][-1].reward += reward[i]
                        episodes[i][-1].done = True
                        finished_episodes.append(deepcopy(episodes[i]))
                        finished_episodes_tool_uses.append(
                            info[i].get("prev_ep_tool_use_counter", 0)
                            if done[i]
                            else info[i].get("tool_use_counter", 0)
                        )
                        finished_episodes_tool_success.append(
                            info[i].get("prev_ep_tool_success_counter", 0)
                            if done[i]
                            else info[i].get("tool_success_counter", 0)
                        )
                    episodes[i].clear()
                    if not done[i]:
                        next_obs[i] = env.envs[i].reset()[0]
                else:
                    transition = Transition(
                        obs=obs[i],
                        action=action[i],
                        reward=reward[i],
                        done=done[i],
                        prompt=extra[i]["formatted_observation"],
                        prompt_ids=extra[i]["prompt_ids"],
                        response=extra[i]["response"],
                        response_ids=extra[i]["response_ids"],
                        attention_mask=extra[i]["attention_mask"],
                        position_ids=extra[i]["position_ids"],
                        response_is_truncated=extra[i]["response_is_truncated"],
                        action_is_formatted=extra[i]["action_is_formatted"],
                    )
                    episodes[i].append(transition)
                    if done[i]:
                        finished_episodes.append(deepcopy(episodes[i]))
                        finished_episodes_tool_uses.append(
                            info[i].get("prev_ep_tool_use_counter", 0)
                        )
                        finished_episodes_tool_success.append(
                            info[i].get("prev_ep_tool_success_counter", 0)
                        )
                        episodes[i].clear()

            obs = next_obs
            if len(tree.flatten(finished_episodes)) >= min_steps:
                break

        info = {
            "actor/num_generation_failed": num_generation_failed,
            "actor/prop_generation_failed": (
                num_generation_failed / len(finished_episodes)
                if self.config.actor_rollout_ref.keep_generation_failed
                else num_generation_failed
                / (len(finished_episodes) + num_generation_failed)
            ),
            "actor/num_tool_uses": np.mean(finished_episodes_tool_uses),
            "actor/num_tool_success": np.mean(finished_episodes_tool_success),
        }
        if (
            self.config.actor_rollout_ref.dump_experience_every > 0
            and self.step_count % self.config.actor_rollout_ref.dump_experience_every
            == 0
        ):
            _to_dump = {}
            for i, ep in enumerate(finished_episodes):
                key = f"episode{i}"
                _to_dump[key] = []
                for transition in ep:
                    _to_dump[key].append(transition.format())
            with open(
                os.path.join(
                    self.game_state_save_path,
                    f"actor{self.actor_id}_step{self.step_count}.json",
                ),
                "w",
            ) as f:
                json.dump(
                    _to_dump,
                    f,
                    indent=4,
                )
        self.step_count += 1
        return finished_episodes, info, env_id

    def agent_act(
        self, vec_observation: List[str], prompt_template: Optional[str] = None
    ) -> Tuple[str, dict]:
        """Use the current LLM as a policy to act.

        Args:
            vec_observation: Vectorized observation from TextArena environment.

        Returns:
            Tuple[str, dict]: Action and extra data.

        """
        if prompt_template is None:
            prompt_template = self.config.actor_rollout_ref.prompt_template

        formatted_observations = []
        for observation in vec_observation:
            observation = TEMPLATE_FACTORY[prompt_template](observation)
            if self.config.actor_rollout_ref.apply_chat_template:
                observation = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": observation}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            formatted_observations.append(observation)

        # Subsample to remove observations that exceed max model length
        idss = self.tokenizer(formatted_observations).input_ids
        exceeds_lengths = [
            len(ids) >= self.config.actor_rollout_ref.rollout.max_model_len
            for ids in idss
        ]
        sub_formatted_observations = [
            o for o, e in zip(formatted_observations, exceeds_lengths) if not e
        ]

        outs = self.tokenizer(
            sub_formatted_observations,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            padding_side="left",
        )
        outs["position_ids"] = compute_position_id_with_mask(outs.attention_mask)
        batch: DataProto = DataProto.from_single_dict(outs)

        prompts = batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=[],
        )

        # # Move to device & Generate
        # prompts = prompts.to(get_device_id())
        # meta_info = {
        #     "eos_token_id": (
        #         self.generation_config.eos_token_id
        #         if self.generation_config is not None
        #         else self.tokenizer.eos_token_id
        #     ),
        #     "pad_token_id": (
        #         self.generation_config.pad_token_id
        #         if self.generation_config is not None
        #         else self.tokenizer.pad_token_id
        #     ),
        # }
        # prompts.meta_info.update(meta_info)
        # timing_generate = {}
        # with self.rollout_sharding_manager:
        #     log_gpu_memory_usage(
        #         "After entering rollout sharding manager", logger=logger
        #     )

        #     prompts = self.rollout_sharding_manager.preprocess_data(prompts)
        #     with simple_timer("generate_sequences", timing_generate):
        #         output = self.rollout.generate_sequences(prompts=prompts)

        #     log_gpu_memory_usage("After rollout generation", logger=logger)

        #     output = self.rollout_sharding_manager.postprocess_data(output)

        # timing_generate.update(self.rollout_sharding_manager.timing)
        # # We calculate the average timing across all ranks
        # # to make sure meta_info["timing"] is the same
        # timing_generate = reduce_timing(timing_generate)
        # output.meta_info["timing"] = timing_generate
        # output = output.to("cpu")
        output = self.actor_rollout_wg.generate_sequences(prompts)

        executable_actions = []
        extras = []
        sub_i = 0

        for i, exceeds_length in enumerate(exceeds_lengths):
            if exceeds_length:
                # if prompt exceeds max model length we skipped the generation
                executable_actions.append(INVALID_ACTION)
                extras.append({"generation_failed": True})
            else:
                token_ids = output.batch["responses"][sub_i].tolist()
                prompt_token_ids = output.batch["prompts"][sub_i].tolist()
                attention_mask = output.batch["attention_mask"][sub_i].tolist()
                position_ids = output.batch["position_ids"][sub_i].tolist()
                raw_action = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                response_is_truncated = self.tokenizer.eos_token_id not in token_ids

                # Valid extraction = proper eos + proper format
                # Only used for metric logging
                extracted_action = (
                    INVALID_ACTION
                    if response_is_truncated
                    else self.extract_action(raw_action, prompt_template)
                )
                executable_actions.append(
                    INVALID_ACTION if response_is_truncated else raw_action
                )
                extras.append(
                    {
                        "formatted_observation": formatted_observations[i],
                        "prompt_ids": prompt_token_ids,
                        "response": raw_action,
                        "response_ids": token_ids,
                        "attention_mask": attention_mask,
                        "position_ids": position_ids,
                        "response_is_truncated": response_is_truncated,
                        "action_is_formatted": extracted_action != INVALID_ACTION,
                        "generation_failed": False,
                        "generation_max_length_reached": (
                            len(prompt_token_ids) + len(token_ids)
                            >= self.config.actor_rollout_ref.rollout.max_model_len
                        ),
                    }
                )
                sub_i += 1
        return executable_actions, extras  # type: ignore

    def extract_action(self, text: str, prompt_template: str) -> str:
        """
        Extract and format the actual action from the model's output.

        This method handles different template formats and ensures the action
        is properly formatted for the environment.

        Args:
            text: Raw text output from the model

        Returns:
            Cleaned and formatted action string ready for the environment
        """
        if not text:
            return ""  # Handle empty text case

        try:
            formatted_action = None
            if prompt_template in ["qwen3_game", "qwen3_general"] or (
                prompt_template in ["no", "na"]
                and "qwen" in self.config.actor_rollout_ref.model.path.lower()
            ):
                formatted_action = extract_last_boxed_answer(text)
                if formatted_action is None:
                    formatted_action = text.strip()
            elif prompt_template == "qwen3_tool":
                formatted_action = extract_action_parameters(text)
            elif prompt_template == "code":
                code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
                if not code_blocks:
                    formatted_action = None
                else:
                    formatted_action = code_blocks[-1].strip()
            else:
                raise NotImplementedError

            if formatted_action is None:
                formatted_action = INVALID_ACTION

            return formatted_action

        except Exception as e:
            logging.error(f"Error in extract_action: {e}")
            # Return invalid action if extraction fails.
            return INVALID_ACTION

    def prepare_trajectories(self, episode: Sequence[Transition]) -> List[dict]:
        """
        Prepare language trajectories (transitions of episode).

        Args:
            episode: A complete episode of the agent environment interaction.

        Returns:
            List of trajectory data
        """
        trajectory_data = []
        rewards = [t.reward for t in episode]

        # Compute returns
        returns = np.zeros_like(rewards, dtype=np.float32)
        cur = 0.0
        for i in reversed(range(len(rewards))):
            cur = rewards[i] + self.config.actor_rollout_ref.gamma * cur
            returns[i] = cur

        # Distribute turn-based returns to token-level returns
        for i, step_data in enumerate(episode):
            # Add trajectory data
            trajectory_data.append(
                dict(
                    prompt=step_data.prompt,
                    prompt_ids=step_data.prompt_ids,
                    response=step_data.response,
                    response_ids=step_data.response_ids,
                    attention_mask=step_data.attention_mask,
                    position_ids=step_data.position_ids,
                    adv=returns[i],
                    info={
                        "actor/action_is_formatted": step_data.action_is_formatted,
                        "actor/step_reward": rewards[i],
                        "actor/discount_factor": self.config.actor_rollout_ref.gamma,
                        "actor/discounted_step_return": returns[i],
                        "actor/response_is_truncated": step_data.response_is_truncated,
                    },
                )
            )

        return trajectory_data


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    """Ray remote class for executing distributed PPO training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.
    """

    def run(self, config):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")

        pprint(OmegaConf.to_container(config, resolve=True))

        OmegaConf.resolve(config)

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(
            local_path, trust_remote_code=trust_remote_code, use_fast=True
        )

        # Version validation for vllm.
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError(
                        "PPO LoRA is not supported before vllm 0.7.3"
                    )

        # Define worker classes based on the actor strategy.
        # [GEM Notes] We only support FSDP for now.
        actor_rollout_cls = GEMActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup

        # Map roles to their corresponding remote worker classes.
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
        }

        # Define the resource pool specification.
        # Map roles to the resource pool.
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # We should adopt a multi-source reward function here:
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # finally, we combine all the rewards together
        # The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # Add a reference policy worker if KL loss or KL reward is used.
        if (
            config.algorithm.use_kl_in_reward
            or config.actor_rollout_ref.actor.use_kl_loss
        ):
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping
        )

        # Initialize the PPO trainer.
        trainer = ReinforceGEMTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            # [GEM Notes] We don't need explicit reward functions.
            # reward_fn=reward_fn,
            # val_reward_fn=val_reward_fn,
            device_name=config.trainer.device,
        )
        # Initialize the workers of the trainer.
        trainer.init_workers()
        # Start the training process.
        trainer.fit()


if __name__ == "__main__":
    main()
