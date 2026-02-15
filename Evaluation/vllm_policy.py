"""
vLLM Policy Wrapper for EnvEval Difficulty Evaluation.

This module provides a policy wrapper that uses vLLM for inference,
compatible with PassAtKEvaluator (both fixed and progressive difficulty modes).

Refactored to align with training inference patterns:
- Uses same TEMPLATE_FACTORY as training
- Uses same action extraction (extract_last_boxed_answer / extract_action_parameters / code blocks)
- Simplified interface for use with ObservationWrapper
"""

import threading
from typing import Optional
import os
import re
import multiprocessing as mp

# vLLM uses multiprocessing workers; CUDA cannot be re-initialized after fork.
# Make evaluation scripts robust even when launched outside the provided bash wrappers.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
try:
    mp.set_start_method("spawn")
except RuntimeError:
    # Start method was already set by the parent process (e.g., interactive session).
    pass

# Import unified action extraction from gem
import sys
gem_path = os.path.join(os.path.dirname(__file__), '..', '..', 'gem')
sys.path.insert(0, os.path.abspath(gem_path))
from gem.utils.parsing import extract_action_parameters, extract_last_boxed_answer
# from gem.utils.constants import INVALID_ACTION
INVALID_ACTION = "<｜INVALID_ACTION｜>"
from gem.utils.prompt_templates import TEMPLATE_FACTORY


class VLLMPolicyWrapper:
    """
    Policy wrapper using vLLM for local model inference.

    Works with ObservationWrapper which handles history management.
    Call signature: policy_fn(observation: str, temperature: float) -> str
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        max_tokens: int = 2048,
        max_model_len: Optional[int] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        prompt_template: str = "qwen3_game",
        apply_chat_template: bool = False,
        trust_remote_code: bool = True,
        return_raw: bool = True,
        **vllm_kwargs
    ):
        """
        Initialize vLLM policy wrapper with thread safety.

        FIX: Added initialization state tracking for thread safety.

        Args:
            model_name: HuggingFace model name or local path
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            max_tokens: Maximum tokens to generate
            max_model_len: Max model length for truncation/skip checks (aligns with training behavior)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization ratio (0.0-1.0)
            prompt_template: Template type ("qwen3_game", "no", "qwen3_general", "code", "qwen3_tool")
            apply_chat_template: Whether to apply tokenizer's chat template
            trust_remote_code: Whether to trust remote code in model
            return_raw: Return raw generation text to the env (training-aligned)
            **vllm_kwargs: Additional arguments passed to vLLM
        """
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.max_model_len = int(max_model_len) if max_model_len is not None else None
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.prompt_template = prompt_template
        self.apply_chat_template = apply_chat_template
        self.trust_remote_code = trust_remote_code
        self.return_raw = bool(return_raw)

        # Keep passing max_model_len to vLLM engine even if the caller provided it
        # via the explicit argument (common in evaluation scripts).
        if self.max_model_len is not None:
            vllm_kwargs = dict(vllm_kwargs)
            vllm_kwargs.setdefault("max_model_len", self.max_model_len)
        self.vllm_kwargs = vllm_kwargs

        # Last-call debug hooks (used by Pass@K per-step logging).
        # - raw responses are the direct vLLM generations (before boxed extraction/wrapping)
        # - actions are the strings returned to the environment
        self.last_raw_responses: list[str] = []
        self.last_actions: list[str] = []
        self.last_extracted_actions: list[str] = []
        self.last_response_is_truncated: list[bool] = []
        self.last_generation_failed: list[bool] = []

        # Lazy initialization
        self.llm = None
        self.sampling_params = None
        self.tokenizer = None
        self._lock = threading.Lock()

        # FIX: Add initialization state tracking for thread safety
        self._initializing = False
        self._initialization_error = None

        print(f"VLLMPolicyWrapper initialized with model: {model_name}")
        print(f"  Template: {prompt_template}")
        print(f"  Return raw: {self.return_raw}")
        print(f"  Note: vLLM will be loaded on first policy call (lazy init)")

    def _initialize_llm(self):
        """
        Lazy initialization of vLLM model with thread safety.

        FIX: Prevents concurrent initialization and handles errors properly.
        """
        # FIX: Prevent concurrent initialization
        if self._initializing:
            # Wait for ongoing initialization
            import time
            max_wait = 60  # seconds
            waited = 0
            while self._initializing and waited < max_wait:
                time.sleep(0.1)
                waited += 0.1

            if self._initialization_error:
                raise self._initialization_error

            return

        self._initializing = True

        # Preflight: vLLM 0.8.x requires at least one visible CUDA GPU.
        # If no GPU is available, vLLM may appear to "hang" while waiting for worker outputs.
        try:
            import torch

            visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if visible_gpus == 0:
                self._initialization_error = RuntimeError(
                    "No CUDA GPU detected (torch.cuda.is_available() is False / device_count is 0). "
                    "vLLM requires a visible GPU. If you're in Docker/Slurm, ensure GPUs are "
                    "requested and passed through (e.g., `--gpus all` / a GPU partition), and "
                    "check `CUDA_VISIBLE_DEVICES` and `nvidia-smi`."
                )
                raise self._initialization_error

            if self.tensor_parallel_size > visible_gpus:
                self._initialization_error = RuntimeError(
                    f"tensor_parallel_size={self.tensor_parallel_size} exceeds visible GPUs={visible_gpus}. "
                    "Reduce tensor_parallel_size or make more GPUs visible (CUDA_VISIBLE_DEVICES)."
                )
                raise self._initialization_error
        except ImportError:
            # If torch isn't available, let vLLM initialization fail with its own error.
            pass

        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            self._initialization_error = ImportError(
                "vLLM is not installed. Install it with: pip install vllm>=0.8.0"
            )
            self._initializing = False
            raise self._initialization_error

        try:
            print(f"\nInitializing vLLM model: {self.model_name}")
            print(f"  - Tensor parallel size: {self.tensor_parallel_size}")
            print(f"  - GPU memory utilization: {self.gpu_memory_utilization}")

            # Create LLM instance
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                trust_remote_code=self.trust_remote_code,
                **self.vllm_kwargs
            )

            # Create sampling parameters
            self.sampling_params = SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_tokens=self.max_tokens,
                n=1
            )

            # Get tokenizer
            self.tokenizer = self.llm.get_tokenizer()

            print("vLLM model loaded successfully!\n")

            self._initialization_error = None

        except Exception as e:
            self._initialization_error = e
            print(f"Error initializing vLLM: {e}")
            raise

        finally:
            self._initializing = False

    def ensure_initialized(self) -> None:
        """
        Public, thread-safe initializer.

        Useful when the caller wants to deterministically load the model once
        (e.g., before launching parallel evaluation workers).
        """
        if self.llm is None:
            with self._lock:
                if self.llm is None:
                    self._initialize_llm()

    def __del__(self):
        """
        Cleanup GPU resources on deletion.

        FIX: Explicitly releases GPU memory to prevent leaks.
        """
        if hasattr(self, 'llm') and self.llm is not None:
            try:
                # FIX: Explicitly cleanup vLLM resources
                del self.llm
                self.llm = None

                # Force garbage collection
                import gc
                gc.collect()

                # Clear CUDA cache if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except ImportError:
                    pass

            except Exception as e:
                print(f"Warning: Error during GPU cleanup: {e}")

    def __call__(self, observation: str, temperature: Optional[float] = None):
        return self.act_batch([observation], temperature=temperature)[0]

    def _format_prompt(self, observation: str) -> str:
        prompt = TEMPLATE_FACTORY[self.prompt_template](observation)

        if self.apply_chat_template and self.tokenizer:
            try:
                prompt = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as e:
                print(f"Warning: Failed to apply chat template: {e}")

        return prompt

    def extract_action(self, text: str) -> str:
        """
        Training-aligned action extraction (used for metrics/debug only).

        NOTE: For full alignment, the default behavior is to return the raw
        model generation to the environment (see `return_raw=True`), since GEM
        envs typically parse the action from the full assistant output.
        """
        if not text:
            return ""

        formatted_action: Optional[str] = None
        if self.prompt_template in ["qwen3_game", "qwen3_general"] or (
            self.prompt_template == "no" and "qwen" in self.model_name.lower()
        ):
            formatted_action = extract_last_boxed_answer(text)
            if formatted_action is None:
                formatted_action = text.strip()
        elif self.prompt_template == "qwen3_tool":
            formatted_action = extract_action_parameters(text)
        elif self.prompt_template == "code":
            code_blocks = re.findall(r"```(?:\\w+)?\\n(.*?)```", text, re.DOTALL)
            formatted_action = code_blocks[-1].strip() if code_blocks else None
        else:
            raise NotImplementedError(f"Unknown prompt_template: {self.prompt_template}")

        if formatted_action is None:
            formatted_action = INVALID_ACTION
        return formatted_action

    def act_batch(self, observations: list[str], temperature: Optional[float] = None) -> list[str]:
        """
        Batched action generation for vectorized evaluation.

        Args:
            observations: List of observations (already formatted by wrappers)
            temperature: Optional override temperature

        Returns:
            List of actions in \\boxed{...} format, same length/order as inputs.
        """
        if temperature is None:
            temperature = self.temperature

        self.ensure_initialized()

        prompts = [self._format_prompt(o) for o in observations]

        exceeds_lengths = [False for _ in prompts]
        if self.max_model_len is not None and self.tokenizer is not None:
            try:
                idss = self.tokenizer(prompts).input_ids
                exceeds_lengths = [len(ids) >= self.max_model_len for ids in idss]
            except Exception:
                exceeds_lengths = [False for _ in prompts]

        sub_prompts = [p for p, e in zip(prompts, exceeds_lengths) if not e]

        sampling_params = self.sampling_params
        if temperature != self.temperature:
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_tokens=self.max_tokens,
                n=1,
            )

        try:
            sub_outputs = self.llm.generate(sub_prompts, sampling_params=sampling_params)
        except Exception as e:
            print(f"Error during vLLM batch generation: {e}")
            sub_outputs = []

        raw_responses: list[str] = []
        actions: list[str] = []
        extracted_actions: list[str] = []
        truncated_flags: list[bool] = []
        generation_failed_flags: list[bool] = []

        sub_i = 0
        for exceeds in exceeds_lengths:
            if exceeds:
                raw_responses.append("")
                actions.append(INVALID_ACTION)
                extracted_actions.append(INVALID_ACTION)
                truncated_flags.append(False)
                generation_failed_flags.append(True)
                continue

            if sub_i >= len(sub_outputs):
                raw_responses.append("")
                actions.append(INVALID_ACTION)
                extracted_actions.append(INVALID_ACTION)
                truncated_flags.append(False)
                generation_failed_flags.append(True)
                continue

            out = sub_outputs[sub_i]
            sub_i += 1

            raw = out.outputs[0].text if out.outputs else ""
            finish_reason = getattr(out.outputs[0], "finish_reason", None) if out.outputs else None
            response_is_truncated = bool(finish_reason == "length")

            raw_responses.append(str(raw))
            extracted = INVALID_ACTION if response_is_truncated else self.extract_action(str(raw))
            extracted_actions.append(str(extracted))
            truncated_flags.append(response_is_truncated)
            generation_failed_flags.append(False)

            if response_is_truncated:
                actions.append(INVALID_ACTION)
            else:
                actions.append(str(raw) if self.return_raw else str(extracted))

        self.last_raw_responses = raw_responses
        self.last_actions = actions
        self.last_extracted_actions = extracted_actions
        self.last_response_is_truncated = truncated_flags
        self.last_generation_failed = generation_failed_flags
        return actions

    def __repr__(self) -> str:
        return (
            f"VLLMPolicyWrapper("
            f"model={self.model_name}, "
            f"template={self.prompt_template}, "
            f"temp={self.temperature}, "
            f"loaded={self.llm is not None})"
        )


def create_vllm_policy(
    model_name: str = "Qwen/Qwen3-8B",
    temperature: float = 0.8,
    **kwargs
) -> VLLMPolicyWrapper:
    """
    Convenience function to create a vLLM policy.

    Args:
        model_name: HuggingFace model name or local path
        temperature: Sampling temperature
        **kwargs: Additional arguments passed to VLLMPolicyWrapper

    Returns:
        VLLMPolicyWrapper instance

    Example:
        policy = create_vllm_policy("Qwen/Qwen3-8B", temperature=0.6, prompt_template="qwen3_game")
        evaluator = PassAtKEvaluator(policy_fn=policy, ...)
    """
    return VLLMPolicyWrapper(
        model_name=model_name,
        temperature=temperature,
        **kwargs
    )
