import json
import os
import random
import re
import sys
import ast
import copy
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gem.core import Env
from gem.utils.constants import LanguageGameReward


_BOXED_RE = re.compile(r"\\boxed\{(.*)\}\s*$", re.DOTALL)

_BFCL_DATA_CACHE: Dict[Tuple[str, str, str, bool], Tuple[List[dict], List[dict]]] = {}


def _maybe_unbox(action: str) -> str:
    match = _BOXED_RE.search(action.strip())
    if match:
        return match.group(1).strip()
    return action.strip()


_BOXED_INSTRUCTION = (
    "Additionally, wrap your entire output in \\\\boxed{...}. "
    "Do not write anything outside the \\\\boxed{...} wrapper."
)


def _default_bfcl_root() -> Path:
    # gem/gem/envs/BFCL/bfcl_env.py -> BFCL/
    return Path(__file__).resolve().parent / "gorilla" / "berkeley-function-call-leaderboard"


@dataclass(frozen=True)
class _BFCLRuntime:
    # Minimal surface we need from BFCL.
    ReturnFormat: Any
    Language: Any
    ast_parse: Any
    ast_checker: Any
    multi_turn_checker: Any
    execute_multi_turn_func_call: Any
    agentic_checker: Any
    load_dataset_entry: Any
    load_ground_truth_entry: Any
    extract_memory_backend_type: Any
    parse_prompt_variation_params: Any
    is_function_calling_format_output: Any
    is_empty_output: Any
    system_prompt_pre_processing_chat_model: Any
    default_decode_ast_prompting: Any
    default_decode_execute_prompting: Any
    is_empty_execute_response: Any
    DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING: str
    MAXIMUM_STEP_LIMIT: int


def _import_bfcl(bfcl_root: Path, project_root: Optional[Path]) -> _BFCLRuntime:
    bfcl_root = bfcl_root.resolve()
    if not bfcl_root.exists():
        raise FileNotFoundError(f"BFCL root not found: {bfcl_root}")

    # `bfcl_eval.utils` imports `bfcl_eval.constants.eval_config` which reads BFCL_PROJECT_ROOT.
    if project_root is not None:
        os.environ["BFCL_PROJECT_ROOT"] = str(Path(project_root).resolve())

    if str(bfcl_root) not in sys.path:
        sys.path.insert(0, str(bfcl_root))

    from bfcl_eval.constants.enums import Language, ReturnFormat
    from bfcl_eval.eval_checker.agentic_eval.agentic_checker import agentic_checker
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
        execute_multi_turn_func_call,
        is_empty_execute_response,
    )
    from bfcl_eval.model_handler.utils import ast_parse
    from bfcl_eval.model_handler.utils import parse_prompt_variation_params
    from bfcl_eval.model_handler.utils import (
        default_decode_ast_prompting,
        default_decode_execute_prompting,
        system_prompt_pre_processing_chat_model,
    )
    from bfcl_eval.constants.default_prompts import (
        DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING,
        MAXIMUM_STEP_LIMIT,
    )
    from bfcl_eval.utils import (
        extract_memory_backend_type,
        is_empty_output,
        is_function_calling_format_output,
        load_dataset_entry,
        load_ground_truth_entry,
    )

    return _BFCLRuntime(
        ReturnFormat=ReturnFormat,
        Language=Language,
        ast_parse=ast_parse,
        ast_checker=ast_checker,
        multi_turn_checker=multi_turn_checker,
        execute_multi_turn_func_call=execute_multi_turn_func_call,
        agentic_checker=agentic_checker,
        load_dataset_entry=load_dataset_entry,
        load_ground_truth_entry=load_ground_truth_entry,
        extract_memory_backend_type=extract_memory_backend_type,
        parse_prompt_variation_params=parse_prompt_variation_params,
        is_function_calling_format_output=is_function_calling_format_output,
        is_empty_output=is_empty_output,
        system_prompt_pre_processing_chat_model=system_prompt_pre_processing_chat_model,
        default_decode_ast_prompting=default_decode_ast_prompting,
        default_decode_execute_prompting=default_decode_execute_prompting,
        is_empty_execute_response=is_empty_execute_response,
        DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING=DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING,
        MAXIMUM_STEP_LIMIT=int(MAXIMUM_STEP_LIMIT),
    )


class BFCLEnv(Env):
    """
    BFCL wrapper in GEM's Env interface.

    - Each episode samples one BFCL test entry.
    - The agent action is the model's function-call output as a string.

    This wrapper follows BFCL's native prompting semantics:
    - The observation is a fully formatted chat prompt (system prompt + user/tool history),
      matching BFCL's prompting runner format.
    - For multi-turn categories, a turn ends when the model output is empty/undecodable.
    """

    def __init__(
        self,
        test_category: str = "simple_python",
        bfcl_root: Optional[str] = None,
        bfcl_project_root: Optional[str] = None,
        model_name: str = "gorilla-openfunctions-v2",
        prompt_variant_index: int = 0,
        sampling: str = "random",
        allowed_ids: Optional[List[str]] = None,
        max_steps: int = 64,
        execute_tools: Optional[bool] = None,
        include_prereq: bool = False,
        seed: Optional[int] = None,
        **_,
    ):
        super().__init__()

        self.test_category = test_category
        self._base_model_name = model_name
        # BFCL executable backends cache tool instances globally keyed by (model_name, test_entry_id, class_name).
        # Add a per-env nonce to avoid accidental cross-run contamination.
        self._instance_nonce = secrets.token_hex(4)
        self.model_name = f"{model_name}_{self._instance_nonce}"
        self.prompt_variant_index = prompt_variant_index
        self.sampling = sampling
        self.allowed_ids = set(allowed_ids) if allowed_ids else None
        self.max_steps = max_steps
        self.execute_tools = execute_tools
        self.include_prereq = include_prereq

        self._bfcl_root = Path(bfcl_root) if bfcl_root else _default_bfcl_root()
        self._bfcl_project_root = (
            Path(bfcl_project_root) if bfcl_project_root else None
        )
        self._bfcl = _import_bfcl(self._bfcl_root, self._bfcl_project_root)

        self._rng = random.Random(seed)
        self._step_count = 0
        self._dataset: List[dict] = []
        self._gt_by_id: Dict[str, dict] = {}
        self._gt_list: Optional[List[dict]] = None
        self._id_to_index: Dict[str, int] = {}

        self._current: Optional[dict] = None
        self._current_gt: Optional[dict] = None
        self._current_index: Optional[int] = None

        # Multi-turn / agentic runtime state
        self._turn_index: int = 0
        self._turn_decoded_calls: List[List[List[str]]] = []
        self._turn_execution_results: List[List[str]] = []
        self._last_freeform_message: Optional[str] = None

        # BFCL-native prompting protocol state
        self._prompt_messages: List[dict] = []
        self._steps_in_current_turn: int = 0

        self._load()

    def _load(self) -> None:
        root_key = str(self._bfcl_root.resolve())
        project_key = (
            str(self._bfcl_project_root.resolve()) if self._bfcl_project_root else ""
        )
        cache_key = (root_key, project_key, self.test_category, bool(self.include_prereq))
        cached = _BFCL_DATA_CACHE.get(cache_key)
        if cached is None:
            dataset = self._bfcl.load_dataset_entry(
                self.test_category, include_prereq=self.include_prereq
            )
            try:
                ground_truth = self._bfcl.load_ground_truth_entry(self.test_category)
            except FileNotFoundError:
                ground_truth = []
            _BFCL_DATA_CACHE[cache_key] = (dataset, ground_truth)
        else:
            dataset, ground_truth = cached
        gt_by_id = {row["id"]: row for row in ground_truth} if ground_truth else {}

        if self.allowed_ids is not None:
            if self.test_category == "format_sensitivity" and ground_truth:
                filtered_dataset: List[dict] = []
                filtered_gt: List[dict] = []
                for row, gt in zip(dataset, ground_truth):
                    if row.get("id") in self.allowed_ids:
                        filtered_dataset.append(row)
                        filtered_gt.append(gt)
                dataset = filtered_dataset
                ground_truth = filtered_gt
                gt_by_id = {}
            else:
                dataset = [row for row in dataset if row.get("id") in self.allowed_ids]

        if not dataset:
            raise ValueError(
                f"No BFCL entries loaded for category={self.test_category} "
                f"(allowed_ids={len(self.allowed_ids) if self.allowed_ids else 'all'})."
            )

        # Populate minimal initial_config for agentic categories so tool execution can work.
        if self.test_category.startswith("memory_"):
            model_result_dir = (
                (self._bfcl_project_root or self._bfcl_root) / "result" / "gem_env"
            )
            model_result_dir.mkdir(parents=True, exist_ok=True)
            for entry in dataset:
                if "initial_config" in entry and entry["initial_config"] is not None:
                    continue
                involved_classes = entry.get("involved_classes") or []
                if not involved_classes:
                    continue
                entry["initial_config"] = {
                    involved_classes[0]: {
                        "model_result_dir": model_result_dir,
                        "scenario": entry.get("scenario", ""),
                        "test_id": entry.get("id", ""),
                        "test_category": self.test_category,
                    }
                }

        if self.test_category.startswith("web_search"):
            for entry in dataset:
                if "initial_config" in entry and entry["initial_config"] is not None:
                    continue
                involved_classes = entry.get("involved_classes") or []
                if not involved_classes:
                    continue
                entry["initial_config"] = {
                    involved_classes[0]: {
                        "show_snippet": False
                        if "no_snippet" in self.test_category
                        else True
                    }
                }

        self._dataset = dataset
        self._gt_by_id = gt_by_id
        self._gt_list = (
            ground_truth if self.test_category == "format_sensitivity" else None
        )
        self._id_to_index = {row.get("id"): i for i, row in enumerate(self._dataset) if row.get("id")}

    def dataset_size(self) -> int:
        return len(self._dataset)

    def dataset_id_at(self, index: int) -> Optional[str]:
        if 0 <= int(index) < len(self._dataset):
            return self._dataset[int(index)].get("id")
        return None

    def _is_multi_turn(self) -> bool:
        return self.test_category.startswith("multi_turn")

    def _is_agentic(self) -> bool:
        return self.test_category.startswith("memory_") or self.test_category.startswith(
            "web_search"
        )

    def _is_format_sensitivity(self) -> bool:
        return self.test_category == "format_sensitivity"

    def _is_relevance_or_irrelevance(self) -> bool:
        return "relevance" in self.test_category or "irrelevance" in self.test_category

    def _effective_execute_tools(self) -> bool:
        if self.execute_tools is not None:
            return self.execute_tools
        if self._is_multi_turn():
            return True
        if self.test_category.startswith("memory_"):
            # Safe-ish by default; heavy deps for vector backend might be missing.
            backend = self._bfcl.extract_memory_backend_type(self.test_category)
            return backend in {"kv", "rec_sum"}
        if self.test_category.startswith("web_search"):
            # Web search typically needs network; default to offline.
            return False
        return False

    @staticmethod
    def _strip_reasoning_qwen(text: str) -> str:
        # Match BFCL Qwen handlers: if the model emits <think>...</think>, keep only the final answer part.
        if not isinstance(text, str):
            return str(text)
        if "</think>" in text:
            return text.split("</think>", 1)[1].lstrip("\n")
        return text

    @staticmethod
    def _inject_boxed_instruction(messages: List[dict]) -> None:
        for msg in messages:
            if msg.get("role") != "system":
                continue
            content = msg.get("content")
            if not isinstance(content, str):
                continue
            if "\\boxed" in content:
                return
            msg["content"] = f"{content.rstrip()}\n\n{_BOXED_INSTRUCTION}\n"
            return

    def _format_prompt_qwen(self, messages: List[dict]) -> str:
        """
        Match BFCL's Qwen prompting formatter (`bfcl_eval.model_handler.local_inference.qwen.QwenHandler._format_prompt`).
        """
        formatted_prompt = ""

        if messages and messages[0].get("role") == "system":
            formatted_prompt += (
                f"<|im_start|>system\n{messages[0].get('content','')}<|im_end|>\n"
            )

        last_query_index = len(messages) - 1
        for offset, message in enumerate(reversed(messages)):
            idx = len(messages) - 1 - offset
            if (
                message.get("role") == "user"
                and isinstance(message.get("content"), str)
                and not (
                    message["content"].startswith("<tool_response>")
                    and message["content"].endswith("</tool_response>")
                )
            ):
                last_query_index = idx
                break

        for idx, message in enumerate(messages):
            role = message.get("role")
            content = message.get("content", "")

            if role == "user" or (role == "system" and idx != 0):
                formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

            elif role == "assistant":
                reasoning_content = ""
                if isinstance(message.get("reasoning_content"), str) and message.get(
                    "reasoning_content"
                ):
                    reasoning_content = message["reasoning_content"]

                elif isinstance(content, str) and "</think>" in content:
                    parts = content.split("</think>")
                    reasoning_content = (
                        parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                    )
                    content = parts[-1].lstrip("\n")

                if idx > last_query_index:
                    if idx == len(messages) - 1 or reasoning_content:
                        formatted_prompt += (
                            f"<|im_start|>{role}\n<think>\n"
                            + reasoning_content.strip("\n")
                            + "\n</think>\n\n"
                            + str(content).lstrip("\n")
                        )
                    else:
                        formatted_prompt += f"<|im_start|>{role}\n{content}"
                else:
                    formatted_prompt += f"<|im_start|>{role}\n{content}"

                formatted_prompt += "<|im_end|>\n"

            elif role == "tool":
                prev_role = messages[idx - 1].get("role") if idx > 0 else None
                next_role = (
                    messages[idx + 1].get("role") if idx < len(messages) - 1 else None
                )

                if idx == 0 or prev_role != "tool":
                    formatted_prompt += "<|im_start|>user"

                formatted_prompt += f"\n<tool_response>\n{content}\n</tool_response>"

                if idx == len(messages) - 1 or next_role != "tool":
                    formatted_prompt += "<|im_end|>\n"

        formatted_prompt += "<|im_start|>assistant\n"
        return formatted_prompt

    def _prompting_next_turn_messages(self, entry: dict, turn_idx: int) -> List[dict]:
        holdout_function: dict = entry.get("missed_function", {}) or {}
        if str(turn_idx) in holdout_function:
            return [
                {
                    "role": "user",
                    "content": self._bfcl.DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING.format(
                        functions=holdout_function[str(turn_idx)]
                    ),
                }
            ]

        question = entry.get("question", [])
        if isinstance(question, list) and 0 <= int(turn_idx) < len(question):
            return copy.deepcopy(question[int(turn_idx)])
        return [{"role": "user", "content": "(missing question)"}]

    @staticmethod
    def _num_turns(entry: dict) -> int:
        question = entry.get("question", [])
        return len(question) if isinstance(question, list) else 1

    def _infer_language(self) -> Any:
        # BFCL uses language-specific categories only for "simple_*".
        if "simple_java" in self.test_category:
            return self._bfcl.Language.JAVA
        if "simple_javascript" in self.test_category:
            return self._bfcl.Language.JAVASCRIPT
        return self._bfcl.Language.PYTHON

    def _infer_return_format(self) -> Any:
        if "simple_java" in self.test_category:
            return self._bfcl.ReturnFormat.JAVA
        if "simple_javascript" in self.test_category:
            return self._bfcl.ReturnFormat.JAVASCRIPT
        return self._bfcl.ReturnFormat.PYTHON

    def _ground_truth_id(self, entry_id: str) -> str:
        if self.test_category.startswith("web_search"):
            return entry_id.replace(self.test_category, "web_search", 1)
        if self.test_category.startswith("memory_"):
            backend = self._bfcl.extract_memory_backend_type(self.test_category)
            return entry_id.replace(f"memory_{backend}", "memory", 1)
        return entry_id

    def reset(
        self,
        seed: Optional[int] = None,
        entry_index: Optional[int] = None,
        bfcl_id: Optional[str] = None,
        **_,
    ) -> Tuple[str, dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            self._rng.seed(seed)

        self._step_count = 0
        self._turn_index = 0
        self._turn_decoded_calls = []
        self._turn_execution_results = []
        self._last_freeform_message = None
        self._prompt_messages = []
        self._steps_in_current_turn = 0

        if entry_index is not None and bfcl_id is not None:
            raise ValueError("Pass at most one of entry_index or bfcl_id.")

        if bfcl_id is not None:
            if bfcl_id not in self._id_to_index:
                raise ValueError(f"Unknown bfcl_id: {bfcl_id}")
            self._current_index = self._id_to_index[bfcl_id]
        elif entry_index is not None:
            if not (0 <= int(entry_index) < len(self._dataset)):
                raise ValueError(
                    f"entry_index out of range: {entry_index} (len={len(self._dataset)})"
                )
            self._current_index = int(entry_index)
        else:
            if self.sampling == "random":
                self._current_index = self._rng.randrange(len(self._dataset))
            elif self.sampling == "sequential":
                self._current_index = self._rng.randrange(len(self._dataset))
            else:
                raise ValueError(f"Unsupported sampling: {self.sampling}")

        assert self._current_index is not None
        self._current = self._dataset[self._current_index]

        bfcl_id = self._current.get("id")
        if self._gt_list is not None:
            # Format-sensitivity ground truth is aligned by index, not by id.
            self._current_gt = self._gt_list[self._current_index]
        else:
            self._current_gt = self._gt_by_id.get(self._ground_truth_id(bfcl_id))

        if self._is_multi_turn() or self._is_agentic():
            self._turn_decoded_calls = [[] for _ in range(self._num_turns(self._current))]
        functions = self._current.get("function", []) or []
        entry_id = self._current.get("id") or ""

        if self._is_multi_turn() or self._is_agentic():
            first_turn = self._prompting_next_turn_messages(self._current, 0)
            # Only the first turn receives the BFCL system prompt (+ tools), matching BFCL prompting runners.
            first_turn = self._bfcl.system_prompt_pre_processing_chat_model(
                first_turn, functions, entry_id
            )
            self._prompt_messages.extend(first_turn)
        else:
            question = self._current.get("question", [])
            if not question:
                prompts = [{"role": "user", "content": "(missing question)"}]
            else:
                idx = min(int(self.prompt_variant_index), len(question) - 1)
                prompts = copy.deepcopy(question[idx])
            prompts = self._bfcl.system_prompt_pre_processing_chat_model(
                prompts, functions, entry_id
            )
            self._prompt_messages.extend(prompts)

        self._inject_boxed_instruction(self._prompt_messages)
        obs = self._format_prompt_qwen(self._prompt_messages)
        return obs, {"bfcl_id": bfcl_id, "entry_index": self._current_index}

    def _finalize_episode(self) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        assert self._current is not None
        bfcl_id = self._current.get("id")

        # Agentic: check final answer string.
        if self._is_agentic():
            if not self._current_gt:
                obs = "No ground truth available for this entry."
                return obs, LanguageGameReward.fail_reward, True, False, {
                    "bfcl_id": bfcl_id,
                }
            if not self._last_freeform_message:
                obs = "No final answer message produced."
                return obs, LanguageGameReward.fail_reward, True, False, {
                    "bfcl_id": bfcl_id,
                }
            possible_answers = self._current_gt.get("ground_truth", [])
            checker = self._bfcl.agentic_checker(self._last_freeform_message, possible_answers)
            ok = bool(checker.get("valid"))
            reward = LanguageGameReward.success_reward if ok else LanguageGameReward.fail_reward
            obs = "Correct." if ok else "Incorrect."
            return obs, reward, True, False, {
                "bfcl_id": bfcl_id,
                "bfcl_category": self.test_category,
                "bfcl_checker": checker,
                "final_message": self._last_freeform_message,
            }

        # Multi-turn: execute BFCL checker against ground truth tool sequences.
        if self._is_multi_turn():
            if not self._current_gt:
                obs = "No ground truth available for this entry."
                return obs, LanguageGameReward.fail_reward, True, False, {
                    "bfcl_id": bfcl_id,
                }
            ground_truth_turns = self._current_gt.get("ground_truth", [])
            checker = self._bfcl.multi_turn_checker(
                multi_turn_model_result_list_decoded=self._turn_decoded_calls,
                multi_turn_ground_truth_list=ground_truth_turns,
                test_entry=self._current,
                test_category=self.test_category,
                model_name=self.model_name,
            )
            ok = bool(checker.get("valid"))
            reward = LanguageGameReward.success_reward if ok else LanguageGameReward.fail_reward
            obs = "Correct." if ok else "Incorrect."
            return obs, reward, True, False, {
                "bfcl_id": bfcl_id,
                "bfcl_category": self.test_category,
                "bfcl_checker": checker,
                "decoded_calls": self._turn_decoded_calls,
            }

        # Fallback: terminate.
        obs = "Done."
        return obs, LanguageGameReward.internal_step_reward, True, False, {
            "bfcl_id": bfcl_id,
        }

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self._step_count += 1

        if not self._current:
            obs = "No active BFCL test case. Call reset() first."
            return obs, LanguageGameReward.fail_reward, True, False, {"bfcl_id": None}

        bfcl_id = self._current.get("id")

        if self._step_count >= self.max_steps:
            obs = "Truncated: max steps reached."
            return obs, LanguageGameReward.fail_reward, False, True, {"bfcl_id": bfcl_id}

        raw_action = self._strip_reasoning_qwen(action if isinstance(action, str) else str(action))
        raw_action = _maybe_unbox(raw_action)

        # Format sensitivity: BFCL-native decode based on per-entry config and run AST checker.
        if self._is_format_sensitivity():
            if not self._current_gt:
                obs = "No ground truth available for this entry."
                return obs, LanguageGameReward.fail_reward, True, False, {"bfcl_id": bfcl_id}

            entry_id = str(bfcl_id or "")
            if ":" not in entry_id or len(entry_id.split(":")) != 3:
                obs = "Invalid format_sensitivity entry id."
                return obs, LanguageGameReward.fail_reward, True, False, {"bfcl_id": entry_id}

            config = entry_id.split(":")[1]
            try:
                (
                    return_format_str,
                    has_tool_call_tag,
                    _func_doc_fmt,
                    _prompt_fmt,
                    _style,
                ) = self._bfcl.parse_prompt_variation_params(config)
                return_format = self._bfcl.ReturnFormat(return_format_str)
            except Exception as e:
                obs = f"Failed to parse format config: {e}"
                return obs, LanguageGameReward.fail_reward, True, False, {"bfcl_id": entry_id}

            try:
                decoded = self._bfcl.ast_parse(
                    raw_action, language=return_format, has_tool_call_tag=has_tool_call_tag
                )
            except Exception as e:
                obs = f"Decode failed: {e}"
                return obs, LanguageGameReward.format_error_reward, True, False, {
                    "bfcl_id": entry_id,
                    "decode_error": str(e),
                }

            if not self._bfcl.is_function_calling_format_output(decoded):
                obs = "Wrong output format."
                return obs, LanguageGameReward.format_error_reward, True, False, {
                    "bfcl_id": entry_id,
                    "decoded": decoded,
                }

            prompt_function = self._current.get("function", [])
            possible_answer_item = self._current_gt.get("ground_truth", [])
            checker_category = entry_id.split(":")[-1]
            checker = self._bfcl.ast_checker(
                func_description=prompt_function,
                model_output=decoded,
                possible_answer=possible_answer_item,
                language=self._bfcl.Language.PYTHON,
                test_category=checker_category,
                model_name=self.model_name,
            )
            ok = bool(checker.get("valid"))
            reward = LanguageGameReward.success_reward if ok else LanguageGameReward.fail_reward
            obs = "Correct." if ok else "Incorrect."
            return obs, reward, True, False, {
                "bfcl_id": entry_id,
                "bfcl_category": self.test_category,
                "format_config": config,
                "decoded_calls": decoded,
                "bfcl_checker": checker,
            }

        # Multi-turn / agentic prompting loop (BFCL-native).
        if self._is_multi_turn() or self._is_agentic():
            # BFCL prompting runner enforces a per-turn step limit.
            if self._steps_in_current_turn > self._bfcl.MAXIMUM_STEP_LIMIT:
                obs = "Forced quit: maximum BFCL prompting steps reached."
                return obs, LanguageGameReward.fail_reward, True, False, {
                    "bfcl_id": bfcl_id,
                    "error_type": "bfcl_prompting:forced_quit",
                }

            self._prompt_messages.append({"role": "assistant", "content": raw_action})

            decoded_calls: Optional[List[str]] = None
            decode_error: Optional[str] = None
            try:
                decoded_calls = self._bfcl.default_decode_execute_prompting(
                    raw_action, has_tool_call_tag=False
                )
            except Exception as e:
                decode_error = str(e)

            # End current turn on decode failure or empty output (BFCL semantics).
            if decode_error is not None or self._bfcl.is_empty_execute_response(
                decoded_calls or []
            ):
                if self._is_agentic() and decode_error is not None:
                    self._last_freeform_message = raw_action

                self._turn_index += 1
                self._steps_in_current_turn = 0
                if self._turn_index >= self._num_turns(self._current):
                    return self._finalize_episode()

                next_turn = self._prompting_next_turn_messages(self._current, self._turn_index)
                self._prompt_messages.extend(next_turn)
                obs = self._format_prompt_qwen(self._prompt_messages)
                return obs, LanguageGameReward.internal_step_reward, False, False, {
                    "bfcl_id": bfcl_id,
                    "decode_error": decode_error,
                }

            # Record this sub-step in the current turn for BFCL evaluation.
            if self._turn_index >= len(self._turn_decoded_calls):
                self._turn_decoded_calls.append([])
            self._turn_decoded_calls[self._turn_index].append(list(decoded_calls))

            execution_results: List[str] = []
            if self._effective_execute_tools():
                try:
                    execution_results, _instances = self._bfcl.execute_multi_turn_func_call(
                        func_call_list=list(decoded_calls),
                        initial_config=self._current.get("initial_config", {}) or {},
                        involved_classes=self._current.get("involved_classes", []) or [],
                        model_name=f"gem_{self.model_name}",
                        test_entry_id=self._current.get("id") or "unknown",
                        long_context=("long_context" in self.test_category),
                        is_evaL_run=False,
                    )
                except Exception as e:
                    execution_results = [f"Error during tool execution: {e}"]

            for execution_result, decoded_model_response in zip(execution_results, decoded_calls):
                self._prompt_messages.append(
                    {"role": "tool", "name": decoded_model_response, "content": execution_result}
                )

            self._steps_in_current_turn += 1
            obs = self._format_prompt_qwen(self._prompt_messages)
            return obs, LanguageGameReward.internal_step_reward, False, False, {
                "bfcl_id": bfcl_id,
                "executed_calls": decoded_calls,
                "execution_results": execution_results,
            }

        # Relevance/Irrelevance: success depends on whether a tool call is present.
        if self._is_relevance_or_irrelevance():
            decoded = None
            decode_error = None
            try:
                decoded = self._bfcl.ast_parse(
                    raw_action,
                    language=self._bfcl.ReturnFormat.PYTHON,
                    has_tool_call_tag=False,
                )
                has_tool_call = not self._bfcl.is_empty_output(decoded)
            except Exception as e:
                decode_error = str(e)
                has_tool_call = False
            ok = (not has_tool_call) if "irrelevance" in self.test_category else bool(has_tool_call)
            reward = LanguageGameReward.success_reward if ok else LanguageGameReward.fail_reward
            obs = "Correct." if ok else "Incorrect."
            return obs, reward, True, False, {
                "bfcl_id": bfcl_id,
                "has_tool_call": has_tool_call,
                "decoded": decoded,
                "decode_error": decode_error,
            }

        # Single-turn AST-style evaluation (BFCL-native prompting decode).
        if not self._current_gt:
            obs = "No ground truth available for this entry."
            return obs, LanguageGameReward.fail_reward, True, False, {"bfcl_id": bfcl_id}

        try:
            decoded = self._bfcl.default_decode_ast_prompting(
                raw_action,
                language=self._infer_return_format(),
                has_tool_call_tag=False,
            )
        except Exception as e:
            obs = f"Decode failed: {e}"
            return obs, LanguageGameReward.format_error_reward, True, False, {
                "bfcl_id": bfcl_id,
                "decode_error": str(e),
            }

        func_description = self._current.get("function", [])
        possible_answer = self._current_gt.get("ground_truth", [])
        checker_result = self._bfcl.ast_checker(
            func_description=func_description,
            model_output=decoded,
            possible_answer=possible_answer,
            language=self._infer_language(),
            test_category=self.test_category,
            model_name=self.model_name,
        )
        ok = bool(checker_result.get("valid"))
        reward = LanguageGameReward.success_reward if ok else LanguageGameReward.fail_reward
        obs = "Correct." if ok else "Incorrect."
        info: Dict[str, Any] = {
            "bfcl_id": bfcl_id,
            "bfcl_category": self.test_category,
            "bfcl_checker": checker_result,
            "decoded_calls": decoded,
        }
        return obs, reward, True, False, info

    def sample_random_action(self) -> str:
        # A minimal syntactically-valid placeholder.
        return "some_function(param=1)"

    def spawn(self, same_state: bool = False, **kwargs) -> "BFCLEnv":
        child = BFCLEnv(
            test_category=self.test_category,
            bfcl_root=str(self._bfcl_root),
            bfcl_project_root=str(self._bfcl_project_root) if self._bfcl_project_root else None,
            model_name=self._base_model_name,
            prompt_variant_index=self.prompt_variant_index,
            sampling=self.sampling,
            allowed_ids=list(self.allowed_ids) if self.allowed_ids else None,
            max_steps=self.max_steps,
            execute_tools=self.execute_tools,
            include_prereq=self.include_prereq,
            **kwargs,
        )
        if same_state:
            child._rng.setstate(self._rng.getstate())
            child._current = self._current
            child._current_gt = self._current_gt
            child._step_count = self._step_count
        return child
