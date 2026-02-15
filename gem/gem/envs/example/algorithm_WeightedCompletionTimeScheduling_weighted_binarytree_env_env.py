from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class WeightedCompletionTimeSchedulingEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 8,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 8

        self.complexity_params = {
            # Number of jobs: more jobs increases combinatorial action space and evaluation depth
            "num_jobs": (3, 15),
            # Max processing time: larger values increase arithmetic difficulty and ratio diversity
            "processing_time_max": (6, 50),
            # Max weight: larger values increase arithmetic difficulty and ratio diversity
            "weight_max": (5, 100),
        }

        self.param_variance = {
            "num_jobs": 1,                # ±1 around interpolated count
            "processing_time_max": 5,     # ±5 around interpolated cap
            "weight_max": 10,             # ±10 around interpolated cap
        }

        self.num_jobs: int = 0
        self.processing_time_max: int = 0
        self.weight_max: int = 0

        self.turn_count: int = 0
        self.jobs: list = []
        self.opt_order: list = []
        self.opt_value: int = 0
        self.best_value: Optional[int] = None
        self.best_order: Optional[list] = None
        self.last_order: Optional[list] = None
        self.last_order_text: Optional[str] = None

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for name, (min_val, max_val) in self.complexity_params.items():
            center = min_val + (max_val - min_val) * normalized
            val = center
            if self.enable_param_randomization:
                var = self.param_variance.get(name, 0)
                if var > 0:
                    val = center + random.uniform(-var, var)
            if min_val <= max_val:
                val = max(min_val, min(max_val, val))
            else:
                val = max(max_val, min(min_val, val))
            setattr(self, name, int(round(val)))

    def _get_instructions(self) -> str:
        example = self.sample_random_action()
        return (
            "Algorithm Scheduling Game: Minimize total weighted completion time on a single machine.\n"
            "You are given jobs with processing times p_i and weights w_i. A schedule is an ordering of all jobs.\n"
            "Evaluation: For an order J1, J2, ..., the completion time of job k is the sum of processing times up to k.\n"
            "Objective = sum over jobs of (w_i * C_i). Lower is better; your goal is the global optimum.\n"
            "Rules:\n"
            "- Submit a complete permutation that uses each job exactly once.\n"
            "- Jobs are identified by integers starting at 1.\n"
            "- You may try multiple times until success or timeout.\n"
            "Format your action as \\boxed{ordered-job-ids}, e.g., \\boxed{1-3-2} or \\boxed{2,1,3}.\n"
            f"Example submission: {example}\n"
        )

    def get_task_suffix(self) -> str:
        lines = []
        lines.append("Current instance:")
        for j in self.jobs:
            lines.append(f"- Job {j['id']}: p={j['p']} w={j['w']}")
        status = []
        status.append(f"Turns used: {self.turn_count}/{self.max_turns}")
        if self.best_value is not None:
            status.append(f"Best value so far: {self.best_value}")
        else:
            status.append("Best value so far: N/A")
        suffix = "\n".join(lines + status)
        return suffix + "\nEnter a complete order in \\boxed{...} format."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        self.turn_count = 0
        self.jobs = []
        self.opt_order = []
        self.opt_value = 0
        self.best_value = None
        self.best_order = None
        self.last_order = None
        self.last_order_text = None

        for i in range(1, self.num_jobs + 1):
            p = random.randint(1, self.processing_time_max)
            w = random.randint(1, self.weight_max)
            self.jobs.append({"id": i, "p": p, "w": w, "ratio": p / float(w)})

        self.opt_order = [j["id"] for j in sorted(self.jobs, key=lambda x: (x["ratio"], x["id"]))]
        self.opt_value = self._objective(self.opt_order)

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _objective(self, order: list) -> int:
        total = 0
        time_acc = 0
        job_map = {j["id"]: j for j in self.jobs}
        for job_id in order:
            p = job_map[job_id]["p"]
            w = job_map[job_id]["w"]
            time_acc += p
            total += w * time_acc
        return int(total)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if parsed is None:
            obs = (
                f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}} with a full permutation. "
                "Episode terminated."
            )
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        order = parsed
        self.last_order = order
        self.last_order_text = "-".join(str(x) for x in order)

        valid_set = set(order)
        expected_set = set(range(1, self.num_jobs + 1))
        if len(order) != self.num_jobs or valid_set != expected_set:
            obs = (
                f"At turn {self.turn_count}, invalid permutation: expected each job id 1..{self.num_jobs} exactly once. "
                "Episode terminated."
            )
            return obs, -0.25, True, False, {"suffix": self.get_task_suffix()}

        val = self._objective(order)
        if self.best_value is None or val < self.best_value:
            self.best_value = val
            self.best_order = order

        if val == self.opt_value:
            obs = (
                f"Success! Optimal total weighted completion time achieved: {val}. "
                f"Accepted order: {self.last_order_text}."
            )
            return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns}). Episode timed out."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        obs = (
            f"Attempt {self.turn_count} evaluated: total weighted completion time = {val}. "
            f"Best so far = {self.best_value}. Try again."
        )
        return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Any]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        ids = re.findall(r'\d+', content)
        if not ids:
            return None
        order = [int(x) for x in ids]
        return order

    def sample_random_action(self) -> str:
        if self.num_jobs <= 0:
            return "\\boxed{1}"
        perm = list(range(1, self.num_jobs + 1))
        random.shuffle(perm)
        return "\\boxed{" + "-".join(str(x) for x in perm) + "}"


class WeightedCompletionTimeSchedulingEnvWithFeedback(WeightedCompletionTimeSchedulingEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        text = obs.lower()

        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint: Optional[str] = None

        if "invalid action format" in text or "use \\boxed" in text:
            error_type = "FormatError"
            error_detail["issue"] = "missing_or_bad_boxed_format"
            hint = "Use \\boxed{1-2-3-...} with all job IDs in order."

        elif "invalid permutation" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "not_a_complete_permutation"
            error_detail["expected_ids"] = list(range(1, getattr(self, "num_jobs", 0) + 1))
            if self.last_order is not None:
                error_detail["received_ids"] = self.last_order
            hint = "Include every job ID exactly once from 1..N; avoid duplicates or missing IDs."

        elif "reached max turns" in text or truncated:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Order jobs by smaller processing-time-to-weight ratio to approach optimum quickly."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        else:
            error_type = "WrongDecision"
            error_detail["last_value"] = self._safe_int_extract(text)
            error_detail["best_so_far"] = getattr(self, "best_value", None)
            error_detail["job_count"] = getattr(self, "num_jobs", None)
            error_detail["last_order"] = getattr(self, "last_order", None)
            hint = "Try ordering jobs by ascending p_i / w_i; break ties by job ID."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["instance_size"] = getattr(self, "num_jobs", None)
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def _safe_int_extract(self, text: str) -> Optional[int]:
        m = re.search(r"completion time\s*=\s*(\d+)", text)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
        return None

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Compute C_i by accumulating processing times along the order; aim to minimize sum w_i * C_i.",
            "turn": 0,
            "instance_size": getattr(self, "num_jobs", None),
        }
        return obs, info