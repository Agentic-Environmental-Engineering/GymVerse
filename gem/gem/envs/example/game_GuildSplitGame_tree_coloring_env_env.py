from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List, Set


class GuildSplitGameEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 100,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 100

        self.complexity_params = {
            "num_entities": (6, 12),          # Roster size: larger N = more combinations and harder
            "selection_size": (2, 6),         # Team size: larger k increases combination complexity = harder
            "free_eval_limit": (5, 1),        # REVERSED: fewer free evaluations = harder (resource constraint)
            "max_edge_weight": (5, 15),       # Larger weight range yields subtler differences and harder optimization
        }

        self.param_variance = {
            "num_entities": 1,        # ±1 to vary roster size slightly
            "selection_size": 0,      # small range → fix at interpolated value for stability
            "free_eval_limit": 0,     # small range → fix at interpolated value
            "max_edge_weight": 2,     # ±2 (~20% of small range) introduces variety without chaos
        }

        self.turn_count: int = 0
        self.num_entities: int = 0
        self.selection_size: int = 0
        self.free_eval_limit: int = 0
        self.max_edge_weight: int = 0

        self.entities: List[str] = []
        self.weights: List[List[int]] = []
        self.best_value: int = 0
        self.best_subset: List[int] = []
        self.evals_used: int = 0

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for param_name, (min_val, max_val) in self.complexity_params.items():
            center_value = min_val + (max_val - min_val) * normalized
            actual_value = center_value
            if self.enable_param_randomization:
                variance = self.param_variance.get(param_name, 0)
                if variance > 0:
                    actual_value = center_value + random.uniform(-variance, variance)
            # Clamp with reversed support
            lo = min(min_val, max_val)
            hi = max(min_val, max_val)
            if actual_value < lo:
                actual_value = lo
            if actual_value > hi:
                actual_value = hi
            setattr(self, param_name, int(round(actual_value)))
        # Feasibility adjustments
        if self.selection_size >= self.num_entities:
            self.selection_size = max(1, self.num_entities - 1)
        if self.selection_size < 1:
            self.selection_size = 1
        if self.free_eval_limit < 0:
            self.free_eval_limit = 0
        if self.max_edge_weight < 1:
            self.max_edge_weight = 1

    def _get_instructions(self) -> str:
        return (
            "Guild Split Challenge\n"
            "Goal: Select exactly K distinct guild members (IDs) to form Team A to maximize rivalry across the split.\n"
            "Scoring: The score is the sum of rivalry weights between Team A and the remaining members.\n"
            "Actions:\n"
            "- Evaluate: check the score of a candidate split without ending the episode. Use: \\boxed{evaluate: id1,id2,...}\n"
            "- Submit: finalize your split and end the episode. Use: \\boxed{submit: id1,id2,...}\n"
            "- Help: show instructions again. Use: \\boxed{help}\n"
            "Rules:\n"
            "- You must list exactly K distinct IDs each time you evaluate or submit.\n"
            "- Format errors or protocol violations end the episode with a penalty.\n"
            f"For example: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        roster_lines = []
        for i in range(self.num_entities):
            roster_lines.append(f"{i+1}: Hero{i+1}")
        edge_lines = []
        for i in range(self.num_entities):
            for j in range(i + 1, self.num_entities):
                edge_lines.append(f"{i+1}-{j+1}: {self.weights[i][j]}")
        suffix = (
            "Instance:\n"
            f"- Roster size N={self.num_entities}\n"
            f"- Selection size K={self.selection_size}\n"
            f"- Free evaluations remaining={max(0, self.free_eval_limit - self.evals_used)}\n"
            "- Members (ID: Name):\n"
            + ", ".join(roster_lines)
            + "\n- Rivalry weights (i-j: w):\n"
            + ", ".join(edge_lines)
            + "\nEnter your action in \\boxed{...} format: evaluate: ids OR submit: ids OR help"
        )
        return suffix

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()
        self.turn_count = 0
        self.evals_used = 0
        self.entities = [f"Hero{i+1}" for i in range(self.num_entities)]
        self.weights = [[0 for _ in range(self.num_entities)] for _ in range(self.num_entities)]
        for i in range(self.num_entities):
            for j in range(i + 1, self.num_entities):
                w = random.randint(1, self.max_edge_weight)
                self.weights[i][j] = w
                self.weights[j][i] = w
        self.best_value, self.best_subset = self._compute_optimal()
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)
        if parsed is None:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}} with evaluate/submit/help."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
        cmd = parsed["cmd"]
        ids = parsed.get("ids", [])

        if cmd == "help":
            obs = "Instructions:\n" + self._get_instructions()
            if self.turn_count >= self.max_turns:
                return f"{obs}\nReached max turns ({self.max_turns}).", 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if cmd in ("evaluate", "submit"):
            valid, reason = self._validate_ids(ids)
            if not valid:
                obs = f"Protocol violation: {reason}. Episode ends."
                return obs, -0.25, True, False, {"suffix": self.get_task_suffix()}

            score = self._cut_value(ids)
            if cmd == "evaluate":
                self.evals_used += 1
                penalty = 0.0
                if self.evals_used > self.free_eval_limit:
                    penalty = -0.05
                obs = (
                    f"Evaluation: split {self._fmt_ids(ids)} yields score={score}. "
                    f"Optimal score is hidden. {'Penalty applied.' if penalty < 0 else 'No penalty.'}"
                )
                if self.turn_count >= self.max_turns:
                    return f"{obs}\nReached max turns ({self.max_turns}).", 0.0, True, True, {"suffix": self.get_task_suffix()}
                return obs, penalty, False, False, {"suffix": self.get_task_suffix()}

            if cmd == "submit":
                if score == self.best_value:
                    obs = (
                        f"Success! Your split {self._fmt_ids(ids)} attains the optimal score={score}. "
                        f"Episode terminated."
                    )
                    return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
                else:
                    obs = (
                        f"Failed. Your split {self._fmt_ids(ids)} scores {score}, "
                        f"but the optimal score={self.best_value}. Episode terminated."
                    )
                    return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        obs = f"Unsupported action '{cmd}'. Episode ends."
        return obs, -0.25, True, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action:
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        m = pattern.search(action)
        if not m:
            return None
        content = m.group(1).strip().lower()
        if content in ("help", "h"):
            return {"cmd": "help"}

        # Detect command and ids
        cmd = None
        ids_part = ""
        if content.startswith("submit:"):
            cmd = "submit"
            ids_part = content[len("submit:"):].strip()
        elif content.startswith("evaluate:"):
            cmd = "evaluate"
            ids_part = content[len("evaluate:"):].strip()
        elif content.startswith("eval:"):
            cmd = "evaluate"
            ids_part = content[len("eval:"):].strip()
        elif content.startswith("score:"):
            cmd = "evaluate"
            ids_part = content[len("score:"):].strip()
        else:
            # Try to interpret pure list as submit
            if re.search(r'\d', content):
                cmd = "submit"
                ids_part = content
            else:
                return {"cmd": content}

        ids = self._parse_ids(ids_part)
        return {"cmd": cmd, "ids": ids}

    def sample_random_action(self) -> str:
        ids = list(range(1, max(2, self.selection_size) + 1))
        return f"\\boxed{{evaluate: {','.join(str(x) for x in ids)}}}"

    def _parse_ids(self, text: str) -> List[int]:
        nums = re.findall(r'\d+', text)
        return [int(x) for x in nums]

    def _validate_ids(self, ids: List[int]) -> Tuple[bool, str]:
        if len(ids) != self.selection_size:
            return False, f"expected exactly {self.selection_size} IDs"
        seen: Set[int] = set()
        for x in ids:
            if x < 1 or x > self.num_entities:
                return False, f"ID {x} out of range 1..{self.num_entities}"
            if x in seen:
                return False, f"duplicate ID {x}"
            seen.add(x)
        return True, ""

    def _cut_value(self, subset_ids: List[int]) -> int:
        sset = set(x - 1 for x in subset_ids)
        complement = [i for i in range(self.num_entities) if i not in sset]
        total = 0
        for i in sset:
            for j in complement:
                total += self.weights[i][j]
        return total

    def _fmt_ids(self, ids: List[int]) -> str:
        return "[" + ", ".join(str(x) for x in ids) + "]"

    def _compute_optimal(self) -> Tuple[int, List[int]]:
        n = self.num_entities
        k = self.selection_size
        combo = list(range(k))  # 0-indexed internal IDs
        best_score = -1
        best_subset = []
        while True:
            ids = [x + 1 for x in combo]
            score = self._cut_value(ids)
            if score > best_score:
                best_score = score
                best_subset = ids[:]
            if not self._next_combination(combo, n, k):
                break
        return best_score, best_subset

    def _next_combination(self, combo: List[int], n: int, k: int) -> bool:
        i = k - 1
        while i >= 0 and combo[i] == n - k + i:
            i -= 1
        if i < 0:
            return False
        combo[i] += 1
        for j in range(i + 1, k):
            combo[j] = combo[j - 1] + 1
        return True


class GuildSplitGameEnvWithFeedback(GuildSplitGameEnv):
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
            error_detail["issue"] = "missing_boxed_format"
            hint = "Wrap your command in \\boxed{...} and use 'evaluate:' or 'submit:'."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            # Extract reason if present
            m = re.search(r'protocol violation: (.+?)\. episode', text)
            if m:
                error_detail["violation"] = m.group(1)
            else:
                error_detail["violation"] = "unknown"
            hint = f"Provide exactly {self.selection_size} unique IDs in range 1..{self.num_entities}."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["cmd"] = "unknown_command"
            hint = "Use 'evaluate:', 'submit:', or 'help'."
        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["max_turns"] = self.max_turns
            hint = "Plan fewer evaluations and submit earlier."
        elif "failed" in text and "optimal score" in text:
            error_type = "WrongDecision"
            # Extract scores
            got_m = re.search(r"scores (\d+)", text)
            opt_m = re.search(r"optimal score=(\d+)", text)
            if got_m:
                error_detail["got_score"] = int(got_m.group(1))
            if opt_m:
                error_detail["optimal_score"] = int(opt_m.group(1))
            hint = "Evaluate promising splits and favor members with high rivalry to those excluded."
        elif "success" in text and "optimal score" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None
        elif "evaluation:" in text:
            error_type = "OK"
            error_detail["outcome"] = "evaluation"
            # Extract score
            sc = re.search(r"yields score=(\d+)", text)
            if sc:
                error_detail["score"] = int(sc.group(1))
            hint = "Use remaining free evaluations to compare alternatives before submitting."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "N": self.num_entities,
                "K": self.selection_size,
                "free_eval_remaining": max(0, self.free_eval_limit - self.evals_used),
            }
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": f"Start by evaluating a reasonable split of {self.selection_size} unique IDs.",
            "turn": 0,
            "state": {
                "N": self.num_entities,
                "K": self.selection_size,
                "free_eval_remaining": self.free_eval_limit,
            },
        }
        return obs, info