from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class RuneRingPuzzleScalediffEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = False,
        max_turns: Optional[int] = 100,
        **_,
    ):
        super().__init__()
        self.complexity = int(max(1, min(10, complexity)))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 100

        # Evolvable parameters
        self.complexity_params = {
            "ring_size": (6, 25),            # widened upper bound to hit L10 endpoint
            "move_budget": (9, 2),           # REVERSED: fewer moves harder
            "scramble_depth": (2, 10),
            "max_rotate_step": (1, 5),
            "op_set_richness": (1, 4),
        }

        # Variance settings
        self.param_variance = {
            "ring_size": 0,
            "move_budget": 0,
            "scramble_depth": 0,
            "max_rotate_step": 0,
            "op_set_richness": 0,
        }

        # Placeholder attributes
        self.ring_size: int = 0
        self.move_budget: int = 0
        self.scramble_depth: int = 0
        self.max_rotate_step: int = 0
        self.op_set_richness: int = 0

        # Other state
        self.turn_count: int = 0
        self.current_ring: Optional[list] = None
        self.target_ring: Optional[list] = None
        self.history: list = []
        self.allowed_ops: list = []
        self._last_parsed_action: Optional[Dict[str, Any]] = None

        self.reset()

    def _apply_complexity_params(self):
        table = {

            1: (6, 8, 2, 1, 1),

            2: (9, 7, 3, 1, 1),

            3: (11, 6, 4, 2, 2),

            4: (13, 6, 5, 2, 2),

            5: (15, 5, 6, 3, 2),

            6: (17, 5, 7, 3, 3),

            7: (19, 4, 8, 4, 3),

            8: (21, 4, 9, 4, 3),

            9: (23, 3, 10, 5, 4),

            10: (25, 2, 10, 5, 4),

        }

        level = int(self.complexity)
        params = table.get(level, table[max(table.keys())])
        (ring_size, move_budget, scramble_depth, max_rotate_step, op_set_richness) = params

    def _get_instructions(self) -> str:
        ops_desc = []
        if "toggle" in self.allowed_ops:
            ops_desc.append(f"- toggle:i → flip rune at position i (1..{self.ring_size})")
        if "invert" in self.allowed_ops:
            ops_desc.append(f"- invert:s-e → flip all runes from s to e (1..{self.ring_size}, s<=e)")
        if "rotate" in self.allowed_ops:
            ops_desc.append(f"- rotate:+k or rotate:-k → rotate ring by k (1..{self.max_rotate_step})")
        if "mirror" in self.allowed_ops:
            ops_desc.append("- mirror → reverse the ring order")

        return (
            "You are playing Rune Ring Puzzle.\n"
            "Goal: transform the current ring of runes to exactly match the target ring within the allowed move budget.\n"
            "Moves are deterministic and consume 1 budget each (commit is free). If you submit an invalid move or exceed budget, the episode ends with no reward.\n"
            "Operators:\n" + "\n".join(ops_desc) + "\n"
            "Use commit when you believe the rings match. There is no partial reward.\n"
            "Format your action as \\boxed{operator:params} or \\boxed{mirror} or \\boxed{commit}.\n"
            f"Example: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        ring_str = "".join(str(x) for x in self.current_ring)
        target_str = "".join(str(x) for x in self.target_ring)
        ops_list = ", ".join(self.allowed_ops)
        return (
            f"Target ring: {target_str}\n"
            f"Current ring: {ring_str}\n"
            f"Ring size: {self.ring_size}, Remaining budget: {self.move_budget}\n"
            f"Allowed ops: {ops_list} (rotate step up to {self.max_rotate_step})\n"
            "Enter your action in \\boxed{...} format: one of "
            "toggle:i, invert:s-e, rotate:+k or rotate:-k, mirror, commit."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()

        # Determine allowed operators based on richness
        richness = self.op_set_richness
        base_ops = ["toggle"]
        if richness >= 2:
            base_ops.append("invert")
        if richness >= 3:
            base_ops.append("rotate")
        if richness >= 4:
            base_ops.append("mirror")
        self.allowed_ops = base_ops

        # Initialize ring and target via solvable scramble from baseline
        self.current_ring = [0 for _ in range(self.ring_size)]
        # Generate target by applying scramble moves to a copy of baseline
        baseline = [0 for _ in range(self.ring_size)]
        # Ensure scramble depth does not exceed move_budget for solvability
        self.scramble_depth = min(self.scramble_depth, self.move_budget)
        target = baseline[:]
        for _ in range(self.scramble_depth):
            op_name = random.choice([op for op in self.allowed_ops if op != "mirror" or self.ring_size > 1])
            if op_name == "toggle":
                i = random.randint(1, self.ring_size)
                target[i - 1] ^= 1
            elif op_name == "invert":
                s = random.randint(1, self.ring_size)
                e = random.randint(s, self.ring_size)
                for idx in range(s - 1, e):
                    target[idx] ^= 1
            elif op_name == "rotate":
                k = random.randint(1, self.max_rotate_step)
                if random.random() < 0.5:
                    k = -k
                k_mod = k % self.ring_size
                target = target[-k_mod:] + target[:-k_mod]
            elif op_name == "mirror":
                target = list(reversed(target))
        self.target_ring = target

        self.turn_count = 0
        self.history = []
        self._last_parsed_action = None

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if not parsed:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        act_type = parsed["type"]
        params = parsed.get("params", {})
        self._last_parsed_action = parsed

        # Unsupported action
        if act_type not in self.allowed_ops and act_type != "commit":
            obs = f"Protocol violation: Unsupported action '{act_type}'."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        # Budget check: non-commit actions consume budget
        if act_type != "commit":
            if self.move_budget <= 0:
                obs = "Protocol violation: budget exceeded. No moves remaining."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        # Validate and apply actions
        if act_type == "toggle":
            i = params.get("i")
            if not isinstance(i, int) or i < 1 or i > self.ring_size:
                obs = f"Protocol violation: toggle index out of range (1..{self.ring_size})."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            self.current_ring[i - 1] ^= 1
            self.move_budget -= 1
            self.history.append(("toggle", i))

        elif act_type == "invert":
            s = params.get("s")
            e = params.get("e")
            if not isinstance(s, int) or not isinstance(e, int) or s < 1 or e < 1 or s > e or e > self.ring_size:
                obs = f"Protocol violation: invert requires s-e with 1<=s<=e<= {self.ring_size}."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            for idx in range(s - 1, e):
                self.current_ring[idx] ^= 1
            self.move_budget -= 1
            self.history.append(("invert", s, e))

        elif act_type == "rotate":
            k = params.get("k")
            if not isinstance(k, int) or k == 0 or abs(k) > self.max_rotate_step:
                obs = f"Protocol violation: rotate step must be non-zero and within ±{self.max_rotate_step}."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            k_mod = k % self.ring_size
            self.current_ring = self.current_ring[-k_mod:] + self.current_ring[:-k_mod]
            self.move_budget -= 1
            self.history.append(("rotate", k))

        elif act_type == "mirror":
            self.current_ring = list(reversed(self.current_ring))
            self.move_budget -= 1
            self.history.append(("mirror",))

        elif act_type == "commit":
            if self.current_ring == self.target_ring:
                obs = "Success! Configuration matches the target. Episode finished."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = "Commit submitted, but configuration does not match target. Episode failed."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        # Max turns check after action
        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns}). Episode timed out."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        # Continue
        obs = (
            f"At turn {self.turn_count}, applied {act_type}. "
            f"Remaining budget: {self.move_budget}."
        )
        return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        extracted = matches[-1].group(1).strip().lower()

        if extracted == "commit":
            return {"type": "commit"}
        if extracted == "mirror":
            return {"type": "mirror"}

        if ":" not in extracted:
            return None
        op, param_str = extracted.split(":", 1)
        op = op.strip()

        if op == "toggle":
            try:
                i = int(param_str.strip())
                return {"type": "toggle", "params": {"i": i}}
            except ValueError:
                return None

        if op == "invert":
            # Expect s-e
            m = re.match(r'\s*(\d+)\s*-\s*(\d+)\s*$', param_str)
            if not m:
                return None
            s = int(m.group(1))
            e = int(m.group(2))
            return {"type": "invert", "params": {"s": s, "e": e}}

        if op == "rotate":
            # Expect +k or -k or plain integer
            m = re.match(r'\s*([+-]?\d+)\s*$', param_str)
            if not m:
                return None
            k = int(m.group(1))
            return {"type": "rotate", "params": {"k": k}}

        return None

    def sample_random_action(self) -> str:
        if not self.allowed_ops:
            return "\\boxed{commit}"
        op = random.choice(self.allowed_ops + ["commit"])
        if op == "toggle":
            i = random.randint(1, max(1, self.ring_size))
            return f"\\boxed{{toggle:{i}}}"
        if op == "invert":
            s = random.randint(1, max(1, self.ring_size))
            e = random.randint(s, max(1, self.ring_size))
            return f"\\boxed{{invert:{s}-{e}}}"
        if op == "rotate":
            k = random.randint(1, max(1, self.max_rotate_step))
            sign = random.choice(["+", "-"])
            return f"\\boxed{{rotate:{sign}{k}}}"
        if op == "mirror":
            return "\\boxed{mirror}"
        return "\\boxed{commit}"


class RuneRingPuzzleScalediffEnvWithFeedback(RuneRingPuzzleScalediffEnv):
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
            error_detail["issue"] = "missing_or_malformed_boxed"
            hint = "Format your move as \\boxed{toggle:3}, \\boxed{invert:2-5}, \\boxed{rotate:+2}, \\boxed{mirror}, or \\boxed{commit}."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "unsupported action" in text:
                error_detail["violation"] = "unsupported_action"
                got = self._last_parsed_action["type"] if self._last_parsed_action else None
                error_detail["got"] = got
                hint = f"Use only allowed ops: {', '.join(self.allowed_ops)} or commit."
            elif "budget exceeded" in text:
                error_detail["violation"] = "budget_exceeded"
                hint = "Commit when you think the rings match; non-commit moves consume budget."
            elif "toggle index out of range" in text:
                error_detail["violation"] = "index_out_of_range"
                hint = f"Choose i between 1 and {self.ring_size} for toggle."
            elif "invert requires s-e" in text:
                error_detail["violation"] = "segment_out_of_range"
                hint = f"Use invert:s-e with 1<=s<=e<= {self.ring_size}."
            elif "rotate step" in text:
                error_detail["violation"] = "rotate_step_invalid"
                hint = f"Use rotate:+k or rotate:-k where 1<=k<= {self.max_rotate_step}."
            else:
                hint = "Check operator parameters and budget before acting."

        elif "commit submitted, but configuration does not match" in text or "episode failed" in text:
            error_type = "WrongDecision"
            error_detail["expected"] = "".join(str(x) for x in self.target_ring)
            error_detail["got"] = "".join(str(x) for x in self.current_ring)
            hint = "Compare the current and target strings and choose an operator that reduces mismatches before committing."

        elif "reached max turns" in text or "timed out" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Act efficiently: prioritize moves that change many mismatched positions or commit sooner."

        elif "success" in text and "matches the target" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "ring_size": self.ring_size,
                "remaining_budget": self.move_budget,
                "allowed_ops": list(self.allowed_ops),
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
            "hint": "Start by identifying indices where current and target differ; use invert over a segment or toggle single runes.",
            "turn": 0,
            "state": {
                "ring_size": self.ring_size,
                "remaining_budget": self.move_budget,
                "allowed_ops": list(self.allowed_ops),
            },
        }
        return obs, info
