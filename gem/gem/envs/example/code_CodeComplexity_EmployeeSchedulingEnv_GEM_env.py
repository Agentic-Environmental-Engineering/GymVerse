from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class CodeComplexityEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 50,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, complexity))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 50

        self.complexity_params = {
            # Function length: larger function means more lines and structure to inspect → harder
            "function_length": (12, 60),
            # Number of if blocks: more decision points → harder
            "num_if_blocks": (1, 8),
            # Maximum elif per if: more branching inside each if → harder
            "max_elif_per_if": (0, 3),
            # Number of loops (for/while): more decision points → harder
            "num_loops": (0, 6),
            # Number of switch/case blocks: more branching → harder
            "num_cases_blocks": (0, 3),
            # Cases per block: more case labels → harder
            "cases_per_block": (2, 6),
            # Boolean operators (&&, ||) inside conditions: each adds a decision point → harder
            "num_boolean_ops": (0, 12),
            # Ternary operators: each adds a decision point → harder
            "num_ternary_ops": (0, 6),
            # REVERSED: query budget for count/peek operations; fewer queries → harder
            "query_budget": (7, 2),
            # REVERSED: number of preview lines shown in suffix; less preview → harder
            "preview_lines": (6, 2),
            # REVERSED: maximum lines allowed per peek; smaller span → harder
            "peek_span_limit": (6, 3),
        }

        self.param_variance = {
            "function_length": 5,
            "num_if_blocks": 1,
            "max_elif_per_if": 1,
            "num_loops": 1,
            "num_cases_blocks": 1,
            "cases_per_block": 1,
            "num_boolean_ops": 2,
            "num_ternary_ops": 1,
            "query_budget": 1,
            "preview_lines": 0,
            "peek_span_limit": 0,
        }

        self.function_length: int = 0
        self.num_if_blocks: int = 0
        self.max_elif_per_if: int = 0
        self.num_loops: int = 0
        self.num_cases_blocks: int = 0
        self.cases_per_block: int = 0
        self.num_boolean_ops: int = 0
        self.num_ternary_ops: int = 0
        self.query_budget: int = 0
        self.preview_lines: int = 0
        self.peek_span_limit: int = 0

        self.turn_count: int = 0
        self.lines: list = []
        self.allowed_keywords = ["if", "elif", "for", "while", "case", "boolean", "ternary"]

        self.counts: Dict[str, int] = {}
        self.true_complexity: int = 0
        self.last_submission: Optional[int] = None

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
            # clamp to range; support reversed params
            lo, hi = (min_val, max_val) if min_val <= max_val else (max_val, min_val)
            actual_value = max(lo, min(hi, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _generate_function(self):
        lines = []
        lines.append("function f(a, b, c) {")
        # base fillers
        fillers = max(0, self.function_length - 2)  # minus header + return

        # distribute constructs
        num_if = self.num_if_blocks
        num_loops = self.num_loops
        num_cases = self.num_cases_blocks
        num_ternary = self.num_ternary_ops

        # elifs distribution
        total_elif = 0
        elif_per_if = []
        for _ in range(num_if):
            x = 0 if self.max_elif_per_if <= 0 else random.randint(0, self.max_elif_per_if)
            elif_per_if.append(x)
            total_elif += x

        # boolean ops distribution across condition-bearing constructs
        cond_slots = num_if + total_elif + num_loops
        boolean_remaining = self.num_boolean_ops
        boolean_assignments = []
        if cond_slots > 0:
            for i in range(cond_slots):
                boolean_assignments.append(0)
            # greedily distribute
            idxs = list(range(cond_slots))
            while boolean_remaining > 0:
                i = random.choice(idxs)
                boolean_assignments[i] += 1
                boolean_remaining -= 1

        # generate if/elif blocks
        cond_index = 0
        for i in range(num_if):
            ops_in_if = boolean_assignments[cond_index] if cond_index < len(boolean_assignments) else 0
            cond_index += 1
            cond = self._make_condition(ops_in_if)
            lines.append(f"  if ({cond}) {{")
            lines.append("    x += 1;")
            lines.append("  }")
            for k in range(elif_per_if[i]):
                ops_in_elif = boolean_assignments[cond_index] if cond_index < len(boolean_assignments) else 0
                cond_index += 1
                cond_e = self._make_condition(ops_in_elif)
                lines.append(f"  else if ({cond_e}) {{")
                lines.append("    y -= 2;")
                lines.append("  }")

        # generate loops
        for i in range(num_loops):
            loop_type = random.choice(["for", "while"])
            if loop_type == "for":
                lines.append(f"  for (i = 0; i < n; i++) {{")
                lines.append("    z = z + i;")
                lines.append("  }")
            else:
                ops_in_while = boolean_assignments[cond_index] if cond_index < len(boolean_assignments) else 0
                cond_index += 1
                cond_w = self._make_condition(ops_in_while)
                lines.append(f"  while ({cond_w}) {{")
                lines.append("    w = w ^ 2;")
                lines.append("    break;")
                lines.append("  }")

        # generate switch/case blocks
        for _ in range(num_cases):
            lines.append("  switch (t) {")
            for c in range(self.cases_per_block):
                lines.append(f"    case {c}:")
                lines.append("      t = t + 1;")
                lines.append("      break;")
            lines.append("    default:")
            lines.append("      t = 0;")
            lines.append("  }")

        # generate ternary ops
        for _ in range(num_ternary):
            cond = self._make_condition(0)
            lines.append(f"  r = {cond} ? a : b;")

        # filler lines to reach approximate length
        while len(lines) < self.function_length - 1:
            lines.append(f"  s{len(lines)} = s{max(0, len(lines)-2)} + 1;")

        lines.append("  return r;")
        lines.append("}")
        self.lines = lines

        # compute counts and true complexity
        counts = {
            "if": num_if,
            "elif": total_elif,
            "for": sum(1 for l in lines if l.strip().startswith("for ")),
            "while": sum(1 for l in lines if l.strip().startswith("while ")),
            "case": num_cases * self.cases_per_block,
            "boolean": self.num_boolean_ops,
            "ternary": num_ternary,
        }
        # loops counted via num_loops, but to be robust we recompute for/while individually
        self.counts = counts
        self.true_complexity = (
            1
            + counts["if"]
            + counts["elif"]
            + counts["for"]
            + counts["while"]
            + counts["case"]
            + counts["boolean"]
            + counts["ternary"]
        )

    def _make_condition(self, extra_ops: int) -> str:
        base = random.choice(["a > b", "n < m", "x == y", "k != 0"])
        ops = []
        for _ in range(extra_ops):
            ops.append(random.choice(["&&", "||"]))
        cond = base
        for op in ops:
            cond += f" {op} " + random.choice(["c > d", "u <= v", "p != q", "i < j"])
        return cond

    def _get_instructions(self) -> str:
        return (
            "Static Code Analysis Game.\n"
            "Goal: compute the cyclomatic complexity of the given pseudo-code function.\n"
            "Cyclomatic complexity = 1 + (#if) + (#elif) + (#for) + (#while) + (#case labels) + (#boolean operators &&/||) + (#ternary ?: operators).\n"
            "You have a limited query budget to inspect the code.\n"
            "Actions:\n"
            "- count: KEYWORD   where KEYWORD in {if, elif, for, while, case, boolean, ternary}\n"
            "- peek: START SPAN   reveal SPAN consecutive lines starting from 1-based START (SPAN <= peek_span_limit)\n"
            "- submit: complexity N   submit your final integer answer\n"
            "Use \\boxed{...} to send actions. Example: "
            + self.sample_random_action()
            + "\n"
        )

    def get_task_suffix(self) -> str:
        preview_n = min(self.preview_lines, len(self.lines))
        preview = "\n".join(f"{i+1}: {self.lines[i]}" for i in range(preview_n))
        return (
            f"Function preview (first {preview_n} lines):\n{preview}\n"
            f"Remaining query budget: {self.query_budget}\n"
            f"Allowed count keywords: {', '.join(self.allowed_keywords)}\n"
            f"Peek span limit: {self.peek_span_limit}\n"
            "Enter your action in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()
        self.turn_count = 0
        self.last_submission = None
        self._generate_function()
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if not parsed:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}"
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        if parsed["type"] == "count":
            keyword = parsed["keyword"]
            if keyword not in self.allowed_keywords:
                obs = f"Unsupported action: unknown keyword '{keyword}'."
                return obs, -0.25, True, False, {"suffix": self.get_task_suffix()}
            if self.query_budget <= 0:
                obs = "Protocol violation: no budget remaining for queries."
                return obs, -0.25, True, False, {"suffix": self.get_task_suffix()}
            self.query_budget -= 1
            value = self.counts.get(keyword, 0)
            obs = (
                f"At turn {self.turn_count}, counted keyword '{keyword}' = {value}. "
                f"Budget remaining: {self.query_budget}."
            )
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        elif parsed["type"] == "peek":
            start = parsed["start"]
            span = parsed["span"]
            if self.query_budget <= 0:
                obs = "Protocol violation: no budget remaining for queries."
                return obs, -0.25, True, False, {"suffix": self.get_task_suffix()}
            if span < 1 or span > self.peek_span_limit:
                obs = f"Protocol violation: peek span {span} exceeds limit {self.peek_span_limit}."
                return obs, -0.25, True, False, {"suffix": self.get_task_suffix()}
            if start < 1 or start > len(self.lines):
                obs = f"Protocol violation: start {start} out of range (1..{len(self.lines)})."
                return obs, -0.25, True, False, {"suffix": self.get_task_suffix()}
            end = start + span - 1
            if end > len(self.lines):
                obs = f"Protocol violation: peek end {end} exceeds file length {len(self.lines)}."
                return obs, -0.25, True, False, {"suffix": self.get_task_suffix()}
            self.query_budget -= 1
            segment = "\n".join(f"{i}: {self.lines[i-1]}" for i in range(start, end + 1))
            obs = (
                f"At turn {self.turn_count}, peeked lines {start}-{end}:\n{segment}\n"
                f"Budget remaining: {self.query_budget}."
            )
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        elif parsed["type"] == "submit":
            value = parsed["value"]
            if not isinstance(value, int):
                obs = "Protocol violation: submission must be an integer complexity value."
                return obs, -0.25, True, False, {"suffix": self.get_task_suffix()}
            self.last_submission = value
            if value == self.true_complexity:
                obs = f"Success! Correct complexity {value}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Failed! Submitted {value} but true complexity is {self.true_complexity}."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = "Unsupported action."
            return obs, -0.25, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns})"
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Any]:
        if not action:
            return None
        pattern = re.compile(r"\\boxed\{(.+?)\}", re.IGNORECASE | re.DOTALL)
        m = pattern.search(action)
        if not m:
            return None
        content = m.group(1).strip()
        lower = content.lower()

        if lower.startswith("count:"):
            kw = content.split(":", 1)[1].strip().lower()
            return {"type": "count", "keyword": kw}

        if lower.startswith("peek:"):
            rest = content.split(":", 1)[1].strip()
            parts = re.split(r"\s+", rest)
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                start = int(parts[0])
                span = int(parts[1])
                return {"type": "peek", "start": start, "span": span}
            return None

        if lower.startswith("submit:"):
            rest = content.split(":", 1)[1].strip()
            # accept "complexity 12" or "complexity=12"
            m2 = re.search(r"complexity\s*=?\s*(-?\d+)", rest, re.IGNORECASE)
            if m2:
                val = int(m2.group(1))
                return {"type": "submit", "value": val}
            return None

        return None

    def sample_random_action(self) -> str:
        choice = random.choice(["count", "peek", "submit"])
        if choice == "count":
            kw = random.choice(self.allowed_keywords)
            return f"\\boxed{{count: {kw}}}"
        elif choice == "peek":
            start = random.randint(1, max(1, min(len(self.lines), self.preview_lines)))
            span = random.randint(1, max(1, self.peek_span_limit))
            return f"\\boxed{{peek: {start} {span}}}"
        else:
            guess = max(1, self.true_complexity + random.choice([-2, -1, 0, 1, 2]))
            return f"\\boxed{{submit: complexity {guess}}}"


class CodeComplexityEnvWithFeedback(CodeComplexityEnv):
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
            hint = "Wrap your command in \\boxed{...} and follow the allowed action formats."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "no budget" in text:
                error_detail["violation"] = "no_budget"
                hint = "Avoid extra queries; compute using known formula. Consider submitting your current estimate."
            elif "span" in text:
                error_detail["violation"] = "peek_span_exceeded"
                hint = f"Use a span <= {self.peek_span_limit}."
            elif "out of range" in text or "exceeds file length" in text:
                error_detail["violation"] = "peek_bounds"
                hint = f"Choose start and span so that end <= {len(self.lines)} and start >= 1."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["allowed"] = {
                "count": self.allowed_keywords,
                "peek_span_limit": self.peek_span_limit,
                "submit": "submit: complexity N",
            }
            hint = "Use 'count: KEYWORD', 'peek: START SPAN', or 'submit: complexity N'."

        elif "failed!" in text:
            error_type = "WrongDecision"
            error_detail["expected"] = self.true_complexity
            error_detail["got"] = self.last_submission
            hint = (
                "Recompute using: 1 + if + elif + for + while + case + boolean + ternary. "
                "Use counts strategically before submitting."
            )

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["max_turns"] = self.max_turns
            hint = "Act earlier: prioritize high-value counts and peek small spans."

        elif "success!" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = "Good job."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "budget": self.query_budget,
                "preview_lines": self.preview_lines,
                "peek_span_limit": self.peek_span_limit,
                "true_complexity": self.true_complexity,
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
            "hint": "Start by counting a construct (e.g., \\boxed{count: if}) to reduce uncertainty.",
            "turn": 0,
            "state": {
                "budget": self.query_budget,
                "preview_lines": self.preview_lines,
                "peek_span_limit": self.peek_span_limit,
            },
        }
        return obs, info