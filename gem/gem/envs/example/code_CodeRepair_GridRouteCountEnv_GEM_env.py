from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class CodeRepairEnv(Env):
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
            'chain_len': (2, 6),           # Function length: more steps increases reasoning and patching difficulty
            'num_bugs': (1, 3),            # More bugs require more fixes and increase search space
            'test_suite_size': (2, 8),     # Larger test suite increases verification burden
            'allowed_edits': (6, 2),       # REVERSED: fewer edit budget increases difficulty
            'visibility_level': (2, 0),    # REVERSED: less visibility makes inspection harder
            'const_range': (5, 20),        # Larger constants create larger numeric variation and harder inference
            'param_steps': (0, 2),         # Steps bound to external parameters (a or b) increase dependency complexity
        }

        self.param_variance = {
            'chain_len': 0,
            'num_bugs': 0,
            'test_suite_size': 1,
            'allowed_edits': 0,
            'visibility_level': 0,
            'const_range': 2,
            'param_steps': 0,
        }

        self.chain_len: int = 0
        self.num_bugs: int = 0
        self.test_suite_size: int = 0
        self.allowed_edits: int = 0
        self.visibility_level: int = 0
        self.const_range: int = 0
        self.param_steps: int = 0

        self.turn_count: int = 0
        self.initialized: bool = False
        self.edits_left: int = 0

        self.correct_ops: list = []
        self.correct_consts: list = []
        self.step_types: list = []    # 'const' or 'param'
        self.param_tags: list = []    # None, 'a', or 'b'

        self.current_ops: list = []
        self.current_consts: list = []

        self.test_suite: list = []    # List of tuples (x,a,b,expected)
        self.query_x: int = 0
        self.query_a: int = 0
        self.query_b: int = 0
        self.correct_query_answer: int = 0

        self.last_run_output: Optional[int] = None
        self.last_test_pass_count: Optional[int] = None

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for param_name, (min_val, max_val) in self.complexity_params.items():
            center_value = min_val + (max_val - min_val) * normalized
            variance = self.param_variance.get(param_name, 0)
            if self.enable_param_randomization and variance > 0:
                actual_value = center_value + random.uniform(-variance, variance)
            else:
                actual_value = center_value
            low = min(min_val, max_val)
            high = max(min_val, max_val)
            actual_value = max(low, min(high, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _compute(self, ops: list, consts: list, x: int, a: int, b: int) -> int:
        val = x
        for i in range(self.chain_len):
            if self.step_types[i] == 'const':
                arg = consts[i]
            else:
                arg = a if self.param_tags[i] == 'a' else b
            op = ops[i]
            if op == 'ADD':
                val = val + arg
            elif op == 'SUB':
                val = val - arg
            elif op == 'MUL':
                val = val * arg
            else:
                val = val  # Should not occur; fallback noop
        return val

    def _build_instance(self):
        ops_choices = ['ADD', 'SUB', 'MUL']
        self.correct_ops = [random.choice(ops_choices) for _ in range(self.chain_len)]
        self.step_types = ['const'] * self.chain_len
        self.param_tags = [None] * self.chain_len
        if self.param_steps > 0:
            indices = random.sample(range(self.chain_len), k=min(self.param_steps, self.chain_len))
            for idx in indices:
                self.step_types[idx] = 'param'
                self.param_tags[idx] = random.choice(['a', 'b'])
        self.correct_consts = []
        for i in range(self.chain_len):
            if self.step_types[i] == 'const':
                self.correct_consts.append(random.randint(1, self.const_range))
            else:
                self.correct_consts.append(None)

        # Tests and query
        self.test_suite = []
        for _ in range(self.test_suite_size):
            tx = random.randint(0, 9)
            ta = random.randint(0, 9)
            tb = random.randint(0, 9)
            expected = self._compute(self.correct_ops, self.correct_consts, tx, ta, tb)
            self.test_suite.append((tx, ta, tb, expected))
        self.query_x = random.randint(0, 9)
        self.query_a = random.randint(0, 9)
        self.query_b = random.randint(0, 9)
        self.correct_query_answer = self._compute(self.correct_ops, self.correct_consts, self.query_x, self.query_a, self.query_b)

        # Apply bugs within edit budget
        bug_count = min(self.num_bugs, self.allowed_edits, self.chain_len)
        self.current_ops = list(self.correct_ops)
        self.current_consts = list(self.correct_consts)
        if bug_count > 0:
            bug_indices = random.sample(range(self.chain_len), k=bug_count)
            for idx in bug_indices:
                if self.step_types[idx] == 'const':
                    if random.random() < 0.5:
                        wrong_op = random.choice([o for o in ops_choices if o != self.correct_ops[idx]])
                        self.current_ops[idx] = wrong_op
                    else:
                        wrong_const = random.randint(1, self.const_range)
                        while wrong_const == self.correct_consts[idx]:
                            wrong_const = random.randint(1, self.const_range)
                        self.current_consts[idx] = wrong_const
                else:
                    wrong_op = random.choice([o for o in ops_choices if o != self.correct_ops[idx]])
                    self.current_ops[idx] = wrong_op

    def _get_instructions(self) -> str:
        example = self.sample_random_action()
        return (
            "CodeRepair: You work on a small arithmetic DSL function built as a pipeline of steps. "
            "Some steps are buggy (wrong operator or wrong constant). Your goal is to submit the correct output "
            f"for the query input (x={self.query_x}, a={self.query_a}, b={self.query_b}) using the repaired function.\n"
            "Commands:\n"
            "- INIT\n"
            "- SHOW\n"
            "- PATCH opK=ADD|SUB|MUL  (K is 1..N)\n"
            "- PATCH cK=<int>         (K is 1..N) for constant steps only\n"
            "- TEST                   (run unit tests; shows pass count)\n"
            "- RUN x=<int> a=<int> b=<int>\n"
            "- SUBMIT <int>           (final answer = output for the query input)\n"
            f"Use \\boxed{{...}} for every action. Example: {example}\n"
        )

    def get_task_suffix(self) -> str:
        steps = self.chain_len
        vis = self.visibility_level
        summary = []
        summary.append(f"Steps: {steps}, Tests: {self.test_suite_size}, Edits left: {self.edits_left}")
        summary.append(f"Query input: x={self.query_x}, a={self.query_a}, b={self.query_b}")
        if self.initialized:
            if vis >= 2:
                parts = []
                for i in range(steps):
                    if self.step_types[i] == 'const':
                        parts.append(f"{i+1}:{self.current_ops[i]} {self.current_consts[i]}")
                    else:
                        parts.append(f"{i+1}:{self.current_ops[i]} param({self.param_tags[i]})")
                summary.append("Current pipeline: " + " | ".join(parts))
            elif vis == 1:
                parts = []
                for i in range(steps):
                    if self.step_types[i] == 'const':
                        parts.append(f"{i+1}:{self.current_ops[i]} ?")
                    else:
                        parts.append(f"{i+1}:{self.current_ops[i]} param(?)")
                summary.append("Pipeline overview: " + " | ".join(parts))
            else:
                summary.append("Pipeline overview: hidden at current visibility level.")
        else:
            summary.append("Project not initialized. Use INIT.")
        if self.last_run_output is not None:
            summary.append(f"Last RUN output: {self.last_run_output}")
        if self.last_test_pass_count is not None:
            summary.append(f"Last TEST pass: {self.last_test_pass_count}/{self.test_suite_size}")
        summary.append("Enter your action in \\boxed{...} format.")
        return "\n".join(summary)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()
        self.turn_count = 0
        self.initialized = False
        self.edits_left = self.allowed_edits
        self.correct_ops = []
        self.correct_consts = []
        self.step_types = []
        self.param_tags = []
        self.current_ops = []
        self.current_consts = []
        self.test_suite = []
        self.query_x = 0
        self.query_a = 0
        self.query_b = 0
        self.correct_query_answer = 0
        self.last_run_output = None
        self.last_test_pass_count = None

        self._build_instance()
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if not parsed:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}} with a supported command."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        cmd = parsed.get("type")

        if cmd in ("SHOW", "PATCH", "RUN", "TEST", "SUBMIT") and not self.initialized:
            obs = "Protocol violation: project not initialized. Use INIT first."
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        reward = 0.0

        if cmd == "INIT":
            self.initialized = True
            obs = f"Project initialized. Steps={self.chain_len}, tests={self.test_suite_size}, edits={self.edits_left}."
        elif cmd == "SHOW":
            obs = "Pipeline displayed."
        elif cmd == "PATCH":
            k = parsed.get("index")
            if not (1 <= k <= self.chain_len):
                obs = f"Protocol violation: index out of range. K must be 1..{self.chain_len}."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if self.edits_left <= 0:
                obs = "Protocol violation: no edits remaining."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if parsed.get("target") == "op":
                new_op = parsed.get("value")
                if new_op not in ("ADD", "SUB", "MUL"):
                    obs = "Unsupported action: operator must be one of ADD,SUB,MUL."
                    return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
                self.current_ops[k - 1] = new_op
                self.edits_left -= 1
                obs = f"Patched op{k}={new_op}. Edits left={self.edits_left}."
            else:
                if self.step_types[k - 1] != 'const':
                    obs = "Protocol violation: cannot patch cK on a param step."
                    return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
                new_const = parsed.get("value")
                self.current_consts[k - 1] = new_const
                self.edits_left -= 1
                obs = f"Patched c{k}={new_const}. Edits left={self.edits_left}."
        elif cmd == "RUN":
            x = parsed.get("x")
            a = parsed.get("a")
            b = parsed.get("b")
            self.last_run_output = self._compute(self.current_ops, self.current_consts, x, a, b)
            obs = f"Run completed: output={self.last_run_output} for x={x}, a={a}, b={b}."
        elif cmd == "TEST":
            pass_count = 0
            for (tx, ta, tb, expected) in self.test_suite:
                out = self._compute(self.current_ops, self.current_consts, tx, ta, tb)
                if out == expected:
                    pass_count += 1
            self.last_test_pass_count = pass_count
            if self.visibility_level >= 2:
                obs = f"Test results: {pass_count}/{self.test_suite_size} passed."
            else:
                obs = f"Test pass count: {pass_count}/{self.test_suite_size}."
        elif cmd == "SUBMIT":
            submitted = parsed.get("value")
            if submitted == self.correct_query_answer:
                obs = "Success! Final answer is correct."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = "Failed! Final answer is incorrect."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
        else:
            obs = "Unsupported action: unknown command."
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            truncated = True
            terminated = True
            obs = f"Reached max turns ({self.max_turns})."
            return obs, 0.0, terminated, truncated, {"suffix": self.get_task_suffix()}

        return self._compose_observation(obs), reward, False, False, {"suffix": self.get_task_suffix()}

    def _compose_observation(self, header: str) -> str:
        return f"{header}\n{self.get_task_suffix()}"

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action:
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()

        if re.fullmatch(r'(?i)INIT', content):
            return {"type": "INIT"}

        if re.fullmatch(r'(?i)SHOW', content):
            return {"type": "SHOW"}

        if re.fullmatch(r'(?i)TEST', content):
            return {"type": "TEST"}

        m_run = re.fullmatch(r'(?i)RUN\s+x=(\-?\d+)\s+a=(\-?\d+)\s+b=(\-?\d+)', content)
        if m_run:
            x = int(m_run.group(1))
            a = int(m_run.group(2))
            b = int(m_run.group(3))
            return {"type": "RUN", "x": x, "a": a, "b": b}

        m_patch_op = re.fullmatch(r'(?i)PATCH\s+op(\d+)\s*=\s*(ADD|SUB|MUL)', content)
        if m_patch_op:
            idx = int(m_patch_op.group(1))
            val = m_patch_op.group(2).upper()
            return {"type": "PATCH", "target": "op", "index": idx, "value": val}

        m_patch_const = re.fullmatch(r'(?i)PATCH\s+c(\d+)\s*=\s*(-?\d+)', content)
        if m_patch_const:
            idx = int(m_patch_const.group(1))
            val = int(m_patch_const.group(2))
            return {"type": "PATCH", "target": "const", "index": idx, "value": val}

        m_submit = re.fullmatch(r'(?i)SUBMIT\s+(-?\d+)', content)
        if m_submit:
            val = int(m_submit.group(1))
            return {"type": "SUBMIT", "value": val}

        return None

    def sample_random_action(self) -> str:
        choices = [
            "\\boxed{INIT}",
            "\\boxed{SHOW}",
            "\\boxed{PATCH op1=ADD}",
            "\\boxed{PATCH c1=3}",
            "\\boxed{TEST}",
            "\\boxed{RUN x=2 a=1 b=4}",
            "\\boxed{SUBMIT 42}",
        ]
        return random.choice(choices)


class CodeRepairEnvWithFeedback(CodeRepairEnv):
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
            hint = "Use \\boxed{COMMAND ...} exactly. For example: \\boxed{INIT} or \\boxed{PATCH op2=MUL}."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "not initialized" in text:
                error_detail["violation"] = "missing_init"
                hint = "Start with \\boxed{INIT} before SHOW, PATCH, RUN, TEST, SUBMIT."
            elif "no edits remaining" in text:
                error_detail["violation"] = "edit_budget_exhausted"
                hint = "Use \\boxed{TEST} and \\boxed{SHOW} earlier to prioritize essential patches."
            elif "index out of range" in text:
                error_detail["violation"] = "bad_index"
                hint = "Use an index K between 1 and the number of steps reported after INIT."
            elif "cannot patch ck on a param step" in text:
                error_detail["violation"] = "wrong_patch_target"
                hint = "For param steps, patch the operator (opK) rather than cK."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_command_or_operator"
            hint = "Valid operators: ADD, SUB, MUL. Valid commands: INIT, SHOW, PATCH, TEST, RUN, SUBMIT."
        elif "failed! final answer is incorrect" in text or ("failed!" in text and "final answer" in text):
            error_type = "WrongDecision"
            error_detail["expected"] = getattr(self, "correct_query_answer", None)
            error_detail["got"] = self._extract_submitted_from_obs(text)
            hint = "Run the function with the query inputs using \\boxed{RUN x=... a=... b=...}. Use \\boxed{TEST} to adjust patches until most tests pass."
        elif "success!" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
        elif truncated:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Plan: INIT → SHOW → TEST → PATCH (strategically) → RUN on query → SUBMIT."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "edits_left": getattr(self, "edits_left", None),
                "tests_total": getattr(self, "test_suite_size", None),
                "visibility_level": getattr(self, "visibility_level", None),
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
            "hint": "Start with \\boxed{INIT}, then \\boxed{SHOW} and \\boxed{TEST} to plan patches.",
            "turn": 0,
            "state": {
                "edits_left": getattr(self, "edits_left", None),
                "tests_total": getattr(self, "test_suite_size", None),
                "visibility_level": getattr(self, "visibility_level", None),
            },
        }
        return obs, info

    def _extract_submitted_from_obs(self, text: str) -> Optional[int]:
        m = re.search(r'submit\s+(-?\d+)', text)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
        return None