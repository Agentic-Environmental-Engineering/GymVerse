from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class CodeSynthesisEnv(Env):
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

        # Evolvable parameters
        self.complexity_params = {
            # Length of the hidden target pipeline; longer pipelines increase search space and ordering complexity
            "pipeline_length": (1, 8),
            # Number of tests; more tests provide richer constraints but require more reasoning
            "num_tests": (2, 12),
            # Typical test input size; larger arrays make behavior less trivial and outputs longer to inspect
            "input_size": (5, 40),
            # Range for numeric parameters in operations (e.g., thresholds, add/mult amounts); larger range increases parameter search
            "param_range": (5, 30),
            # REVERSED: number of allowed preview inspections; fewer inspections make it harder to deduce the pipeline
            "inspect_budget": (6, 2),
            # REVERSED: max elements shown per preview; smaller previews reduce information per inspection
            "preview_limit": (5, 2),
        }

        # Variance settings
        self.param_variance = {
            "pipeline_length": 1,
            "num_tests": 2,
            "input_size": 5,
            "param_range": 4,
            "inspect_budget": 0,
            "preview_limit": 0,
        }

        # Placeholder attributes (set in _apply_complexity_params)
        self.pipeline_length: int = 0
        self.num_tests: int = 0
        self.input_size: int = 0
        self.param_range: int = 0
        self.inspect_budget: int = 0
        self.preview_limit: int = 0

        # Domain state
        self.turn_count: int = 0
        self.agent_pipeline: List[Tuple[str, Optional[int]]] = []
        self.target_pipeline: List[Tuple[str, Optional[int]]] = []
        self.tests: List[List[int]] = []
        self.last_run_result: Optional[List[int]] = None
        self.inspect_budget_remaining: int = 0

        # Operation sets
        self.param_ops = {"FILTER_GT", "FILTER_LT", "FILTER_EQ", "MAP_ADD", "MAP_SUB", "MAP_MULT", "TAKE", "DROP"}
        self.no_param_ops = {"SORT_ASC", "SORT_DESC", "REVERSE", "UNIQUE"}

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
            # Clamp within range, support reversed ranges
            lo = min(min_val, max_val)
            hi = max(min_val, max_val)
            actual_value = max(lo, min(hi, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        return (
            "Code Synthesis Game:\n"
            "You must reconstruct a hidden functional pipeline that transforms integer arrays. "
            "Compose operations to match the behavior and finally submit the exact pipeline.\n"
            "Available operations:\n"
            "- Param ops: FILTER_GT k, FILTER_LT k, FILTER_EQ k, MAP_ADD k, MAP_SUB k, MAP_MULT k, TAKE n, DROP n\n"
            "- No-param ops: SORT_ASC, SORT_DESC, REVERSE, UNIQUE\n"
            "Commands:\n"
            "- ADD <OP> [PARAM]\n"
            "- REMOVE <INDEX>\n"
            "- CLEAR\n"
            "- RUN <TEST_INDEX>\n"
            "- PREVIEW <TEST_INDEX> <K>  (uses budget; K <= preview_limit)\n"
            "- SHOW LAST\n"
            "- SUBMIT CURRENT\n"
            "- SUBMIT PIPELINE: <OP [PARAM]> | <OP [PARAM]> | ...\n"
            "Use \\boxed{...} to send actions. Example: "
            f"{self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        pipeline_str = self._pipeline_to_str(self.agent_pipeline)
        return (
            f"Status:\n"
            f"- Tests: {self.num_tests}\n"
            f"- Preview budget remaining: {self.inspect_budget_remaining}\n"
            f"- Preview limit per action: {self.preview_limit}\n"
            f"- Current pipeline: {pipeline_str if pipeline_str else '[empty]'}\n"
            f"- Turn: {self.turn_count}/{self.max_turns}\n"
            "Enter your action in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        self.turn_count = 0
        self.agent_pipeline = []
        self.last_run_result = None
        self.inspect_budget_remaining = self.inspect_budget

        # Generate target pipeline
        self.target_pipeline = self._generate_target_pipeline(self.pipeline_length, self.param_range)
        # Generate tests
        self.tests = []
        for _ in range(self.num_tests):
            size = max(1, self.input_size + random.randint(-2, 3))
            vals = [random.randint(-self.param_range, self.param_range) for _ in range(size)]
            self.tests.append(vals)

        # Ensure at least one non-empty output
        if not self._has_nonempty_output(self.target_pipeline, self.tests):
            tries = 0
            while tries < 20:
                self.target_pipeline = self._generate_target_pipeline(self.pipeline_length, self.param_range)
                if self._has_nonempty_output(self.target_pipeline, self.tests):
                    break
                tries += 1

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if parsed is None:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        cmd = parsed.get("cmd", "").upper()
        reward = 0.0

        if cmd == "ADD":
            op = parsed.get("op")
            param = parsed.get("param")
            if op is None:
                obs = f"Protocol violation: missing operation for ADD."
                reward = -0.2
            elif op in self.param_ops and param is None:
                obs = f"Protocol violation: parameter required for {op}."
                reward = -0.2
            elif op in self.no_param_ops and param is not None:
                obs = f"Protocol violation: {op} does not take a parameter."
                reward = -0.2
            elif (op not in self.param_ops) and (op not in self.no_param_ops):
                obs = f"Unsupported action: unknown operation {op}."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            else:
                self.agent_pipeline.append((op, param))
                obs = f"Added step {op}{(' ' + str(param)) if param is not None else ''}. Current pipeline: {self._pipeline_to_str(self.agent_pipeline)}"

        elif cmd == "REMOVE":
            idx = parsed.get("index")
            if idx is None or idx < 1 or idx > len(self.agent_pipeline):
                obs = "Protocol violation: invalid index for REMOVE."
                reward = -0.2
            else:
                removed = self.agent_pipeline.pop(idx - 1)
                obs = f"Removed step {idx}: {removed[0]}{(' ' + str(removed[1])) if removed[1] is not None else ''}. Current pipeline: {self._pipeline_to_str(self.agent_pipeline)}"

        elif cmd == "CLEAR":
            self.agent_pipeline = []
            obs = "Workspace cleared. Current pipeline: [empty]"

        elif cmd == "RUN":
            ti = parsed.get("test_index")
            if ti is None or ti < 1 or ti > len(self.tests):
                obs = "Protocol violation: invalid test index for RUN."
                reward = -0.2
            else:
                arr = list(self.tests[ti - 1])
                out = self._apply_pipeline(self.agent_pipeline, arr)
                self.last_run_result = out
                obs = f"Ran test {ti}. Output length: {len(out)}. Use PREVIEW to inspect first elements."

        elif cmd == "PREVIEW":
            ti = parsed.get("test_index")
            k = parsed.get("k")
            if self.inspect_budget_remaining <= 0:
                obs = "Protocol violation: no preview budget remaining."
                reward = -0.2
            elif ti is None or ti < 1 or ti > len(self.tests):
                obs = "Protocol violation: invalid test index for PREVIEW."
                reward = -0.2
            elif k is None or k < 1 or k > self.preview_limit:
                obs = f"Protocol violation: K must be between 1 and {self.preview_limit}."
                reward = -0.2
            else:
                arr = list(self.tests[ti - 1])
                out = self._apply_pipeline(self.agent_pipeline, arr)
                self.last_run_result = out
                preview = out[:k]
                self.inspect_budget_remaining -= 1
                obs = f"Preview test {ti} (first {k}): {preview}. Budget remaining: {self.inspect_budget_remaining}"

        elif cmd == "SHOW":
            sub = parsed.get("sub")
            if sub == "LAST":
                if self.last_run_result is None:
                    obs = "Protocol violation: nothing to show; run a test first."
                    reward = -0.2
                else:
                    k = min(3, self.preview_limit)
                    preview = self.last_run_result[:k]
                    obs = f"Last run summary: length={len(self.last_run_result)}, first {k}={preview}."
            else:
                obs = f"Unsupported action: SHOW {sub}."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}

        elif cmd == "SUBMIT":
            mode = parsed.get("mode")
            if mode == "CURRENT":
                submitted = list(self.agent_pipeline)
            else:
                spec = parsed.get("spec_steps")
                if spec is None or len(spec) == 0:
                    obs = "Protocol violation: SUBMIT requires a pipeline specification."
                    reward = -0.2
                    spec = None
                submitted = spec

            if submitted is not None:
                if self._pipelines_equal(submitted, self.target_pipeline):
                    obs = "Success! Your submitted pipeline exactly matches the hidden target."
                    return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
                else:
                    obs = "Failed! Your submitted pipeline does not match the target."
                    return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = f"Unsupported action: {cmd}."
            return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            truncated = True
            terminated = True
            obs = f"Reached max turns ({self.max_turns})."
            return obs, 0.0, terminated, truncated, {"suffix": self.get_task_suffix()}

        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action:
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        content_upper = content.upper()

        # SUBMIT parsing
        if content_upper.startswith("SUBMIT"):
            # SUBMIT CURRENT
            if "CURRENT" in content_upper:
                return {"cmd": "SUBMIT", "mode": "CURRENT"}
            # SUBMIT PIPELINE: ...
            m = re.search(r"SUBMIT(?:\s+PIPELINE)?\s*:\s*(.+)", content, flags=re.IGNORECASE | re.DOTALL)
            if m:
                spec_str = m.group(1).strip()
                steps = [s.strip() for s in spec_str.split("|") if s.strip()]
                parsed_steps = []
                for s in steps:
                    toks = s.strip().split()
                    if len(toks) == 0:
                        continue
                    op = toks[0].upper()
                    if op in self.param_ops:
                        if len(toks) < 2:
                            return {"cmd": "SUBMIT", "spec_steps": []}
                        try:
                            param = int(toks[1])
                        except ValueError:
                            return {"cmd": "SUBMIT", "spec_steps": []}
                        parsed_steps.append((op, param))
                    elif op in self.no_param_ops:
                        parsed_steps.append((op, None))
                    else:
                        return {"cmd": "SUBMIT", "spec_steps": []}
                return {"cmd": "SUBMIT", "spec_steps": parsed_steps}
            return {"cmd": "SUBMIT", "spec_steps": []}

        # ADD
        if content_upper.startswith("ADD"):
            toks = content.split()
            if len(toks) < 2:
                return {"cmd": "ADD"}
            op = toks[1].upper()
            if op in self.param_ops:
                if len(toks) < 3:
                    return {"cmd": "ADD", "op": op}
                try:
                    param = int(toks[2])
                except ValueError:
                    return {"cmd": "ADD", "op": op}
                return {"cmd": "ADD", "op": op, "param": param}
            else:
                return {"cmd": "ADD", "op": op}

        # REMOVE
        if content_upper.startswith("REMOVE"):
            toks = content.split()
            idx = None
            if len(toks) >= 2:
                try:
                    idx = int(toks[1])
                except ValueError:
                    idx = None
            return {"cmd": "REMOVE", "index": idx}

        # CLEAR
        if content_upper.strip() == "CLEAR":
            return {"cmd": "CLEAR"}

        # RUN
        if content_upper.startswith("RUN"):
            toks = content.split()
            ti = None
            if len(toks) >= 2:
                try:
                    ti = int(toks[1])
                except ValueError:
                    ti = None
            return {"cmd": "RUN", "test_index": ti}

        # PREVIEW
        if content_upper.startswith("PREVIEW"):
            toks = content.split()
            ti = None
            k = None
            if len(toks) >= 2:
                try:
                    ti = int(toks[1])
                except ValueError:
                    ti = None
            if len(toks) >= 3:
                try:
                    k = int(toks[2])
                except ValueError:
                    k = None
            return {"cmd": "PREVIEW", "test_index": ti, "k": k}

        # SHOW LAST
        if content_upper.strip().startswith("SHOW"):
            if "LAST" in content_upper:
                return {"cmd": "SHOW", "sub": "LAST"}
            toks = content_upper.strip().split()
            sub = toks[1] if len(toks) >= 2 else ""
            return {"cmd": "SHOW", "sub": sub}

        return {"cmd": content_upper.split()[0] if content_upper.split() else ""}

    def sample_random_action(self) -> str:
        ex = random.choice([
            "\\boxed{ADD FILTER_GT 3}",
            "\\boxed{ADD SORT_ASC}",
            "\\boxed{RUN 1}",
            "\\boxed{PREVIEW 2 3}",
            "\\boxed{REMOVE 1}",
            "\\boxed{CLEAR}",
            "\\boxed{SHOW LAST}",
            "\\boxed{SUBMIT CURRENT}",
            "\\boxed{SUBMIT PIPELINE: FILTER_GT 3 | MAP_ADD 2 | SORT_ASC}",
        ])
        return ex

    def _pipeline_to_str(self, pipeline: List[Tuple[str, Optional[int]]]) -> str:
        parts = []
        for op, param in pipeline:
            if param is None:
                parts.append(op)
            else:
                parts.append(f"{op} {param}")
        return " | ".join(parts)

    def _apply_pipeline(self, pipeline: List[Tuple[str, Optional[int]]], arr: List[int]) -> List[int]:
        res = list(arr)
        for op, param in pipeline:
            if op == "FILTER_GT":
                res = [x for x in res if x > (param or 0)]
            elif op == "FILTER_LT":
                res = [x for x in res if x < (param or 0)]
            elif op == "FILTER_EQ":
                res = [x for x in res if x == (param or 0)]
            elif op == "MAP_ADD":
                res = [x + (param or 0) for x in res]
            elif op == "MAP_SUB":
                res = [x - (param or 0) for x in res]
            elif op == "MAP_MULT":
                p = param if param is not None else 1
                res = [x * p for x in res]
            elif op == "TAKE":
                n = max(0, param or 0)
                res = res[:n]
            elif op == "DROP":
                n = max(0, param or 0)
                res = res[n:]
            elif op == "SORT_ASC":
                res = sorted(res)
            elif op == "SORT_DESC":
                res = sorted(res, reverse=True)
            elif op == "REVERSE":
                res = list(reversed(res))
            elif op == "UNIQUE":
                seen = set()
                out = []
                for x in res:
                    if x not in seen:
                        seen.add(x)
                        out.append(x)
                res = out
            else:
                # Ignore unknown in application; but generation only uses known ops
                pass
        return res

    def _generate_target_pipeline(self, length: int, prange: int) -> List[Tuple[str, Optional[int]]]:
        ops_all = list(self.param_ops | self.no_param_ops)
        pipeline: List[Tuple[str, Optional[int]]] = []
        for _ in range(length):
            op = random.choice(ops_all)
            param = None
            if op in {"FILTER_GT", "FILTER_LT"}:
                param = random.randint(max(0, prange // -1), prange) if op == "FILTER_GT" else random.randint(-prange, prange)
                # bias towards moderate thresholds
                if op == "FILTER_GT":
                    param = random.randint(0, prange)
                else:
                    param = random.randint(-prange, prange)
            elif op == "FILTER_EQ":
                param = random.randint(-prange, prange)
            elif op in {"MAP_ADD", "MAP_SUB"}:
                param = random.randint(-prange, prange)
            elif op == "MAP_MULT":
                param = random.randint(2, max(2, min(5, prange)))
            elif op in {"TAKE", "DROP"}:
                param = random.randint(0, max(1, self.input_size))
            else:
                param = None
            pipeline.append((op, param))
        return pipeline

    def _has_nonempty_output(self, pipeline: List[Tuple[str, Optional[int]]], tests: List[List[int]]) -> bool:
        for arr in tests:
            out = self._apply_pipeline(pipeline, arr)
            if len(out) > 0:
                return True
        return False

    def _pipelines_equal(self, a: List[Tuple[str, Optional[int]]], b: List[Tuple[str, Optional[int]]]) -> bool:
        if len(a) != len(b):
            return False
        for (op1, p1), (op2, p2) in zip(a, b):
            if op1 != op2:
                return False
            if p1 != p2:
                return False
        return True


class CodeSynthesisEnvWithFeedback(CodeSynthesisEnv):
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
            hint = "Wrap your command in \\boxed{...} using the listed syntax."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_command_or_op"
            hint = "Use commands: ADD, REMOVE, CLEAR, RUN, PREVIEW, SHOW LAST, SUBMIT CURRENT, SUBMIT PIPELINE: ..."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "parameter required" in text:
                error_detail["violation"] = "missing_parameter"
                hint = "Provide integer parameter for param ops (e.g., ADD FILTER_GT 3)."
            elif "invalid index" in text:
                error_detail["violation"] = "bad_index"
                hint = "Use 1-based indices within current pipeline length or test count."
            elif "no preview budget" in text:
                error_detail["violation"] = "no_budget"
                hint = "Plan previews before submission; you have limited budget."
            elif "k must be between" in text:
                error_detail["violation"] = "k_out_of_bounds"
                hint = "Choose K within preview_limit shown in status."
            elif "nothing to show" in text:
                error_detail["violation"] = "no_last_run"
                hint = "Run a test first: \\boxed{RUN 1} then \\boxed{SHOW LAST}."
            else:
                error_detail["violation"] = "general_protocol"
                hint = "Follow the command formats exactly and respect limits."

        elif "failed!" in text:
            error_type = "WrongDecision"
            error_detail["submitted_length"] = len(self.agent_pipeline)
            error_detail["target_length"] = len(self.target_pipeline)
            hint = "Check operation order and parameters. Use RUN and PREVIEW on multiple tests to refine."

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["turn_limit"] = self.max_turns
            hint = "Avoid unnecessary actions; preview sparingly and submit when confident."

        elif "success!" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "preview_budget_remaining": getattr(self, "inspect_budget_remaining", None),
                "current_pipeline_length": len(getattr(self, "agent_pipeline", [])),
                "preview_limit": getattr(self, "preview_limit", None),
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
            "hint": "Start by running a test and previewing small outputs to infer operations (e.g., \\boxed{RUN 1}).",
            "turn": 0,
            "state": {
                "preview_budget_remaining": getattr(self, "inspect_budget_remaining", None),
                "current_pipeline_length": len(getattr(self, "agent_pipeline", [])),
                "preview_limit": getattr(self, "preview_limit", None),
            },
        }
        return obs, info