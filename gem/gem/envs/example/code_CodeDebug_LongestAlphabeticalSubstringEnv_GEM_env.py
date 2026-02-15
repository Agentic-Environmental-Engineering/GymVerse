from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class CodeDebugEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 60,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 60

        self.complexity_params = {
            # Total lines in the file: larger files are harder (more search space)
            "num_lines": (8, 50),
            # Number of bug types in the curriculum: more options increases hypothesis space
            "num_bug_types": (3, 7),
            # Number of tests to run: larger numbers give richer feedback but require stronger fixes
            "tests_count": (6, 20),
            # REVERSED: maximum lines revealed in one range view; smaller spans increase difficulty
            "max_view_span": (10, 3),
        }

        self.param_variance = {
            "num_lines": 5,
            "num_bug_types": 1,
            "tests_count": 2,
            "max_view_span": 0,
        }

        self.num_lines: int = 0
        self.num_bug_types: int = 0
        self.tests_count: int = 0
        self.max_view_span: int = 0

        self.turn_count: int = 0
        self.baseline_lines: Optional[list] = None
        self.working_lines: Optional[list] = None
        self.best_lines: Optional[list] = None
        self.changed_lines: Optional[set] = None
        self.bug_type: Optional[str] = None
        self.bug_types_available: Optional[list] = None
        self.target_line_index: Optional[int] = None
        self.fix_line_text: Optional[str] = None
        self.last_test_pass: Optional[int] = None
        self.note_memory: str = ""
        self.last_search_result: Optional[int] = None

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
            if min_val <= max_val:
                actual_value = max(min_val, min(max_val, actual_value))
            else:
                actual_value = max(max_val, min(min_val, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _build_program(self):
        all_bug_types = [
            "OffByOneRange",
            "WrongOperator",
            "MissingReturn",
            "IncorrectInitialization",
            "VariableNameMismatch",
            "SwapOperands",
            "ShadowingBug",
        ]
        k = max(1, min(len(all_bug_types), self.num_bug_types))
        self.bug_types_available = all_bug_types[:k]
        self.bug_type = random.choice(self.bug_types_available)

        core_lines = [
            "# Sum numbers from 1..n",
            "def sum_to_n(n):",
            None,  # placeholder for initialization
            None,  # placeholder for loop header
            None,  # placeholder for accumulation
            None,  # placeholder for return
        ]

        # Inject bug variants and fixes for the same function
        if self.bug_type == "OffByOneRange":
            init = "    total = 0"
            loop_bug = "    for i in range(1, n):"
            loop_fix = "    for i in range(1, n+1):"
            acc = "        total += i"
            ret = "    return total"
            core_lines[2] = init
            core_lines[3] = loop_bug
            core_lines[4] = acc
            core_lines[5] = ret
            bug_line_local = 3
            fix_text = loop_fix

        elif self.bug_type == "WrongOperator":
            init = "    total = 0"
            loop = "    for i in range(1, n+1):"
            acc_bug = "        total -= i"
            acc_fix = "        total += i"
            ret = "    return total"
            core_lines[2] = init
            core_lines[3] = loop
            core_lines[4] = acc_bug
            core_lines[5] = ret
            bug_line_local = 4
            fix_text = acc_fix

        elif self.bug_type == "MissingReturn":
            init = "    total = 0"
            loop = "    for i in range(1, n+1):"
            acc = "        total += i"
            ret_bug = "    return"
            ret_fix = "    return total"
            core_lines[2] = init
            core_lines[3] = loop
            core_lines[4] = acc
            core_lines[5] = ret_bug
            bug_line_local = 5
            fix_text = ret_fix

        elif self.bug_type == "IncorrectInitialization":
            init_bug = "    total = 1"
            loop = "    for i in range(1, n+1):"
            acc = "        total += i"
            ret = "    return total"
            core_lines[2] = init_bug
            core_lines[3] = loop
            core_lines[4] = acc
            core_lines[5] = ret
            bug_line_local = 2
            fix_text = "    total = 0"

        elif self.bug_type == "VariableNameMismatch":
            init_bug = "    totl = 0"
            loop = "    for i in range(1, n+1):"
            acc = "        total += i"
            ret = "    return total"
            core_lines[2] = init_bug
            core_lines[3] = loop
            core_lines[4] = acc
            core_lines[5] = ret
            bug_line_local = 2
            fix_text = "    total = 0"

        elif self.bug_type == "SwapOperands":
            init = "    total = 0"
            loop = "    for i in range(1, n+1):"
            acc_bug = "        total = i - total"
            acc_fix = "        total += i"
            ret = "    return total"
            core_lines[2] = init
            core_lines[3] = loop
            core_lines[4] = acc_bug
            core_lines[5] = ret
            bug_line_local = 4
            fix_text = acc_fix

        else:  # ShadowingBug
            init = "    total = 0"
            loop_bug = "    for total in range(1, n+1):"
            loop_fix = "    for i in range(1, n+1):"
            acc = "        total += i"
            ret = "    return total"
            core_lines[2] = init
            core_lines[3] = loop_bug
            core_lines[4] = acc
            core_lines[5] = ret
            bug_line_local = 3
            fix_text = loop_fix

        filler_count = max(0, self.num_lines - len(core_lines))
        filler = []
        for i in range(filler_count):
            c = random.choice([
                "# comment: performance note",
                "# TODO: review naming",
                "# refactor opportunity",
                "# audit: check tests",
                "# doc: sum_to_n usage",
            ])
            filler.append(c)
        self.baseline_lines = filler + core_lines
        self.working_lines = list(self.baseline_lines)
        self.best_lines = None
        self.changed_lines = set()
        self.target_line_index = len(filler) + bug_line_local
        self.fix_line_text = fix_text

    def _get_instructions(self) -> str:
        actions = [
            "- VIEW line=K",
            "- VIEW range=START:END (span limited)",
            "- VIEW meta=true",
            "- SEARCH pattern=TEXT",
            "- COMPARE line=K",
            "- PATCH line=K text=\"...\"",
            "- REVERT",
            "- NOTE msg=\"your hypothesis\"",
            "- KEEP",
            "- TEST",
            "- SUBMIT",
            "- HELP",
        ]
        return (
            "You are in a code debugging environment.\n"
            "Goal: fix a single bug by patching exactly one line and submit the correct fix.\n"
            "You can inspect lines, search patterns, run tests, and update a working patch.\n"
            "Allowed actions:\n"
            + "\n".join(actions)
            + "\nFormat your action as \\boxed{COMMAND arg=value ...}.\n"
            f"Example: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        status = []
        status.append(f"Turn: {self.turn_count}/{self.max_turns}")
        status.append(f"File lines: {len(self.baseline_lines) if self.baseline_lines else 0}")
        status.append(f"Changed lines: {sorted(self.changed_lines) if self.changed_lines else []}")
        if self.last_test_pass is not None:
            status.append(f"Last tests: {self.last_test_pass}/{self.tests_count}")
        if self.note_memory:
            status.append(f"Note: {self.note_memory[:60]}")
        actions_hint = "Enter actions in \\boxed{...}. Example: \\boxed{VIEW line=3}"
        return " | ".join(status) + "\n" + actions_hint

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()

        self.turn_count = 0
        self.last_test_pass = None
        self.last_search_result = None
        self.note_memory = ""
        self._build_program()

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if not parsed:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}"
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        cmd = parsed["cmd"]
        args = parsed.get("args", {})
        reward = 0.0
        obs = ""

        def line_in_bounds(idx: int) -> bool:
            return 0 <= idx < len(self.working_lines)

        if cmd == "help":
            obs = "Actions: VIEW, PATCH, COMPARE, SEARCH, NOTE, KEEP, REVERT, TEST, SUBMIT. Use \\boxed{...} with arg=value."
        elif cmd == "view":
            if "meta" in args:
                bt = ", ".join(self.bug_types_available) if self.bug_types_available else ""
                obs = f"Meta: lines={len(self.working_lines)}, bug_types=[{bt}], max_view_span={self.max_view_span}"
            elif "line" in args:
                try:
                    k = int(args["line"])
                    if not line_in_bounds(k):
                        obs = "Protocol violation: line index out of bounds"
                    else:
                        obs = f"Line {k}: {self.working_lines[k]}"
                except:
                    obs = "Protocol violation: invalid line index"
            elif "range" in args or ("start" in args and "end" in args):
                if "range" in args:
                    rng = args["range"]
                    try:
                        start_str, end_str = rng.split(":")
                        start = int(start_str)
                        end = int(end_str)
                    except:
                        start, end = -1, -2
                else:
                    try:
                        start = int(args["start"])
                        end = int(args["end"])
                    except:
                        start, end = -1, -2
                span = max(0, end - start + 1)
                if span <= 0:
                    obs = "Protocol violation: invalid range"
                elif span > self.max_view_span:
                    obs = f"Protocol violation: requested span exceeds max_view_span={self.max_view_span}"
                elif not (line_in_bounds(start) and line_in_bounds(end)):
                    obs = "Protocol violation: range out of bounds"
                else:
                    chunk = [f"{i}: {self.working_lines[i]}" for i in range(start, end + 1)]
                    obs = "Range view:\n" + "\n".join(chunk)
            else:
                obs = "Protocol violation: VIEW requires 'line=K', 'range=START:END', or 'meta=true'"
        elif cmd == "search":
            pattern = args.get("pattern", "")
            if not pattern:
                obs = "Protocol violation: SEARCH requires pattern=TEXT"
            else:
                found = -1
                for i, line in enumerate(self.working_lines):
                    if pattern.lower() in line.lower():
                        found = i
                        break
                self.last_search_result = found
                obs = f"Search result: first_index={found}"
        elif cmd == "compare":
            try:
                k = int(args.get("line", "-1"))
                if not line_in_bounds(k):
                    obs = "Protocol violation: line index out of bounds"
                else:
                    eq = self.working_lines[k] == self.baseline_lines[k]
                    obs = f"Compare line {k}: equal_to_baseline={eq}"
            except:
                obs = "Protocol violation: invalid line index"
        elif cmd == "patch":
            if "line" not in args or "text" not in args:
                obs = "Protocol violation: PATCH requires line=K and text=\"...\""
            else:
                try:
                    k = int(args["line"])
                    if not line_in_bounds(k):
                        obs = "Protocol violation: line index out of bounds"
                    else:
                        self.working_lines[k] = args["text"]
                        self.changed_lines.add(k)
                        obs = f"Patched line {k}"
                except:
                    obs = "Protocol violation: invalid line index"
        elif cmd == "revert":
            self.working_lines = list(self.baseline_lines)
            self.changed_lines.clear()
            obs = "Reverted working patch to baseline"
        elif cmd == "note":
            msg = args.get("msg", "")
            if not msg:
                obs = "Protocol violation: NOTE requires msg=\"...\""
            else:
                self.note_memory = msg
                obs = "Note recorded"
        elif cmd == "keep":
            self.best_lines = list(self.working_lines)
            obs = "Saved current patch as best"
        elif cmd == "test":
            base_pass_map = {
                "OffByOneRange": int(self.tests_count * 0.5),
                "WrongOperator": int(self.tests_count * 0.25),
                "MissingReturn": 0,
                "IncorrectInitialization": int(self.tests_count * 0.3),
                "VariableNameMismatch": 0,
                "SwapOperands": int(self.tests_count * 0.1),
                "ShadowingBug": int(self.tests_count * 0.1),
            }
            hint_map = {
                "OffByOneRange": "n+1",
                "WrongOperator": "+=",
                "MissingReturn": "return total",
                "IncorrectInitialization": "total = 0",
                "VariableNameMismatch": "total = 0",
                "SwapOperands": "+=",
                "ShadowingBug": "for i in",
            }
            fixed_exact = self.working_lines[self.target_line_index] == self.fix_line_text
            hint_sub = hint_map.get(self.bug_type, "")
            contains_hint = hint_sub and (hint_sub in self.working_lines[self.target_line_index])
            if fixed_exact:
                passed = self.tests_count
            elif contains_hint:
                passed = min(self.tests_count, base_pass_map[self.bug_type] + int(self.tests_count * 0.33))
            else:
                passed = base_pass_map[self.bug_type]
            self.last_test_pass = passed
            obs = f"Tests run: {self.tests_count}, passed: {passed}"
        elif cmd == "submit":
            if self.working_lines[self.target_line_index] == self.fix_line_text:
                obs = f"Success! Fixed {self.bug_type} at line {self.target_line_index}"
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Failed! Incorrect fix for {self.bug_type} at line {self.target_line_index}"
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
        else:
            obs = f"Unsupported action: {cmd}"
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            truncated = True
            terminated = True
            obs = f"Reached max turns ({self.max_turns})"
            return obs, 0.0, terminated, truncated, {"suffix": self.get_task_suffix()}

        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Any]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()

        # command token
        if not content:
            return None
        parts = content.split()
        cmd = parts[0].strip().lower()
        rest = content[len(parts[0]) :].strip()

        # parse key=value pairs; special-case text= and msg= to capture remainder if quoted or not
        args: Dict[str, Any] = {}
        # Handle 'range=START:END'
        rng_match = re.search(r'range\s*=\s*([0-9]+)\s*:\s*([0-9]+)', rest, re.IGNORECASE)
        if rng_match:
            args["range"] = f"{rng_match.group(1)}:{rng_match.group(2)}"

        # Handle line=K
        line_match = re.search(r'line\s*=\s*([0-9]+)', rest, re.IGNORECASE)
        if line_match:
            args["line"] = line_match.group(1)

        # meta=true
        if re.search(r'meta\s*=\s*true', rest, re.IGNORECASE):
            args["meta"] = "true"

        # start and end
        start_match = re.search(r'start\s*=\s*([0-9]+)', rest, re.IGNORECASE)
        end_match = re.search(r'end\s*=\s*([0-9]+)', rest, re.IGNORECASE)
        if start_match and end_match:
            args["start"] = start_match.group(1)
            args["end"] = end_match.group(1)

        # pattern=...
        pat_match = re.search(r'pattern\s*=\s*"(.*?)"', rest, re.IGNORECASE | re.DOTALL)
        if pat_match:
            args["pattern"] = pat_match.group(1)
        else:
            pat_match2 = re.search(r'pattern\s*=\s*([^\s]+)', rest, re.IGNORECASE)
            if pat_match2:
                args["pattern"] = pat_match2.group(1)

        # text=...
        txt_match = re.search(r'text\s*=\s*"(.*?)"', rest, re.IGNORECASE | re.DOTALL)
        if txt_match:
            args["text"] = txt_match.group(1)
        else:
            txt_match2 = re.search(r'text\s*=\s*(.+)', rest, re.IGNORECASE | re.DOTALL)
            if txt_match2:
                args["text"] = txt_match2.group(1).strip()

        # msg=...
        msg_match = re.search(r'msg\s*=\s*"(.*?)"', rest, re.IGNORECASE | re.DOTALL)
        if msg_match:
            args["msg"] = msg_match.group(1)
        else:
            msg_match2 = re.search(r'msg\s*=\s*(.+)', rest, re.IGNORECASE | re.DOTALL)
            if msg_match2:
                args["msg"] = msg_match2.group(1).strip()

        return {"cmd": cmd, "args": args}

    def sample_random_action(self) -> str:
        if self.baseline_lines:
            k = random.randint(0, len(self.baseline_lines) - 1)
        else:
            k = 3
        return f"\\boxed{{VIEW line={k}}}"


class CodeDebugEnvWithFeedback(CodeDebugEnv):
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
            hint = "Use \\boxed{COMMAND arg=value}. Example: \\boxed{VIEW line=3}"
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "out of bounds" in text:
                error_detail["violation"] = "index_out_of_bounds"
                hint = "Check valid line indices with \\boxed{VIEW meta=true} and retry."
            elif "invalid range" in text:
                error_detail["violation"] = "invalid_range"
                hint = "Use range=START:END with START<=END and within file bounds."
            elif "exceeds max_view_span" in text:
                error_detail["violation"] = "span_too_large"
                hint = f"Request fewer lines per view (max {self.max_view_span})."
            elif "requires" in text:
                error_detail["violation"] = "missing_arguments"
                hint = "Provide required args. For PATCH, include line=K and text=\"...\"."
            else:
                hint = "Re-read the action format and ensure arguments are valid."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["cmd"] = "unknown"
            hint = "Use one of: VIEW, PATCH, COMPARE, SEARCH, NOTE, KEEP, REVERT, TEST, SUBMIT."
        elif "failed!" in text:
            error_type = "WrongDecision"
            error_detail["expected_fix_line"] = self.fix_line_text
            error_detail["target_line"] = self.target_line_index
            hint = f"Inspect line {self.target_line_index} via VIEW; consider common bugs like {self.bug_type}."
        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Plan queries efficiently; use SEARCH to narrow candidates, PATCH, TEST, then SUBMIT."
        elif "success!" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None
        else:
            error_type = "OK"
            if "tests run:" in text:
                error_detail["phase"] = "testing"
                error_detail["last_pass"] = self.last_test_pass

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "changed_lines": sorted(self.changed_lines) if self.changed_lines else [],
                "tests_count": self.tests_count,
                "last_test_pass": self.last_test_pass,
                "target_line": self.target_line_index,
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
            "hint": "Start with \\boxed{VIEW meta=true} to learn file size and bug types.",
            "turn": 0,
        }
        return obs, info