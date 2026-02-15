from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class AlgorithmRepairEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 20,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 20

        self.complexity_params = {
            # number of independent bugs injected into the program; more bugs = harder to fix
            'num_bugs': (1, 6),
            # number of hidden test cases; more tests = harder generalization
            'num_tests': (3, 12),
            # number of distractor no-op/comment lines inserted; more noise = harder
            'noise_lines': (0, 4),
            # REVERSED: maximum number of edit actions allowed; fewer allowed edits = harder
            'edit_budget': (8, 2),
            # typical array size per test; larger arrays = more varied execution = harder
            'arr_size': (4, 12),
            # number of possible tasks in the sampling pool (from 1 to 4); more variety = harder
            'task_set_size': (1, 4),
        }
        self.param_variance = {
            'num_bugs': 1,
            'num_tests': 2,
            'noise_lines': 1,
            'edit_budget': 2,
            'arr_size': 2,
            'task_set_size': 0,
        }

        self.turn_count: int = 0

        self.original_program: list = []
        self.current_program: list = []
        self.correct_program: list = []
        self.target_task: str = ""
        self.test_suite: list = []
        self.optimal_edit_cost: int = 0
        self.edits_made: int = 0
        self.last_eval: Dict[str, Any] = {}

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
            if min(self.complexity_params[param_name][0], self.complexity_params[param_name][1]) == self.complexity_params[param_name][0]:
                actual_value = max(min_val, min(max_val, actual_value))
            else:
                actual_value = max(max_val, min(min_val, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        example = self.sample_random_action()
        return (
            "Algorithm Repair Game\n"
            "You are given a buggy algorithm in a constrained DSL. Your goal is to repair it so that it computes the specified target function correctly on hidden tests, while minimizing line edits relative to the original program.\n"
            "Allowed DSL lines:\n"
            "- INIT var value        (value in {0,1,FIRST})\n"
            "- LOOP i 0 n            (loop i from 0 to n-1)\n"
            "- LOOP i 1 n            (loop i from 1 to n-1)\n"
            "- ENDLOOP\n"
            "- ADD sum arr[i]\n"
            "- SUB sum arr[i]\n"
            "- MUL prod arr[i]\n"
            "- MAX max arr[i]\n"
            "- COUNT_POS count arr[i]\n"
            "- RETURN var            (var in {sum,prod,count,max})\n"
            "- NOP                   (no operation)\n"
            "- COMMENT text          (ignored)\n"
            "Actions:\n"
            "- EDIT k: <line>\n"
            "- INSERT p: <line>      (insert after line p; use 0 to insert at start)\n"
            "- DELETE k\n"
            "- SHOW                  (print current program)\n"
            "- RUN                   (submit for evaluation; episode ends)\n"
            "Use \\boxed{...} to submit actions. For example: "
            f"{example}\n"
        )

    def get_task_suffix(self) -> str:
        listing = self._program_to_string(self.current_program)
        return (
            f"Target task: {self.target_task}\n"
            f"Edit actions remaining: {max(0, self.edit_budget - self.edits_made)} (budget={self.edit_budget})\n"
            f"Turns left: {max(0, self.max_turns - self.turn_count)}\n"
            f"Program:\n{listing}\n"
            "Enter your action in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()

        self.turn_count = 0
        self.edits_made = 0
        self.last_eval = {}

        tasks_pool_all = ['SUM', 'PRODUCT', 'COUNT_POS', 'MAX']
        pool_size = max(1, min(len(tasks_pool_all), self.task_set_size))
        tasks_pool = tasks_pool_all[:pool_size]
        self.target_task = random.choice(tasks_pool)

        self.correct_program = self._make_correct_program(self.target_task)
        self.original_program = list(self.correct_program)
        buggable_indices = [i for i in range(len(self.original_program))]
        num_bugs = min(self.num_bugs, len(buggable_indices))
        random.shuffle(buggable_indices)
        bug_idxs = buggable_indices[:num_bugs]
        for idx in bug_idxs:
            self.original_program[idx] = self._make_bug(self.original_program[idx], self.target_task)

        for _ in range(self.noise_lines):
            pos = random.randint(0, len(self.original_program))
            self.original_program.insert(pos, "COMMENT note")

        self.current_program = list(self.original_program)
        self.optimal_edit_cost = num_bugs

        # Ensure edit budget is at least the optimal cost to keep solvable
        if self.edit_budget < self.optimal_edit_cost:
            self.edit_budget = self.optimal_edit_cost

        self.test_suite = self._make_test_suite(self.target_task, self.num_tests, self.arr_size)

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)
        if parsed is None:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        terminated = False
        truncated = False
        reward = 0.0
        info: Dict[str, Any] = {"suffix": self.get_task_suffix()}

        if parsed["type"] == "SHOW":
            obs = self.get_task_suffix()
            return obs, 0.0, False, False, info

        if parsed["type"] in ("EDIT", "INSERT", "DELETE"):
            if self.edits_made >= self.edit_budget:
                obs = "Protocol violation: edit budget exhausted. No further edit actions allowed; you may RUN."
                return obs, 0.0, False, False, info

            if parsed["type"] == "EDIT":
                idx = parsed["index"]
                if not (1 <= idx <= len(self.current_program)):
                    obs = "Protocol violation: line index out of bounds for EDIT."
                    return obs, 0.0, False, False, info
                self.current_program[idx - 1] = parsed["line"]
                self.edits_made += 1
                obs = f"Applied EDIT at line {idx}."
            elif parsed["type"] == "INSERT":
                pos = parsed["pos"]
                if not (0 <= pos <= len(self.current_program)):
                    obs = "Protocol violation: position out of bounds for INSERT."
                    return obs, 0.0, False, False, info
                self.current_program.insert(pos, parsed["line"])
                self.edits_made += 1
                obs = f"Applied INSERT after line {pos}."
            elif parsed["type"] == "DELETE":
                idx = parsed["index"]
                if not (1 <= idx <= len(self.current_program)):
                    obs = "Protocol violation: line index out of bounds for DELETE."
                    return obs, 0.0, False, False, info
                del self.current_program[idx - 1]
                self.edits_made += 1
                obs = f"Applied DELETE at line {idx}."

            obs += "\n" + self.get_task_suffix()
            return obs, 0.0, False, False, info

        if parsed["type"] == "RUN":
            structural_ok, structural_msg = self._check_structure(self.current_program)
            if not structural_ok:
                edit_dist = self._edit_distance(self.original_program, self.current_program)
                self.last_eval = {
                    "format_ok": True,
                    "structural_ok": False,
                    "structural_msg": structural_msg,
                    "valid": False,
                    "edit_distance": edit_dist,
                    "optimal_cost": self.optimal_edit_cost,
                }
                obs = f"Failed! Program structure invalid: {structural_msg}\nEdit distance used: {edit_dist}, optimal required: {self.optimal_edit_cost}."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

            syntax_ok = self._check_syntax(self.current_program)
            if not syntax_ok:
                edit_dist = self._edit_distance(self.original_program, self.current_program)
                self.last_eval = {
                    "format_ok": False,
                    "structural_ok": True,
                    "valid": False,
                    "edit_distance": edit_dist,
                    "optimal_cost": self.optimal_edit_cost,
                }
                obs = f"Failed! Program syntax invalid.\nEdit distance used: {edit_dist}, optimal required: {self.optimal_edit_cost}."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

            passed = self._run_on_tests(self.current_program, self.test_suite)
            edit_dist = self._edit_distance(self.original_program, self.current_program)
            self.last_eval = {
                "format_ok": True,
                "structural_ok": True,
                "valid": bool(passed),
                "edit_distance": edit_dist,
                "optimal_cost": self.optimal_edit_cost,
            }

            if passed and edit_dist == self.optimal_edit_cost:
                obs = f"Success! Valid and optimal fix applied.\nEdit distance used: {edit_dist}, optimal required: {self.optimal_edit_cost}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            elif passed and edit_dist != self.optimal_edit_cost:
                obs = f"Valid but not optimal.\nEdit distance used: {edit_dist}, optimal required: {self.optimal_edit_cost}."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Failed! Candidate does not satisfy tests.\nEdit distance used: {edit_dist}, optimal required: {self.optimal_edit_cost}."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        obs = "Unsupported action. Use EDIT, INSERT, DELETE, SHOW, or RUN."
        if self.turn_count >= self.max_turns:
            truncated = True
            terminated = True
            obs = f"Reached max turns ({self.max_turns})."
            return obs, 0.0, terminated, truncated, {"suffix": self.get_task_suffix()}
        return obs + "\n" + self.get_task_suffix(), reward, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Any]:
        if not action:
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        m = pattern.search(action)
        if not m:
            return None
        content = m.group(1).strip()
        if re.fullmatch(r'RUN', content, re.IGNORECASE):
            return {"type": "RUN"}
        if re.fullmatch(r'SHOW', content, re.IGNORECASE):
            return {"type": "SHOW"}

        edit_match = re.match(r'^EDIT\s+(\d+)\s*:\s*(.+)$', content, re.IGNORECASE)
        if edit_match:
            idx = int(edit_match.group(1))
            line = edit_match.group(2).strip()
            return {"type": "EDIT", "index": idx, "line": line}

        insert_match = re.match(r'^INSERT\s+(\d+)\s*:\s*(.+)$', content, re.IGNORECASE)
        if insert_match:
            pos = int(insert_match.group(1))
            line = insert_match.group(2).strip()
            return {"type": "INSERT", "pos": pos, "line": line}

        delete_match = re.match(r'^DELETE\s+(\d+)$', content, re.IGNORECASE)
        if delete_match:
            idx = int(delete_match.group(1))
            return {"type": "DELETE", "index": idx}

        return {"type": "UNSUPPORTED", "raw": content}

    def sample_random_action(self) -> str:
        choices = []
        if self.current_program:
            ln = random.randint(1, len(self.current_program))
            choices.append(f"\\boxed{{EDIT {ln}: NOP}}")
            choices.append(f"\\boxed{{DELETE {ln}}}")
        ins_pos = 0 if not self.current_program else random.randint(0, len(self.current_program))
        choices.append(f"\\boxed{{INSERT {ins_pos}: COMMENT temp}}")
        choices.append("\\boxed{SHOW}")
        choices.append("\\boxed{RUN}")
        return random.choice(choices)

    def _program_to_string(self, program: list) -> str:
        return "\n".join([f"{i+1}: {line}" for i, line in enumerate(program)])

    def _make_correct_program(self, task: str) -> list:
        if task == 'SUM':
            return ["INIT sum 0", "LOOP i 0 n", "ADD sum arr[i]", "ENDLOOP", "RETURN sum"]
        if task == 'PRODUCT':
            return ["INIT prod 1", "LOOP i 0 n", "MUL prod arr[i]", "ENDLOOP", "RETURN prod"]
        if task == 'COUNT_POS':
            return ["INIT count 0", "LOOP i 0 n", "COUNT_POS count arr[i]", "ENDLOOP", "RETURN count"]
        if task == 'MAX':
            return ["INIT max FIRST", "LOOP i 1 n", "MAX max arr[i]", "ENDLOOP", "RETURN max"]
        return ["INIT sum 0", "LOOP i 0 n", "ADD sum arr[i]", "ENDLOOP", "RETURN sum"]

    def _make_bug(self, line: str, task: str) -> str:
        cmds = ["ADD sum arr[i]", "SUB sum arr[i]", "MUL prod arr[i]", "MAX max arr[i]", "COUNT_POS count arr[i]"]
        if line.startswith("INIT"):
            parts = line.split()
            var = parts[1]
            val = parts[2] if len(parts) > 2 else "0"
            if task in ('SUM', 'COUNT_POS') and val == "0":
                return f"INIT {var} 1"
            if task == 'PRODUCT' and val == "1":
                return f"INIT {var} 0"
            if task == 'MAX':
                return f"INIT {var} 0"
            return f"INIT {var} 1"
        if line.startswith("LOOP"):
            if " 0 n" in line:
                return "LOOP i 1 n"
            else:
                return "LOOP i 0 n"
        if line.startswith("ADD"):
            return random.choice([c for c in cmds if c != "ADD sum arr[i]"])
        if line.startswith("MUL"):
            return random.choice([c for c in cmds if c != "MUL prod arr[i]"])
        if line.startswith("COUNT_POS"):
            return random.choice([c for c in cmds if c != "COUNT_POS count arr[i]"])
        if line.startswith("MAX"):
            return random.choice([c for c in cmds if c != "MAX max arr[i]"])
        if line.startswith("RETURN"):
            return "RETURN i"
        return "NOP"

    def _make_test_suite(self, task: str, num_tests: int, arr_size: int) -> list:
        suite = []
        for _ in range(num_tests):
            size = max(2, int(round(random.uniform(max(2, arr_size - 2), arr_size + 2))))
            arr = [random.randint(-5, 7) for _ in range(size)]
            suite.append(arr)
        return suite

    def _check_structure(self, program: list) -> Tuple[bool, str]:
        loop_count = sum(1 for l in program if l.strip().startswith("LOOP"))
        end_count = sum(1 for l in program if l.strip().startswith("ENDLOOP"))
        ret_count = sum(1 for l in program if l.strip().startswith("RETURN"))
        if loop_count != 1 or end_count != 1 or ret_count != 1:
            return False, "Program must contain exactly one LOOP, one ENDLOOP, and one RETURN."
        loop_idx = next(i for i, l in enumerate(program) if l.strip().startswith("LOOP"))
        end_idx = next(i for i, l in enumerate(program) if l.strip().startswith("ENDLOOP"))
        if not (loop_idx < end_idx):
            return False, "ENDLOOP must come after LOOP."
        return True, "OK"

    def _check_syntax(self, program: list) -> bool:
        for line in program:
            s = line.strip()
            if not s:
                return False
            if s.startswith("COMMENT") or s == "NOP":
                continue
            if s.startswith("INIT"):
                parts = s.split()
                if len(parts) != 3:
                    return False
                if parts[1] not in ('sum', 'prod', 'count', 'max'):
                    return False
                if parts[2] not in ('0', '1', 'FIRST'):
                    return False
            elif s.startswith("LOOP"):
                parts = s.split()
                if len(parts) != 4:
                    return False
                if parts[1] != 'i' or parts[2] not in ('0', '1') or parts[3] != 'n':
                    return False
            elif s == "ENDLOOP":
                pass
            elif s.startswith("ADD"):
                if s != "ADD sum arr[i]":
                    return False
            elif s.startswith("SUB"):
                if s != "SUB sum arr[i]":
                    return False
            elif s.startswith("MUL"):
                if s != "MUL prod arr[i]":
                    return False
            elif s.startswith("MAX"):
                if s != "MAX max arr[i]":
                    return False
            elif s.startswith("COUNT_POS"):
                if s != "COUNT_POS count arr[i]":
                    return False
            elif s.startswith("RETURN"):
                parts = s.split()
                if len(parts) != 2:
                    return False
                if parts[1] not in ('sum', 'prod', 'count', 'max'):
                    return False
            else:
                return False
        return True

    def _run_on_tests(self, program: list, tests: list) -> bool:
        expected_fn = self._expected_function(self.target_task)
        for arr in tests:
            try:
                out = self._execute(program, arr)
            except Exception:
                return False
            exp = expected_fn(arr)
            if out != exp:
                return False
        return True

    def _expected_function(self, task: str):
        if task == 'SUM':
            return lambda arr: sum(arr)
        if task == 'PRODUCT':
            prod = lambda arr: (1 if len(arr) == 0 else self._prod(arr))
            return prod
        if task == 'COUNT_POS':
            return lambda arr: sum(1 for x in arr if x > 0)
        if task == 'MAX':
            return lambda arr: max(arr)
        return lambda arr: sum(arr)

    def _prod(self, arr):
        p = 1
        for x in arr:
            p *= x
        return p

    def _execute(self, program: list, arr: list) -> int:
        n = len(arr)
        env: Dict[str, Any] = {'sum': None, 'prod': None, 'count': None, 'max': None, 'i': 0}
        pc = 0
        loop_idx = None
        end_idx = None
        for i, line in enumerate(program):
            if line.strip().startswith("LOOP"): loop_idx = i
            if line.strip() == "ENDLOOP": end_idx = i
        if loop_idx is None or end_idx is None or loop_idx >= end_idx:
            raise RuntimeError("bad structure")

        # pre-loop segment
        for line in program[:loop_idx]:
            self._exec_line(line.strip(), env, arr, n)

        loop_line = program[loop_idx].strip()
        parts = loop_line.split()
        start = int(parts[2])
        indices = range(start, n)
        for i in indices:
            env['i'] = i
            for line in program[loop_idx + 1:end_idx]:
                self._exec_line(line.strip(), env, arr, n)

        # post-loop segment
        ret_value = None
        for line in program[end_idx + 1:]:
            s = line.strip()
            if s.startswith("RETURN"):
                ret_var = s.split()[1]
                ret_value = env.get(ret_var)
                break
            else:
                self._exec_line(s, env, arr, n)
        if ret_value is None:
            raise RuntimeError("no return")
        return int(ret_value)

    def _exec_line(self, s: str, env: Dict[str, Any], arr: list, n: int):
        if s.startswith("COMMENT") or s == "NOP":
            return
        if s.startswith("INIT"):
            _, var, val = s.split()
            if val == 'FIRST':
                env[var] = arr[0]
            else:
                env[var] = int(val)
            return
        if s.startswith("ADD"):
            env['sum'] = (0 if env['sum'] is None else env['sum']) + arr[env['i']]
            return
        if s.startswith("SUB"):
            env['sum'] = (0 if env['sum'] is None else env['sum']) - arr[env['i']]
            return
        if s.startswith("MUL"):
            env['prod'] = (1 if env['prod'] is None else env['prod']) * arr[env['i']]
            return
        if s.startswith("MAX"):
            cur = env['max']
            val = arr[env['i']]
            if cur is None:
                env['max'] = val
            else:
                env['max'] = cur if cur >= val else val
            return
        if s.startswith("COUNT_POS"):
            env['count'] = (0 if env['count'] is None else env['count']) + (1 if arr[env['i']] > 0 else 0)
            return
        if s.startswith("LOOP") or s == "ENDLOOP" or s.startswith("RETURN"):
            return
        raise RuntimeError("syntax error during exec")

    def _edit_distance(self, a: list, b: list) -> int:
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        return dp[m][n]


class AlgorithmRepairEnvWithFeedback(AlgorithmRepairEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        text = obs.lower()
        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint = None

        if "invalid action format" in text or "use \\boxed" in text:
            error_type = "FormatError"
            error_detail["issue"] = "missing_boxed_format"
            hint = "Wrap the action in \\boxed{...}, e.g., \\boxed{EDIT 3: ADD sum arr[i]}."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["supported"] = ["EDIT k: <line>", "INSERT p: <line>", "DELETE k", "SHOW", "RUN"]
            hint = "Use one of EDIT, INSERT, DELETE, SHOW, or RUN."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "budget exhausted" in text:
                error_detail["violation"] = "edit_budget_exhausted"
                hint = "Stop editing and use \\boxed{RUN} to submit."
            elif "out of bounds" in text:
                error_detail["violation"] = "index_out_of_bounds"
                hint = "Use valid 1-based line indices within the program length."
            elif "structure invalid" in text or "must contain exactly one loop" in text:
                error_detail["violation"] = "invalid_structure"
                hint = "Ensure exactly one LOOP, one ENDLOOP, and one RETURN remain."

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = getattr(self, "max_turns", None)
            hint = "Plan edits and RUN earlier to avoid timeout."

        elif "failed! candidate does not satisfy tests" in text or "program syntax invalid" in text:
            error_type = "WrongDecision"
            if hasattr(self, "last_eval") and self.last_eval:
                error_detail["edit_distance"] = self.last_eval.get("edit_distance")
                error_detail["optimal_cost"] = self.last_eval.get("optimal_cost")
                error_detail["structural_ok"] = self.last_eval.get("structural_ok")
                error_detail["format_ok"] = self.last_eval.get("format_ok", True)
            hint = "Check INIT constants, loop start index (0 vs 1), operation line (ADD/MUL/MAX/COUNT_POS), and the RETURN variable."

        elif "valid but not optimal" in text:
            error_type = "NotOptimal"
            if hasattr(self, "last_eval") and self.last_eval:
                error_detail["edit_distance"] = self.last_eval.get("edit_distance")
                error_detail["optimal_cost"] = self.last_eval.get("optimal_cost")
            hint = "Minimize line changes: revert superfluous inserts/deletes or non-essential edits."

        elif "success!" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = "Good job. Aim to replicate this performance across tasks."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "task": getattr(self, "target_task", None),
                "edit_budget": getattr(self, "edit_budget", None),
                "edits_made": getattr(self, "edits_made", None),
                "optimal_edit_cost": getattr(self, "optimal_edit_cost", None),
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
            "hint": "Start by inspecting with \\boxed{SHOW}, then EDIT minimal lines and \\boxed{RUN}.",
            "turn": 0,
        }
        return obs, info