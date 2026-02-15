from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class CodeGraphRecursionEnv(Env):
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

        # Evolvable parameters
        self.complexity_params = {
            'num_files': (1, 8),                    # More files => larger codebase => harder
            'max_funcs_per_file': (3, 25),          # More functions per file => more nodes => harder
            'max_calls_per_func': (1, 6),           # More calls per function => denser graph => harder
            'num_recursive_components': (0, 10),    # More cycles/SCCs => more recursion to detect => harder
            'obs_line_limit': (40, 8),              # REVERSED: fewer lines per open => less visibility => harder
            'noise_calls_per_file': (0, 6),         # More external/noise calls => more misleading content => harder
        }

        # Randomization variance settings
        self.param_variance = {
            'num_files': 1,
            'max_funcs_per_file': 3,
            'max_calls_per_func': 1,
            'num_recursive_components': 1,
            'obs_line_limit': 3,
            'noise_calls_per_file': 1,
        }

        # Placeholder attributes (filled in reset via _apply_complexity_params)
        self.num_files: int = 0
        self.max_funcs_per_file: int = 0
        self.max_calls_per_func: int = 0
        self.num_recursive_components: int = 0
        self.obs_line_limit: int = 0
        self.noise_calls_per_file: int = 0

        # State
        self.turn_count: int = 0
        self.files: list = []
        self.code_files: Dict[str, list] = {}
        self.functions_by_file: Dict[str, list] = {}
        self.func_to_file: Dict[str, str] = {}
        self.function_calls: Dict[str, set] = {}
        self.recursive_funcs_set: set = set()
        self.marked_recursive: set = set()
        self.true_recursive_count: int = 0

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
                    if min_val > max_val:
                        actual_value = max(max_val, min(min_val, actual_value))
                    else:
                        actual_value = max(min_val, min(max_val, actual_value))
                    actual_value = int(round(actual_value))
                else:
                    actual_value = int(round(center_value))
            else:
                actual_value = int(round(center_value))
            setattr(self, param_name, actual_value)

    def _get_instructions(self) -> str:
        return (
            "You are analyzing a synthetic codebase to determine recursion.\n"
            "Goal: submit the total number of recursive functions (functions that call themselves, or are in mutual recursion cycles).\n"
            "Available commands (use \\boxed{...}):\n"
            "- ls\n"
            "- open <filename> <start>:<end>\n"
            "- functions <filename>\n"
            "- calls <function_name>\n"
            "- mark_rec <function_name>\n"
            "- query_marked\n"
            "- summary\n"
            "- help\n"
            "- submit total=<K>\n"
            "A recursive function is one that either has a self-call or belongs to a cycle of calls.\n"
            "Format your actions like: " + self.sample_random_action() + "\n"
        )

    def get_task_suffix(self) -> str:
        remaining = max(0, (self.max_turns or 0) - self.turn_count)
        preview_files = ", ".join(self.files[:min(3, len(self.files))]) if self.files else "(none)"
        return (
            f"Context: files={len(self.files)} [{preview_files}], "
            f"marked={len(self.marked_recursive)}, "
            f"turn={self.turn_count}, remaining={remaining}. "
            "Enter your action using \\boxed{...}."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        self.turn_count = 0
        self.files = [f"file_{i+1}.py" for i in range(self.num_files)]
        self.code_files = {fn: [] for fn in self.files}
        self.functions_by_file = {fn: [] for fn in self.files}
        self.func_to_file = {}
        self.function_calls = {}
        self.recursive_funcs_set = set()
        self.marked_recursive = set()
        all_funcs = []

        for fn in self.files:
            count = random.randint(max(1, self.max_funcs_per_file - 2), self.max_funcs_per_file)
            for j in range(count):
                fname = f"f{self.files.index(fn)+1}_{j+1}"
                self.functions_by_file[fn].append(fname)
                self.func_to_file[fname] = fn
                self.function_calls[fname] = set()
                all_funcs.append(fname)

        # Initial random calls
        for f in all_funcs:
            k = random.randint(0, self.max_calls_per_func)
            candidates = [x for x in all_funcs if x != f]
            random.shuffle(candidates)
            for c in candidates[:k]:
                self.function_calls[f].add(c)

        # Create recursive components (cycles)
        unused = set(all_funcs)
        created = 0
        while created < self.num_recursive_components and unused:
            choice = random.random()
            if choice < 0.5 and len(unused) >= 1:
                # Direct recursion
                f = random.choice(list(unused))
                self.function_calls[f].add(f)
                unused.discard(f)
                created += 1
            else:
                # Mutual cycle of size 2-4 (bounded by remaining)
                max_cycle_size = min(4, len(unused))
                if max_cycle_size < 2:
                    break
                size = random.randint(2, max_cycle_size)
                cycle_nodes = random.sample(list(unused), size)
                for i in range(size):
                    a = cycle_nodes[i]
                    b = cycle_nodes[(i + 1) % size]
                    self.function_calls[a].add(b)
                for n in cycle_nodes:
                    unused.discard(n)
                created += 1

        # Add noise calls (external) in code text only
        noise_registry: Dict[str, int] = {fn: 0 for fn in self.files}

        # Build text for files
        for fn in self.files:
            lines = []
            funcs = self.functions_by_file[fn]
            for fname in funcs:
                lines.append(f"def {fname}():")
                callees = sorted(list(self.function_calls[fname]))
                for cal in callees:
                    lines.append(f"    call {cal}()")
                # add noise calls per file spread across functions
                remain_noise = max(0, self.noise_calls_per_file - noise_registry[fn])
                extra_noise = random.randint(0, min(remain_noise, 2))
                for ni in range(extra_noise):
                    lines.append(f"    call external_{fn.replace('.py','')}_{noise_registry[fn]+ni+1}()")
                noise_registry[fn] += extra_noise
                lines.append("    return 0")
                lines.append("")  # blank line spacer
            self.code_files[fn] = lines

        # Compute recursive function set via SCC
        self.recursive_funcs_set = self._compute_recursive_functions(self.function_calls)
        self.true_recursive_count = len(self.recursive_funcs_set)

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _compute_recursive_functions(self, graph: Dict[str, set]) -> set:
        index = 0
        indices: Dict[str, int] = {}
        lowlink: Dict[str, int] = {}
        stack: list = []
        onstack: Dict[str, bool] = {}
        result_sccs: list = []
        nodes = list(graph.keys())

        def strongconnect(v: str):
            nonlocal index
            indices[v] = index
            lowlink[v] = index
            index += 1
            stack.append(v)
            onstack[v] = True
            for w in graph.get(v, set()):
                if w not in indices:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif onstack.get(w, False):
                    lowlink[v] = min(lowlink[v], indices[w])
            if lowlink[v] == indices[v]:
                scc = []
                while True:
                    w = stack.pop()
                    onstack[w] = False
                    scc.append(w)
                    if w == v:
                        break
                result_sccs.append(scc)

        for v in nodes:
            if v not in indices:
                strongconnect(v)

        recursive = set()
        for scc in result_sccs:
            if len(scc) > 1:
                recursive.update(scc)
            else:
                v = scc[0]
                if v in graph.get(v, set()):
                    recursive.add(v)
        return recursive

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if parsed is None:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        cmd = parsed[0]

        if cmd == "unsupported":
            obs = f"Error: Unsupported action '{parsed[1]}'."
            return obs, -0.25, True, False, {"suffix": self.get_task_suffix()}

        if cmd == "ls":
            obs = f"Files: {', '.join(self.files) if self.files else '(none)'}."
            reward = 0.0
        elif cmd == "open":
            fn, s, e = parsed[1], parsed[2], parsed[3]
            if fn not in self.code_files:
                obs = f"Error: Unknown file '{fn}'."
                return obs, -0.25, True, False, {"suffix": self.get_task_suffix()}
            if s < 1 or e < s:
                obs = f"Error: Invalid line range {s}:{e}."
                return obs, -0.25, True, False, {"suffix": self.get_task_suffix()}
            lines = self.code_files[fn]
            total = len(lines)
            if s > total:
                obs = f"Error: Start line {s} exceeds file length {total}."
                return obs, -0.25, True, False, {"suffix": self.get_task_suffix()}
            max_end = min(e, s + self.obs_line_limit - 1, total)
            snippet = []
            for i in range(s, max_end + 1):
                snippet.append(f"{i:03d}: {lines[i-1]}")
            obs = f"Open {fn} lines {s}:{max_end} (limit {self.obs_line_limit}):\n" + "\n".join(snippet)
            reward = 0.0
        elif cmd == "functions":
            fn = parsed[1]
            if fn not in self.functions_by_file:
                obs = f"Error: Unknown file '{fn}'."
                return obs, -0.25, True, False, {"suffix": self.get_task_suffix()}
            funcs = self.functions_by_file[fn]
            obs = f"Functions in {fn}: {', '.join(funcs) if funcs else '(none)'}."
            reward = 0.0
        elif cmd == "calls":
            fname = parsed[1]
            if fname not in self.function_calls:
                obs = f"Error: Unknown function '{fname}'."
                return obs, -0.25, True, False, {"suffix": self.get_task_suffix()}
            callees = sorted(list(self.function_calls[fname]))
            selfrec = "yes" if fname in self.function_calls[fname] else "no"
            obs = f"Calls of {fname}: {', '.join(callees) if callees else '(none)'}; self-recursive={selfrec}."
            reward = 0.0
        elif cmd == "mark_rec":
            fname = parsed[1]
            if fname not in self.function_calls:
                obs = f"Error: Unknown function '{fname}'."
                return obs, -0.25, True, False, {"suffix": self.get_task_suffix()}
            self.marked_recursive.add(fname)
            obs = f"Marked {fname} as recursive candidate. Total marked={len(self.marked_recursive)}."
            reward = 0.0
        elif cmd == "query_marked":
            obs = f"Marked candidates: {len(self.marked_recursive)}."
            reward = 0.0
        elif cmd == "summary":
            obs = (
                f"Summary: files={len(self.files)}, total_functions={sum(len(v) for v in self.functions_by_file.values())}, "
                f"marked={len(self.marked_recursive)}."
            )
            reward = 0.0
        elif cmd == "help":
            obs = "Commands: ls; open <filename> <start>:<end>; functions <filename>; calls <function>; mark_rec <function>; query_marked; summary; submit total=<K>."
            reward = 0.0
        elif cmd == "submit":
            k = parsed[1]
            if k == self.true_recursive_count:
                obs = f"Success! Correct total recursive functions = {k}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Failed. Your total={k}, correct={self.true_recursive_count}."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
        else:
            obs = "Error: Unsupported action."
            return obs, -0.25, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            obs_timeout = f"Reached max turns ({self.max_turns})."
            return obs_timeout, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Any]:
        if not action:
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        if content.lower() == "ls":
            return ("ls",)
        m = re.match(r'^open\s+([A-Za-z0-9_\.\-]+)\s+(\d+):(\d+)$', content)
        if m:
            fn = m.group(1)
            s = int(m.group(2))
            e = int(m.group(3))
            return ("open", fn, s, e)
        m = re.match(r'^functions\s+([A-Za-z0-9_\.\-]+)$', content)
        if m:
            return ("functions", m.group(1))
        m = re.match(r'^calls\s+([A-Za-z0-9_]+)$', content)
        if m:
            return ("calls", m.group(1))
        m = re.match(r'^mark_rec\s+([A-Za-z0-9_]+)$', content)
        if m:
            return ("mark_rec", m.group(1))
        if content.lower() == "query_marked":
            return ("query_marked",)
        if content.lower() == "summary":
            return ("summary",)
        if content.lower() == "help":
            return ("help",)
        m = re.match(r'^submit\s+total\s*=\s*(\d+)$', content, re.IGNORECASE)
        if m:
            return ("submit", int(m.group(1)))
        return ("unsupported", content)

    def sample_random_action(self) -> str:
        examples = [
            "\\boxed{ls}",
            "\\boxed{summary}",
            "\\boxed{help}",
            "\\boxed{submit total=0}",
        ]
        return random.choice(examples)


class CodeGraphRecursionEnvWithFeedback(CodeGraphRecursionEnv):
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
            hint = "Wrap your command in \\boxed{...}, e.g., \\boxed{ls}."
        elif text.startswith("error: unsupported action"):
            error_type = "UnsupportedAction"
            raw = None
            m = re.search(r"unsupported action '(.+?)'", obs, re.IGNORECASE)
            if m:
                raw = m.group(1)
            error_detail["raw"] = raw
            hint = "Use supported commands: ls, open <file> <s:e>, functions <file>, calls <func>, mark_rec <func>, query_marked, summary, submit total=<K>."
        elif "error: unknown file" in text or "error: unknown function" in text or "error: invalid line range" in text or "exceeds file length" in text:
            error_type = "ProtocolViolation"
            if "unknown file" in text:
                error_detail["violation"] = "unknown_file"
                hint = "List files with ls, then use functions <filename> to discover functions."
            elif "unknown function" in text:
                error_detail["violation"] = "unknown_function"
                hint = "Use functions <filename> to see available function names before calls/mark_rec."
            elif "invalid line range" in text or "exceeds file length" in text:
                error_detail["violation"] = "invalid_open_range"
                hint = "Check file length and request a valid range. You can open fewer lines; the environment limits lines per open."
        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Plan fewer exploratory steps and submit when confident."
        elif "failed." in text:
            error_type = "WrongDecision"
            m_given = re.search(r"your total=(\d+)", obs, re.IGNORECASE)
            m_correct = re.search(r"correct=(\d+)", obs, re.IGNORECASE)
            if m_given:
                error_detail["got"] = int(m_given.group(1))
            if m_correct:
                error_detail["expected"] = int(m_correct.group(1))
            hint = "Count functions with self-calls and those in mutual cycles. Query calls and infer cycles; marked set can help tracking."
        elif "success!" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None
        else:
            error_type = "OK"
            error_detail["outcome"] = "step_ok"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["marked"] = len(getattr(self, "marked_recursive", set()))
            diagnostic["files"] = len(getattr(self, "files", []))
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start by listing files with ls, then inspect functions and calls.",
            "turn": 0,
            "marked": 0,
            "files": len(getattr(self, "files", [])),
        }
        return obs, info