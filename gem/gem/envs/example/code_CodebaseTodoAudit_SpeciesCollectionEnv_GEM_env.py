from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List, Set


class CodebaseTodoAuditEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 80,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 80

        self.complexity_params = {
            # Number of files in the codebase: larger graph = more exploration = harder
            "num_files": (6, 24),
            # Max imports per file (out-degree): denser graph = larger reachable set = harder
            "max_out_degree": (1, 4),
            # Max TODOs per file: higher values make counting more error-prone = harder
            "todo_max_per_file": (2, 6),
            # REVERSED: % of files having non-zero TODOs; fewer non-zero makes searching harder
            "nonzero_todo_ratio_percent": (70, 30),
            # REVERSED: % quota for marking reachable files; fewer marks allowed = harder
            "mark_quota_percent": (100, 40),
        }

        self.param_variance = {
            "num_files": 1,                   # medium discrete range → ±1
            "max_out_degree": 0,              # small range → fixed
            "todo_max_per_file": 1,           # medium discrete range → ±1
            "nonzero_todo_ratio_percent": 5,  # larger range → ±5%
            "mark_quota_percent": 5,          # larger range → ±5%
        }

        self.num_files: int = 0
        self.max_out_degree: int = 0
        self.todo_max_per_file: int = 0
        self.nonzero_todo_ratio_percent: int = 0
        self.mark_quota_percent: int = 0

        self.turn_count: int = 0
        self.files: Dict[str, Dict[str, Any]] = {}
        self.start_file: str = ""
        self.known_files: Set[str] = set()
        self.inspected_files: Set[str] = set()
        self.marked_files: Set[str] = set()
        self.collected_sum: int = 0
        self.reachable_set: Set[str] = set()
        self.total_reachable_todo: int = 0
        self.mark_limit: int = 0

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
                    low = min(min_val, max_val)
                    high = max(min_val, max_val)
                    actual_value = max(low, min(high, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        return (
            "Codebase TODO Audit Game\n"
            "Goal: Determine the total number of TODO tags across all files reachable from the start module via imports.\n"
            "Hidden environment: A codebase import graph; each file has a TODO count.\n"
            "You can only see information by exploring.\n"
            "\n"
            "Actions:\n"
            "- task: Show these instructions again.\n"
            "- neighbors <file>: Reveal which files <file> imports; discovered files become known.\n"
            "- inspect <file>: Reveal the TODO count of a known file.\n"
            "- mark <file>: Collect the TODO count of an inspected file (consumes mark quota).\n"
            "- list_known: Show currently known files.\n"
            "- count: Show the current collected TODO sum across marked files.\n"
            "- submit <integer>: Submit your final answer for the total TODO count across all reachable files.\n"
            "\n"
            f"Start module: {self.start_file}\n"
            f"Mark quota: You can mark at most {self.mark_limit} files this episode.\n"
            "Format: Always respond with \\boxed{your_action}. Example:\n"
            f"{self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        known_sorted = sorted(self.known_files)
        inspected_sorted = sorted(self.inspected_files)
        marked_sorted = sorted(self.marked_files)
        remaining_quota = max(0, self.mark_limit - len(self.marked_files))
        return (
            f"State:\n"
            f"- Start: {self.start_file}\n"
            f"- Known files ({len(known_sorted)}): {', '.join(known_sorted) if known_sorted else '(none)'}\n"
            f"- Inspected files ({len(inspected_sorted)}): {', '.join(inspected_sorted) if inspected_sorted else '(none)'}\n"
            f"- Marked files ({len(marked_sorted)}): {', '.join(marked_sorted) if marked_sorted else '(none)'}\n"
            f"- Collected TODO sum: {self.collected_sum}\n"
            f"- Mark quota remaining: {remaining_quota}\n"
            f"- Turn: {self.turn_count}/{self.max_turns}\n"
            "Enter your action in \\boxed{...} as one of:\n"
            "task | neighbors <file> | inspect <file> | mark <file> | list_known | count | submit <integer>"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        names = [f"mod{idx}.py" for idx in range(self.num_files)]
        self.start_file = random.choice(names)
        self.files = {}
        for name in names:
            self.files[name] = {"neighbors": [], "todo": 0}

        for name in names:
            out_deg = random.randint(0, self.max_out_degree)
            possible = [n for n in names if n != name]
            if out_deg > len(possible):
                out_deg = len(possible)
            self.files[name]["neighbors"] = random.sample(possible, out_deg)

        self.reachable_set = self._bfs_reachable(self.start_file)
        target_min_reach = max(2, self.num_files // 4)
        attempts = 0
        while len(self.reachable_set) < target_min_reach and attempts < 100:
            out_nodes = [n for n in names if n not in self.reachable_set]
            if not out_nodes:
                break
            src = random.choice(list(self.reachable_set))
            dst = random.choice(out_nodes)
            if dst not in self.files[src]["neighbors"]:
                self.files[src]["neighbors"].append(dst)
            self.reachable_set = self._bfs_reachable(self.start_file)
            attempts += 1

        p_nonzero = max(0.0, min(1.0, self.nonzero_todo_ratio_percent / 100.0))
        for name in names:
            if random.random() < p_nonzero:
                self.files[name]["todo"] = random.randint(1, self.todo_max_per_file)
            else:
                self.files[name]["todo"] = 0

        if all(self.files[n]["todo"] == 0 for n in self.reachable_set):
            pick = random.choice(list(self.reachable_set))
            self.files[pick]["todo"] = random.randint(1, self.todo_max_per_file)

        self.total_reachable_todo = sum(self.files[n]["todo"] for n in self.reachable_set)
        self.mark_limit = max(1, int(round(len(self.reachable_set) * self.mark_quota_percent / 100.0)))

        self.turn_count = 0
        self.known_files = {self.start_file}
        self.inspected_files = set()
        self.marked_files = set()
        self.collected_sum = 0

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False
        reward = 0.0

        parsed = self._parse_action(action)
        if parsed is None:
            obs = "Invalid action format. Use \\boxed{...} with supported commands."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        if parsed.get("type") == "invalid":
            reason = parsed.get("reason", "unsupported")
            if reason == "unsupported":
                obs = "Unsupported action. This ends the episode."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            elif reason == "missing_argument":
                obs = "Protocol violation: missing argument for command."
                reward = -0.1
                return obs, reward, False, False, {"suffix": self.get_task_suffix()}
            elif reason == "bad_submit":
                obs = "Protocol violation: submit expects an integer. This ends the episode."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        cmd = parsed["type"]

        if cmd == "task":
            obs = self._get_instructions()
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if cmd == "list_known":
            known_sorted = sorted(self.known_files)
            obs = f"Known files ({len(known_sorted)}): {', '.join(known_sorted)}"
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if cmd == "neighbors":
            file = parsed["file"]
            if file not in self.known_files:
                obs = "Protocol violation: file not known. Discover via neighbors of known files first."
                reward = -0.1
                return obs, reward, False, False, {"suffix": self.get_task_suffix()}
            neighbors = self.files.get(file, {}).get("neighbors", [])
            for n in neighbors:
                self.known_files.add(n)
            neigh_text = ", ".join(neighbors) if neighbors else "(none)"
            obs = f"Neighbors of {file}: {neigh_text}."
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if cmd == "inspect":
            file = parsed["file"]
            if file not in self.known_files:
                obs = "Protocol violation: file not known. Use neighbors <known_file> to discover it."
                reward = -0.1
                return obs, reward, False, False, {"suffix": self.get_task_suffix()}
            todo = self.files[file]["todo"]
            self.inspected_files.add(file)
            obs = f"Inspection of {file}: TODO={todo}."
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if cmd == "mark":
            file = parsed["file"]
            if file not in self.known_files:
                obs = "Protocol violation: file not known. Discover it first."
                reward = -0.1
                return obs, reward, False, False, {"suffix": self.get_task_suffix()}
            if file not in self.inspected_files:
                obs = "Protocol violation: mark requires prior inspect."
                reward = -0.1
                return obs, reward, False, False, {"suffix": self.get_task_suffix()}
            if file in self.marked_files:
                obs = "Protocol violation: file already marked."
                reward = -0.05
                return obs, reward, False, False, {"suffix": self.get_task_suffix()}
            if len(self.marked_files) >= self.mark_limit:
                obs = "Protocol violation: mark quota exceeded."
                reward = -0.1
                return obs, reward, False, False, {"suffix": self.get_task_suffix()}
            self.marked_files.add(file)
            todo = self.files[file]["todo"]
            self.collected_sum += todo
            remaining = max(0, self.mark_limit - len(self.marked_files))
            obs = f"Marked {file}: collected {todo}. Total collected={self.collected_sum}. Quota left={remaining}."
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if cmd == "count":
            obs = f"Currently collected TODO sum: {self.collected_sum} across {len(self.marked_files)} files."
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if cmd == "submit":
            val = parsed["value"]
            expected = self.total_reachable_todo
            if val == expected:
                obs = f"Success! Correct summary {val}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Failed! Incorrect summary. Expected {expected}; Got {val}."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        obs = "Unsupported action. This ends the episode."
        return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        # Unreachable

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action:
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        extracted = matches[-1].group(1).strip()
        if not extracted:
            return {"type": "invalid", "reason": "unsupported"}

        tokens = extracted.split()
        if len(tokens) == 0:
            return {"type": "invalid", "reason": "unsupported"}

        cmd = tokens[0].lower()

        if cmd == "task":
            return {"type": "task"}
        if cmd == "list_known":
            return {"type": "list_known"}
        if cmd == "neighbors":
            if len(tokens) < 2:
                return {"type": "invalid", "reason": "missing_argument"}
            return {"type": "neighbors", "file": tokens[1]}
        if cmd == "inspect":
            if len(tokens) < 2:
                return {"type": "invalid", "reason": "missing_argument"}
            return {"type": "inspect", "file": tokens[1]}
        if cmd == "mark":
            if len(tokens) < 2:
                return {"type": "invalid", "reason": "missing_argument"}
            return {"type": "mark", "file": tokens[1]}
        if cmd == "count":
            return {"type": "count"}
        if cmd == "submit":
            if len(tokens) < 2:
                return {"type": "invalid", "reason": "bad_submit"}
            val_str = tokens[1]
            try:
                val = int(val_str)
            except Exception:
                return {"type": "invalid", "reason": "bad_submit"}
            return {"type": "submit", "value": val}

        return {"type": "invalid", "reason": "unsupported"}

    def sample_random_action(self) -> str:
        if self.start_file:
            return f"\\boxed{{neighbors {self.start_file}}}"
        return "\\boxed{task}"

    def _bfs_reachable(self, start: str) -> Set[str]:
        visited: Set[str] = set()
        queue: List[str] = [start]
        while queue:
            cur = queue.pop(0)
            if cur in visited:
                continue
            visited.add(cur)
            for nei in self.files.get(cur, {}).get("neighbors", []):
                if nei not in visited:
                    queue.append(nei)
        return visited


class CodebaseTodoAuditEnvWithFeedback(CodebaseTodoAuditEnv):
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
            hint = "Wrap your command in \\boxed{...} and use supported actions."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_command"
            hint = "Use one of: task, neighbors <file>, inspect <file>, mark <file>, list_known, count, submit <integer>."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "missing argument" in text:
                error_detail["violation"] = "missing_argument"
                hint = "Provide the required <file> or integer argument for the command."
            elif "file not known" in text:
                error_detail["violation"] = "unknown_file"
                hint = "Discover files via neighbors <known_file> first (start with the start module)."
            elif "mark requires prior inspect" in text:
                error_detail["violation"] = "mark_requires_inspect"
                hint = "First run inspect <file>, then mark <file>."
            elif "already marked" in text:
                error_detail["violation"] = "already_marked"
                hint = "Mark a different inspected file; use count to track what you have collected."
            elif "quota exceeded" in text:
                error_detail["violation"] = "quota_exceeded"
                hint = "You reached the mark quota; use neighbors/inspect wisely and submit when ready."
            elif "submit expects an integer" in text:
                error_detail["violation"] = "bad_submit"
                hint = "Provide an integer value, e.g., \\boxed{submit 7}."

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["outcome"] = "timeout"
            hint = "Plan earlier: discover neighbors, inspect selectively, track count, and submit before the turn limit."

        elif "failed!" in text:
            error_type = "WrongDecision"
            got = None
            expected = None
            m_expected = re.search(r'expected\s+(\d+)', obs, re.IGNORECASE)
            m_got = re.search(r'got\s+(\d+)', obs, re.IGNORECASE)
            if m_expected:
                expected = int(m_expected.group(1))
            if m_got:
                got = int(m_got.group(1))
            error_detail["expected"] = expected
            error_detail["got"] = got
            hint = "Explore from the start module: use neighbors to discover reachable files, inspect them, and mark selectively to track the sum before submitting."

        elif "success!" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "known_files": len(getattr(self, "known_files", [])),
                "inspected_files": len(getattr(self, "inspected_files", [])),
                "marked_files": len(getattr(self, "marked_files", [])),
                "mark_quota_remaining": max(0, getattr(self, "mark_limit", 0) - len(getattr(self, "marked_files", []))),
                "start_file": getattr(self, "start_file", None),
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
            "hint": f"Start by exploring neighbors of {self.start_file} using \\boxed{{neighbors {self.start_file}}}.",
            "turn": 0,
            "state": {
                "known_files": 1,
                "inspected_files": 0,
                "marked_files": 0,
                "mark_quota_remaining": self.mark_limit,
                "start_file": self.start_file,
            },
        }
        return obs, info