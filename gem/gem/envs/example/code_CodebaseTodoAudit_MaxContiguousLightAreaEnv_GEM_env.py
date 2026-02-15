from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class CodebaseTodoAuditEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 100,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 100

        # Evolvable parameters
        self.complexity_params = {
            'num_files': (3, 20),             # Number of files: more files increases search space
            'avg_lines': (20, 200),           # Average lines per file: longer files harder to scan
            'todo_perc': (15, 5),             # REVERSED: per-thousand probability (%) of inserting TODO; fewer TODOs → harder
            'distractor_perc': (0, 10),       # Distractor rate per-thousand: more false-lookalikes → harder exploration
            'preview_lines': (10, 3),         # REVERSED: fewer preview lines in open → harder partial view
        }
        # Parameter variance for randomization
        self.param_variance = {
            'num_files': 1,        # ±1 file variation
            'avg_lines': 15,       # ±15 lines variation (~10% at high end)
            'todo_perc': 2,        # ±2 per-thousand variation
            'distractor_perc': 2,  # ±2 per-thousand variation
            'preview_lines': 1,    # ±1 line variation
        }

        # Placeholder attributes (set in _apply_complexity_params)
        self.num_files: int = 0
        self.avg_lines: int = 0
        self.todo_perc: int = 0
        self.distractor_perc: int = 0
        self.preview_lines: int = 0

        # Other state
        self.turn_count: int = 0
        self.codebase: list = []
        self.marked_ids: set = set()
        self.total_todos: int = 0
        self.languages = ["python", "javascript", "c", "java", "go"]
        self.lang_ext = {
            "python": "py",
            "javascript": "js",
            "c": "c",
            "java": "java",
            "go": "go",
        }

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
            # Clamp to range, support reversed
            if min_val <= max_val:
                actual_value = max(min_val, min(max_val, actual_value))
            else:
                actual_value = max(max_val, min(min_val, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _generate_code_line(self, lang: str) -> str:
        # Simple synthetic line generation with tokens
        base_snippets = {
            "python": ["def foo():", "x = 1", "print(x)", "for i in range(10):", "return x", "class Bar:", "if x>0:"],
            "javascript": ["function foo(){", "let x = 1;", "console.log(x);", "for(let i=0;i<10;i++){", "return x;", "class Bar{}", "if(x>0){"],
            "c": ["int x=0;", "printf(\"%d\", x);", "for(int i=0;i<10;i++){", "return x;", "struct Bar{int y;};", "if(x>0){"],
            "java": ["class Foo{", "int x=1;", "System.out.println(x);", "for(int i=0;i<10;i++){", "return x;", "if(x>0){"],
            "go": ["func foo(){", "x := 1", "fmt.Println(x)", "for i:=0;i<10;i++{", "return x", "type Bar struct{}", "if x>0 {"],
        }
        line = random.choice(base_snippets.get(lang, ["code"]))
        # Insert distractors occasionally
        d_rate = self.distractor_perc / 1000.0
        if random.random() < d_rate:
            distractors = ["NOTODO", "todo", "ToDo", "TODOx", "TOD O", "/*TODOn*/"]
            line += "  " + random.choice(distractors)
        return line

    def _generate_todo_line(self, lang: str) -> str:
        comment_style = {
            "python": "# TODO: implement",
            "javascript": "// TODO: implement",
            "c": "// TODO: implement",
            "java": "// TODO: implement",
            "go": "// TODO: implement",
        }
        return comment_style.get(lang, "// TODO: implement")

    def _build_codebase(self):
        self.codebase = []
        self.total_todos = 0
        for idx in range(1, self.num_files + 1):
            lang = random.choice(self.languages)
            ext = self.lang_ext[lang]
            name = f"file_{idx}.{ext}"
            # Lines per file around avg_lines with ±30%
            min_lines = max(3, int(self.avg_lines * 0.7))
            max_lines = max(min_lines + 1, int(self.avg_lines * 1.3))
            n_lines = random.randint(min_lines, max_lines)
            lines = []
            todo_count = 0
            t_rate = self.todo_perc / 1000.0
            for _ in range(n_lines):
                if random.random() < t_rate:
                    line = self._generate_todo_line(lang)
                else:
                    line = self._generate_code_line(lang)
                lines.append(line)
            # Count exact TODO tokens (word-boundary)
            todo_count = sum(len(re.findall(r'\bTODO\b', ln)) for ln in lines)
            self.total_todos += todo_count
            self.codebase.append({
                "id": idx,
                "name": name,
                "lang": lang,
                "lines": lines,
                "todo_count": todo_count,
                "n_lines": n_lines,
            })

    def _get_instructions(self) -> str:
        example = self.sample_random_action()
        return (
            "You are auditing a codebase to determine the total number of TODO markers across all files.\n"
            "Actions (use one per turn inside \\boxed{...}):\n"
            "- list: show files and basic metadata\n"
            "- open <id>: preview the first few lines of a file\n"
            "- mark <id>: mark a file (for aggregate operations)\n"
            "- unmark <id>: remove a mark from a file\n"
            "- count <id>: return the exact TODO count for that file\n"
            "- sum_marked: sum TODOs over currently marked files\n"
            "- parity: reveal whether the global total TODO is even or odd\n"
            "- meta: show metadata about this environment\n"
            "- submit <N>: submit your final integer answer for total TODOs\n"
            "Invalid formats or unsupported actions incur penalties and may end the episode.\n"
            f"Example action: {example}\n"
        )

    def get_task_suffix(self) -> str:
        marked_sorted = sorted(list(self.marked_ids))
        return (
            f"State: files={self.num_files}, turns_used={self.turn_count}/{self.max_turns}, "
            f"marked={marked_sorted}. Enter one action using \\boxed{{...}}."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()
        self.turn_count = 0
        self.marked_ids = set()
        self._build_codebase()
        obs = self._get_instructions()
        info = {"suffix": self.get_task_suffix()}
        return obs, info

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if parsed is None:
            obs = f"Turn {self.turn_count}: invalid action format. Use \\boxed{{...}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        cmd = parsed.get("type")
        reward = 0.0
        obs = ""

        if cmd == "list":
            lines = []
            lines.append("Files:")
            for f in self.codebase:
                lines.append(f"- id={f['id']}, name={f['name']}, lang={f['lang']}, lines={f['n_lines']}")
            obs = "\n".join(lines)

        elif cmd == "open":
            fid = parsed.get("id")
            if not isinstance(fid, int) or fid < 1 or fid > self.num_files:
                obs = f"Protocol violation: file id {fid} out of range."
                reward = -0.2
            else:
                f = self.codebase[fid - 1]
                preview_n = min(self.preview_lines, f["n_lines"])
                preview = "\n".join([f"{i+1}: {f['lines'][i]}" for i in range(preview_n)])
                obs = (
                    f"Open {f['name']} ({f['lang']}, {f['n_lines']} lines). Preview first {preview_n} lines:\n{preview}"
                )

        elif cmd == "mark":
            fid = parsed.get("id")
            if not isinstance(fid, int) or fid < 1 or fid > self.num_files:
                obs = f"Protocol violation: file id {fid} out of range."
                reward = -0.2
            else:
                self.marked_ids.add(fid)
                obs = f"Marked file {fid}."

        elif cmd == "unmark":
            fid = parsed.get("id")
            if not isinstance(fid, int) or fid < 1 or fid > self.num_files:
                obs = f"Protocol violation: file id {fid} out of range."
                reward = -0.2
            else:
                if fid in self.marked_ids:
                    self.marked_ids.remove(fid)
                    obs = f"Unmarked file {fid}."
                else:
                    obs = f"Protocol violation: file {fid} not marked."
                    reward = -0.2

        elif cmd == "count":
            fid = parsed.get("id")
            if not isinstance(fid, int) or fid < 1 or fid > self.num_files:
                obs = f"Protocol violation: file id {fid} out of range."
                reward = -0.2
            else:
                f = self.codebase[fid - 1]
                cnt = f["todo_count"]
                obs = f"File {fid} has {cnt} TODO markers."

        elif cmd == "sum_marked":
            if len(self.marked_ids) == 0:
                obs = "Protocol violation: no files marked."
                reward = -0.2
            else:
                total_marked = sum(self.codebase[fid - 1]["todo_count"] for fid in self.marked_ids)
                obs = f"Sum over marked files {sorted(list(self.marked_ids))}: {total_marked}"

        elif cmd == "parity":
            parity_str = "even" if (self.total_todos % 2 == 0) else "odd"
            obs = f"Global total TODO parity is {parity_str}."

        elif cmd == "meta":
            obs = (
                f"Meta: num_files={self.num_files}, avg_lines≈{self.avg_lines}, preview_lines={self.preview_lines}, "
                f"todo_rate≈{self.todo_perc}/1000, distractor_rate≈{self.distractor_perc}/1000."
            )

        elif cmd == "submit":
            guess = parsed.get("value")
            if not isinstance(guess, int) or guess < 0:
                obs = f"Protocol violation: submit requires a non-negative integer."
                reward = -0.5
                terminated = True
            else:
                if guess == self.total_todos:
                    obs = f"Success! Correct total TODOs = {guess}."
                    reward = 1.0
                    terminated = True
                else:
                    obs = f"Failed! Incorrect total. Expected {self.total_todos}, got {guess}."
                    reward = -1.0
                    terminated = True

        else:
            obs = f"Unsupported action: {cmd}."
            reward = -1.0
            terminated = True

        if not terminated and self.turn_count >= self.max_turns:
            truncated = True
            terminated = True
            obs = f"Reached max turns ({self.max_turns})."

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Any]:
        if not action:
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        extracted = matches[-1].group(1).strip()
        tokens = extracted.split()
        if len(tokens) == 0:
            return None
        cmd = tokens[0].lower()

        if cmd == "list":
            return {"type": "list"}
        if cmd == "open" and len(tokens) == 2 and tokens[1].isdigit():
            return {"type": "open", "id": int(tokens[1])}
        if cmd == "mark" and len(tokens) == 2 and tokens[1].isdigit():
            return {"type": "mark", "id": int(tokens[1])}
        if cmd == "unmark" and len(tokens) == 2 and tokens[1].isdigit():
            return {"type": "unmark", "id": int(tokens[1])}
        if cmd == "count" and len(tokens) == 2 and tokens[1].isdigit():
            return {"type": "count", "id": int(tokens[1])}
        if cmd == "sum_marked":
            return {"type": "sum_marked"}
        if cmd == "parity":
            return {"type": "parity"}
        if cmd == "meta":
            return {"type": "meta"}
        if cmd == "submit" and len(tokens) == 2 and re.fullmatch(r'\d+', tokens[1]):
            return {"type": "submit", "value": int(tokens[1])}
        return {"type": cmd}  # unsupported or malformed

    def sample_random_action(self) -> str:
        if self.num_files > 0:
            fid = random.randint(1, self.num_files)
            choices = [
                f"list",
                f"open {fid}",
                f"mark {fid}",
                f"count {fid}",
                "sum_marked",
                "parity",
                "meta",
                f"submit {random.randint(0, max(0, self.total_todos + 3))}",
            ]
        else:
            choices = ["list", "meta"]
        return f"\\boxed{{{random.choice(choices)}}}"


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
            error_detail["issue"] = "missing_or_wrong_boxed_format"
            hint = "Wrap a single action in \\boxed{...}, e.g., \\boxed{list}."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "out of range" in text:
                error_detail["violation"] = "file_id_out_of_range"
                hint = "Use file ids between 1 and the number of files shown by \\boxed{list}."
            elif "not marked" in text:
                error_detail["violation"] = "unmark_non_marked"
                hint = "Mark a file first with \\boxed{mark <id>} before unmarking."
            elif "submit requires a non-negative integer" in text:
                error_detail["violation"] = "bad_submit_argument"
                hint = "Submit an integer like \\boxed{submit 12}."
            elif "no files marked" in text:
                error_detail["violation"] = "sum_marked_without_marks"
                hint = "Mark files using \\boxed{mark <id>} then use \\boxed{sum_marked}."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["action"] = "unknown_command"
            hint = "Use one of: list, open <id>, mark <id>, unmark <id>, count <id>, sum_marked, parity, meta, submit <N>."

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Plan fewer exploratory steps. Use \\boxed{count <id>} to get exact per-file counts quickly."

        elif "failed! incorrect total" in text:
            error_type = "WrongDecision"
            # Extract expected and got
            m_expected = re.search(r'expected (\d+)', obs, re.IGNORECASE)
            m_got = re.search(r'got (\d+)', obs, re.IGNORECASE)
            if m_expected:
                error_detail["expected"] = int(m_expected.group(1))
            if m_got:
                error_detail["got"] = int(m_got.group(1))
            hint = "Compute per-file counts using \\boxed{count <id>} and sum them before submitting."

        elif "success! correct total" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = "Good job."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "num_files": getattr(self, "num_files", None),
                "marked": sorted(list(getattr(self, "marked_ids", set()))),
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
            "hint": "Start with \\boxed{list} to see file ids, then use \\boxed{count <id>} to gather totals.",
            "turn": 0,
            "state": {
                "num_files": getattr(self, "num_files", None),
                "marked": [],
            },
        }
        return obs, info