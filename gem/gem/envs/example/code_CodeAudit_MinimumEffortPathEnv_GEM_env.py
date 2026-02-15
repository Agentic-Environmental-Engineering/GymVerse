from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class CodeAuditEnv(Env):
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
            "code_length": (12, 40),        # Total lines of code: longer files increase search space and reasoning load → harder
            "num_symbols": (3, 12),         # Distinct variable names: more names increase tracking complexity → harder
            "nesting_depth": (0, 3),        # Max nesting constructs per function: deeper nesting complicates control-flow → harder
            "free_hint_budget": (2, 0),     # REVERSED: free hints allowed; fewer hints require more independent reasoning → harder
            "issue_density": (1, 4),        # Issues per function type included (thresholded); more types present → harder
        }

        # Variance settings
        self.param_variance = {
            "code_length": 3,        # ~±3 lines (8% of range)
            "num_symbols": 1,        # ±1 symbol
            "nesting_depth": 0,      # small range; keep fixed at level-determined
            "free_hint_budget": 0,   # small range; keep deterministic
            "issue_density": 1,      # ±1 issue type
        }

        # Placeholders
        self.code_length: int = 0
        self.num_symbols: int = 0
        self.nesting_depth: int = 0
        self.free_hint_budget: int = 0
        self.issue_density: int = 0

        # State
        self.turn_count: int = 0
        self.code_lines: Dict[int, str] = {}
        self.line_meta: Dict[int, Dict[str, Any]] = {}
        self.symbol_to_lines: Dict[str, list] = {}
        self.functions: Dict[int, Dict[str, Any]] = {}
        self.true_issue_count: int = 0
        self.marks: Dict[int, set] = {}
        self.worklist: list = []
        self.hints_used: int = 0

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for name, (min_val, max_val) in self.complexity_params.items():
            center = min_val + (max_val - min_val) * normalized
            actual = center
            variance = self.param_variance.get(name, 0)
            if self.enable_param_randomization and variance > 0:
                actual = center + random.uniform(-variance, variance)
            # Clamp respecting reversed params
            lo, hi = (min_val, max_val) if min_val <= max_val else (max_val, min_val)
            actual = max(lo, min(hi, actual))
            setattr(self, name, int(round(actual)))

    def _generate_code(self):
        self.code_lines = {}
        self.line_meta = {}
        self.symbol_to_lines = {}
        self.functions = {}
        self.true_issue_count = 0

        var_pool = [chr(ord("a") + i) for i in range(26)]
        symbols = var_pool[:max(3, self.num_symbols)]
        L = self.code_length
        F = max(1, min(5, self.nesting_depth + 2))  # 2..5 functions by nesting
        line_no = 1

        def add_line(text, func_id=None, used=None, assigned=None, is_return=False, has_magic=False):
            nonlocal line_no
            if line_no > L:
                return False
            self.code_lines[line_no] = text
            m = {
                "function_id": func_id,
                "used_symbols": set(used or []),
                "assigned_symbols": set(assigned or []),
                "is_return": bool(is_return),
                "has_magic": bool(has_magic),
                "is_unreachable": False,  # filled after flow analysis
            }
            self.line_meta[line_no] = m
            for s in m["used_symbols"] | m["assigned_symbols"]:
                self.symbol_to_lines.setdefault(s, []).append(line_no)
            line_no += 1
            return True

        # Plan: per function add a header and inject issues based on issue_density thresholding
        for k in range(1, F + 1):
            if line_no > L:
                break
            func_id = k
            self.functions[func_id] = {"start": line_no, "end": None}
            add_line(f"def func{func_id}():", func_id=func_id)
            # Basic assignment to anchor a safe symbol
            if not add_line(f"    {symbols[0]} = 0", func_id=func_id, assigned=[symbols[0]]):
                break

            # Add nesting scaffolding
            for d in range(self.nesting_depth):
                if not add_line(f"    if cond{d}:", func_id=func_id):
                    break
                if not add_line(f"        pass", func_id=func_id):
                    break

            # Thresholded issue inclusion (increasing difficulty)
            # 1: use-before-assign
            if self.issue_density >= 1 and line_no <= L:
                s = random.choice(symbols[1:]) if len(symbols) > 1 else symbols[0]
                add_line(f"    print({s})", func_id=func_id, used=[s])

            # 2: magic-number
            if self.issue_density >= 2 and line_no <= L:
                t = random.choice(symbols)
                val = random.choice([10, 42, 99])
                add_line(f"    {t} = {val}", func_id=func_id, assigned=[t], has_magic=True)

            # 3: unreachable (return then subsequent real line)
            did_return = False
            if self.issue_density >= 3 and line_no <= L:
                r = random.choice(symbols)
                if add_line(f"    return {r}", func_id=func_id, used=[r], is_return=True):
                    did_return = True
            if did_return and line_no <= L:
                u = random.choice(symbols)
                add_line(f"    {u} = 5  # post-return", func_id=func_id, assigned=[u])

            # 4: unused-variable
            if self.issue_density >= 4 and line_no <= L:
                w = random.choice(symbols)
                add_line(f"    {w} = 1  # may be unused", func_id=func_id, assigned=[w])

            # Filler to approach L
            while line_no <= L and (line_no - self.functions[func_id]["start"]) < max(5, L // F - 1):
                choice = random.choice(["pass", "use", "assign"])
                if choice == "pass":
                    if not add_line("    pass", func_id=func_id):
                        break
                elif choice == "use":
                    s = random.choice(symbols)
                    if not add_line(f"    use({s})", func_id=func_id, used=[s]):
                        break
                else:
                    s = random.choice(symbols)
                    if not add_line(f"    {s} = 0", func_id=func_id, assigned=[s]):
                        break

            self.functions[func_id]["end"] = line_no - 1

        # If still short, pad with top-level comments/pass
        while line_no <= L:
            add_line("# padding")
            add_line("pass")

        # Flow and issue analysis per function
        for fid, fr in self.functions.items():
            start, end = fr["start"], fr["end"] if fr["end"] is not None else start
            assigned_so_far = set()
            used_total = set()
            assigned_total = set()
            seen_return = False
            for ln in range(start, end + 1):
                meta = self.line_meta.get(ln, {})
                if not meta:
                    continue
                if meta.get("is_return", False):
                    seen_return = True
                # use-before-assign
                for s in meta["used_symbols"]:
                    used_total.add(s)
                    if s not in assigned_so_far:
                        self.true_issue_count += 1
                # track assignments
                for a in meta["assigned_symbols"]:
                    assigned_so_far.add(a)
                    assigned_total.add(a)
                # magic-number
                if meta.get("has_magic", False):
                    self.true_issue_count += 1
                # unreachable
                if seen_return and not meta.get("is_return", False):
                    # mark unreachable if the line is within function and not a no-op comment
                    text = self.code_lines.get(ln, "")
                    if text.strip() not in ("pass", "") and not text.strip().startswith("def"):
                        meta["is_unreachable"] = True
                        self.true_issue_count += 1
            # unused-variable
            for a in assigned_total:
                if a not in used_total:
                    self.true_issue_count += 1

        # Ensure solvability: at least 1 issue
        if self.true_issue_count < 1:
            # Inject a guaranteed magic-number at last function body line
            last_fid = max(self.functions.keys())
            ln = self.functions[last_fid]["end"]
            t = symbols[0]
            self.code_lines[ln] = f"    {t} = 99"
            self.line_meta[ln]["assigned_symbols"].add(t)
            self.line_meta[ln]["has_magic"] = True
            self.true_issue_count += 1

    def _get_instructions(self) -> str:
        return (
            "You are auditing a hidden code snippet.\n"
            "Goal: determine the total number of issues in the code.\n"
            "Issue types considered:\n"
            "- use-before-assign (using a variable before any assignment in the same function)\n"
            "- unreachable (statements after a return in the same function)\n"
            "- unused-variable (assigned in a function but never used there)\n"
            "- magic-number (literal value >=10 assigned)\n"
            "Actions:\n"
            "- QUERY LINE i\n"
            "- QUERY SYMBOL name\n"
            "- LIST NEIGHBORS i\n"
            "- MARK i type\n"
            "- PUSH i\n"
            "- POP\n"
            "- STATUS\n"
            "- HINT\n"
            "- ANSWER n\n"
            "Use \\boxed{...} to submit actions. Example: \\boxed{QUERY LINE 3}\n"
        )

    def get_task_suffix(self) -> str:
        remaining_hints = max(0, self.free_hint_budget - self.hints_used)
        return (
            f"Turns: {self.turn_count}/{self.max_turns}\n"
            f"Marks: {sum(len(v) for v in self.marks.values())} entries across {len(self.marks)} lines\n"
            f"Worklist size: {len(self.worklist)}\n"
            f"Hints remaining: {remaining_hints}\n"
            "Enter an action with \\boxed{...} (e.g., \\boxed{QUERY LINE 1})."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()

        self.turn_count = 0
        self.marks = {}
        self.worklist = []
        self.hints_used = 0

        self._generate_code()

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)
        if not parsed:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}"
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        act_type = parsed["type"]
        terminated = False
        truncated = False
        reward = 0.0
        obs = ""

        if act_type == "QUERY_LINE":
            idx = parsed["index"]
            if idx not in self.code_lines:
                obs = f"Failed! Invalid line index {idx}."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            text = self.code_lines[idx]
            meta = self.line_meta[idx]
            obs = (
                f"Line {idx}: {text}\n"
                f"- function_id: {meta.get('function_id')}\n"
                f"- used_symbols: {sorted(list(meta.get('used_symbols', [])))}\n"
                f"- assigned_symbols: {sorted(list(meta.get('assigned_symbols', [])))}\n"
            )

        elif act_type == "QUERY_SYMBOL":
            name = parsed["name"]
            lines = self.symbol_to_lines.get(name, [])
            obs = f"Symbol '{name}' occurs on lines: {sorted(lines)}"

        elif act_type == "LIST_NEIGHBORS":
            idx = parsed["index"]
            if idx not in self.code_lines:
                obs = f"Failed! Invalid line index {idx}."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            meta = self.line_meta[idx]
            fid = meta.get("function_id")
            if not fid or fid not in self.functions:
                next_line = idx + 1 if (idx + 1) in self.code_lines else None
                obs = f"Neighbors of line {idx}: next={next_line}"
            else:
                start = self.functions[fid]["start"]
                end = self.functions[fid]["end"]
                if meta.get("is_return", False):
                    obs = f"Neighbors of line {idx}: return encountered; subsequent lines may be unreachable until function end {end}"
                else:
                    next_in_func = idx + 1 if idx + 1 <= end else None
                    obs = f"Neighbors of line {idx} within function {fid}: next={next_in_func}, function_end={end}"

        elif act_type == "MARK":
            idx = parsed["index"]
            cat = parsed["category"]
            valid_cats = {"use-before-assign", "unreachable", "unused-variable", "magic-number"}
            if idx not in self.code_lines or cat not in valid_cats:
                obs = "Failed! Unsupported action or invalid parameters."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            self.marks.setdefault(idx, set()).add(cat)
            obs = f"Marked line {idx} as {cat}. Total marks on this line: {len(self.marks[idx])}"

        elif act_type == "PUSH":
            idx = parsed["index"]
            if idx not in self.code_lines:
                obs = f"Failed! Invalid line index {idx}."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            self.worklist.append(idx)
            obs = f"Pushed line {idx}. Worklist size: {len(self.worklist)}"

        elif act_type == "POP":
            if self.worklist:
                idx = self.worklist.pop()
                obs = f"Popped line {idx}. Worklist size: {len(self.worklist)}"
            else:
                obs = "Worklist empty. Nothing to pop."

        elif act_type == "STATUS":
            remaining_hints = max(0, self.free_hint_budget - self.hints_used)
            obs = (
                f"Status:\n"
                f"- marks_total: {sum(len(v) for v in self.marks.values())}\n"
                f"- marked_lines: {sorted(list(self.marks.keys()))}\n"
                f"- worklist: {self.worklist}\n"
                f"- hints_remaining: {remaining_hints}\n"
                f"- code_length: {self.code_length}\n"
            )

        elif act_type == "HINT":
            remaining_hints = max(0, self.free_hint_budget - self.hints_used)
            if remaining_hints <= 0:
                obs = "No hints remaining. Use queries and reasoning to determine the count."
            else:
                self.hints_used += 1
                present = set()
                for ln, meta in self.line_meta.items():
                    fid = meta.get("function_id")
                    if not fid:
                        continue
                    if meta.get("has_magic", False):
                        present.add("magic-number")
                # Assess presence by re-running minimal checks (without revealing count)
                # Use-before-assign: if any use before assignment occurred during analysis
                # Unreachable: if any line marked unreachable
                # Unused-variable: infer likely if assignments exist without use in function
                uba_present = False
                unreach_present = False
                unused_present = False
                # Scan functions
                for fid, fr in self.functions.items():
                    start, end = fr["start"], fr["end"]
                    assigned_so_far = set()
                    used_total = set()
                    seen_return = False
                    for ln in range(start, end + 1):
                        meta = self.line_meta.get(ln, {})
                        if meta.get("is_return", False):
                            seen_return = True
                        for s in meta.get("used_symbols", []):
                            used_total.add(s)
                            if s not in assigned_so_far:
                                uba_present = True
                        for a in meta.get("assigned_symbols", []):
                            assigned_so_far.add(a)
                        if meta.get("is_unreachable", False) or (seen_return and not meta.get("is_return", False)):
                            unreach_present = True
                    for a in assigned_so_far:
                        if a not in used_total:
                            unused_present = True
                if uba_present:
                    present.add("use-before-assign")
                if unreach_present:
                    present.add("unreachable")
                if unused_present:
                    present.add("unused-variable")
                obs = f"Hint: issue categories present include {sorted(list(present))}."

        elif act_type == "ANSWER":
            n = parsed["value"]
            if n == self.true_issue_count:
                obs = f"Success! Correct issue count {n}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Failed! Wrong answer {n}. True count was {self.true_issue_count}."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = "Failed! Unsupported action."
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

        # End of step

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r"\\boxed\{(.+?)\}", re.IGNORECASE | re.DOTALL)
        m = pattern.findall(action)
        if not m:
            return None
        content = m[-1].strip()
        # Try patterns
        # QUERY LINE i
        ql = re.match(r"^\s*QUERY\s+LINE\s+(\d+)\s*$", content, re.IGNORECASE)
        if ql:
            return {"type": "QUERY_LINE", "index": int(ql.group(1))}
        # QUERY SYMBOL name
        qs = re.match(r"^\s*QUERY\s+SYMBOL\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", content, re.IGNORECASE)
        if qs:
            return {"type": "QUERY_SYMBOL", "name": qs.group(1)}
        # LIST NEIGHBORS i
        ln = re.match(r"^\s*LIST\s+NEIGHBORS\s+(\d+)\s*$", content, re.IGNORECASE)
        if ln:
            return {"type": "LIST_NEIGHBORS", "index": int(ln.group(1))}
        # MARK i type
        mk = re.match(r"^\s*MARK\s+(\d+)\s+([a-z\-]+)\s*$", content, re.IGNORECASE)
        if mk:
            return {"type": "MARK", "index": int(mk.group(1)), "category": mk.group(2).lower()}
        # PUSH i
        pu = re.match(r"^\s*PUSH\s+(\d+)\s*$", content, re.IGNORECASE)
        if pu:
            return {"type": "PUSH", "index": int(pu.group(1))}
        # POP
        po = re.match(r"^\s*POP\s*$", content, re.IGNORECASE)
        if po:
            return {"type": "POP"}
        # STATUS
        st = re.match(r"^\s*STATUS\s*$", content, re.IGNORECASE)
        if st:
            return {"type": "STATUS"}
        # HINT
        hi = re.match(r"^\s*HINT\s*$", content, re.IGNORECASE)
        if hi:
            return {"type": "HINT"}
        # ANSWER n
        an = re.match(r"^\s*ANSWER\s+(\d+)\s*$", content, re.IGNORECASE)
        if an:
            return {"type": "ANSWER", "value": int(an.group(1))}
        return {"type": "UNKNOWN", "raw": content}

    def sample_random_action(self) -> str:
        choices = []
        if self.code_lines:
            idx = random.choice(list(self.code_lines.keys()))
            choices.extend([
                f"\\boxed{{QUERY LINE {idx}}}",
                f"\\boxed{{PUSH {idx}}}",
                f"\\boxed{{LIST NEIGHBORS {idx}}}",
                f"\\boxed{{MARK {idx} unreachable}}",
            ])
        sym = None
        if self.symbol_to_lines:
            sym = random.choice(list(self.symbol_to_lines.keys()))
            choices.append(f"\\boxed{{QUERY SYMBOL {sym}}}")
        choices.extend([f"\\boxed{{STATUS}}", f"\\boxed{{HINT}}", f"\\boxed{{POP}}", f"\\boxed{{ANSWER {max(1, self.true_issue_count - 1)}}}"])
        return random.choice(choices)


class CodeAuditEnvWithFeedback(CodeAuditEnv):
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
                hint = "Wrap your command in \\boxed{...}, e.g., \\boxed{QUERY LINE 1}."
            elif text.startswith("failed! unsupported action"):
                error_type = "UnsupportedAction"
                error_detail["action"] = "unknown_or_invalid"
                hint = "Use one of: QUERY LINE i, QUERY SYMBOL name, LIST NEIGHBORS i, MARK i type, PUSH i, POP, STATUS, HINT, ANSWER n."
            elif text.startswith("failed! invalid line index"):
                error_type = "UnsupportedAction"
                error_detail["action"] = "index_out_of_range"
                hint = "Check code_length in STATUS and query a valid line number."
            elif text.startswith("no hints remaining"):
                error_type = "ProtocolViolation"
                error_detail["violation"] = "hint_budget_exhausted"
                hint = "Use queries (QUERY LINE, QUERY SYMBOL) and LIST NEIGHBORS to gather evidence."
            elif text.startswith("failed! wrong answer"):
                error_type = "WrongDecision"
                # Try to extract numbers
                m = re.search(r"wrong answer (\d+). true count was (\d+)", text)
                if m:
                    error_detail["got"] = int(m.group(1))
                    error_detail["expected"] = int(m.group(2))
                hint = "Cross-check unreachable statements after returns and variables used before assignment; then recompute."
            elif "reached max turns" in text:
                error_type = "Timeout"
                error_detail["limit"] = self.max_turns
                hint = "Prioritize informative queries and avoid redundant actions."
            elif "success" in text:
                error_type = "OK"
                error_detail["outcome"] = "success"
                hint = None
            else:
                error_type = "OK"
                error_detail["outcome"] = "step"

            diagnostic = {"error_type": error_type}
            if self.feedback_level >= 1:
                diagnostic["error_detail"] = error_detail
                diagnostic["turn"] = getattr(self, "turn_count", None)
                diagnostic["state"] = {
                    "marks_total": sum(len(v) for v in self.marks.values()),
                    "worklist_size": len(self.worklist),
                    "hints_remaining": max(0, self.free_hint_budget - self.hints_used),
                    "code_length": self.code_length,
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
                "hint": "Start by querying a header and a few lines: \\boxed{QUERY LINE 1}, then use \\boxed{STATUS}.",
                "turn": 0,
                "state": {
                    "marks_total": 0,
                    "worklist_size": 0,
                    "hints_remaining": max(0, self.free_hint_budget - self.hints_used),
                    "code_length": self.code_length,
                },
            }
            return obs, info