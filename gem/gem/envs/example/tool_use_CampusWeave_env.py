from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


def _step_range_overlap(level: int) -> Tuple[int, int]:
    return 2 * level, 2 * level + 2  # level1:2-4 ... level10:20-22


class CampusWeaveEnv(Env):
    """
    Overlap版：教育场景（课程、学生、作业、考试），步数 2N~2N+2。
    """

    def __init__(self, complexity: int = 1, max_turns: Optional[int] = 200, **_):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.max_turns = max_turns if max_turns is not None else 200
        self.min_required_steps, self.max_required_steps = _step_range_overlap(self.complexity)
        self._init_database()
        self.reset()

    # ---------- 数据与工具 ----------
    def _init_database(self) -> None:
        self.tools: Dict[str, Dict[str, Any]] = {
            "open_table": {"parameters": ["name"], "returns": "Set current table", "usage": "tool:open_table(name=<table>)"},
            "filter_rows": {"parameters": ["column", "op", "value"], "returns": "Filter rows", "usage": "tool:filter_rows(column=<col>, op=eq|gt|lt, value=<val>)"},
            "join_table": {"parameters": ["name", "left_on", "right_on", "how"], "returns": "Join table", "usage": "tool:join_table(name=<table>, left_on=<col>, right_on=<col>, how=inner)"},
            "select_columns": {"parameters": ["names"], "returns": "Project columns", "usage": "tool:select_columns(names=<c1,c2,...>)"},
            "compute_count": {"parameters": ["column"], "returns": "Count rows/non-null", "usage": "tool:compute_count(column=<optional>)"},
            "compute_sum": {"parameters": ["column"], "returns": "Sum numeric column", "usage": "tool:compute_sum(column=<col>)"},
            "unique_count": {"parameters": ["column"], "returns": "Distinct count", "usage": "tool:unique_count(column=<col>)"},
            "snapshot": {"parameters": ["name"], "returns": "Save snapshot", "usage": "tool:snapshot(name=<snap>)"},
            "reset_state": {"parameters": [], "returns": "Clear state", "usage": "tool:reset_state()"},
            "show_preview": {"parameters": ["n"], "returns": "Preview rows", "usage": "tool:show_preview(n=<int>)"},
        }

        n_students = 30 + self.complexity * 10
        n_courses = 10 + self.complexity * 5
        n_assignments = 25 + self.complexity * 10
        n_exams = 20 + self.complexity * 8
        majors = ["cs", "math", "bio", "econ", "history"]
        terms = ["spring", "fall"]

        self.tables: Dict[str, List[Dict[str, Any]]] = {}
        self.tables["students"] = []
        for sid in range(1, n_students + 1):
            self.tables["students"].append(
                {"student_id": sid, "major": random.choice(majors), "gpa": round(random.uniform(2.0, 4.0), 2)}
            )

        self.tables["courses"] = []
        for cid in range(1, n_courses + 1):
            self.tables["courses"].append(
                {"course_id": cid, "dept": random.choice(majors), "credits": random.randint(2, 5), "term": random.choice(terms)}
            )

        self.tables["assignments"] = []
        for aid in range(1, n_assignments + 1):
            self.tables["assignments"].append(
                {"assign_id": aid, "course_id": random.randint(1, n_courses), "score": random.randint(50, 100)}
            )

        self.tables["exams"] = []
        for eid in range(1, n_exams + 1):
            self.tables["exams"].append(
                {"exam_id": eid, "course_id": random.randint(1, n_courses), "score": random.randint(40, 100)}
            )

        self.execution_state: Dict[str, Any] = {}
        self.current_table_name: Optional[str] = None
        self.current_table: Optional[List[Dict[str, Any]]] = None
        self.required_steps: int = self.min_required_steps
        self.task: Dict[str, Any] = {}
        self.turn_count: int = 0
        self.steps_taken: int = 0

    def _get_instructions(self) -> str:
        tools_str = "\n".join([f"- {n}: {m['usage']}" for n, m in self.tools.items()])
        return (
            "You analyze academic data (students, courses, assignments, exams) to compute a numeric metric.\n"
            "Use tools to open, join, filter, and aggregate tables.\n"
            "Actions must be in \\boxed{...}; use 'tool:' for tool calls, 'answer:' for final.\n"
            "Available tools:\n" + tools_str +
            "\nFinal submission: \\boxed{answer:<number>}.\n"
            "Meet the minimum tool-call count before answering."
        )

    def get_task_suffix(self) -> str:
        ct = self.current_table_name if self.current_table_name else "None"
        return (
            f"Task: {self._describe_task()} | Current table: {ct} | "
            f"Tool calls: {self.steps_taken}/{self.required_steps} | Turns: {self.turn_count}/{self.max_turns} | "
            "Use \\boxed{tool:...} or \\boxed{answer:...}."
        )

    def _describe_task(self) -> str:
        base = self.task.get("base_table", "?")
        metric = self.task.get("metric", "?")
        col = self.task.get("metric_column")
        clauses = []
        for op in self.task.get("ops", []):
            if op["op"] == "join":
                clauses.append(f"join {op['table']} on {op['left_on']}={op['right_on']}")
            elif op["op"] == "filter":
                clauses.append(f"filter {op['column']} {op['operator']} {op['value']}")
        metric_str = f"{metric}({col})" if col else metric
        return f"From {base}, apply: " + ("; ".join(clauses) if clauses else "no filters") + f"; then compute {metric_str}"

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self.turn_count = 0
        self.steps_taken = 0
        self.current_table_name = None
        self.current_table = None
        self.required_steps = random.randint(self.min_required_steps, self.max_required_steps)
        self.task = self._generate_task_requiring_n_steps(self.required_steps)
        self.task["solution"] = self._compute_ground_truth(self.task)
        obs = self._get_instructions()
        return obs, {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        if self.turn_count > self.max_turns:
            return "Timeout: maximum turns reached.", 0.0, True, True, {"suffix": self.get_task_suffix()}
        parsed = self._parse_action(action)
        if parsed is None:
            return "Invalid action format. Use \\boxed{tool:...} or \\boxed{answer:...}.", LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
        if parsed["type"] == "tool":
            name = parsed["name"]; args = parsed["args"]
            if name not in self.tools:
                return f"Unsupported tool: {name}.", -0.2, True, False, {"suffix": self.get_task_suffix()}
            try:
                result = self._execute_tool(name, args)
                self.steps_taken += 1
                return f"Tool {name} executed. Result: {result}", 0.0, False, False, {"suffix": self.get_task_suffix()}
            except ValueError as ve:
                return f"Protocol violation: {str(ve)}", 0.0, False, False, {"suffix": self.get_task_suffix()}
            except Exception as e:
                return f"Execution error: {str(e)}", -0.1, False, False, {"suffix": self.get_task_suffix()}
        else:
            val = parsed["value"]
            correct = self._compare_answer(val, self.task["solution"])
            if not correct:
                return f"Wrong answer. Your submitted answer: {val}.", 0.0, True, False, {"suffix": self.get_task_suffix()}
            if self.steps_taken < self.required_steps:
                return f"Protocol violation: insufficient tool usage ({self.steps_taken}/{self.required_steps}).", 0.0, True, False, {"suffix": self.get_task_suffix()}
            return f"Success: correct final answer {val}.", 1.0, True, False, {"suffix": self.get_task_suffix()}

    # ---------- Parsing ----------
    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        m = re.search(r"\\boxed\{(.+?)\}", action.strip(), flags=re.S)
        if not m:
            return None
        content = m.group(1).strip()
        if content.lower().startswith("tool:"):
            content = content[5:].strip()
        if content.lower().startswith("answer:"):
            val_str = content[7:].strip()
            try:
                if re.match(r"^-?\d+$", val_str):
                    return {"type": "answer", "value": int(val_str)}
                return {"type": "answer", "value": float(val_str)}
            except Exception:
                return None
        if "(" in content and content.endswith(")"):
            name = content.split("(", 1)[0].strip()
            args_str = content[len(name) + 1 : -1].strip()
            args = {}
            if args_str:
                parts = [p.strip() for p in self._split_args(args_str)]
                for p in parts:
                    if "=" not in p:
                        return None
                    k, v = p.split("=", 1)
                    args[k.strip()] = self._parse_value(v.strip())
            return {"type": "tool", "name": name, "args": args}
        if re.match(r"^-?\d+(\.\d+)?$", content):
            try:
                if "." in content:
                    return {"type": "answer", "value": float(content)}
                else:
                    return {"type": "answer", "value": int(content)}
            except Exception:
                return None
        return None

    def _split_args(self, s: str) -> List[str]:
        parts, buf = [], ""
        nest, in_quote = 0, False
        quote_char = ""
        for ch in s:
            if in_quote:
                buf += ch
                if ch == quote_char:
                    in_quote = False
                continue
            if ch in ("'", '"'):
                in_quote = True
                quote_char = ch
                buf += ch
                continue
            if ch == "(":
                nest += 1
            if ch == ")":
                nest = max(nest - 1, 0)
            if ch == "," and nest == 0:
                parts.append(buf.strip()); buf = ""; continue
            buf += ch
        if buf.strip():
            parts.append(buf.strip())
        return parts

    def _parse_value(self, v: str) -> Any:
        if v.lower() in ("true", "false"):
            return v.lower() == "true"
        if v.startswith("'") and v.endswith("'"):
            return v[1:-1]
        if v.startswith('"') and v.endswith('"'):
            return v[1:-1]
        if re.match(r"^-?\d+$", v):
            return int(v)
        if re.match(r"^-?\d+\.\d+$", v):
            return float(v)
        return v

    def sample_random_action(self) -> str:
        if self.current_table_name is None:
            return f"\\boxed{{tool:open_table(name='{random.choice(list(self.tables.keys()))}')}}"
        col = random.choice(list(self.current_table[0].keys())) if self.current_table else "student_id"
        return f"\\boxed{{tool:filter_rows(column='{col}', op=eq, value='{random.choice(['cs','spring','high'])}')}}"

    # ---------- 执行 ----------
    def _execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        if name == "open_table":
            tname = args.get("name")
            if tname not in self.tables:
                raise ValueError(f"table not found: {tname}")
            self.current_table_name = tname
            self.current_table = [dict(r) for r in self.tables[tname]]
            return f"Opened table '{tname}' with {len(self.current_table)} rows."
        if name == "reset_state":
            self.current_table_name = None
            self.current_table = None
            self.execution_state.clear()
            return "State reset. No active table."
        if name == "show_preview":
            n = int(args.get("n", 5))
            if self.current_table is None:
                raise ValueError("no active table")
            preview = self.current_table[: max(0, n)]
            return f"Preview {len(preview)} rows: {preview}"
        if self.current_table is None:
            raise ValueError("no active table")
        if name == "filter_rows":
            col, op, val = args.get("column"), args.get("op"), args.get("value")
            if col is None or op is None:
                raise ValueError("missing filter parameters")

            def keep(row):
                if col not in row:
                    return False
                rv = row[col]
                if op == "eq":
                    return str(rv) == str(val)
                try:
                    rvn = float(rv); vn = float(val)
                except Exception:
                    rvn = vn = None
                if op == "gt" and rvn is not None and vn is not None:
                    return rvn > vn
                if op == "lt" and rvn is not None and vn is not None:
                    return rvn < vn
                return False

            before = len(self.current_table)
            self.current_table = [r for r in self.current_table if keep(r)]
            after = len(self.current_table)
            return f"Filtered rows: {before} -> {after}."
        if name == "join_table":
            tname = args.get("name"); left_on = args.get("left_on"); right_on = args.get("right_on"); how = args.get("how", "inner")
            if tname not in self.tables:
                raise ValueError(f"table not found: {tname}")
            if left_on is None or right_on is None:
                raise ValueError("missing join keys")
            right_index = {}
            for r in self.tables[tname]:
                right_index.setdefault(r.get(right_on), []).append(r)
            joined = []
            for l in self.current_table:
                matches = right_index.get(l.get(left_on), [])
                if matches:
                    for r in matches:
                        merged = dict(l)
                        for k, v in r.items():
                            if k in merged:
                                merged[f"{tname}_{k}"] = v
                            else:
                                merged[k] = v
                        joined.append(merged)
                elif how == "left":
                    joined.append(dict(l))
            self.current_table = joined
            return f"Joined with '{tname}' using {left_on}={right_on}. Result rows: {len(self.current_table)}."
        if name == "select_columns":
            names = args.get("names")
            cols = [c.strip() for c in names.split(",")] if isinstance(names, str) else list(names or [])
            if not cols:
                raise ValueError("no columns specified")
            self.current_table = [{k: r.get(k) for k in cols} for r in self.current_table]
            return f"Selected columns: {','.join(cols)}."
        if name == "compute_count":
            column = args.get("column", None)
            if column is None:
                cnt = len(self.current_table); self.execution_state["last_metric"] = cnt
                return f"Counted {cnt} rows."
            cnt = sum(1 for r in self.current_table if r.get(column) is not None)
            self.execution_state["last_metric"] = cnt
            return f"Counted {cnt} non-null in '{column}'."
        if name == "compute_sum":
            column = args.get("column")
            if column is None:
                raise ValueError("sum requires 'column'")
            total = 0.0
            for r in self.current_table:
                v = r.get(column)
                try:
                    total += float(v)
                except Exception:
                    pass
            total = round(total, 6); self.execution_state["last_metric"] = total
            return f"Summed {column} = {total}."
        if name == "unique_count":
            column = args.get("column")
            if column is None:
                raise ValueError("unique_count requires 'column'")
            uniq = set()
            for r in self.current_table:
                uniq.add(r.get(column))
            cnt = len(uniq); self.execution_state["last_metric"] = cnt
            return f"Distinct values in {column}: {cnt}."
        if name == "snapshot":
            snap = args.get("name")
            if not snap:
                raise ValueError("snapshot requires 'name'")
            self.execution_state.setdefault("snapshots", {})[snap] = [dict(r) for r in self.current_table]
            return f"Snapshot '{snap}' saved with {len(self.current_table)} rows."
        raise ValueError(f"unknown tool '{name}'")

    def _apply_ops(self, base_rows: List[Dict[str, Any]], ops: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows = [dict(r) for r in base_rows]
        for op in ops:
            if op["op"] == "join":
                tname = op["table"]; left_on = op["left_on"]; right_on = op["right_on"]
                right_index = {}
                for r in self.tables[tname]:
                    right_index.setdefault(r.get(right_on), []).append(r)
                new_rows = []
                for l in rows:
                    matches = right_index.get(l.get(left_on), [])
                    for r in matches:
                        merged = dict(l)
                        for k, v in r.items():
                            if k in merged:
                                merged[f"{tname}_{k}"] = v
                            else:
                                merged[k] = v
                        new_rows.append(merged)
                rows = new_rows
            elif op["op"] == "filter":
                col = op["column"]; operator = op["operator"]; val = op["value"]
                def keep(r):
                    rv = r.get(col)
                    if operator == "eq":
                        return str(rv) == str(val)
                    try:
                        rvn = float(rv); vn = float(val)
                    except Exception:
                        return False
                    if operator == "gt":
                        return rvn > vn
                    if operator == "lt":
                        return rvn < vn
                    return False
                rows = [r for r in rows if keep(r)]
        return rows

    def _compute_ground_truth(self, task: Dict[str, Any]) -> float:
        rows = self._apply_ops(self.tables[task["base_table"]], task["ops"])
        metric = task["metric"]; col = task.get("metric_column")
        if metric == "count":
            return float(len(rows))
        if metric == "sum":
            total = 0.0
            for r in rows:
                v = r.get(col)
                try:
                    total += float(v)
                except Exception:
                    pass
            return round(total, 6)
        if metric == "unique":
            uniq = set()
            for r in rows:
                uniq.add(r.get(col))
            return float(len(uniq))
        return 0.0

    def _compare_answer(self, submitted: Any, solution: float) -> bool:
        try:
            sv = float(submitted)
            if abs(sv - round(solution)) < 1e-9 and abs(solution - round(solution)) < 1e-9:
                return True
            return abs(sv - solution) < 1e-6
        except Exception:
            return False

    # ---------- 任务生成 ----------
    def _generate_task_requiring_n_steps(self, steps: int) -> Dict[str, Any]:
        base_table = random.choice(["students", "courses", "assignments", "exams"])
        ops: List[Dict[str, Any]] = []
        candidate_filters: List[Dict[str, Any]] = []

        # 大幅增加候选操作，确保level 10（需要45-46个ops）能够完美执行
        if base_table == "students":
            # major filters (5 values)
            majors = ["cs", "math", "bio", "econ", "history"]
            candidate_filters.append({"op": "filter", "column": "major", "operator": "eq", "value": random.choice(majors)})
            candidate_filters.append({"op": "filter", "column": "major", "operator": "eq", "value": random.choice(majors)})
            candidate_filters.append({"op": "filter", "column": "major", "operator": "eq", "value": random.choice(majors)})
            # gpa filters (multiple thresholds)
            candidate_filters.append({"op": "filter", "column": "gpa", "operator": "gt", "value": round(random.uniform(2.5, 3.0), 1)})
            candidate_filters.append({"op": "filter", "column": "gpa", "operator": "gt", "value": round(random.uniform(3.0, 3.5), 1)})
            candidate_filters.append({"op": "filter", "column": "gpa", "operator": "gt", "value": round(random.uniform(3.5, 4.0), 1)})
            candidate_filters.append({"op": "filter", "column": "gpa", "operator": "lt", "value": round(random.uniform(3.0, 3.5), 1)})
            candidate_filters.append({"op": "filter", "column": "gpa", "operator": "lt", "value": round(random.uniform(2.5, 3.0), 1)})
            candidate_filters.append({"op": "filter", "column": "gpa", "operator": "eq", "value": round(random.uniform(2.0, 4.0), 1)})
            # student_id filters
            candidate_filters.append({"op": "filter", "column": "student_id", "operator": "gt", "value": random.randint(10, 30)})
            candidate_filters.append({"op": "filter", "column": "student_id", "operator": "lt", "value": random.randint(50, 100)})
            candidate_filters.append({"op": "filter", "column": "student_id", "operator": "eq", "value": random.randint(1, 100)})
            # Add more filters dynamically to reach ~36 filters
            for _ in range(24):
                col = random.choice(["major", "gpa", "student_id"])
                if col == "major":
                    candidate_filters.append({"op": "filter", "column": col, "operator": "eq", "value": random.choice(majors)})
                elif col == "gpa":
                    op = random.choice(["eq", "gt", "lt"])
                    candidate_filters.append({"op": "filter", "column": col, "operator": op, "value": round(random.uniform(2.0, 4.0), 1)})
                else:  # student_id
                    op = random.choice(["eq", "gt", "lt"])
                    candidate_filters.append({"op": "filter", "column": col, "operator": op, "value": random.randint(1, 150)})

        if base_table == "courses":
            # dept filters (5 values)
            depts = ["cs", "math", "bio", "econ", "history"]
            candidate_filters.append({"op": "filter", "column": "dept", "operator": "eq", "value": random.choice(depts)})
            candidate_filters.append({"op": "filter", "column": "dept", "operator": "eq", "value": random.choice(depts)})
            candidate_filters.append({"op": "filter", "column": "dept", "operator": "eq", "value": random.choice(depts)})
            # term filters (2 values)
            candidate_filters.append({"op": "filter", "column": "term", "operator": "eq", "value": random.choice(["spring", "fall"])})
            candidate_filters.append({"op": "filter", "column": "term", "operator": "eq", "value": random.choice(["spring", "fall"])})
            # credits filters (2-5)
            candidate_filters.append({"op": "filter", "column": "credits", "operator": "gt", "value": random.randint(2, 4)})
            candidate_filters.append({"op": "filter", "column": "credits", "operator": "lt", "value": random.randint(3, 5)})
            candidate_filters.append({"op": "filter", "column": "credits", "operator": "eq", "value": random.randint(2, 5)})
            candidate_filters.append({"op": "filter", "column": "credits", "operator": "eq", "value": random.randint(2, 5)})
            # course_id filters
            candidate_filters.append({"op": "filter", "column": "course_id", "operator": "gt", "value": random.randint(3, 15)})
            candidate_filters.append({"op": "filter", "column": "course_id", "operator": "lt", "value": random.randint(20, 50)})
            candidate_filters.append({"op": "filter", "column": "course_id", "operator": "eq", "value": random.randint(1, 60)})
            # Add more filters dynamically to reach ~36 filters
            for _ in range(24):
                col = random.choice(["dept", "term", "credits", "course_id"])
                if col == "dept":
                    candidate_filters.append({"op": "filter", "column": col, "operator": "eq", "value": random.choice(depts)})
                elif col == "term":
                    candidate_filters.append({"op": "filter", "column": col, "operator": "eq", "value": random.choice(["spring", "fall"])})
                elif col == "credits":
                    op = random.choice(["eq", "gt", "lt"])
                    candidate_filters.append({"op": "filter", "column": col, "operator": op, "value": random.randint(2, 5)})
                else:  # course_id
                    op = random.choice(["eq", "gt", "lt"])
                    candidate_filters.append({"op": "filter", "column": col, "operator": op, "value": random.randint(1, 100)})

        if base_table == "assignments":
            # score filters (50-100)
            candidate_filters.append({"op": "filter", "column": "score", "operator": "gt", "value": random.randint(60, 80)})
            candidate_filters.append({"op": "filter", "column": "score", "operator": "gt", "value": random.randint(80, 95)})
            candidate_filters.append({"op": "filter", "column": "score", "operator": "lt", "value": random.randint(70, 90)})
            candidate_filters.append({"op": "filter", "column": "score", "operator": "lt", "value": random.randint(60, 75)})
            candidate_filters.append({"op": "filter", "column": "score", "operator": "eq", "value": random.randint(50, 100)})
            # assign_id filters
            candidate_filters.append({"op": "filter", "column": "assign_id", "operator": "gt", "value": random.randint(10, 30)})
            candidate_filters.append({"op": "filter", "column": "assign_id", "operator": "lt", "value": random.randint(50, 100)})
            candidate_filters.append({"op": "filter", "column": "assign_id", "operator": "eq", "value": random.randint(1, 120)})
            # course_id filters
            candidate_filters.append({"op": "filter", "column": "course_id", "operator": "gt", "value": random.randint(3, 15)})
            candidate_filters.append({"op": "filter", "column": "course_id", "operator": "lt", "value": random.randint(20, 50)})
            candidate_filters.append({"op": "filter", "column": "course_id", "operator": "eq", "value": random.randint(1, 60)})
            # Add more filters dynamically to reach ~36 filters
            for _ in range(25):
                col = random.choice(["score", "assign_id", "course_id"])
                op = random.choice(["eq", "gt", "lt"])
                if col == "score":
                    candidate_filters.append({"op": "filter", "column": col, "operator": op, "value": random.randint(50, 100)})
                elif col == "assign_id":
                    candidate_filters.append({"op": "filter", "column": col, "operator": op, "value": random.randint(1, 150)})
                else:  # course_id
                    candidate_filters.append({"op": "filter", "column": col, "operator": op, "value": random.randint(1, 100)})

        if base_table == "exams":
            # score filters (40-100)
            candidate_filters.append({"op": "filter", "column": "score", "operator": "gt", "value": random.randint(55, 75)})
            candidate_filters.append({"op": "filter", "column": "score", "operator": "gt", "value": random.randint(75, 95)})
            candidate_filters.append({"op": "filter", "column": "score", "operator": "lt", "value": random.randint(70, 90)})
            candidate_filters.append({"op": "filter", "column": "score", "operator": "lt", "value": random.randint(50, 70)})
            candidate_filters.append({"op": "filter", "column": "score", "operator": "eq", "value": random.randint(40, 100)})
            # exam_id filters
            candidate_filters.append({"op": "filter", "column": "exam_id", "operator": "gt", "value": random.randint(5, 25)})
            candidate_filters.append({"op": "filter", "column": "exam_id", "operator": "lt", "value": random.randint(40, 80)})
            candidate_filters.append({"op": "filter", "column": "exam_id", "operator": "eq", "value": random.randint(1, 100)})
            # course_id filters
            candidate_filters.append({"op": "filter", "column": "course_id", "operator": "gt", "value": random.randint(3, 15)})
            candidate_filters.append({"op": "filter", "column": "course_id", "operator": "lt", "value": random.randint(20, 50)})
            candidate_filters.append({"op": "filter", "column": "course_id", "operator": "eq", "value": random.randint(1, 60)})
            # Add more filters dynamically to reach ~36 filters
            for _ in range(25):
                col = random.choice(["score", "exam_id", "course_id"])
                op = random.choice(["eq", "gt", "lt"])
                if col == "score":
                    candidate_filters.append({"op": "filter", "column": col, "operator": op, "value": random.randint(40, 100)})
                elif col == "exam_id":
                    candidate_filters.append({"op": "filter", "column": col, "operator": op, "value": random.randint(1, 120)})
                else:  # course_id
                    candidate_filters.append({"op": "filter", "column": col, "operator": op, "value": random.randint(1, 100)})

        joins: List[Dict[str, Any]] = []
        if base_table == "students":
            # Add joins from students to other tables (students doesn't have direct foreign keys in the original design)
            # Use major/gpa as join keys (semantic joins, might not be realistic but expands ops)
            joins.append({"op": "join", "table": "courses", "left_on": "major", "right_on": "dept"})
            joins.append({"op": "join", "table": "courses", "left_on": "student_id", "right_on": "course_id"})
            joins.append({"op": "join", "table": "assignments", "left_on": "student_id", "right_on": "assign_id"})
            joins.append({"op": "join", "table": "exams", "left_on": "student_id", "right_on": "exam_id"})
            joins.append({"op": "join", "table": "assignments", "left_on": "major", "right_on": "score"})
            joins.append({"op": "join", "table": "exams", "left_on": "major", "right_on": "score"})
            joins.append({"op": "join", "table": "courses", "left_on": "gpa", "right_on": "credits"})
            joins.append({"op": "join", "table": "assignments", "left_on": "gpa", "right_on": "course_id"})
            joins.append({"op": "join", "table": "exams", "left_on": "gpa", "right_on": "course_id"})
            joins.append({"op": "join", "table": "courses", "left_on": "student_id", "right_on": "credits"})
            joins.append({"op": "join", "table": "assignments", "left_on": "student_id", "right_on": "course_id"})
            joins.append({"op": "join", "table": "exams", "left_on": "student_id", "right_on": "course_id"})
        if base_table == "courses":
            joins.append({"op": "join", "table": "assignments", "left_on": "course_id", "right_on": "course_id"})
            joins.append({"op": "join", "table": "exams", "left_on": "course_id", "right_on": "course_id"})
            joins.append({"op": "join", "table": "students", "left_on": "dept", "right_on": "major"})
            joins.append({"op": "join", "table": "students", "left_on": "course_id", "right_on": "student_id"})
            joins.append({"op": "join", "table": "assignments", "left_on": "dept", "right_on": "score"})
            joins.append({"op": "join", "table": "exams", "left_on": "dept", "right_on": "score"})
            joins.append({"op": "join", "table": "students", "left_on": "credits", "right_on": "gpa"})
            joins.append({"op": "join", "table": "assignments", "left_on": "credits", "right_on": "assign_id"})
            joins.append({"op": "join", "table": "exams", "left_on": "credits", "right_on": "exam_id"})
            joins.append({"op": "join", "table": "students", "left_on": "course_id", "right_on": "student_id"})
            joins.append({"op": "join", "table": "assignments", "left_on": "term", "right_on": "score"})
            joins.append({"op": "join", "table": "exams", "left_on": "term", "right_on": "score"})
        if base_table == "assignments":
            joins.append({"op": "join", "table": "courses", "left_on": "course_id", "right_on": "course_id"})
            joins.append({"op": "join", "table": "exams", "left_on": "course_id", "right_on": "course_id"})
            joins.append({"op": "join", "table": "students", "left_on": "assign_id", "right_on": "student_id"})
            joins.append({"op": "join", "table": "students", "left_on": "course_id", "right_on": "student_id"})
            joins.append({"op": "join", "table": "courses", "left_on": "score", "right_on": "credits"})
            joins.append({"op": "join", "table": "exams", "left_on": "score", "right_on": "score"})
            joins.append({"op": "join", "table": "students", "left_on": "score", "right_on": "major"})
            joins.append({"op": "join", "table": "courses", "left_on": "assign_id", "right_on": "credits"})
            joins.append({"op": "join", "table": "exams", "left_on": "assign_id", "right_on": "exam_id"})
            joins.append({"op": "join", "table": "students", "left_on": "score", "right_on": "gpa"})
            joins.append({"op": "join", "table": "courses", "left_on": "score", "right_on": "dept"})
            joins.append({"op": "join", "table": "exams", "left_on": "course_id", "right_on": "exam_id"})
        if base_table == "exams":
            joins.append({"op": "join", "table": "courses", "left_on": "course_id", "right_on": "course_id"})
            joins.append({"op": "join", "table": "assignments", "left_on": "course_id", "right_on": "course_id"})
            joins.append({"op": "join", "table": "students", "left_on": "exam_id", "right_on": "student_id"})
            joins.append({"op": "join", "table": "students", "left_on": "course_id", "right_on": "student_id"})
            joins.append({"op": "join", "table": "courses", "left_on": "score", "right_on": "credits"})
            joins.append({"op": "join", "table": "assignments", "left_on": "score", "right_on": "score"})
            joins.append({"op": "join", "table": "students", "left_on": "score", "right_on": "major"})
            joins.append({"op": "join", "table": "courses", "left_on": "exam_id", "right_on": "credits"})
            joins.append({"op": "join", "table": "assignments", "left_on": "exam_id", "right_on": "assign_id"})
            joins.append({"op": "join", "table": "students", "left_on": "score", "right_on": "gpa"})
            joins.append({"op": "join", "table": "courses", "left_on": "score", "right_on": "dept"})
            joins.append({"op": "join", "table": "assignments", "left_on": "course_id", "right_on": "assign_id"})

        while len(ops) < max(1, steps - 2):
            choice = random.choice(["filter", "join"])
            if choice == "filter" and candidate_filters:
                ops.append(candidate_filters.pop(0))
            elif choice == "join" and joins:
                ops.append(joins.pop(0))
            elif candidate_filters:
                ops.append(candidate_filters.pop(0))
            elif joins:
                ops.append(joins.pop(0))
            else:
                break
        ops = ops[: max(0, steps - 2)]

        metric = random.choice(["count", "sum", "unique"])
        metric_col = None
        if metric == "sum":
            metric_col = random.choice(["gpa", "credits", "score"])
        if metric == "unique":
            metric_col = random.choice(["major", "dept", "term"])

        return {"base_table": base_table, "ops": ops, "metric": metric, "metric_column": metric_col}


class CampusWeaveEnvWithFeedback(CampusWeaveEnv):
    def __init__(self, complexity: int = 1, feedback_level: int = 2, **kwargs):
        super().__init__(complexity=complexity, **kwargs)
        self.feedback_level = feedback_level

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        if not terminated and not truncated and self.feedback_level > 0:
            info["suffix"] = self.get_task_suffix()
        if self.feedback_level > 1:
            info["diagnostic"] = {"required_steps": self.required_steps, "steps_taken": self.steps_taken}
        return obs, reward, terminated, truncated, info
