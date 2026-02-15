from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


def _step_range_overlap(level: int) -> Tuple[int, int]:
    return 2 * level, 2 * level + 2  # level1:2-4 ... level10:20-22


class ChoreCanvasEnv(Env):
    """
    Overlap版：居家/生活事务调度与汇总场景，步数区间 2N~2N+2。
    """

    def __init__(self, complexity: int = 1, max_turns: Optional[int] = 220, **_):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.max_turns = max_turns if max_turns is not None else 220
        self.min_required_steps, self.max_required_steps = _step_range_overlap(self.complexity)
        self._init_database()
        self.reset()

    def evolve(self, step_success_rate: float, **kwargs) -> int:
        """Override evolve to properly update step ranges when complexity changes."""
        old_complexity = self.complexity
        new_complexity = super().evolve(step_success_rate, **kwargs)

        # If complexity changed, update step ranges and database
        if new_complexity != old_complexity:
            self.min_required_steps, self.max_required_steps = _step_range_overlap(new_complexity)
            self._init_database()

        return new_complexity

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

        rooms = ["kitchen", "bathroom", "living", "garage", "yard"]
        chores = ["clean", "repair", "organize", "shop", "cook"]
        stores = ["market", "hardware", "pharmacy", "bakery", "online"]
        utilities = ["electric", "water", "gas", "internet"]
        family = ["alice", "bob", "carol", "dave"]

        n_tasks = 24 + self.complexity * 10
        n_groceries = 20 + self.complexity * 8
        n_bills = 18 + self.complexity * 6
        n_appts = 16 + self.complexity * 5

        self.tables: Dict[str, List[Dict[str, Any]]] = {}
        self.tables["tasks"] = []
        for tid in range(1, n_tasks + 1):
            self.tables["tasks"].append(
                {
                    "task_id": tid,
                    "room": random.choice(rooms),
                    "type": random.choice(chores),
                    "effort": random.randint(1, 8),
                    "assignee": random.choice(family),
                }
            )

        self.tables["groceries"] = []
        for gid in range(1, n_groceries + 1):
            self.tables["groceries"].append(
                {
                    "item_id": gid,
                    "store": random.choice(stores),
                    "category": random.choice(["produce", "cleaning", "pet", "snack"]),
                    "cost": round(random.uniform(2, 80), 2),
                    "priority": random.choice(["low", "med", "high"]),
                }
            )

        self.tables["bills"] = []
        for bid in range(1, n_bills + 1):
            self.tables["bills"].append(
                {
                    "bill_id": bid,
                    "utility": random.choice(utilities),
                    "amount": round(random.uniform(20, 250), 2),
                    "status": random.choice(["due", "paid", "overdue"]),
                }
            )

        self.tables["appointments"] = []
        for aid in range(1, n_appts + 1):
            self.tables["appointments"].append(
                {
                    "appt_id": aid,
                    "topic": random.choice(["doctor", "school", "repair", "delivery"]),
                    "day": random.choice(["mon", "tue", "wed", "thu", "fri"]),
                    "slot": random.choice(["am", "pm"]),
                    "attendee": random.choice(family),
                }
            )

        self.execution_state: Dict[str, Any] = {}
        self.current_table_name: Optional[str] = None
        self.current_table: Optional[List[Dict[str, Any]]] = None
        self.required_steps: int = self.min_required_steps
        self.task: Dict[str, Any] = {}
        self.turn_count: int = 0
        self.steps_taken: int = 0

    # ---------- 指令 ----------
    def _get_instructions(self) -> str:
        tools_str = "\n".join([f"- {n}: {m['usage']}" for n, m in self.tools.items()])
        return (
            "You coordinate household chores, shopping, bills, and appointments to compute a metric.\n"
            "Use tools to open, join, filter, and aggregate; actions must be in \\boxed{...} with 'tool:' or 'answer:'.\n"
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

    # ---------- 环节 ----------
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
        nest = 0
        in_quote = False
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
        col = random.choice(list(self.current_table[0].keys())) if self.current_table else "task_id"
        return f"\\boxed{{tool:filter_rows(column='{col}', op=eq, value='{random.choice(['kitchen','paid','high'])}')}}"

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

    # ---------- 真值 ----------
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
        base_table = random.choice(["tasks", "groceries", "bills", "appointments"])
        ops: List[Dict[str, Any]] = []
        candidate_filters = []

        # 大幅增加候选操作，确保level 10（需要45-46个ops）能够完美执行
        if base_table == "tasks":
            # room filters (5 options)
            candidate_filters.append({"op": "filter", "column": "room", "operator": "eq", "value": random.choice(["kitchen", "bathroom", "living", "garage", "yard"])})
            candidate_filters.append({"op": "filter", "column": "room", "operator": "eq", "value": random.choice(["kitchen", "bathroom", "living", "garage", "yard"])})
            # type filters (5 options)
            candidate_filters.append({"op": "filter", "column": "type", "operator": "eq", "value": random.choice(["clean", "repair", "organize", "shop", "cook"])})
            candidate_filters.append({"op": "filter", "column": "type", "operator": "eq", "value": random.choice(["clean", "repair", "organize", "shop", "cook"])})
            # assignee filters (4 options)
            candidate_filters.append({"op": "filter", "column": "assignee", "operator": "eq", "value": random.choice(["alice", "bob", "carol", "dave"])})
            candidate_filters.append({"op": "filter", "column": "assignee", "operator": "eq", "value": random.choice(["alice", "bob", "carol", "dave"])})
            # effort filters (multiple thresholds)
            candidate_filters.append({"op": "filter", "column": "effort", "operator": "gt", "value": random.randint(1, 3)})
            candidate_filters.append({"op": "filter", "column": "effort", "operator": "gt", "value": random.randint(4, 6)})
            candidate_filters.append({"op": "filter", "column": "effort", "operator": "lt", "value": random.randint(5, 7)})
            candidate_filters.append({"op": "filter", "column": "effort", "operator": "lt", "value": random.randint(2, 4)})
            # task_id filters
            candidate_filters.append({"op": "filter", "column": "task_id", "operator": "gt", "value": random.randint(10, 30)})
            candidate_filters.append({"op": "filter", "column": "task_id", "operator": "lt", "value": random.randint(50, 80)})
            # Additional filters for higher complexity (add many more to reach 48 ops)
            candidate_filters.append({"op": "filter", "column": "room", "operator": "eq", "value": random.choice(["kitchen", "bathroom", "living"])})
            candidate_filters.append({"op": "filter", "column": "type", "operator": "eq", "value": random.choice(["repair", "organize"])})
            candidate_filters.append({"op": "filter", "column": "assignee", "operator": "eq", "value": random.choice(["alice", "carol"])})
            candidate_filters.append({"op": "filter", "column": "effort", "operator": "eq", "value": random.randint(3, 6)})
            candidate_filters.append({"op": "filter", "column": "task_id", "operator": "eq", "value": random.randint(20, 60)})
            # Add more filters to reach ~36 filters total (with 12 joins = 48 ops)
            for _ in range(19):
                col = random.choice(["room", "type", "assignee", "effort", "task_id"])
                if col in ["room", "type", "assignee"]:
                    val = random.choice(["kitchen", "bathroom", "living", "garage", "yard", "clean", "repair", "organize", "shop", "cook", "alice", "bob", "carol", "dave"])
                    candidate_filters.append({"op": "filter", "column": col, "operator": "eq", "value": val})
                elif col == "effort":
                    op = random.choice(["eq", "gt", "lt"])
                    candidate_filters.append({"op": "filter", "column": col, "operator": op, "value": random.randint(1, 8)})
                else:  # task_id
                    op = random.choice(["eq", "gt", "lt"])
                    candidate_filters.append({"op": "filter", "column": col, "operator": op, "value": random.randint(1, 100)})

        if base_table == "groceries":
            # store filters (5 options)
            candidate_filters.append({"op": "filter", "column": "store", "operator": "eq", "value": random.choice(["market", "hardware", "pharmacy", "bakery", "online"])})
            candidate_filters.append({"op": "filter", "column": "store", "operator": "eq", "value": random.choice(["market", "hardware", "pharmacy", "bakery", "online"])})
            # category filters (4 options)
            candidate_filters.append({"op": "filter", "column": "category", "operator": "eq", "value": random.choice(["produce", "cleaning", "pet", "snack"])})
            candidate_filters.append({"op": "filter", "column": "category", "operator": "eq", "value": random.choice(["produce", "cleaning", "pet", "snack"])})
            # priority filters (3 options)
            candidate_filters.append({"op": "filter", "column": "priority", "operator": "eq", "value": random.choice(["low", "med", "high"])})
            candidate_filters.append({"op": "filter", "column": "priority", "operator": "eq", "value": random.choice(["low", "med", "high"])})
            # cost filters (multiple thresholds)
            candidate_filters.append({"op": "filter", "column": "cost", "operator": "lt", "value": random.randint(15, 35)})
            candidate_filters.append({"op": "filter", "column": "cost", "operator": "lt", "value": random.randint(40, 60)})
            candidate_filters.append({"op": "filter", "column": "cost", "operator": "gt", "value": random.randint(10, 30)})
            candidate_filters.append({"op": "filter", "column": "cost", "operator": "gt", "value": random.randint(35, 55)})
            # item_id filters
            candidate_filters.append({"op": "filter", "column": "item_id", "operator": "gt", "value": random.randint(5, 20)})
            candidate_filters.append({"op": "filter", "column": "item_id", "operator": "lt", "value": random.randint(40, 70)})
            # Additional filters for higher complexity
            candidate_filters.append({"op": "filter", "column": "store", "operator": "eq", "value": random.choice(["hardware", "pharmacy"])})
            candidate_filters.append({"op": "filter", "column": "category", "operator": "eq", "value": random.choice(["cleaning", "pet"])})
            candidate_filters.append({"op": "filter", "column": "priority", "operator": "eq", "value": "med"})
            candidate_filters.append({"op": "filter", "column": "cost", "operator": "eq", "value": random.randint(20, 50)})
            candidate_filters.append({"op": "filter", "column": "item_id", "operator": "eq", "value": random.randint(10, 50)})
            # Add more filters to reach ~36 filters total
            for _ in range(19):
                col = random.choice(["store", "category", "priority", "cost", "item_id"])
                if col in ["store", "category", "priority"]:
                    val = random.choice(["market", "hardware", "pharmacy", "bakery", "online", "produce", "cleaning", "pet", "snack", "low", "med", "high"])
                    candidate_filters.append({"op": "filter", "column": col, "operator": "eq", "value": val})
                elif col == "cost":
                    op = random.choice(["eq", "gt", "lt"])
                    candidate_filters.append({"op": "filter", "column": col, "operator": op, "value": random.randint(2, 80)})
                else:  # item_id
                    op = random.choice(["eq", "gt", "lt"])
                    candidate_filters.append({"op": "filter", "column": col, "operator": op, "value": random.randint(1, 100)})

        if base_table == "bills":
            # utility filters (4 options)
            candidate_filters.append({"op": "filter", "column": "utility", "operator": "eq", "value": random.choice(["electric", "water", "gas", "internet"])})
            candidate_filters.append({"op": "filter", "column": "utility", "operator": "eq", "value": random.choice(["electric", "water", "gas", "internet"])})
            # status filters (3 options)
            candidate_filters.append({"op": "filter", "column": "status", "operator": "eq", "value": random.choice(["due", "paid", "overdue"])})
            candidate_filters.append({"op": "filter", "column": "status", "operator": "eq", "value": random.choice(["due", "paid", "overdue"])})
            # amount filters (multiple thresholds)
            candidate_filters.append({"op": "filter", "column": "amount", "operator": "gt", "value": random.randint(30, 70)})
            candidate_filters.append({"op": "filter", "column": "amount", "operator": "gt", "value": random.randint(80, 150)})
            candidate_filters.append({"op": "filter", "column": "amount", "operator": "lt", "value": random.randint(100, 180)})
            candidate_filters.append({"op": "filter", "column": "amount", "operator": "lt", "value": random.randint(50, 90)})
            # bill_id filters
            candidate_filters.append({"op": "filter", "column": "bill_id", "operator": "gt", "value": random.randint(5, 20)})
            candidate_filters.append({"op": "filter", "column": "bill_id", "operator": "lt", "value": random.randint(30, 60)})
            # Additional filters for higher complexity
            candidate_filters.append({"op": "filter", "column": "utility", "operator": "eq", "value": random.choice(["water", "gas"])})
            candidate_filters.append({"op": "filter", "column": "status", "operator": "eq", "value": "paid"})
            candidate_filters.append({"op": "filter", "column": "amount", "operator": "eq", "value": random.randint(60, 120)})
            candidate_filters.append({"op": "filter", "column": "bill_id", "operator": "eq", "value": random.randint(10, 40)})
            # Add more filters to reach ~36 filters total
            for _ in range(22):
                col = random.choice(["utility", "status", "amount", "bill_id"])
                if col in ["utility", "status"]:
                    val = random.choice(["electric", "water", "gas", "internet", "due", "paid", "overdue"])
                    candidate_filters.append({"op": "filter", "column": col, "operator": "eq", "value": val})
                elif col == "amount":
                    op = random.choice(["eq", "gt", "lt"])
                    candidate_filters.append({"op": "filter", "column": col, "operator": op, "value": random.randint(20, 250)})
                else:  # bill_id
                    op = random.choice(["eq", "gt", "lt"])
                    candidate_filters.append({"op": "filter", "column": col, "operator": op, "value": random.randint(1, 100)})

        if base_table == "appointments":
            # topic filters (4 options)
            candidate_filters.append({"op": "filter", "column": "topic", "operator": "eq", "value": random.choice(["doctor", "school", "repair", "delivery"])})
            candidate_filters.append({"op": "filter", "column": "topic", "operator": "eq", "value": random.choice(["doctor", "school", "repair", "delivery"])})
            # day filters (5 options)
            candidate_filters.append({"op": "filter", "column": "day", "operator": "eq", "value": random.choice(["mon", "tue", "wed", "thu", "fri"])})
            candidate_filters.append({"op": "filter", "column": "day", "operator": "eq", "value": random.choice(["mon", "tue", "wed", "thu", "fri"])})
            # slot filters (2 options)
            candidate_filters.append({"op": "filter", "column": "slot", "operator": "eq", "value": random.choice(["am", "pm"])})
            candidate_filters.append({"op": "filter", "column": "slot", "operator": "eq", "value": random.choice(["am", "pm"])})
            # attendee filters (4 options)
            candidate_filters.append({"op": "filter", "column": "attendee", "operator": "eq", "value": random.choice(["alice", "bob", "carol", "dave"])})
            candidate_filters.append({"op": "filter", "column": "attendee", "operator": "eq", "value": random.choice(["alice", "bob", "carol", "dave"])})
            # appt_id filters
            candidate_filters.append({"op": "filter", "column": "appt_id", "operator": "gt", "value": random.randint(3, 15)})
            candidate_filters.append({"op": "filter", "column": "appt_id", "operator": "lt", "value": random.randint(25, 50)})
            # Additional filters for higher complexity
            candidate_filters.append({"op": "filter", "column": "topic", "operator": "eq", "value": random.choice(["school", "delivery"])})
            candidate_filters.append({"op": "filter", "column": "day", "operator": "eq", "value": random.choice(["tue", "thu"])})
            candidate_filters.append({"op": "filter", "column": "slot", "operator": "eq", "value": "pm"})
            candidate_filters.append({"op": "filter", "column": "attendee", "operator": "eq", "value": random.choice(["bob", "dave"])})
            candidate_filters.append({"op": "filter", "column": "appt_id", "operator": "eq", "value": random.randint(5, 35)})
            # Add more filters to reach ~36 filters total
            for _ in range(21):
                col = random.choice(["topic", "day", "slot", "attendee", "appt_id"])
                if col in ["topic", "day", "slot", "attendee"]:
                    val = random.choice(["doctor", "school", "repair", "delivery", "mon", "tue", "wed", "thu", "fri", "am", "pm", "alice", "bob", "carol", "dave"])
                    candidate_filters.append({"op": "filter", "column": col, "operator": "eq", "value": val})
                else:  # appt_id
                    op = random.choice(["eq", "gt", "lt"])
                    candidate_filters.append({"op": "filter", "column": col, "operator": op, "value": random.randint(1, 100)})

        joins = []
        if base_table == "tasks":
            joins.append({"op": "join", "table": "appointments", "left_on": "assignee", "right_on": "attendee"})
            joins.append({"op": "join", "table": "groceries", "left_on": "room", "right_on": "store"})
            joins.append({"op": "join", "table": "bills", "left_on": "type", "right_on": "utility"})
            joins.append({"op": "join", "table": "appointments", "left_on": "room", "right_on": "topic"})
            joins.append({"op": "join", "table": "groceries", "left_on": "assignee", "right_on": "priority"})
            joins.append({"op": "join", "table": "bills", "left_on": "assignee", "right_on": "status"})
            # Additional joins for higher complexity
            joins.append({"op": "join", "table": "groceries", "left_on": "type", "right_on": "category"})
            joins.append({"op": "join", "table": "appointments", "left_on": "type", "right_on": "topic"})
            joins.append({"op": "join", "table": "bills", "left_on": "room", "right_on": "utility"})
            joins.append({"op": "join", "table": "groceries", "left_on": "room", "right_on": "category"})
            joins.append({"op": "join", "table": "appointments", "left_on": "assignee", "right_on": "day"})
            joins.append({"op": "join", "table": "bills", "left_on": "type", "right_on": "status"})
        if base_table == "groceries":
            joins.append({"op": "join", "table": "tasks", "left_on": "category", "right_on": "type"})
            joins.append({"op": "join", "table": "appointments", "left_on": "store", "right_on": "topic"})
            joins.append({"op": "join", "table": "bills", "left_on": "priority", "right_on": "status"})
            joins.append({"op": "join", "table": "tasks", "left_on": "store", "right_on": "room"})
            joins.append({"op": "join", "table": "appointments", "left_on": "category", "right_on": "day"})
            joins.append({"op": "join", "table": "bills", "left_on": "store", "right_on": "utility"})
            # Additional joins for higher complexity
            joins.append({"op": "join", "table": "tasks", "left_on": "priority", "right_on": "assignee"})
            joins.append({"op": "join", "table": "appointments", "left_on": "priority", "right_on": "slot"})
            joins.append({"op": "join", "table": "bills", "left_on": "category", "right_on": "utility"})
            joins.append({"op": "join", "table": "tasks", "left_on": "category", "right_on": "room"})
            joins.append({"op": "join", "table": "appointments", "left_on": "store", "right_on": "day"})
            joins.append({"op": "join", "table": "bills", "left_on": "category", "right_on": "status"})
        if base_table == "bills":
            joins.append({"op": "join", "table": "tasks", "left_on": "utility", "right_on": "type"})
            joins.append({"op": "join", "table": "appointments", "left_on": "status", "right_on": "slot"})
            joins.append({"op": "join", "table": "groceries", "left_on": "status", "right_on": "priority"})
            joins.append({"op": "join", "table": "tasks", "left_on": "status", "right_on": "assignee"})
            joins.append({"op": "join", "table": "appointments", "left_on": "utility", "right_on": "topic"})
            joins.append({"op": "join", "table": "groceries", "left_on": "utility", "right_on": "store"})
            # Additional joins for higher complexity
            joins.append({"op": "join", "table": "tasks", "left_on": "utility", "right_on": "room"})
            joins.append({"op": "join", "table": "appointments", "left_on": "utility", "right_on": "day"})
            joins.append({"op": "join", "table": "groceries", "left_on": "utility", "right_on": "category"})
            joins.append({"op": "join", "table": "tasks", "left_on": "status", "right_on": "room"})
            joins.append({"op": "join", "table": "appointments", "left_on": "status", "right_on": "attendee"})
            joins.append({"op": "join", "table": "groceries", "left_on": "status", "right_on": "store"})
        if base_table == "appointments":
            joins.append({"op": "join", "table": "tasks", "left_on": "attendee", "right_on": "assignee"})
            joins.append({"op": "join", "table": "groceries", "left_on": "topic", "right_on": "store"})
            joins.append({"op": "join", "table": "bills", "left_on": "slot", "right_on": "status"})
            joins.append({"op": "join", "table": "tasks", "left_on": "topic", "right_on": "room"})
            joins.append({"op": "join", "table": "groceries", "left_on": "day", "right_on": "category"})
            joins.append({"op": "join", "table": "bills", "left_on": "topic", "right_on": "utility"})
            # Additional joins for higher complexity
            joins.append({"op": "join", "table": "tasks", "left_on": "day", "right_on": "type"})
            joins.append({"op": "join", "table": "groceries", "left_on": "attendee", "right_on": "priority"})
            joins.append({"op": "join", "table": "bills", "left_on": "day", "right_on": "utility"})
            joins.append({"op": "join", "table": "tasks", "left_on": "slot", "right_on": "assignee"})
            joins.append({"op": "join", "table": "groceries", "left_on": "slot", "right_on": "store"})
            joins.append({"op": "join", "table": "bills", "left_on": "attendee", "right_on": "status"})

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
            metric_col = random.choice(["effort", "cost", "amount"])
        if metric == "unique":
            metric_col = random.choice(["room", "store", "status", "day"])

        return {"base_table": base_table, "ops": ops, "metric": metric, "metric_column": metric_col}


class ChoreCanvasEnvWithFeedback(ChoreCanvasEnv):
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
