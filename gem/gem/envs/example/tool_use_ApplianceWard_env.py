from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


def _step_range_overlap(level: int) -> Tuple[int, int]:
    return 2 * level, 2 * level + 2  # level1:2-4 ... level10:20-22


class ApplianceWardEnv(Env):
    """
    Overlap版：家电维修/工单/零件/技师场景，步数 2N~2N+2。
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

        n_tickets = 25 + self.complexity * 10
        n_parts = 20 + self.complexity * 8
        n_techs = 10 + self.complexity * 4
        n_visits = 18 + self.complexity * 7
        appliance_types = ["fridge", "washer", "dryer", "oven", "ac", "tv"]
        skills = ["electrical", "plumbing", "hvac", "general"]
        regions = ["north", "south", "east", "west"]

        self.tables: Dict[str, List[Dict[str, Any]]] = {}
        self.tables["technicians"] = []
        for tid in range(1, n_techs + 1):
            self.tables["technicians"].append(
                {"tech_id": tid, "skill": random.choice(skills), "region": random.choice(regions), "rating": round(random.uniform(3.0, 5.0), 2)}
            )

        self.tables["tickets"] = []
        for kid in range(1, n_tickets + 1):
            self.tables["tickets"].append(
                {
                    "ticket_id": kid,
                    "appliance": random.choice(appliance_types),
                    "severity": random.choice(["low", "med", "high"]),
                    "region": random.choice(regions),
                    "status": random.choice(["open", "assigned", "resolved"]),
                    "tech_id": random.randint(1, n_techs),
                }
            )

        self.tables["parts"] = []
        for pid in range(1, n_parts + 1):
            self.tables["parts"].append(
                {
                    "part_id": pid,
                    "appliance": random.choice(appliance_types),
                    "cost": round(random.uniform(5, 200), 2),
                    "stock": random.randint(0, 30),
                }
            )

        self.tables["visits"] = []
        for vid in range(1, n_visits + 1):
            self.tables["visits"].append(
                {
                    "visit_id": vid,
                    "ticket_id": random.randint(1, n_tickets),
                    "duration_h": random.randint(1, 6),
                    "success": random.choice([True, False]),
                }
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
            "You triage appliance repair data (tickets, technicians, parts, visits) to compute a numeric metric.\n"
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
        col = random.choice(list(self.current_table[0].keys())) if self.current_table else "ticket_id"
        return f"\\boxed{{tool:filter_rows(column='{col}', op=eq, value='{random.choice(['open','fridge','hvac'])}')}}"

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
        base_table = random.choice(["tickets", "technicians", "parts", "visits"])
        ops: List[Dict[str, Any]] = []
        candidate_filters: List[Dict[str, Any]] = []

        # 大幅增加候选操作，确保level 10（需要45-46个ops）能够完美执行
        appliance_types = ["fridge", "washer", "dryer", "oven", "ac", "tv"]
        severities = ["low", "med", "high"]
        statuses = ["open", "assigned", "resolved"]
        regions = ["north", "south", "east", "west"]
        skills = ["electrical", "plumbing", "hvac", "general"]

        if base_table == "tickets":
            # appliance filters (6 values)
            candidate_filters.append({"op": "filter", "column": "appliance", "operator": "eq", "value": random.choice(appliance_types)})
            candidate_filters.append({"op": "filter", "column": "appliance", "operator": "eq", "value": random.choice(appliance_types)})
            candidate_filters.append({"op": "filter", "column": "appliance", "operator": "eq", "value": random.choice(appliance_types)})
            # severity filters (3 values)
            candidate_filters.append({"op": "filter", "column": "severity", "operator": "eq", "value": random.choice(severities)})
            candidate_filters.append({"op": "filter", "column": "severity", "operator": "eq", "value": random.choice(severities)})
            # region filters (4 values)
            candidate_filters.append({"op": "filter", "column": "region", "operator": "eq", "value": random.choice(regions)})
            candidate_filters.append({"op": "filter", "column": "region", "operator": "eq", "value": random.choice(regions)})
            # status filters (3 values)
            candidate_filters.append({"op": "filter", "column": "status", "operator": "eq", "value": random.choice(statuses)})
            candidate_filters.append({"op": "filter", "column": "status", "operator": "eq", "value": random.choice(statuses)})
            # ticket_id filters
            candidate_filters.append({"op": "filter", "column": "ticket_id", "operator": "gt", "value": random.randint(10, 30)})
            candidate_filters.append({"op": "filter", "column": "ticket_id", "operator": "lt", "value": random.randint(50, 100)})
            candidate_filters.append({"op": "filter", "column": "ticket_id", "operator": "eq", "value": random.randint(1, 120)})
            # tech_id filters
            candidate_filters.append({"op": "filter", "column": "tech_id", "operator": "gt", "value": random.randint(3, 10)})
            candidate_filters.append({"op": "filter", "column": "tech_id", "operator": "lt", "value": random.randint(15, 40)})
            candidate_filters.append({"op": "filter", "column": "tech_id", "operator": "eq", "value": random.randint(1, 50)})
            # Add more filters dynamically to reach ~36 filters
            for _ in range(21):
                col = random.choice(["appliance", "severity", "region", "status", "ticket_id", "tech_id"])
                if col in ["appliance", "severity", "region", "status"]:
                    vals = {"appliance": appliance_types, "severity": severities, "region": regions, "status": statuses}
                    candidate_filters.append({"op": "filter", "column": col, "operator": "eq", "value": random.choice(vals[col])})
                else:
                    op = random.choice(["eq", "gt", "lt"])
                    candidate_filters.append({"op": "filter", "column": col, "operator": op, "value": random.randint(1, 150)})

        if base_table == "technicians":
            # skill filters (4 values)
            candidate_filters.append({"op": "filter", "column": "skill", "operator": "eq", "value": random.choice(skills)})
            candidate_filters.append({"op": "filter", "column": "skill", "operator": "eq", "value": random.choice(skills)})
            candidate_filters.append({"op": "filter", "column": "skill", "operator": "eq", "value": random.choice(skills)})
            # region filters (4 values)
            candidate_filters.append({"op": "filter", "column": "region", "operator": "eq", "value": random.choice(regions)})
            candidate_filters.append({"op": "filter", "column": "region", "operator": "eq", "value": random.choice(regions)})
            # rating filters (3.0-5.0)
            candidate_filters.append({"op": "filter", "column": "rating", "operator": "gt", "value": round(random.uniform(3.0, 3.5), 1)})
            candidate_filters.append({"op": "filter", "column": "rating", "operator": "gt", "value": round(random.uniform(3.5, 4.0), 1)})
            candidate_filters.append({"op": "filter", "column": "rating", "operator": "gt", "value": round(random.uniform(4.0, 4.5), 1)})
            candidate_filters.append({"op": "filter", "column": "rating", "operator": "lt", "value": round(random.uniform(4.0, 4.5), 1)})
            candidate_filters.append({"op": "filter", "column": "rating", "operator": "lt", "value": round(random.uniform(3.5, 4.0), 1)})
            candidate_filters.append({"op": "filter", "column": "rating", "operator": "eq", "value": round(random.uniform(3.0, 5.0), 1)})
            # tech_id filters
            candidate_filters.append({"op": "filter", "column": "tech_id", "operator": "gt", "value": random.randint(3, 10)})
            candidate_filters.append({"op": "filter", "column": "tech_id", "operator": "lt", "value": random.randint(15, 40)})
            candidate_filters.append({"op": "filter", "column": "tech_id", "operator": "eq", "value": random.randint(1, 50)})
            # Add more filters dynamically to reach ~36 filters
            for _ in range(22):
                col = random.choice(["skill", "region", "rating", "tech_id"])
                if col == "skill":
                    candidate_filters.append({"op": "filter", "column": col, "operator": "eq", "value": random.choice(skills)})
                elif col == "region":
                    candidate_filters.append({"op": "filter", "column": col, "operator": "eq", "value": random.choice(regions)})
                elif col == "rating":
                    op = random.choice(["eq", "gt", "lt"])
                    candidate_filters.append({"op": "filter", "column": col, "operator": op, "value": round(random.uniform(3.0, 5.0), 1)})
                else:  # tech_id
                    op = random.choice(["eq", "gt", "lt"])
                    candidate_filters.append({"op": "filter", "column": col, "operator": op, "value": random.randint(1, 100)})

        if base_table == "parts":
            # appliance filters (6 values)
            candidate_filters.append({"op": "filter", "column": "appliance", "operator": "eq", "value": random.choice(appliance_types)})
            candidate_filters.append({"op": "filter", "column": "appliance", "operator": "eq", "value": random.choice(appliance_types)})
            candidate_filters.append({"op": "filter", "column": "appliance", "operator": "eq", "value": random.choice(appliance_types)})
            # cost filters (5-200)
            candidate_filters.append({"op": "filter", "column": "cost", "operator": "lt", "value": random.randint(40, 100)})
            candidate_filters.append({"op": "filter", "column": "cost", "operator": "lt", "value": random.randint(100, 160)})
            candidate_filters.append({"op": "filter", "column": "cost", "operator": "gt", "value": random.randint(20, 80)})
            candidate_filters.append({"op": "filter", "column": "cost", "operator": "gt", "value": random.randint(80, 150)})
            candidate_filters.append({"op": "filter", "column": "cost", "operator": "eq", "value": random.randint(5, 200)})
            # stock filters (0-30)
            candidate_filters.append({"op": "filter", "column": "stock", "operator": "gt", "value": random.randint(5, 15)})
            candidate_filters.append({"op": "filter", "column": "stock", "operator": "gt", "value": random.randint(15, 25)})
            candidate_filters.append({"op": "filter", "column": "stock", "operator": "lt", "value": random.randint(10, 20)})
            candidate_filters.append({"op": "filter", "column": "stock", "operator": "lt", "value": random.randint(5, 12)})
            candidate_filters.append({"op": "filter", "column": "stock", "operator": "eq", "value": random.randint(0, 30)})
            # part_id filters
            candidate_filters.append({"op": "filter", "column": "part_id", "operator": "gt", "value": random.randint(5, 20)})
            candidate_filters.append({"op": "filter", "column": "part_id", "operator": "lt", "value": random.randint(40, 80)})
            candidate_filters.append({"op": "filter", "column": "part_id", "operator": "eq", "value": random.randint(1, 100)})
            # Add more filters dynamically to reach ~36 filters
            for _ in range(20):
                col = random.choice(["appliance", "cost", "stock", "part_id"])
                if col == "appliance":
                    candidate_filters.append({"op": "filter", "column": col, "operator": "eq", "value": random.choice(appliance_types)})
                else:
                    op = random.choice(["eq", "gt", "lt"])
                    if col == "cost":
                        candidate_filters.append({"op": "filter", "column": col, "operator": op, "value": random.randint(5, 200)})
                    elif col == "stock":
                        candidate_filters.append({"op": "filter", "column": col, "operator": op, "value": random.randint(0, 30)})
                    else:  # part_id
                        candidate_filters.append({"op": "filter", "column": col, "operator": op, "value": random.randint(1, 120)})

        if base_table == "visits":
            # duration_h filters (1-6)
            candidate_filters.append({"op": "filter", "column": "duration_h", "operator": "gt", "value": random.randint(1, 3)})
            candidate_filters.append({"op": "filter", "column": "duration_h", "operator": "gt", "value": random.randint(3, 5)})
            candidate_filters.append({"op": "filter", "column": "duration_h", "operator": "lt", "value": random.randint(3, 5)})
            candidate_filters.append({"op": "filter", "column": "duration_h", "operator": "lt", "value": random.randint(2, 4)})
            candidate_filters.append({"op": "filter", "column": "duration_h", "operator": "eq", "value": random.randint(1, 6)})
            # success filters (True/False)
            candidate_filters.append({"op": "filter", "column": "success", "operator": "eq", "value": True})
            candidate_filters.append({"op": "filter", "column": "success", "operator": "eq", "value": False})
            candidate_filters.append({"op": "filter", "column": "success", "operator": "eq", "value": random.choice([True, False])})
            # visit_id filters
            candidate_filters.append({"op": "filter", "column": "visit_id", "operator": "gt", "value": random.randint(5, 20)})
            candidate_filters.append({"op": "filter", "column": "visit_id", "operator": "lt", "value": random.randint(30, 70)})
            candidate_filters.append({"op": "filter", "column": "visit_id", "operator": "eq", "value": random.randint(1, 100)})
            # ticket_id filters
            candidate_filters.append({"op": "filter", "column": "ticket_id", "operator": "gt", "value": random.randint(10, 30)})
            candidate_filters.append({"op": "filter", "column": "ticket_id", "operator": "lt", "value": random.randint(50, 100)})
            candidate_filters.append({"op": "filter", "column": "ticket_id", "operator": "eq", "value": random.randint(1, 120)})
            # Add more filters dynamically to reach ~36 filters
            for _ in range(22):
                col = random.choice(["duration_h", "success", "visit_id", "ticket_id"])
                if col == "success":
                    candidate_filters.append({"op": "filter", "column": col, "operator": "eq", "value": random.choice([True, False])})
                else:
                    op = random.choice(["eq", "gt", "lt"])
                    if col == "duration_h":
                        candidate_filters.append({"op": "filter", "column": col, "operator": op, "value": random.randint(1, 6)})
                    else:  # visit_id or ticket_id
                        candidate_filters.append({"op": "filter", "column": col, "operator": op, "value": random.randint(1, 150)})

        joins: List[Dict[str, Any]] = []
        if base_table == "tickets":
            joins.append({"op": "join", "table": "technicians", "left_on": "tech_id", "right_on": "tech_id"})
            joins.append({"op": "join", "table": "parts", "left_on": "appliance", "right_on": "appliance"})
            joins.append({"op": "join", "table": "visits", "left_on": "ticket_id", "right_on": "ticket_id"})
            joins.append({"op": "join", "table": "technicians", "left_on": "region", "right_on": "region"})
            joins.append({"op": "join", "table": "parts", "left_on": "ticket_id", "right_on": "part_id"})
            joins.append({"op": "join", "table": "visits", "left_on": "tech_id", "right_on": "visit_id"})
            joins.append({"op": "join", "table": "technicians", "left_on": "severity", "right_on": "skill"})
            joins.append({"op": "join", "table": "parts", "left_on": "status", "right_on": "appliance"})
            joins.append({"op": "join", "table": "visits", "left_on": "severity", "right_on": "success"})
            joins.append({"op": "join", "table": "technicians", "left_on": "ticket_id", "right_on": "tech_id"})
            joins.append({"op": "join", "table": "parts", "left_on": "region", "right_on": "appliance"})
            joins.append({"op": "join", "table": "visits", "left_on": "appliance", "right_on": "ticket_id"})
        if base_table == "technicians":
            joins.append({"op": "join", "table": "tickets", "left_on": "tech_id", "right_on": "tech_id"})
            joins.append({"op": "join", "table": "parts", "left_on": "skill", "right_on": "appliance"})
            joins.append({"op": "join", "table": "visits", "left_on": "tech_id", "right_on": "visit_id"})
            joins.append({"op": "join", "table": "tickets", "left_on": "region", "right_on": "region"})
            joins.append({"op": "join", "table": "parts", "left_on": "tech_id", "right_on": "part_id"})
            joins.append({"op": "join", "table": "visits", "left_on": "region", "right_on": "ticket_id"})
            joins.append({"op": "join", "table": "tickets", "left_on": "skill", "right_on": "severity"})
            joins.append({"op": "join", "table": "parts", "left_on": "region", "right_on": "appliance"})
            joins.append({"op": "join", "table": "visits", "left_on": "skill", "right_on": "success"})
            joins.append({"op": "join", "table": "tickets", "left_on": "tech_id", "right_on": "ticket_id"})
            joins.append({"op": "join", "table": "parts", "left_on": "skill", "right_on": "appliance"})
            joins.append({"op": "join", "table": "visits", "left_on": "tech_id", "right_on": "ticket_id"})
        if base_table == "parts":
            joins.append({"op": "join", "table": "tickets", "left_on": "appliance", "right_on": "appliance"})
            joins.append({"op": "join", "table": "technicians", "left_on": "appliance", "right_on": "skill"})
            joins.append({"op": "join", "table": "visits", "left_on": "part_id", "right_on": "visit_id"})
            joins.append({"op": "join", "table": "tickets", "left_on": "part_id", "right_on": "ticket_id"})
            joins.append({"op": "join", "table": "technicians", "left_on": "part_id", "right_on": "tech_id"})
            joins.append({"op": "join", "table": "visits", "left_on": "appliance", "right_on": "ticket_id"})
            joins.append({"op": "join", "table": "tickets", "left_on": "appliance", "right_on": "status"})
            joins.append({"op": "join", "table": "technicians", "left_on": "appliance", "right_on": "region"})
            joins.append({"op": "join", "table": "visits", "left_on": "stock", "right_on": "duration_h"})
            joins.append({"op": "join", "table": "tickets", "left_on": "cost", "right_on": "tech_id"})
            joins.append({"op": "join", "table": "technicians", "left_on": "stock", "right_on": "rating"})
            joins.append({"op": "join", "table": "visits", "left_on": "part_id", "right_on": "ticket_id"})
        if base_table == "visits":
            joins.append({"op": "join", "table": "tickets", "left_on": "ticket_id", "right_on": "ticket_id"})
            joins.append({"op": "join", "table": "technicians", "left_on": "visit_id", "right_on": "tech_id"})
            joins.append({"op": "join", "table": "parts", "left_on": "visit_id", "right_on": "part_id"})
            joins.append({"op": "join", "table": "tickets", "left_on": "visit_id", "right_on": "ticket_id"})
            joins.append({"op": "join", "table": "technicians", "left_on": "ticket_id", "right_on": "tech_id"})
            joins.append({"op": "join", "table": "parts", "left_on": "ticket_id", "right_on": "part_id"})
            joins.append({"op": "join", "table": "tickets", "left_on": "success", "right_on": "severity"})
            joins.append({"op": "join", "table": "technicians", "left_on": "success", "right_on": "skill"})
            joins.append({"op": "join", "table": "parts", "left_on": "duration_h", "right_on": "stock"})
            joins.append({"op": "join", "table": "tickets", "left_on": "duration_h", "right_on": "tech_id"})
            joins.append({"op": "join", "table": "technicians", "left_on": "duration_h", "right_on": "rating"})
            joins.append({"op": "join", "table": "parts", "left_on": "visit_id", "right_on": "part_id"})

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
            metric_col = random.choice(["rating", "cost", "stock", "duration_h"])
        if metric == "unique":
            metric_col = random.choice(["appliance", "severity", "status", "skill", "region", "success"])

        return {"base_table": base_table, "ops": ops, "metric": metric, "metric_column": metric_col}


class ApplianceWardEnvWithFeedback(ApplianceWardEnv):
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
