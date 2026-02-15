from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class WorkbenchOrchestratorEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        max_turns: Optional[int] = 100,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.max_turns = max_turns if max_turns is not None else 100
        self.min_required_steps = self.complexity
        self.max_required_steps = self.complexity * 2
        self._init_database()
        self.reset()

    def _init_database(self) -> None:
        self.tools: Dict[str, Dict[str, Any]] = {
            "open_table": {
                "parameters": ["name"],
                "returns": "Set current focus table",
                "usage": "tool:open_table(name=<table_name>)",
            },
            "filter_rows": {
                "parameters": ["column", "op", "value"],
                "returns": "Filter rows in current table",
                "usage": "tool:filter_rows(column=<col>, op=eq|gt|lt, value=<val>)",
            },
            "join_table": {
                "parameters": ["name", "left_on", "right_on", "how"],
                "returns": "Join another table into current table",
                "usage": "tool:join_table(name=<table>, left_on=<col>, right_on=<col>, how=inner)",
            },
            "select_columns": {
                "parameters": ["names"],
                "returns": "Project selected columns (comma-separated)",
                "usage": "tool:select_columns(names=<col1,col2,...>)",
            },
            "compute_count": {
                "parameters": ["column"],
                "returns": "Count rows or non-null in column",
                "usage": "tool:compute_count(column=<optional_col>)",
            },
            "compute_sum": {
                "parameters": ["column"],
                "returns": "Sum numeric column",
                "usage": "tool:compute_sum(column=<col>)",
            },
            "unique_count": {
                "parameters": ["column"],
                "returns": "Count distinct values in column",
                "usage": "tool:unique_count(column=<col>)",
            },
            "snapshot": {
                "parameters": ["name"],
                "returns": "Save current table snapshot by name",
                "usage": "tool:snapshot(name=<snap_name>)",
            },
            "reset_state": {
                "parameters": [],
                "returns": "Clear current state",
                "usage": "tool:reset_state()",
            },
            "show_preview": {
                "parameters": ["n"],
                "returns": "Preview first n rows",
                "usage": "tool:show_preview(n=<int>)",
            },
        }
        n_events = 30 + self.complexity * 12
        n_devices = 10 + self.complexity * 5
        n_users = 8 + self.complexity * 4
        n_orders = 20 + self.complexity * 8
        kinds = ["phone", "tablet", "sensor", "laptop"][: min(4, 2 + self.complexity // 4)]
        statuses = ["active", "inactive"]
        plans = ["free", "pro", "premium"]
        regions = ["NA", "EU", "APAC"]
        event_types = ["view", "click", "error"]
        days = [f"D{i}" for i in range(1, 6 + (self.complexity // 2))]

        self.tables: Dict[str, List[Dict[str, Any]]] = {}
        self.tables["users"] = []
        for uid in range(1, n_users + 1):
            self.tables["users"].append(
                {
                    "user_id": uid,
                    "plan": random.choice(plans),
                    "region": random.choice(regions),
                }
            )

        self.tables["devices"] = []
        owner_ids = [u["user_id"] for u in self.tables["users"]]
        for did in range(1, n_devices + 1):
            self.tables["devices"].append(
                {
                    "id": did,
                    "owner_id": random.choice(owner_ids),
                    "status": random.choice(statuses),
                    "kind": random.choice(kinds),
                }
            )

        self.tables["events"] = []
        device_ids = [d["id"] for d in self.tables["devices"]]
        for eid in range(1, n_events + 1):
            self.tables["events"].append(
                {
                    "event_id": eid,
                    "device_id": random.choice(device_ids),
                    "type": random.choice(event_types),
                    "duration": random.randint(1, 300),
                    "day": random.choice(days),
                }
            )

        self.tables["orders"] = []
        order_statuses = ["paid", "refund", "pending"]
        for oid in range(1, n_orders + 1):
            self.tables["orders"].append(
                {
                    "order_id": oid,
                    "user_id": random.choice(owner_ids),
                    "amount": round(random.uniform(5, 500), 2),
                    "status": random.choice(order_statuses),
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
        tools_str = []
        for name, meta in self.tools.items():
            tools_str.append(f"- {name}: {meta['usage']}")
        tool_list = "\n".join(tools_str)
        return (
            "You are orchestrating a data workflow using the provided tools to compute a final numeric answer.\n"
            "Use the tools to open tables, join related data, filter rows, and aggregate values.\n"
            "Actions must be submitted in \\boxed{...} format.\n"
            "Use 'tool:' prefix for tool calls and 'answer:' for final submission.\n"
            "Available tools:\n" + tool_list +
            "\nFinal submission format: \\boxed{answer:<number>} (do not include units).\n"
            "The task requires a minimum number of tool calls before submitting the final answer."
        )

    def get_task_suffix(self) -> str:
        ct = self.current_table_name if self.current_table_name else "None"
        return (
            f"Task: {self._describe_task()} | "
            f"Current table: {ct} | Tool calls: {self.steps_taken}/{self.required_steps} | "
            f"Turns: {self.turn_count}/{self.max_turns} | "
            "Submit actions using \\boxed{tool:...} or \\boxed{answer:...}."
        )

    def _describe_task(self) -> str:
        base = self.task.get("base_table", "?")
        metric = self.task.get("metric", "?")
        col = self.task.get("metric_column", None)
        clauses = []
        for op in self.task.get("ops", []):
            if op["op"] == "join":
                clauses.append(f"join {op['table']} on {op['left_on']}={op['right_on']}")
            elif op["op"] == "filter":
                val = op["value"]
                clauses.append(f"filter {op['column']} {op['operator']} {val}")
            elif op["op"] == "select":
                clauses.append(f"select {','.join(op['names'])}")
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
        required_steps = random.randint(self.min_required_steps, self.max_required_steps)
        self.required_steps = required_steps
        self.task = self._generate_task_requiring_n_steps(required_steps)
        self.task["solution"] = self._compute_ground_truth(self.task)
        obs = self._get_instructions()
        info = {"suffix": self.get_task_suffix()}
        return obs, info

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        if self.turn_count > self.max_turns:
            obs = "Timeout: maximum turns reached."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}
        parsed = self._parse_action(action)
        if parsed is None:
            obs = "Invalid action format. Use \\boxed{tool:...} or \\boxed{answer:...}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
        if parsed["type"] == "tool":
            name = parsed["name"]
            args = parsed["args"]
            if name not in self.tools:
                obs = f"Unsupported tool: {name}."
                return obs, -0.2, True, False, {"suffix": self.get_task_suffix()}
            try:
                result = self._execute_tool(name, args)
                self.steps_taken += 1
                obs = f"Tool {name} executed. Result: {result}"
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            except ValueError as ve:
                obs = f"Protocol violation: {str(ve)}"
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            except Exception as e:
                obs = f"Execution error: {str(e)}"
                return obs, -0.1, False, False, {"suffix": self.get_task_suffix()}
        elif parsed["type"] == "answer":
            val = parsed["value"]
            correct = self._compare_answer(val, self.task["solution"])
            if not correct:
                obs = f"Wrong answer. Your submitted answer: {val}."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            if self.steps_taken < self.required_steps:
                obs = (
                    f"Protocol violation: insufficient tool usage "
                    f"({self.steps_taken}/{self.required_steps}). Your submitted answer: {val}."
                )
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            obs = f"Success: correct final answer {val}."
            return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
        else:
            obs = "Unsupported action type."
            return obs, -0.2, True, False, {"suffix": self.get_task_suffix()}

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
                    k = k.strip()
                    v = v.strip()
                    val = self._parse_value(v)
                    args[k] = val
            return {"type": "tool", "name": name, "args": args}
        # If only a number inside box, treat as answer
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
        parts = []
        buf = ""
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
                buf += ch
                continue
            if ch == ")":
                nest = max(nest - 1, 0)
                buf += ch
                continue
            if ch == "," and nest == 0:
                parts.append(buf.strip())
                buf = ""
                continue
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
            tbl = random.choice(list(self.tables.keys()))
            return f"\\boxed{{tool:open_table(name='{tbl}')}}"
        else:
            col = random.choice(list(self.current_table[0].keys())) if self.current_table else "event_id"
            return f"\\boxed{{tool:filter_rows(column='{col}', op=eq, value='{random.choice(['active','view','paid','EU'])}')}}"

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        if name == "open_table":
            tname = args.get("name")
            if tname not in self.tables:
                raise ValueError(f"table not found: {tname}")
            self.current_table_name = tname
            self.current_table = [dict(row) for row in self.tables[tname]]
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
            col = args.get("column")
            op = args.get("op")
            val = args.get("value")
            if col is None or op is None:
                raise ValueError("missing filter parameters")
            def keep(row):
                if col not in row:
                    return False
                rv = row[col]
                if op == "eq":
                    return str(rv) == str(val)
                try:
                    rvn = float(rv)
                    vn = float(val)
                except Exception:
                    rvn = None
                    vn = None
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
            tname = args.get("name")
            left_on = args.get("left_on")
            right_on = args.get("right_on")
            how = args.get("how", "inner")
            if tname not in self.tables:
                raise ValueError(f"table not found: {tname}")
            if left_on is None or right_on is None:
                raise ValueError("missing join keys")
            right_index = {}
            for r in self.tables[tname]:
                right_index.setdefault(r.get(right_on), []).append(r)
            joined = []
            for l in self.current_table:
                lk = l.get(left_on)
                matches = right_index.get(lk, [])
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
            if isinstance(names, str):
                cols = [c.strip() for c in names.split(",")]
            else:
                cols = list(names) if names else []
            if not cols:
                raise ValueError("no columns specified")
            projected = []
            for r in self.current_table:
                projected.append({k: r.get(k) for k in cols})
            self.current_table = projected
            return f"Selected columns: {','.join(cols)}."
        if name == "compute_count":
            column = args.get("column", None)
            if column is None:
                cnt = len(self.current_table)
                self.execution_state["last_metric"] = cnt
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
            total = round(total, 6)
            self.execution_state["last_metric"] = total
            return f"Summed {column} = {total}."
        if name == "unique_count":
            column = args.get("column")
            if column is None:
                raise ValueError("unique_count requires 'column'")
            uniq = set()
            for r in self.current_table:
                uniq.add(r.get(column))
            cnt = len(uniq)
            self.execution_state["last_metric"] = cnt
            return f"Distinct values in {column}: {cnt}."
        if name == "snapshot":
            snap = args.get("name")
            if not snap:
                raise ValueError("snapshot requires 'name'")
            self.execution_state.setdefault("snapshots", {})[snap] = [dict(r) for r in self.current_table]
            return f"Snapshot '{snap}' saved with {len(self.current_table)} rows."
        raise ValueError(f"unknown tool '{name}'")

    def _generate_task_requiring_n_steps(self, n: int) -> Dict[str, Any]:
        base_options = ["events", "orders"]
        base = random.choice(base_options)
        ops: List[Dict[str, Any]] = []
        # Ensure at least one open step conceptually counted by agent; we don't add it here since it's implicit in task
        remaining = max(1, n - 1)
        # Prefer joins early to make filters meaningful on joined columns
        join_plan = []
        filter_plan = []
        select_plan = []
        if base == "events":
            possible_joins = [
                {"op": "join", "table": "devices", "left_on": "device_id", "right_on": "id"},
                {"op": "join", "table": "users", "left_on": "owner_id", "right_on": "user_id"},  # requires devices joined
                {"op": "join", "table": "orders", "left_on": "user_id", "right_on": "user_id"},
            ]
            # Choose valid subset respecting dependencies
            if remaining >= 1:
                join_plan.append(possible_joins[0])
                remaining -= 1
            if remaining >= 1 and random.random() < 0.7:
                join_plan.append(possible_joins[1])
                remaining -= 1
            if remaining >= 1 and random.random() < 0.5:
                join_plan.append(possible_joins[2])
                remaining -= 1
            # Filters on base and joined tables
            filters_candidates = [
                {"op": "filter", "column": "type", "operator": "eq", "value": random.choice(["error", "click", "view"])},
                {"op": "filter", "column": "day", "operator": "eq", "value": random.choice([d["day"] for d in self.tables["events"]])},
                {"op": "filter", "column": "duration", "operator": "gt", "value": random.randint(10, 150)},
                {"op": "filter", "column": "status", "operator": "eq", "value": random.choice(["active", "inactive"])},
                {"op": "filter", "column": "kind", "operator": "eq", "value": random.choice(["phone", "tablet", "sensor", "laptop"])},
                {"op": "filter", "column": "plan", "operator": "eq", "value": random.choice(["free", "pro", "premium"])},
                {"op": "filter", "column": "region", "operator": "eq", "value": random.choice(["NA", "EU", "APAC"])},
                {"op": "filter", "column": "status", "operator": "eq", "value": random.choice(["paid", "refund", "pending"])},
            ]
            # Ensure fields exist after joins
            # We'll add filters progressively until remaining 0
            for cand in filters_candidates:
                if remaining <= 0:
                    break
                # Only allow filters on joined tables if they exist in join_plan
                col = cand["column"]
                if col in ("status", "kind") and not any(j["table"] == "devices" for j in join_plan):
                    continue
                if col in ("plan", "region") and not any(j["table"] == "users" for j in join_plan):
                    continue
                if col in ("status") and any(j["table"] == "orders" for j in join_plan):
                    # orders.status vs devices.status ambiguity; we rely on filter to hit matching key in merged row:
                    # after join, both may exist; we bias toward 'status' from orders by presence of 'orders_status'
                    # But filtering uses plain key; we allow devices.status only if orders joined creates 'status' collision.
                    pass
                filter_plan.append(cand)
                remaining -= 1
            metric = random.choice(["count", "sum", "unique"])
            if metric == "sum":
                metric_col = "duration"
            elif metric == "unique":
                metric_col = random.choice(["type", "kind", "region"])
                # ensure relevant join exists for kind/region
                if metric_col == "kind" and not any(j["table"] == "devices" for j in join_plan):
                    metric_col = "type"
                if metric_col == "region" and not any(j["table"] == "users" for j in join_plan):
                    metric_col = "type"
            else:
                metric_col = None
        else:  # orders base
            possible_joins = [
                {"op": "join", "table": "users", "left_on": "user_id", "right_on": "user_id"},
            ]
            if remaining >= 1:
                join_plan.append(possible_joins[0])
                remaining -= 1
            filters_candidates = [
                {"op": "filter", "column": "status", "operator": "eq", "value": random.choice(["paid", "refund", "pending"])},
                {"op": "filter", "column": "amount", "operator": "gt", "value": random.randint(20, 200)},
                {"op": "filter", "column": "plan", "operator": "eq", "value": random.choice(["free", "pro", "premium"])},
                {"op": "filter", "column": "region", "operator": "eq", "value": random.choice(["NA", "EU", "APAC"])},
            ]
            for cand in filters_candidates:
                if remaining <= 0:
                    break
                if cand["column"] in ("plan", "region") and not any(j["table"] == "users" for j in join_plan):
                    continue
                filter_plan.append(cand)
                remaining -= 1
            metric = random.choice(["sum", "count"])
            metric_col = "amount" if metric == "sum" else None
        ops = join_plan + filter_plan + select_plan
        return {"base_table": base, "ops": ops, "metric": metric, "metric_column": metric_col}

    def _apply_ops(self, base_rows: List[Dict[str, Any]], ops: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows = [dict(r) for r in base_rows]
        for op in ops:
            if op["op"] == "join":
                tname = op["table"]
                left_on = op["left_on"]
                right_on = op["right_on"]
                right_index = {}
                for r in self.tables[tname]:
                    right_index.setdefault(r.get(right_on), []).append(r)
                new_rows = []
                for l in rows:
                    lk = l.get(left_on)
                    matches = right_index.get(lk, [])
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
                col = op["column"]
                operator = op["operator"]
                val = op["value"]
                def keep(r):
                    rv = r.get(col)
                    if operator == "eq":
                        return str(rv) == str(val)
                    try:
                        rvn = float(rv)
                        vn = float(val)
                    except Exception:
                        return False
                    if operator == "gt":
                        return rvn > vn
                    if operator == "lt":
                        return rvn < vn
                    return False
                rows = [r for r in rows if keep(r)]
            elif op["op"] == "select":
                cols = op["names"]
                rows = [{k: r.get(k) for k in cols} for r in rows]
        return rows

    def _compute_ground_truth(self, task: Dict[str, Any]) -> float:
        base = task["base_table"]
        base_rows = self.tables[base]
        rows = self._apply_ops(base_rows, task["ops"])
        metric = task["metric"]
        col = task.get("metric_column")
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


class WorkbenchOrchestratorEnvWithFeedback(WorkbenchOrchestratorEnv):
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
            hint = "Wrap your action in \\boxed{...}. Use tool:open_table(...) or answer:<number>."
        elif "unsupported tool" in text:
            error_type = "UnsupportedAction"
            m = re.search(r"unsupported tool:\s*([a-z_]+)", text)
            error_detail["tool"] = m.group(1) if m else None
            hint = "Check the available tools list in the instructions and use the correct name."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "no active table" in text:
                error_detail["violation"] = "no_active_table"
                hint = "Call \\boxed{tool:open_table(name='<table>')} before filtering, joining, or aggregating."
            elif "insufficient tool usage" in text:
                error_detail["violation"] = "insufficient_steps"
                hint = "Perform the required number of tool calls (open, join, filter) before submitting the final answer."
            elif "missing filter parameters" in text:
                error_detail["violation"] = "bad_filter_args"
                hint = "Provide column, op, and value for filter_rows."
            elif "table not found" in text:
                error_detail["violation"] = "bad_table_name"
                hint = "Use an existing table name (events, devices, users, orders)."
            else:
                error_detail["violation"] = "other_protocol"
                hint = "Ensure tool arguments are correct and prerequisites are met."
        elif "execution error" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "execution_error"
            hint = "Review tool arguments and state. Use show_preview to inspect current table."
        elif "wrong answer" in text:
            error_type = "WrongDecision"
            m = re.search(r"your submitted answer:\s*([-+]?\d+(\.\d+)?)", obs, flags=re.I)
            got = float(m.group(1)) if m else None
            error_detail["expected"] = self.task.get("solution")
            error_detail["got"] = got
            hint = "Verify joins and filters. Preview rows and use compute_sum/count/unique_count to cross-check before submitting."
        elif "timeout" in text:
            error_type = "Timeout"
            error_detail["outcome"] = "max_turns_reached"
            hint = "Plan your tool sequence and submit before reaching the turn limit."
        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "current_table": getattr(self, "current_table_name", None),
                "steps_taken": getattr(self, "steps_taken", None),
                "required_steps": getattr(self, "required_steps", None),
                "metric": self.task.get("metric"),
                "metric_column": self.task.get("metric_column"),
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
            "hint": "Start by opening the relevant base table with open_table, then join and filter as described in the task.",
            "turn": 0,
            "state": {
                "current_table": None,
                "steps_taken": 0,
                "required_steps": self.required_steps,
                "metric": self.task.get("metric"),
                "metric_column": self.task.get("metric_column"),
            },
        }
        return obs, info