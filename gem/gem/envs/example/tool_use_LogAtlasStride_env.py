from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


def _step_range_overlap(level: int) -> Tuple[int, int]:
    # level1:2-4, level2:4-6 ... level10:20-22
    return 2 * level, 2 * level + 2


class LogAtlasStrideEnv(Env):
    """
    Overlap版：固定阶梯步数区间，基于日志/服务/告警数据进行聚合。
    """

    def __init__(self, complexity: int = 1, max_turns: Optional[int] = 200, **_):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.max_turns = max_turns if max_turns is not None else 200
        self.min_required_steps, self.max_required_steps = _step_range_overlap(self.complexity)
        self._init_database()
        self.reset()

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
        # 数据规模随复杂度线性增长
        n_logs = 40 + self.complexity * 15
        n_services = 12 + self.complexity * 4
        n_incidents = 15 + self.complexity * 6
        tiers = ["gold", "silver", "bronze"]
        regions = ["us", "eu", "apac"]
        levels = ["info", "warn", "error"]
        days = [f"D{i}" for i in range(1, 7 + self.complexity // 2)]

        self.tables: Dict[str, List[Dict[str, Any]]] = {}
        self.tables["services"] = []
        for sid in range(1, n_services + 1):
            self.tables["services"].append(
                {
                    "service_id": sid,
                    "tier": random.choice(tiers),
                    "owner": f"team_{random.randint(1, 6)}",
                    "region": random.choice(regions),
                }
            )

        self.tables["logs"] = []
        service_ids = [s["service_id"] for s in self.tables["services"]]
        for lid in range(1, n_logs + 1):
            self.tables["logs"].append(
                {
                    "log_id": lid,
                    "service_id": random.choice(service_ids),
                    "level": random.choice(levels),
                    "latency_ms": random.randint(10, 600),
                    "day": random.choice(days),
                    "region": random.choice(regions),
                }
            )

        self.tables["incidents"] = []
        for iid in range(1, n_incidents + 1):
            self.tables["incidents"].append(
                {
                    "incident_id": iid,
                    "service_id": random.choice(service_ids),
                    "severity": random.choice(["sev1", "sev2", "sev3"]),
                    "status": random.choice(["open", "mitigated", "closed"]),
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
            "You operate on observability data (services, logs, incidents) to compute a numeric metric.\n"
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
                obs = f"Protocol violation: insufficient tool usage ({self.steps_taken}/{self.required_steps})."
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
                    val = self._parse_value(v.strip())
                    args[k.strip()] = val
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
        col = random.choice(list(self.current_table[0].keys())) if self.current_table else "log_id"
        return f"\\boxed{{tool:filter_rows(column='{col}', op=eq, value='{random.choice(['error','gold','us'])}')}}"

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
                    rvn = float(rv)
                    vn = float(val)
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
            cols = [c.strip() for c in names.split(",")] if isinstance(names, str) else list(names or [])
            if not cols:
                raise ValueError("no columns specified")
            projected = [{k: r.get(k) for k in cols} for r in self.current_table]
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
        base_options = ["logs", "incidents"]
        base = random.choice(base_options)
        ops: List[Dict[str, Any]] = []
        remaining = max(1, n - 1)
        joins, filters = [], []
        if base == "logs":
            join_candidates = [
                {"op": "join", "table": "services", "left_on": "service_id", "right_on": "service_id"},
                {"op": "join", "table": "incidents", "left_on": "service_id", "right_on": "service_id"},
            ]
            if remaining >= 1:
                joins.append(join_candidates[0])
                remaining -= 1
            if remaining >= 1 and random.random() < 0.7:
                joins.append(join_candidates[1])
                remaining -= 1
            filter_candidates = [
                {"op": "filter", "column": "level", "operator": "eq", "value": random.choice(["error", "warn"])},
                {"op": "filter", "column": "latency_ms", "operator": "gt", "value": random.randint(50, 300)},
                {"op": "filter", "column": "tier", "operator": "eq", "value": random.choice(["gold", "silver"])},
                {"op": "filter", "column": "region", "operator": "eq", "value": random.choice(["us", "eu", "apac"])},
                {"op": "filter", "column": "severity", "operator": "eq", "value": random.choice(["sev1", "sev2"])},
                {"op": "filter", "column": "status", "operator": "eq", "value": random.choice(["open", "mitigated"])},
            ]
            for cand in filter_candidates:
                if remaining <= 0:
                    break
                col = cand["column"]
                if col in ("tier",) and not any(j["table"] == "services" for j in joins):
                    continue
                if col in ("severity", "status") and not any(j["table"] == "incidents" for j in joins):
                    continue
                filters.append(cand)
                remaining -= 1
            metric = random.choice(["count", "sum", "unique"])
            metric_col = "latency_ms" if metric == "sum" else random.choice(["level", "region", "tier"]) if metric == "unique" else None
        else:
            join_candidates = [
                {"op": "join", "table": "services", "left_on": "service_id", "right_on": "service_id"},
            ]
            if remaining >= 1:
                joins.append(join_candidates[0])
                remaining -= 1
            filter_candidates = [
                {"op": "filter", "column": "severity", "operator": "eq", "value": random.choice(["sev1", "sev2", "sev3"])},
                {"op": "filter", "column": "status", "operator": "eq", "value": random.choice(["open", "mitigated", "closed"])},
                {"op": "filter", "column": "tier", "operator": "eq", "value": random.choice(["gold", "silver", "bronze"])},
                {"op": "filter", "column": "region", "operator": "eq", "value": random.choice(["us", "eu", "apac"])},
            ]
            for cand in filter_candidates:
                if remaining <= 0:
                    break
                col = cand["column"]
                if col in ("tier", "region") and not any(j["table"] == "services" for j in joins):
                    continue
                filters.append(cand)
                remaining -= 1
            metric = random.choice(["count", "unique"])
            metric_col = random.choice(["severity", "status", "tier", "region"]) if metric == "unique" else None
        ops = joins + filters
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
        return rows

    def _compute_ground_truth(self, task: Dict[str, Any]) -> float:
        base_rows = self.tables[task["base_table"]]
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


class LogAtlasStrideEnvWithFeedback(LogAtlasStrideEnv):
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
            hint = "Wrap action in \\boxed{...}. Use tool:open_table(...) or answer:<number>."
        elif "unsupported tool" in text:
            error_type = "UnsupportedAction"
            m = re.search(r"unsupported tool:\\s*([a-z_]+)", text)
            error_detail["tool"] = m.group(1) if m else None
            hint = "Check available tools and use a valid name."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "no active table" in text:
                error_detail["violation"] = "no_active_table"
                hint = "Open a table first with open_table."
            elif "insufficient tool usage" in text:
                error_detail["violation"] = "insufficient_steps"
                hint = "Meet required tool calls before answering."
            else:
                error_detail["violation"] = "other_protocol"
                hint = "Check tool args and state."
        elif "execution error" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "execution_error"
            hint = "Inspect arguments/state; use show_preview to debug."
        elif "wrong answer" in text:
            error_type = "WrongDecision"
            error_detail["expected"] = self.task.get("solution")
            hint = "Verify joins/filters; recompute metric before answering."
        elif "timeout" in text:
            error_type = "Timeout"
            error_detail["outcome"] = "max_turns_reached"
            hint = "Plan and submit before turn limit."
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
            "hint": "Open the base table, then join/filter per task, then aggregate.",
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
