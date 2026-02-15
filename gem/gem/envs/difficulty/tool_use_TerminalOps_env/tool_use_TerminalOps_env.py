from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


def _step_range_overlap(level: int) -> Tuple[int, int]:
    return 2 * level, 2 * level + 2  # level1:2-4 ... level10:20-22


class TerminalOpsEnv(Env):
    """
    终端/运维场景：文件、进程、日志，加动作工具（run/kill/tail/env），步数随复杂度提升（overlap: 2N~2N+2）。
    模拟环境，不执行真实命令。
    """

    def __init__(self, complexity: int = 1, max_turns: Optional[int] = 220, **_):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.max_turns = max_turns if max_turns is not None else 220
        self.min_required_steps, self.max_required_steps = _step_range_overlap(self.complexity)
        self._init_database()
        self.reset()

    def evolve(self, step_success_rate: float, **kwargs) -> int:
        old_complexity = self.complexity
        new_complexity = super().evolve(step_success_rate, **kwargs)
        if new_complexity != old_complexity:
            self.min_required_steps, self.max_required_steps = _step_range_overlap(new_complexity)
            self._init_database()
        return new_complexity

    def set_complexity(self, complexity: int):
        self.complexity = max(1, min(10, int(complexity)))
        self.min_required_steps, self.max_required_steps = _step_range_overlap(self.complexity)
        self._init_database()

    # ---------- 数据与工具 ----------
    def _init_database(self) -> None:
        self.tools: Dict[str, Dict[str, Any]] = {
            "open_table": {"parameters": ["name"], "returns": "Set active table", "usage": "tool:open_table(name=<table>)"},
            "filter_rows": {"parameters": ["column", "op", "value"], "returns": "Filter rows", "usage": "tool:filter_rows(column=<col>, op=eq|gt|lt, value=<val>)"},
            "join_table": {"parameters": ["name", "left_on", "right_on", "how"], "returns": "Join table", "usage": "tool:join_table(name=<table>, left_on=<col>, right_on=<col>, how=inner)"},
            "select_columns": {"parameters": ["names"], "returns": "Project columns", "usage": "tool:select_columns(names=<c1,c2,...>)"},
            "compute_count": {"parameters": ["column"], "returns": "Count rows/non-null", "usage": "tool:compute_count(column=<optional>)"},
            "compute_sum": {"parameters": ["column"], "returns": "Sum numeric", "usage": "tool:compute_sum(column=<col>)"},
            "compute_avg": {"parameters": ["column"], "returns": "Average numeric", "usage": "tool:compute_avg(column=<col>)"},
            "unique_count": {"parameters": ["column"], "returns": "Distinct count", "usage": "tool:unique_count(column=<col>)"},
            "run_cmd": {"parameters": ["cmd"], "returns": "Simulate running a command", "usage": "tool:run_cmd(cmd=<str>)"},
            "kill_process": {"parameters": ["pid"], "returns": "Simulate killing a process", "usage": "tool:kill_process(pid=<int>)"},
            "tail_log": {"parameters": ["service", "lines"], "returns": "Simulate tailing a log", "usage": "tool:tail_log(service=<str>, lines=<int>)"},
            "set_env": {"parameters": ["key", "value"], "returns": "Set env var (simulated)", "usage": "tool:set_env(key=<str>, value=<str>)"},
            "snapshot": {"parameters": ["name"], "returns": "Save snapshot", "usage": "tool:snapshot(name=<snap>)"},
            "reset_state": {"parameters": [], "returns": "Clear state", "usage": "tool:reset_state()"},
            "show_preview": {"parameters": ["n"], "returns": "Preview rows", "usage": "tool:show_preview(n=<int>)"},
        }

        users = ["root", "deploy", "ops", "svc", "data"]
        services = ["api", "worker", "db", "cache", "proxy"]
        levels = ["info", "warn", "error", "critical"]
        file_types = ["log", "cfg", "bin", "tmp", "data"]

        n_files = 22 + self.complexity * 8
        n_procs = 18 + self.complexity * 7
        n_logs = 24 + self.complexity * 9

        self.tables: Dict[str, List[Dict[str, Any]]] = {}
        self.tables["files"] = []
        for fid in range(1, n_files + 1):
            self.tables["files"].append(
                {
                    "path": f"/srv/app/{random.choice(services)}/{random.choice(['run','cfg','logs'])}/f{fid}",
                    "size": random.randint(4, 5000),
                    "owner": random.choice(users),
                    "type": random.choice(file_types),
                    "mtime": random.randint(1, 96),
                }
            )

        self.tables["processes"] = []
        for pid in range(2000, 2000 + n_procs):
            self.tables["processes"].append(
                {
                    "pid": pid,
                    "user": random.choice(users),
                    "service": random.choice(services),
                    "cpu": round(random.uniform(0.1, 95.0), 2),
                    "mem": round(random.uniform(10, 2048), 1),
                    "status": random.choice(["running", "sleep", "zombie", "stopped"]),
                }
            )

        self.tables["logs"] = []
        for lid in range(1, n_logs + 1):
            self.tables["logs"].append(
                {
                    "log_id": lid,
                    "service": random.choice(services),
                    "level": random.choice(levels),
                    "bytes": random.randint(200, 6000),
                    "lines": random.randint(10, 500),
                }
            )

        self.execution_state: Dict[str, Any] = {"env": {}, "commands": [], "kills": [], "tails": []}
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
            "You manage terminal-like ops data (files/processes/logs) and actions (run/kill/tail/env).\n"
            "Use \\boxed{tool:...}; final answer in \\boxed{answer:<number>}.\n"
            "Available tools:\n" + tools_str +
            "\nMeet the minimum tool-call count before answering."
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
            elif op["op"] == "run":
                clauses.append(f"run '{op['cmd']}'")
            elif op["op"] == "kill":
                clauses.append(f"kill pid {op['pid']}")
            elif op["op"] == "tail":
                clauses.append(f"tail {op['service']} {op['lines']} lines")
            elif op["op"] == "env":
                clauses.append(f"env {op['key']}={op['value']}")
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
        self.execution_state = {"env": {}, "commands": [], "kills": [], "tails": []}
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
        col = random.choice(list(self.current_table[0].keys())) if self.current_table else "service"
        return f"\\boxed{{tool:filter_rows(column='{col}', op=eq, value='{random.choice(['api','error','root'])}')}}"

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
            self.execution_state = {"env": {}, "commands": [], "kills": [], "tails": []}
            return "State reset. No active table."
        if name == "show_preview":
            n = int(args.get("n", 5))
            if self.current_table is None:
                raise ValueError("no active table")
            preview = self.current_table[: max(0, n)]
            return f"Preview {len(preview)} rows: {preview}"
        if self.current_table is None and name not in ("run_cmd", "kill_process", "tail_log", "set_env"):
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
        if name == "compute_avg":
            column = args.get("column")
            if column is None:
                raise ValueError("avg requires 'column'")
            vals = []
            for r in self.current_table:
                v = r.get(column)
                try:
                    vals.append(float(v))
                except Exception:
                    pass
            avg = round(sum(vals) / len(vals), 6) if vals else 0.0
            self.execution_state["last_metric"] = avg
            return f"Averaged {column} = {avg}."
        if name == "unique_count":
            column = args.get("column")
            if column is None:
                raise ValueError("unique_count requires 'column'")
            uniq = set()
            for r in self.current_table:
                uniq.add(r.get(column))
            cnt = len(uniq); self.execution_state["last_metric"] = cnt
            return f"Distinct values in {column}: {cnt}."
        if name == "run_cmd":
            cmd = args.get("cmd")
            self.execution_state["commands"].append(cmd)
            return f"Command '{cmd}' simulated."
        if name == "kill_process":
            pid = args.get("pid")
            self.execution_state["kills"].append(pid)
            return f"Process {pid} kill simulated."
        if name == "tail_log":
            svc = args.get("service"); lines = int(args.get("lines", 10))
            self.execution_state["tails"].append({"service": svc, "lines": lines})
            return f"Tailed {svc} for {lines} lines (simulated)."
        if name == "set_env":
            key = args.get("key"); val = args.get("value")
            self.execution_state["env"][key] = val
            return f"Env {key} set to {val}."
        if name == "snapshot":
            snap = args.get("name")
            if not snap:
                raise ValueError("snapshot requires 'name'")
            self.execution_state.setdefault("snapshots", {})[snap] = [dict(r) for r in self.current_table] if self.current_table else []
            return f"Snapshot '{snap}' saved."
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
            else:
                # run/kill/tail/env 不改变表
                continue
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
        if metric == "avg":
            vals = []
            for r in rows:
                v = r.get(col)
                try:
                    vals.append(float(v))
                except Exception:
                    pass
            return round(sum(vals) / len(vals), 6) if vals else 0.0
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
        base_table = random.choice(["files", "processes", "logs"])
        ops: List[Dict[str, Any]] = []
        candidate_filters = []
        candidate_misc = []
        n_procs = len(self.tables.get("processes", []))

        if base_table == "files":
            for _ in range(6):
                candidate_filters.append({"op": "filter", "column": "owner", "operator": "eq", "value": random.choice(["root", "deploy", "ops", "svc", "data"])})
                candidate_filters.append({"op": "filter", "column": "type", "operator": "eq", "value": random.choice(["log", "cfg", "bin", "tmp", "data"])})
                candidate_filters.append({"op": "filter", "column": "size", "operator": random.choice(["gt", "lt"]), "value": random.randint(100, 3000)})
        if base_table == "processes":
            for _ in range(6):
                candidate_filters.append({"op": "filter", "column": "service", "operator": "eq", "value": random.choice(["api", "worker", "db", "cache", "proxy"])})
                candidate_filters.append({"op": "filter", "column": "cpu", "operator": random.choice(["gt", "lt"]), "value": round(random.uniform(5, 60), 2)})
                candidate_filters.append({"op": "filter", "column": "mem", "operator": random.choice(["gt", "lt"]), "value": round(random.uniform(50, 800), 1)})
                candidate_filters.append({"op": "filter", "column": "status", "operator": "eq", "value": random.choice(["running", "sleep", "zombie", "stopped"])})
        if base_table == "logs":
            for _ in range(6):
                candidate_filters.append({"op": "filter", "column": "service", "operator": "eq", "value": random.choice(["api", "worker", "db", "cache", "proxy"])})
                candidate_filters.append({"op": "filter", "column": "level", "operator": "eq", "value": random.choice(["info", "warn", "error", "critical"])})
                candidate_filters.append({"op": "filter", "column": "bytes", "operator": random.choice(["gt", "lt"]), "value": random.randint(500, 5000)})

        for _ in range(25):
            col = random.choice(["owner", "type", "size", "mtime", "service", "cpu", "mem", "status", "level", "bytes", "lines"])
            if col in ["owner", "type", "service", "status", "level"]:
                val = random.choice(["root", "deploy", "ops", "svc", "data", "log", "cfg", "bin", "tmp", "data", "api", "worker", "db", "cache", "proxy", "running", "sleep", "zombie", "stopped", "info", "warn", "error", "critical"])
                candidate_filters.append({"op": "filter", "column": col, "operator": "eq", "value": val})
            else:
                op = random.choice(["gt", "lt"])
                bound = round(random.uniform(1, 4000), 2)
                candidate_filters.append({"op": "filter", "column": col, "operator": op, "value": bound})

        for _ in range(25):
            candidate_misc.append({"op": "run", "cmd": random.choice(["ls -l", "df -h", "ps aux", "cat /etc/hosts", "tail -n 20 /var/log/syslog"])})
            candidate_misc.append({"op": "kill", "pid": random.randint(2000, 2000 + n_procs - 1)})
            candidate_misc.append({"op": "tail", "service": random.choice(["api", "worker", "db", "cache", "proxy"]), "lines": random.randint(5, 50)})
            candidate_misc.append({"op": "env", "key": random.choice(["ENV", "REGION", "MODE"]), "value": random.choice(["prod", "staging", "test"])})

        candidate_joins = [
            {"op": "join", "table": "processes", "left_on": "service", "right_on": "service"},
            {"op": "join", "table": "logs", "left_on": "service", "right_on": "service"},
            {"op": "join", "table": "files", "left_on": "owner", "right_on": "owner"},
        ]

        random.shuffle(candidate_filters)
        random.shuffle(candidate_joins)
        random.shuffle(candidate_misc)

        while len(ops) < steps - 1:
            choice = random.random()
            if candidate_joins and choice < 0.3:
                ops.append(candidate_joins.pop())
            elif candidate_misc and choice < 0.55:
                ops.append(candidate_misc.pop())
            elif candidate_filters:
                ops.append(candidate_filters.pop())
            else:
                break

        metric = random.choice(["count", "sum", "unique", "avg"])
        metric_col = None
        if metric == "sum":
            metric_col = random.choice(["size", "mtime", "cpu", "mem", "bytes", "lines"])
        if metric == "unique":
            metric_col = random.choice(["owner", "type", "service", "status", "level"])
        if metric == "avg":
            metric_col = random.choice(["size", "cpu", "mem", "bytes", "lines"])

        return {"base_table": base_table, "ops": ops, "metric": metric, "metric_column": metric_col}


class TerminalOpsEnvWithFeedback(TerminalOpsEnv):
    def __init__(self, complexity: int = 1, feedback_level: int = 2, **kwargs):
        super().__init__(complexity=complexity, **kwargs)
        self.feedback_level = feedback_level

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        info["tool_use_counter"] = getattr(self, "steps_taken", 0)
        info["prev_ep_tool_use_counter"] = getattr(self, "steps_taken", 0) if terminated or truncated else 0
        return obs, reward, terminated, truncated, info
