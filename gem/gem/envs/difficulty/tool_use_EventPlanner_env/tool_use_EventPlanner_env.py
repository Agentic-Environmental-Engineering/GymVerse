from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


def _step_range_overlap(level: int) -> Tuple[int, int]:
    return 2 * level, 2 * level + 2  # level1:2-4 ... level10:20-22


class EventPlannerEnv(Env):
    """
    事件/会议策划场景：场地、供应商、日程、与会者，加上动作工具（订场地/雇佣/排期/群发），步数随复杂度提升（overlap: 2N~2N+2）。
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
            "book_venue": {"parameters": ["venue_id", "days"], "returns": "Book a venue", "usage": "tool:book_venue(venue_id=<id>, days=<int>)"},
            "hire_vendor": {"parameters": ["vendor_id", "service"], "returns": "Hire vendor for service", "usage": "tool:hire_vendor(vendor_id=<id>, service=<str>)"},
            "schedule_session": {"parameters": ["session_id", "slot"], "returns": "Schedule a session", "usage": "tool:schedule_session(session_id=<id>, slot=<slot>)"},
            "send_invites": {"parameters": ["group"], "returns": "Send invites to group", "usage": "tool:send_invites(group=<str>)"},
            "snapshot": {"parameters": ["name"], "returns": "Save snapshot", "usage": "tool:snapshot(name=<snap>)"},
            "reset_state": {"parameters": [], "returns": "Clear state", "usage": "tool:reset_state()"},
            "show_preview": {"parameters": ["n"], "returns": "Preview rows", "usage": "tool:show_preview(n=<int>)"},
        }

        cities = ["NYC", "SFO", "LA", "SEA", "CHI", "AUS", "BOS", "ATL"]
        types = ["conf", "expo", "meetup", "workshop"]
        services = ["catering", "av", "security", "staff", "design"]
        slots = ["morning", "noon", "afternoon", "evening"]
        groups = ["vip", "speakers", "sponsors", "attendees"]

        n_venues = 18 + self.complexity * 6
        n_vendors = 20 + self.complexity * 7
        n_sessions = 22 + self.complexity * 8
        n_attendees = 30 + self.complexity * 10

        self.tables: Dict[str, List[Dict[str, Any]]] = {}
        self.tables["venues"] = []
        for vid in range(1, n_venues + 1):
            self.tables["venues"].append(
                {
                    "venue_id": vid,
                    "city": random.choice(cities),
                    "capacity": random.randint(80, 2000),
                    "price": random.randint(800, 12000),
                    "type": random.choice(types),
                    "rating": round(random.uniform(3.0, 5.0), 2),
                }
            )

        self.tables["vendors"] = []
        for vd in range(1, n_vendors + 1):
            self.tables["vendors"].append(
                {
                    "vendor_id": vd,
                    "service": random.choice(services),
                    "city": random.choice(cities),
                    "cost": random.randint(500, 8000),
                    "score": round(random.uniform(2.5, 5.0), 2),
                }
            )

        self.tables["sessions"] = []
        for sid in range(1, n_sessions + 1):
            self.tables["sessions"].append(
                {
                    "session_id": sid,
                    "topic": random.choice(["ai", "web", "ops", "design", "product", "data"]),
                    "speaker": f"spk{random.randint(1, 50)}",
                    "duration": random.randint(20, 120),
                    "slot": random.choice(slots),
                    "track": random.choice(["main", "sideA", "sideB"]),
                }
            )

        self.tables["attendees"] = []
        for aid in range(1, n_attendees + 1):
            self.tables["attendees"].append(
                {
                    "attendee_id": aid,
                    "city": random.choice(cities),
                    "ticket": random.choice(["std", "vip", "student", "sponsor"]),
                    "checked_in": random.choice([True, False]),
                    "group": random.choice(groups),
                }
            )

        self.execution_state: Dict[str, Any] = {"bookings": [], "hires": [], "sessions": {}, "invites": []}
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
            "You plan events: venues, vendors, sessions, attendees, plus booking/scheduling/invite actions.\n"
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
            elif op["op"] == "book":
                clauses.append(f"book venue {op['venue_id']} days {op['days']}")
            elif op["op"] == "hire":
                clauses.append(f"hire vendor {op['vendor_id']} for {op['service']}")
            elif op["op"] == "sched":
                clauses.append(f"schedule session {op['session_id']} slot {op['slot']}")
            elif op["op"] == "invite":
                clauses.append(f"invite group {op['group']}")
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
        self.execution_state = {"bookings": [], "hires": [], "sessions": {}, "invites": []}
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
        col = random.choice(list(self.current_table[0].keys())) if self.current_table else "venue_id"
        return f"\\boxed{{tool:filter_rows(column='{col}', op=eq, value='{random.choice(['NYC','catering','vip'])}')}}"

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
            self.execution_state = {"bookings": [], "hires": [], "sessions": {}, "invites": []}
            return "State reset. No active table."
        if name == "show_preview":
            n = int(args.get("n", 5))
            if self.current_table is None:
                raise ValueError("no active table")
            preview = self.current_table[: max(0, n)]
            return f"Preview {len(preview)} rows: {preview}"
        if self.current_table is None and name not in ("book_venue", "hire_vendor", "schedule_session", "send_invites"):
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
        if name == "book_venue":
            vid = args.get("venue_id"); days = int(args.get("days", 1))
            self.execution_state["bookings"].append({"venue_id": vid, "days": days})
            return f"Venue {vid} booked for {days} days."
        if name == "hire_vendor":
            vd = args.get("vendor_id"); service = args.get("service")
            self.execution_state["hires"].append({"vendor_id": vd, "service": service})
            return f"Vendor {vd} hired for {service}."
        if name == "schedule_session":
            sid = args.get("session_id"); slot = args.get("slot")
            self.execution_state["sessions"][sid] = slot
            return f"Session {sid} scheduled at {slot}."
        if name == "send_invites":
            group = args.get("group")
            self.execution_state["invites"].append(group)
            return f"Invites sent to {group}."
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
                # booking/hire/schedule/invite 不改变表
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
        base_table = random.choice(["venues", "vendors", "sessions", "attendees"])
        ops: List[Dict[str, Any]] = []
        candidate_filters = []
        candidate_misc = []
        n_sessions = len(self.tables.get("sessions", []))
        n_vendors = len(self.tables.get("vendors", []))
        n_venues = len(self.tables.get("venues", []))

        if base_table == "venues":
            for _ in range(6):
                candidate_filters.append({"op": "filter", "column": "city", "operator": "eq", "value": random.choice(["NYC", "SFO", "LA", "SEA", "CHI", "AUS", "BOS", "ATL"])})
                candidate_filters.append({"op": "filter", "column": "capacity", "operator": random.choice(["gt", "lt"]), "value": random.randint(150, 1500)})
                candidate_filters.append({"op": "filter", "column": "price", "operator": random.choice(["gt", "lt"]), "value": random.randint(1000, 9000)})
                candidate_filters.append({"op": "filter", "column": "type", "operator": "eq", "value": random.choice(["conf", "expo", "meetup", "workshop"])})
        if base_table == "vendors":
            for _ in range(6):
                candidate_filters.append({"op": "filter", "column": "service", "operator": "eq", "value": random.choice(["catering", "av", "security", "staff", "design"])})
                candidate_filters.append({"op": "filter", "column": "city", "operator": "eq", "value": random.choice(["NYC", "SFO", "LA", "SEA", "CHI", "AUS", "BOS", "ATL"])})
                candidate_filters.append({"op": "filter", "column": "cost", "operator": random.choice(["gt", "lt"]), "value": random.randint(800, 6000)})
                candidate_filters.append({"op": "filter", "column": "score", "operator": random.choice(["gt", "lt"]), "value": round(random.uniform(3.0, 4.7), 2)})
        if base_table == "sessions":
            for _ in range(6):
                candidate_filters.append({"op": "filter", "column": "topic", "operator": "eq", "value": random.choice(["ai", "web", "ops", "design", "product", "data"])})
                candidate_filters.append({"op": "filter", "column": "slot", "operator": "eq", "value": random.choice(["morning", "noon", "afternoon", "evening"])})
                candidate_filters.append({"op": "filter", "column": "duration", "operator": random.choice(["gt", "lt"]), "value": random.randint(30, 100)})
        if base_table == "attendees":
            for _ in range(6):
                candidate_filters.append({"op": "filter", "column": "city", "operator": "eq", "value": random.choice(["NYC", "SFO", "LA", "SEA", "CHI", "AUS", "BOS", "ATL"])})
                candidate_filters.append({"op": "filter", "column": "ticket", "operator": "eq", "value": random.choice(["std", "vip", "student", "sponsor"])})
                candidate_filters.append({"op": "filter", "column": "checked_in", "operator": "eq", "value": random.choice([True, False])})
                candidate_filters.append({"op": "filter", "column": "group", "operator": "eq", "value": random.choice(["vip", "speakers", "sponsors", "attendees"])})

        for _ in range(25):
            col = random.choice(["city", "capacity", "price", "type", "service", "cost", "score", "topic", "slot", "duration", "ticket", "checked_in", "group"])
            if col in ["city", "type", "service", "topic", "slot", "ticket", "group"]:
                val = random.choice(["NYC", "SFO", "LA", "SEA", "CHI", "AUS", "BOS", "ATL", "conf", "expo", "meetup", "workshop", "catering", "av", "security", "staff", "design", "ai", "web", "ops", "design", "product", "data", "morning", "noon", "afternoon", "evening", "std", "vip", "student", "sponsor", "vip", "speakers", "sponsors", "attendees"])
                candidate_filters.append({"op": "filter", "column": col, "operator": "eq", "value": val})
            elif col == "checked_in":
                candidate_filters.append({"op": "filter", "column": col, "operator": "eq", "value": random.choice([True, False])})
            else:
                op = random.choice(["gt", "lt"])
                bound = random.randint(1, 12000)
                candidate_filters.append({"op": "filter", "column": col, "operator": op, "value": bound})

        for _ in range(25):
            candidate_misc.append({"op": "book", "venue_id": random.randint(1, n_venues), "days": random.randint(1, 5)})
            candidate_misc.append({"op": "hire", "vendor_id": random.randint(1, n_vendors), "service": random.choice(["catering", "av", "security", "staff", "design"])})
            candidate_misc.append({"op": "sched", "session_id": random.randint(1, n_sessions), "slot": random.choice(["morning", "noon", "afternoon", "evening"])})
            candidate_misc.append({"op": "invite", "group": random.choice(["vip", "speakers", "sponsors", "attendees"])})

        candidate_joins = [
            {"op": "join", "table": "venues", "left_on": "venue_id", "right_on": "venue_id"},
            {"op": "join", "table": "vendors", "left_on": "vendor_id", "right_on": "vendor_id"},
            {"op": "join", "table": "sessions", "left_on": "session_id", "right_on": "session_id"},
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
            metric_col = random.choice(["price", "cost", "capacity", "duration"])
        if metric == "unique":
            metric_col = random.choice(["city", "type", "service", "topic", "slot", "ticket", "group"])
        if metric == "avg":
            metric_col = random.choice(["price", "cost", "score", "rating", "duration"])

        return {"base_table": base_table, "ops": ops, "metric": metric, "metric_column": metric_col}


class EventPlannerEnvWithFeedback(EventPlannerEnv):
    def __init__(self, complexity: int = 1, feedback_level: int = 2, **kwargs):
        super().__init__(complexity=complexity, **kwargs)
        self.feedback_level = feedback_level

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        info["tool_use_counter"] = getattr(self, "steps_taken", 0)
        info["prev_ep_tool_use_counter"] = getattr(self, "steps_taken", 0) if terminated or truncated else 0
        return obs, reward, terminated, truncated, info
