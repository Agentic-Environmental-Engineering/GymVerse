from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List

class WorkflowAnvilEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        max_turns: Optional[int] = 30,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.max_turns = max_turns if max_turns is not None else 30

        self.min_required_steps = self.complexity
        self.max_required_steps = self.complexity * 2

        self.tools: Dict[str, Dict[str, Any]] = {}
        self.workspace: Dict[str, Any] = {}
        self.execution_state: Dict[str, Any] = {}
        self.task: Dict[str, Any] = {}
        self.turn_count = 0
        self.steps_taken = 0

        self._init_database()
        self.reset()

    def _init_database(self):
        self.tools = {}
        base_tools = [
            {
                "name": "load_table",
                "description": "Load a CSV-like table from workspace into active state",
                "parameters": [{"name": "file", "type": "string"}],
                "returns": "int",
            },
            {
                "name": "filter_rows",
                "description": "Filter active table by condition",
                "parameters": [
                    {"name": "column", "type": "string"},
                    {"name": "op", "type": "string"},  # ==,!=,>,<,>=,<=,contains,startswith,endswith
                    {"name": "value", "type": "string"},
                ],
                "returns": "int",
            },
            {
                "name": "select_columns",
                "description": "Keep only specified columns in active table",
                "parameters": [{"name": "columns", "type": "string"}],  # comma-separated
                "returns": "int",
            },
            {
                "name": "sort_by",
                "description": "Sort active table by a column",
                "parameters": [
                    {"name": "column", "type": "string"},
                    {"name": "order", "type": "string"},  # asc or desc
                ],
                "returns": "int",
            },
            {
                "name": "compute_aggregate",
                "description": "Compute an aggregate over a column (sum,avg,min,max)",
                "parameters": [
                    {"name": "func", "type": "string"},
                    {"name": "column", "type": "string"},
                ],
                "returns": "float",
            },
            {
                "name": "save_table",
                "description": "Save active table back into workspace",
                "parameters": [{"name": "file", "type": "string"}],
                "returns": "int",
            },
            {
                "name": "read_text",
                "description": "Load a text file content into active text",
                "parameters": [{"name": "file", "type": "string"}],
                "returns": "int",
            },
            {
                "name": "grep",
                "description": "Find lines in a text file matching a pattern (case-insensitive substring)",
                "parameters": [{"name": "file", "type": "string"}, {"name": "pattern", "type": "string"}],
                "returns": "list",
            },
            {
                "name": "summarize_table",
                "description": "Produce a simple summary string of top-K rows using current table and store in active text",
                "parameters": [
                    {"name": "top_k", "type": "int"},
                    {"name": "sort_by", "type": "string"},
                    {"name": "order", "type": "string"},
                ],
                "returns": "string",
            },
            {
                "name": "render_report",
                "description": "Render the report file using the current table, aggregate, and optional grep results",
                "parameters": [
                    {"name": "file", "type": "string"},
                    {"name": "top_k", "type": "int"},
                    {"name": "sort_by", "type": "string"},
                    {"name": "order", "type": "string"},
                ],
                "returns": "string",
            },
        ]
        advanced_tools = [
            {
                "name": "join_tables",
                "description": "Inner join two workspace tables on a key column",
                "parameters": [
                    {"name": "left", "type": "string"},
                    {"name": "right", "type": "string"},
                    {"name": "on", "type": "string"},
                ],
                "returns": "int",
            },
            {
                "name": "group_by_aggregate",
                "description": "Group by a column and aggregate another column",
                "parameters": [
                    {"name": "by", "type": "string"},
                    {"name": "agg_func", "type": "string"},
                    {"name": "target", "type": "string"},
                ],
                "returns": "int",
            },
        ]
        # Scale tools with complexity
        available = base_tools.copy()
        if self.complexity >= 6:
            available += [advanced_tools[0]]
        if self.complexity >= 8:
            available += [advanced_tools[1]]
        for t in available:
            self.tools[t["name"]] = t

        # Build workspace datasets
        def make_inventory(n: int) -> List[Dict[str, Any]]:
            cats = ["Hardware", "Gadget", "Tool", "Accessory", "Material"]
            rows = []
            for i in range(n):
                rows.append({
                    "id": i + 1,
                    "sku": f"SKU{i+1:03d}",
                    "category": random.choice(cats),
                    "price": round(random.uniform(5, 500), 2),
                    "stock": random.randint(0, 500),
                    "rating": round(random.uniform(1.0, 5.0), 2),
                })
            return rows

        def make_sales(n: int) -> List[Dict[str, Any]]:
            regions = ["North", "South", "East", "West"]
            prods = ["Hammer", "Drill", "Saw", "Wrench", "Tape"]
            rows = []
            for i in range(n):
                units = random.randint(1, 50)
                price = random.uniform(10, 300)
                revenue = round(units * price, 2)
                rows.append({
                    "id": i + 1,
                    "region": random.choice(regions),
                    "product": random.choice(prods),
                    "units": units,
                    "revenue": revenue,
                })
            return rows

        inv_count = 20 + (self.complexity - 1) * 5  # grows with complexity
        sales_count = 20 + (self.complexity - 1) * 5

        self.workspace = {}
        self.workspace["inventory.csv"] = make_inventory(inv_count)
        self.workspace["sales.csv"] = make_sales(sales_count)

        # Notes file with keywords
        keywords = ["alpha", "beta", "gamma", "delta", "epsilon", "theta", "omega", "spire", "anvil"]
        chosen_kw = random.choice(keywords)
        noise_kw = [kw for kw in keywords if kw != chosen_kw]
        lines = []
        for _ in range(10 + self.complexity):
            if random.random() < 0.4:
                lines.append(f"This line mentions {random.choice(noise_kw)} tool hints.")
            else:
                lines.append(f"Remember the {chosen_kw} protocol for reports.")
        self.workspace["notes.txt"] = "\n".join(lines)
        self.workspace["_keywords"] = {"primary": chosen_kw}

        self.execution_state = {
            "table": None,
            "table_source": None,
            "aggregate": None,
            "aggregate_meta": None,
            "grep_matches": None,
            "grep_meta": None,
            "last_report_meta": None,
            "report_ready": False,
        }

    def _generate_task_requiring_n_steps(self, required_steps: int) -> Dict[str, Any]:
        dataset = random.choice(["inventory.csv", "sales.csv"])
        if dataset == "inventory.csv":
            filter_type = random.choice(["category", "price_min"])
            if filter_type == "category":
                # pick a category present
                categories = list({row["category"] for row in self.workspace[dataset]})
                value = random.choice(categories)
                filter_cond = {"column": "category", "op": "==", "value": value}
            else:
                value = round(random.uniform(50, 200), 2)
                filter_cond = {"column": "price", "op": ">=", "value": value}
            sort_by = random.choice(["price", "rating", "stock"])
            agg_func = random.choice(["sum", "avg", "max"])
            agg_col = random.choice(["price", "stock"])
        else:
            filter_type = "region"
            regions = list({row["region"] for row in self.workspace[dataset]})
            value = random.choice(regions)
            filter_cond = {"column": "region", "op": "==", "value": value}
            sort_by = random.choice(["revenue", "units"])
            agg_func = random.choice(["sum", "avg", "max"])
            agg_col = random.choice(["revenue", "units"])

        order = random.choice(["asc", "desc"])
        top_k = random.randint(3, min(6, 3 + self.complexity))
        requires_grep = self.complexity >= 5
        grep_pattern = self.workspace["_keywords"]["primary"] if requires_grep else None

        task = {
            "dataset": dataset,
            "filter": filter_cond,
            "sort_by": sort_by,
            "order": order,
            "top_k": top_k,
            "aggregate_func": agg_func,
            "aggregate_column": agg_col,
            "requires_grep": requires_grep,
            "grep_pattern": grep_pattern,
            "required_steps": required_steps,
            "target_report": "report.txt",
        }
        return task

    def _get_instructions(self) -> str:
        tools_list = ", ".join(sorted(self.tools.keys()))
        fc = self.task["filter"]
        instr = []
        instr.append("You are orchestrating tools over a workspace to produce a validated report.")
        instr.append(f"Goal: Use tools to prepare data and render a report file '{self.task['target_report']}' that matches the instance parameters.")
        instr.append(f"Dataset: {self.task['dataset']}")
        instr.append(f"Filter: {fc['column']} {fc['op']} {fc['value']}")
        instr.append(f"Sort: {self.task['sort_by']} ({self.task['order']}); Top-K: {self.task['top_k']}")
        instr.append(f"Aggregate: {self.task['aggregate_func']} of {self.task['aggregate_column']}")
        if self.task["requires_grep"]:
            instr.append(f"Notes require grep match containing: '{self.task['grep_pattern']}' from notes.txt")
        instr.append(f"Available tools: {tools_list}")
        instr.append("Protocol:")
        instr.append(f"- Execute at least {self.task['required_steps']} tool calls before submitting.")
        instr.append("- Typical flow: load_table -> filter_rows -> sort_by -> compute_aggregate -> (optional: grep) -> render_report -> submit.")
        instr.append("Action format: Use \\boxed{...} with either a tool call or submission.")
        instr.append("Examples:")
        instr.append("\\boxed{tool: load_table file=inventory.csv}")
        instr.append("\\boxed{tool: filter_rows column=category op== value=Hardware}")
        instr.append("\\boxed{tool: sort_by column=price order=desc}")
        instr.append("\\boxed{tool: compute_aggregate func=sum column=price}")
        instr.append("\\boxed{tool: render_report file=report.txt top_k=5 sort_by=price order=desc}")
        instr.append("\\boxed{submit: file=report.txt}")
        return "\n".join(instr)

    def get_task_suffix(self) -> str:
        tools_list = ", ".join(sorted(self.tools.keys()))
        suffix = []
        suffix.append(f"Steps: {self.steps_taken}/{self.task['required_steps']}")
        suffix.append(f"Workspace files: {', '.join(sorted([k for k in self.workspace.keys() if not k.startswith('_')]))}")
        suffix.append(f"Tools: {tools_list}")
        suffix.append("Respond using \\boxed{tool: ...} or \\boxed{submit: file=...}.")
        return " | ".join(suffix)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        required_steps = random.randint(self.min_required_steps, self.max_required_steps)
        self.task = self._generate_task_requiring_n_steps(required_steps)
        self.turn_count = 0
        self.steps_taken = 0
        self.execution_state = {
            "table": None,
            "table_source": None,
            "aggregate": None,
            "aggregate_meta": None,
            "grep_matches": None,
            "grep_meta": None,
            "last_report_meta": None,
            "report_ready": False,
        }
        obs = self._get_instructions()
        info = {"suffix": self.get_task_suffix()}
        return obs, info

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if action is None:
            return None
        m = re.search(r"\\boxed\{(.*)\}", action, flags=re.DOTALL)
        if not m:
            return None
        inner = m.group(1).strip()
        # pattern: "tool: NAME key=val key=val" or "submit: file=report.txt"
        if ":" not in inner:
            return None
        head, rest = inner.split(":", 1)
        head = head.strip().lower()
        rest = rest.strip()
        if head == "tool":
            parts = re.split(r"[;\s]+", rest)
            if len(parts) == 0:
                return None
            tool_name = parts[0].strip()
            args = {}
            for token in parts[1:]:
                if "=" in token:
                    k, v = token.split("=", 1)
                    args[k.strip()] = v.strip()
                elif "==" in token:
                    k, v = token.split("==", 1)
                    args[k.strip()] = v.strip()
            return {"type": "tool", "name": tool_name, "args": args}
        elif head == "submit":
            args = {}
            tokens = re.split(r"[;\s]+", rest)
            for tok in tokens:
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    args[k.strip()] = v.strip()
            return {"type": "submit", "args": args}
        else:
            return None

    def _coerce_value(self, v: str):
        if re.fullmatch(r"-?\d+", v):
            return int(v)
        if re.fullmatch(r"-?\d+\.\d+", v):
            return float(v)
        return v

    def _apply_filter(self, rows: List[Dict[str, Any]], column: str, op: str, value: Any) -> List[Dict[str, Any]]:
        def match(x):
            xv = x.get(column)
            if op == "==":
                return xv == value
            if op == "!=":
                return xv != value
            try:
                if op == ">=":
                    return xv >= value
                if op == "<=":
                    return xv <= value
                if op == ">":
                    return xv > value
                if op == "<":
                    return xv < value
            except TypeError:
                pass
            if isinstance(xv, str):
                s = str(xv)
                vs = str(value)
                if op == "contains":
                    return vs.lower() in s.lower()
                if op == "startswith":
                    return s.lower().startswith(vs.lower())
                if op == "endswith":
                    return s.lower().endswith(vs.lower())
            return False
        return [r for r in rows if match(r)]

    def _sort_rows(self, rows: List[Dict[str, Any]], column: str, order: str) -> List[Dict[str, Any]]:
        reverse = True if order.lower() == "desc" else False
        return sorted(rows, key=lambda r: r.get(column, 0), reverse=reverse)

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        if name == "load_table":
            file = args.get("file")
            if file not in self.workspace or not isinstance(self.workspace[file], list):
                raise ValueError(f"Table not found: {file}")
            self.execution_state["table"] = [dict(row) for row in self.workspace[file]]
            self.execution_state["table_source"] = file
            return f"Loaded {len(self.execution_state['table'])} rows from {file}"

        if name == "filter_rows":
            tbl = self.execution_state.get("table")
            if tbl is None:
                raise ValueError("No active table. Use load_table first.")
            column = args.get("column")
            op = args.get("op") or args.get("op==") or "=="
            value = self._coerce_value(args.get("value") if args.get("value") is not None else "")
            filtered = self._apply_filter(tbl, column, op, value)
            self.execution_state["table"] = filtered
            return f"Filtered table to {len(filtered)} rows by {column} {op} {value}"

        if name == "select_columns":
            tbl = self.execution_state.get("table")
            if tbl is None:
                raise ValueError("No active table. Use load_table first.")
            cols = args.get("columns", "")
            cols_list = [c.strip() for c in cols.split(",") if c.strip()]
            new_tbl = [{c: row.get(c) for c in cols_list} for row in tbl]
            self.execution_state["table"] = new_tbl
            return f"Selected columns: {', '.join(cols_list)} (rows: {len(new_tbl)})"

        if name == "sort_by":
            tbl = self.execution_state.get("table")
            if tbl is None:
                raise ValueError("No active table. Use load_table first.")
            column = args.get("column")
            order = args.get("order", "asc").lower()
            sorted_tbl = self._sort_rows(tbl, column, order)
            self.execution_state["table"] = sorted_tbl
            return f"Sorted by {column} {order}. Top row value: {sorted_tbl[0].get(column) if sorted_tbl else 'n/a'}"

        if name == "compute_aggregate":
            tbl = self.execution_state.get("table")
            if tbl is None:
                raise ValueError("No active table. Use load_table first.")
            func = args.get("func", "sum").lower()
            column = args.get("column")
            vals = [r.get(column) for r in tbl if isinstance(r.get(column), (int, float))]
            if not vals:
                raise ValueError(f"No numeric values for {column}")
            if func == "sum":
                agg = float(sum(vals))
            elif func == "avg":
                agg = float(sum(vals) / len(vals))
            elif func == "max":
                agg = float(max(vals))
            elif func == "min":
                agg = float(min(vals))
            else:
                raise ValueError(f"Unsupported aggregate func: {func}")
            self.execution_state["aggregate"] = agg
            self.execution_state["aggregate_meta"] = {"func": func, "column": column}
            return f"Aggregate {func}({column}) = {round(agg, 2)}"

        if name == "save_table":
            tbl = self.execution_state.get("table")
            if tbl is None:
                raise ValueError("No active table. Use load_table first.")
            file = args.get("file")
            self.workspace[file] = [dict(row) for row in tbl]
            return f"Saved table to {file} (rows: {len(tbl)})"

        if name == "read_text":
            file = args.get("file")
            content = self.workspace.get(file)
            if content is None or not isinstance(content, str):
                raise ValueError(f"Text file not found: {file}")
            self.execution_state["text"] = content
            return f"Read text file {file} ({len(content.splitlines())} lines)"

        if name == "grep":
            file = args.get("file")
            pattern = args.get("pattern", "")
            content = self.workspace.get(file)
            if content is None or not isinstance(content, str):
                raise ValueError(f"Text file not found: {file}")
            lines = content.splitlines()
            matches = [ln for ln in lines if pattern.lower() in ln.lower()]
            self.execution_state["grep_matches"] = matches
            self.execution_state["grep_meta"] = {"file": file, "pattern": pattern, "count": len(matches)}
            return f"Grep found {len(matches)} matching lines for pattern '{pattern}' in {file}"

        if name == "summarize_table":
            tbl = self.execution_state.get("table")
            if tbl is None:
                raise ValueError("No active table. Use load_table first.")
            top_k = int(self._coerce_value(args.get("top_k", "3")))
            sort_by = args.get("sort_by")
            order = args.get("order", "desc").lower()
            sorted_tbl = self._sort_rows(tbl, sort_by, order)
            top_rows = sorted_tbl[:top_k]
            summary_lines = [f"id={r.get('id','?')} {sort_by}={r.get(sort_by)}" for r in top_rows]
            summary = "SUMMARY:\n" + "\n".join(summary_lines)
            self.execution_state["text"] = summary
            self.execution_state["top_k_info"] = {"sort_by": sort_by, "order": order, "top_k": top_k}
            return f"Created summary for top {top_k} by {sort_by} {order}"

        if name == "render_report":
            file = args.get("file", "report.txt")
            top_k = int(self._coerce_value(args.get("top_k", str(self.task["top_k"]))))
            sort_by = args.get("sort_by", self.task["sort_by"])
            order = args.get("order", self.task["order"]).lower()

            # prerequisites
            tbl = self.execution_state.get("table")
            agg_meta = self.execution_state.get("aggregate_meta")
            agg_val = self.execution_state.get("aggregate")
            grep_meta = self.execution_state.get("grep_meta")
            if tbl is None:
                raise ValueError("No active table for report.")
            if agg_meta is None or agg_val is None:
                raise ValueError("Aggregate missing. Use compute_aggregate first.")
            if self.task["requires_grep"] and (grep_meta is None or grep_meta.get("count", 0) == 0):
                raise ValueError("Grep prerequisite missing or empty for notes.")

            # Check table matches task filter and sort
            base_tbl = self.workspace[self.task["dataset"]]
            fc = self.task["filter"]
            expected_filtered = self._apply_filter(base_tbl, fc["column"], fc["op"], fc["value"])
            expected_sorted = self._sort_rows(expected_filtered, self.task["sort_by"], self.task["order"])
            expected_top = expected_sorted[:self.task["top_k"]]

            # Compare current active table's top against what the task expects (using provided sort/order/top_k)
            current_sorted = self._sort_rows(tbl, sort_by, order)
            current_top = current_sorted[:top_k]

            top_ok = True
            if len(current_top) != len(expected_top):
                top_ok = False
            else:
                for a, b in zip(current_top, expected_top):
                    if a.get("id") != b.get("id"):
                        top_ok = False
                        break

            agg_ok = (agg_meta["func"] == self.task["aggregate_func"] and agg_meta["column"] == self.task["aggregate_column"])
            grep_ok = True
            if self.task["requires_grep"]:
                grep_ok = grep_meta["pattern"].lower() == self.task["grep_pattern"].lower() and grep_meta["count"] > 0

            report_lines = []
            report_lines.append(f"REPORT: {self.task['dataset']}")
            report_lines.append(f"FILTER: {fc['column']} {fc['op']} {fc['value']}")
            report_lines.append(f"TOPK: {top_k} BY {sort_by} {order}")
            report_lines.append(f"AGGREGATE: {agg_meta['func']} {agg_meta['column']} = {round(agg_val, 2)}")
            if self.task["requires_grep"]:
                report_lines.append(f"NOTES_PATTERN: {grep_meta['pattern']} (matches={grep_meta['count']})")
            for r in current_top:
                report_lines.append(f"ROW id={r.get('id')} {sort_by}={r.get(sort_by)}")
            content = "\n".join(report_lines)
            self.workspace[file] = content

            ready = bool(top_ok and agg_ok and grep_ok)
            self.execution_state["last_report_meta"] = {
                "file": file,
                "top_k": top_k,
                "sort_by": sort_by,
                "order": order,
                "top_ok": top_ok,
                "agg_ok": agg_ok,
                "grep_ok": grep_ok,
            }
            self.execution_state["report_ready"] = ready
            return f"Rendered report to {file}. Checks -> top_ok={top_ok}, agg_ok={agg_ok}, grep_ok={grep_ok}"

        if name == "join_tables":
            left = args.get("left")
            right = args.get("right")
            on = args.get("on")
            left_tbl = self.workspace.get(left)
            right_tbl = self.workspace.get(right)
            if not isinstance(left_tbl, list) or not isinstance(right_tbl, list):
                raise ValueError("Both join inputs must be tables.")
            # build hash for right
            index = {}
            for r in right_tbl:
                key = r.get(on)
                if key is not None:
                    index.setdefault(key, []).append(r)
            joined = []
            for l in left_tbl:
                key = l.get(on)
                if key in index:
                    for r in index[key]:
                        joined.append({**l, **r})
            self.execution_state["table"] = joined
            self.execution_state["table_source"] = f"{left}+{right}"
            return f"Joined {left} and {right} on {on} -> rows: {len(joined)}"

        if name == "group_by_aggregate":
            tbl = self.execution_state.get("table")
            if tbl is None:
                raise ValueError("No active table. Use load_table first.")
            by = args.get("by")
            func = args.get("agg_func", "sum").lower()
            target = args.get("target")
            groups: Dict[Any, List[float]] = {}
            for row in tbl:
                k = row.get(by)
                v = row.get(target)
                if isinstance(v, (int, float)):
                    groups.setdefault(k, []).append(v)
            result = []
            for k, vs in groups.items():
                if func == "sum":
                    val = sum(vs)
                elif func == "avg":
                    val = sum(vs) / len(vs)
                elif func == "max":
                    val = max(vs)
                elif func == "min":
                    val = min(vs)
                else:
                    raise ValueError(f"Unsupported agg func: {func}")
                result.append({by: k, f"{func}_{target}": round(val, 2)})
            self.execution_state["table"] = result
            return f"Grouped by {by} with {func}({target}) -> groups: {len(result)}"

        raise ValueError(f"Unknown tool '{name}'")

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1

        parsed = self._parse_action(action)
        if not parsed:
            obs = "Invalid action format. Use \\boxed{tool: ...} or \\boxed{submit: file=...}."
            info = {"suffix": self.get_task_suffix()}
            return obs, LanguageGameReward.format_error_reward, True, False, info

        if parsed["type"] == "tool":
            name = parsed["name"]
            if name not in self.tools:
                obs = f"Unsupported tool: {name}. Episode terminated."
                info = {"suffix": self.get_task_suffix()}
                return obs, 0.0, True, False, info  # Fixed: was -0.4
            try:
                result = self._execute_tool(name, parsed["args"])
                self.steps_taken += 1
                obs = f"OK: Tool '{name}' executed.\nResult: {result}\nProgress: steps {self.steps_taken}/{self.task['required_steps']}."
                terminated = False
                truncated = False
                reward = 0.0
            except Exception as e:
                obs = f"Execution error in tool '{name}': {e}"
                terminated = True  # Fixed: execution errors should end episode
                truncated = False
                reward = 0.0  # Fixed: was -0.1, failures should be 0.0
            info = {"suffix": self.get_task_suffix()}
        elif parsed["type"] == "submit":
            # Protocol check
            if self.steps_taken < self.task["required_steps"]:
                obs = f"Protocol violation: Need at least {self.task['required_steps']} tool calls before submit. Episode terminated."
                info = {"suffix": self.get_task_suffix()}
                return obs, 0.0, True, False, info  # Fixed: was -0.2

            file = parsed["args"].get("file", self.task["target_report"])
            report_meta = self.execution_state.get("last_report_meta")
            ready = self.execution_state.get("report_ready", False)
            if file != self.task["target_report"]:
                obs = f"Failure: Submitted wrong file '{file}', expected '{self.task['target_report']}'. Episode terminated."
                info = {"suffix": self.get_task_suffix()}
                return obs, 0.0, True, False, info  # Fixed: was -0.25

            if ready and report_meta and report_meta.get("file") == file:
                obs = "SUCCESS: Report accepted. All checks passed. Episode terminated."
                reward = 1.0
                terminated = True
                truncated = False
            else:
                details = report_meta if report_meta else {}
                obs = f"Failure: Report did not meet constraints or was not rendered properly. Details: {details}. Episode terminated."
                reward = 0.0  # Fixed: was -0.3, failures should be 0.0
                terminated = True
                truncated = False
            info = {"suffix": self.get_task_suffix()}
        else:
            obs = "Invalid action type. Use tool or submit."
            info = {"suffix": self.get_task_suffix()}
            return obs, 0.0, True, False, info  # Fixed: was -0.2

        if not terminated and self.turn_count >= self.max_turns:
            obs = f"Timeout: Reached max turns ({self.max_turns}). Episode terminated."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}  # Fixed: was -0.3

        return obs, reward, terminated, truncated, info

    def sample_random_action(self) -> str:
        tool = random.choice(list(self.tools.keys()))
        # generate plausible args
        if tool == "load_table":
            f = random.choice(["inventory.csv", "sales.csv"])
            return f"\\boxed{{tool: load_table file={f}}}"
        if tool == "filter_rows":
            choices = [
                ("category", "==", "Hardware"),
                ("price", ">=", "100"),
                ("region", "==", "North"),
                ("revenue", ">", "500"),
            ]
            c, o, v = random.choice(choices)
            return f"\\boxed{{tool: filter_rows column={c} op== value={v}}}"
        if tool == "sort_by":
            return "\\boxed{tool: sort_by column=price order=desc}"
        if tool == "compute_aggregate":
            return "\\boxed{tool: compute_aggregate func=sum column=price}"
        if tool == "save_table":
            return "\\boxed{tool: save_table file=filtered.csv}"
        if tool == "read_text":
            return "\\boxed{tool: read_text file=notes.txt}"
        if tool == "grep":
            kw = self.workspace["_keywords"]["primary"]
            return f"\\boxed{{tool: grep file=notes.txt pattern={kw}}}"
        if tool == "summarize_table":
            return "\\boxed{tool: summarize_table top_k=5 sort_by=price order=desc}"
        if tool == "render_report":
            return "\\boxed{tool: render_report file=report.txt top_k=5 sort_by=price order=desc}"
        if tool == "join_tables":
            return "\\boxed{tool: join_tables left=inventory.csv right=sales.csv on=id}"
        if tool == "group_by_aggregate":
            return "\\boxed{tool: group_by_aggregate by=category agg_func=sum target=price}"
        return "\\boxed{submit: file=report.txt}"


class WorkflowAnvilEnvWithFeedback(WorkflowAnvilEnv):
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
            hint = "Wrap your action in \\boxed{...} and use 'tool:' or 'submit:' with parameters."

        elif "unsupported tool" in text:
            error_type = "UnsupportedAction"
            # Extract tool name
            m = re.search(r"unsupported tool: ([\w_]+)", obs, flags=re.IGNORECASE)
            error_detail["tool"] = m.group(1) if m else None
            hint = "Use available tools listed in instructions. Call 'load_table' first to start."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            m = re.search(r"need at least (\d+)", obs)
            if m:
                error_detail["required_steps"] = int(m.group(1))
            error_detail["steps_taken"] = getattr(self, "steps_taken", None)
            hint = "Execute enough tool calls to meet the minimum before submitting. Use render_report to finalize."

        elif "execution error" in text:
            error_type = "ExecutionError"
            # Try to capture tool and message
            mtool = re.search(r"tool '([^']+)'", obs, flags=re.IGNORECASE)
            mmsg = re.search(r"execution error.*?: (.+)", obs, flags=re.IGNORECASE)
            error_detail["tool"] = mtool.group(1) if mtool else None
            error_detail["message"] = mmsg.group(1) if mmsg else obs
            hint = "Check prerequisites for the tool: load a table before filtering; compute aggregate before render_report."

        elif "failure" in text and "report" in text:
            error_type = "WrongDecision"
            # Extract report meta booleans
            top_ok = "top_ok=true" in text
            agg_ok = "agg_ok=true" in text
            grep_ok = "grep_ok=true" in text
            error_detail["top_ok"] = top_ok
            error_detail["agg_ok"] = agg_ok
            error_detail["grep_ok"] = grep_ok
            hint = "Ensure table matches the required filter and sort, compute the specified aggregate, and grep the required pattern before render_report."

        elif "timeout" in text:
            error_type = "Timeout"
            error_detail["max_turns"] = getattr(self, "max_turns", None)
            hint = "Plan the sequence: load_table -> filter_rows -> sort_by -> compute_aggregate -> grep (if required) -> render_report -> submit."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["steps_taken"] = getattr(self, "steps_taken", None)
            diagnostic["required_steps"] = self.task.get("required_steps") if hasattr(self, "task") else None
            diagnostic["task"] = {
                "dataset": self.task.get("dataset"),
                "filter": self.task.get("filter"),
                "sort_by": self.task.get("sort_by"),
                "order": self.task.get("order"),
                "top_k": self.task.get("top_k"),
                "aggregate_func": self.task.get("aggregate_func"),
                "aggregate_column": self.task.get("aggregate_column"),
                "requires_grep": self.task.get("requires_grep"),
                "grep_pattern": self.task.get("grep_pattern"),
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
            "hint": "Begin with \\boxed{tool: load_table file=<dataset>} then filter and sort as specified.",
            "turn": 0,
            "steps_taken": 0,
            "required_steps": self.task.get("required_steps"),
        }
        return obs, info