from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List, Union


class WorkbenchMaestroEnv(Env):
    def __init__(self, complexity: int = 1, max_turns: Optional[int] = 30, **_):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.max_turns = max_turns if max_turns is not None else 30

        # Step-based complexity: N to 2N tool calls required
        self.min_required_steps = self.complexity
        self.max_required_steps = self.complexity * 2

        self._init_database()
        self.reset()

    def _init_database(self) -> None:
        # Tool catalog (name -> signature/desc). Used for instructions and validation.
        self.tools = {
            "load": {
                "description": "Load a dataset into the working buffer.",
                "params": ["name"],
                "example": r"\boxed{load(name='sales')}",
            },
            "select": {
                "description": "Select a subset of columns.",
                "params": ["columns"],
                "example": r"\boxed{select(columns=['id','amount'])}",
            },
            "filter": {
                "description": "Filter rows by a condition.",
                "params": ["column", "op", "value"],
                "example": r"\boxed{filter(column='region', op='==', value='North')}",
            },
            "groupby_aggregate": {
                "description": "Group by column(s) and aggregate a target with sum/count/avg/min/max.",
                "params": ["by", "agg", "target"],
                "example": r"\boxed{groupby_aggregate(by=['department'], agg='sum', target='amount')}",
            },
            "sort": {
                "description": "Sort current data by column.",
                "params": ["by", "ascending"],
                "example": r"\boxed{sort(by='amount', ascending=False)}",
            },
            "head": {
                "description": "Keep first n rows.",
                "params": ["n"],
                "example": r"\boxed{head(n=5)}",
            },
            "unique": {
                "description": "List unique values in a column.",
                "params": ["column"],
                "example": r"\boxed{unique(column='category')}",
            },
            "compute_stat": {
                "description": "Compute a stat over a numeric column on current data: sum/count/avg/min/max.",
                "params": ["stat", "column"],
                "example": r"\boxed{compute_stat(stat='sum', column='amount')}",
            },
            "save_as": {
                "description": "Save current working data into a named snapshot.",
                "params": ["name"],
                "example": r"\boxed{save_as(name='sales_view')}",
            },
            "switch_to": {
                "description": "Switch working data to a saved snapshot.",
                "params": ["name"],
                "example": r"\boxed{switch_to(name='sales_view')}",
            },
            "join_with": {
                "description": "Join current data with a saved snapshot on a key, using 'inner' or 'left'.",
                "params": ["name", "on", "how"],
                "example": r"\boxed{join_with(name='employees', on='rep_id', how='inner')}",
            },
            "describe": {
                "description": "Return schema summary of current data.",
                "params": [],
                "example": r"\boxed{describe()}",
            },
            "reset_data": {
                "description": "Clear current working data.",
                "params": [],
                "example": r"\boxed{reset_data()}",
            },
            "submit": {
                "description": "Submit final answer for evaluation.",
                "params": ["answer"],
                "example": r"\boxed{submit(answer=123.45)}",
            },
        }

        # Simulated datasets (scale rows with complexity)
        random.seed(1337 + self.complexity)
        base_rows = 60 + self.complexity * 15
        regions = ["North", "South", "East", "West", "Central"][: 3 + (self.complexity // 3)]
        categories = ["Gadget", "Widget", "Tool", "Accessory", "Component"][: 3 + (self.complexity // 2)]
        departments = ["Sales", "Support", "Engineering", "HR", "Marketing"][: 3 + (self.complexity // 2)]

        # Employees
        self.datasets = {}
        employees = []
        for i in range(25 + self.complexity * 5):
            emp_id = 1000 + i
            employees.append({
                "emp_id": emp_id,
                "name": f"Emp{i}",
                "department": random.choice(departments),
                "hire_year": random.randint(2010, 2023),
                "performance_score": random.randint(50, 100),
            })
        self.datasets["employees"] = employees

        # Sales
        sales = []
        for i in range(base_rows):
            rep = random.choice(employees)["emp_id"]
            month = random.randint(1, 12)
            amount = round(random.uniform(50.0, 5000.0), 2)
            sales.append({
                "sale_id": i + 1,
                "rep_id": rep,
                "region": random.choice(regions),
                "category": random.choice(categories),
                "month": month,
                "quarter": (month - 1) // 3 + 1,
                "amount": amount,
            })
        self.datasets["sales"] = sales

        # Inventory
        inventory = []
        for i in range(40 + self.complexity * 10):
            sku = f"SKU{i+1:04d}"
            inventory.append({
                "sku": sku,
                "category": random.choice(categories),
                "stock": random.randint(0, 200),
                "reorder_level": random.randint(10, 80),
            })
        self.datasets["inventory"] = inventory

        # Execution state
        self.current_data: Optional[List[Dict[str, Any]]] = None
        self.saved_frames: Dict[str, List[Dict[str, Any]]] = {}
        self.last_result: Any = None
        self.turn_count = 0
        self.steps_taken = 0
        self.task: Dict[str, Any] = {}

    def _get_instructions(self) -> str:
        # High-level instructions for the agent
        lines = []
        lines.append("You are a tool orchestrator. Use the available tools to compute the required answer, then submit it.")
        lines.append("Rules:")
        lines.append("- Use boxed calls: \\boxed{tool_name(arg=value, ...)}")
        lines.append("- You must execute at least the required number of tool calls before submitting.")
        lines.append("- Most tools operate on the current working data loaded by load(name=...).")
        lines.append("Available tools:")
        for name, meta in self.tools.items():
            params = ", ".join(meta["params"])
            lines.append(f"- {name}({params}): {meta['description']}")
        lines.append("Finalize with: \\boxed{submit(answer=...)}")
        return "\n".join(lines)

    def get_task_suffix(self) -> str:
        loaded = "yes" if self.current_data is not None else "no"
        saved = sorted(list(self.saved_frames.keys()))
        return (
            f"Turn {self.turn_count}/{self.max_turns} | Steps {self.steps_taken}/{self.task.get('required_steps','?')} | "
            f"Working data loaded: {loaded} | Saved: {saved}"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        required_steps = random.randint(self.min_required_steps, self.max_required_steps)
        self.turn_count = 0
        self.steps_taken = 0
        self.current_data = None
        self.saved_frames = {}
        self.last_result = None

        self.task = self._generate_task_requiring_n_steps(required_steps)
        obs = self._format_task_prompt(self.task)
        return obs, {"suffix": self.get_task_suffix()}

    def _format_task_prompt(self, task: Dict[str, Any]) -> str:
        # Plain prompt including task and reminder of format
        t = task["type"]
        if t == "sum_sales":
            constraints_txt = ", ".join([f"{k}={v}" for k, v in task["constraints"].items()])
            return (
                "Task: Compute the total sales amount matching all constraints and submit the numeric result.\n"
                f"Dataset: sales | Constraints: {constraints_txt}\n"
                "Respond using \\boxed{tool(...)} calls. Submit with \\boxed{submit(answer=NUMBER)}"
            )
        elif t == "top_employees":
            return (
                "Task: List the names of the top N employees by performance_score in the given department and hired before a year.\n"
                f"Dataset: employees | department={task['department']} | top_n={task['top_n']} | hire_year_before={task['hire_year_before']}\n"
                "Return a comma-separated list of names via \\boxed{submit(answer='Name1,Name2,...')}"
            )
        elif t == "inventory_deficit":
            return (
                "Task: Count how many inventory items are below the reorder level within a category, and submit the count.\n"
                f"Dataset: inventory | category={task['category']}\n"
                "Submit with \\boxed{submit(answer=INTEGER)}"
            )
        elif t == "dept_max_sales_via_join":
            return (
                "Task: Join sales with employees on rep_id=emp_id and find the department with the highest total sales. Submit the department name.\n"
                "Datasets: sales, employees\n"
                "Submit with \\boxed{submit(answer='DepartmentName')}"
            )
        else:
            return "Task: Unknown task type."

    def _generate_task_requiring_n_steps(self, n: int) -> Dict[str, Any]:
        # Select a task archetype and parameters, build reference
        # Bias towards join tasks at higher n
        choices = ["sum_sales", "top_employees", "inventory_deficit"]
        if n >= 6:
            choices.append("dept_max_sales_via_join")
        t = random.choice(choices)

        if t == "sum_sales":
            # At least: load + k filters + compute_stat (sum) -> 2 + k steps
            # Build k = max(1, min(n-2, available_constraints))
            constraints = {}
            possible = []
            # Use realistic filters
            any_sale = random.choice(self.datasets["sales"])
            # quarter
            if random.random() < 0.7:
                possible.append(("quarter", any_sale["quarter"]))
            # region
            if random.random() < 0.9:
                possible.append(("region", any_sale["region"]))
            # category
            if random.random() < 0.9:
                possible.append(("category", any_sale["category"]))
            # month exact or range
            if random.random() < 0.4:
                possible.append(("month", any_sale["month"]))
            k = max(1, min(len(possible), max(1, n - 2)))
            for key, val in random.sample(possible, k):
                constraints[key] = val

            # Compute reference
            data = list(self.datasets["sales"])
            for key, val in constraints.items():
                data = [r for r in data if r.get(key) == val]
            total = round(sum(r["amount"] for r in data), 2)
            return {
                "type": "sum_sales",
                "required_steps": n,
                "constraints": constraints,
                "reference": total,
                "answer_type": "number",
            }

        if t == "top_employees":
            dept = random.choice(list({e["department"] for e in self.datasets["employees"]}))
            top_n = min(3 + (n // 3), 5)
            cut = random.randint(2012, 2022)
            # Reference
            filtered = [e for e in self.datasets["employees"] if e["department"] == dept and e["hire_year"] < cut]
            filtered.sort(key=lambda x: x["performance_score"], reverse=True)
            names = [e["name"] for e in filtered[:top_n]]
            return {
                "type": "top_employees",
                "required_steps": n,
                "department": dept,
                "top_n": top_n,
                "hire_year_before": cut,
                "reference": ",".join(names),
                "answer_type": "csv_names",
            }

        if t == "inventory_deficit":
            cat = random.choice(list({x["category"] for x in self.datasets["inventory"]}))
            deficit = [x for x in self.datasets["inventory"] if x["category"] == cat and x["stock"] < x["reorder_level"]]
            return {
                "type": "inventory_deficit",
                "required_steps": n,
                "category": cat,
                "reference": len(deficit),
                "answer_type": "number",
            }

        # dept_max_sales_via_join
        # Reference
        sales = self.datasets["sales"]
        employees = self.datasets["employees"]
        emp_map = {e["emp_id"]: e["department"] for e in employees}
        totals: Dict[str, float] = {}
        for r in sales:
            dep = emp_map.get(r["rep_id"])
            if dep is None:
                continue
            totals[dep] = totals.get(dep, 0.0) + float(r["amount"])
        if totals:
            best_dep = max(totals.items(), key=lambda kv: kv[1])[0]
        else:
            best_dep = "Sales"
        return {
            "type": "dept_max_sales_via_join",
            "required_steps": n,
            "reference": best_dep,
            "answer_type": "string",
        }

    def _parse_action(self, action: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        # Expect \boxed{...}
        if not isinstance(action, str):
            return None
        m = re.search(r"\\boxed\{(.+)\}\s*$", action.strip(), flags=re.DOTALL)
        if not m:
            return None
        inner = m.group(1).strip()
        # Extract tool and args: name(...)
        m2 = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\((.*)\))?\s*$", inner, flags=re.DOTALL)
        if not m2:
            return None
        tool = m2.group(1)
        args_str = m2.group(2) or ""
        args = self._parse_args(args_str)
        if args is None:
            return None
        return tool, args

    def _parse_args(self, s: str) -> Optional[Dict[str, Any]]:
        # Parse comma-separated key=value pairs with simple literals
        args: Dict[str, Any] = {}
        if not s.strip():
            return args
        parts = self._smart_split_commas(s)
        for part in parts:
            if "=" not in part:
                return None
            k, v = part.split("=", 1)
            key = k.strip()
            if not key:
                return None
            val = self._parse_literal(v.strip())
            args[key] = val
        return args

    def _smart_split_commas(self, s: str) -> List[str]:
        parts = []
        buf = []
        depth_br = 0
        depth_sq = 0
        in_str = False
        quote = ""
        for ch in s:
            if in_str:
                buf.append(ch)
                if ch == quote:
                    in_str = False
                continue
            if ch in ("'", '"'):
                in_str = True
                quote = ch
                buf.append(ch)
                continue
            if ch == "(":
                depth_br += 1
            elif ch == ")":
                depth_br = max(0, depth_br - 1)
            elif ch == "[":
                depth_sq += 1
            elif ch == "]":
                depth_sq = max(0, depth_sq - 1)
            if ch == "," and depth_br == 0 and depth_sq == 0:
                parts.append("".join(buf).strip())
                buf = []
            else:
                buf.append(ch)
        if buf:
            parts.append("".join(buf).strip())
        return parts

    def _parse_literal(self, v: str) -> Any:
        # String
        if (len(v) >= 2) and ((v[0] == v[-1] == "'") or (v[0] == v[-1] == '"')):
            return v[1:-1]
        # List
        if v.startswith("[") and v.endswith("]"):
            inner = v[1:-1].strip()
            if not inner:
                return []
            parts = self._smart_split_commas(inner)
            return [self._parse_literal(p.strip()) for p in parts]
        # Bool
        if v.lower() == "true":
            return True
        if v.lower() == "false":
            return False
        # Number
        if re.fullmatch(r"-?\d+", v):
            try:
                return int(v)
            except Exception:
                pass
        if re.fullmatch(r"-?\d+\.\d+", v):
            try:
                return float(v)
            except Exception:
                pass
        # Bare word -> string
        return v

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        info: Dict[str, Any] = {"suffix": self.get_task_suffix()}

        # Timeout check
        if self.turn_count > self.max_turns:
            obs = "Timeout: turn limit reached. Episode truncated."
            return obs, 0.0, True, True, info

        parsed = self._parse_action(action)
        if not parsed:
            obs = "Invalid action format. Use \\boxed{tool_name(arg=value,...)}"
            return obs, float(LanguageGameReward.format_error_reward), True, False, info

        tool, args = parsed
        if tool not in self.tools:
            obs = f"Unsupported action: unknown tool '{tool}'."
            return obs, -0.5, True, False, info

        # Handle submit separately
        if tool == "submit":
            ans = args.get("answer", None)
            # Enforce minimum steps before submission
            if self.steps_taken < self.task["required_steps"]:
                obs = (
                    "Protocol error: submission before meeting required tool calls. "
                    f"Steps taken {self.steps_taken} < required {self.task['required_steps']}."
                )
                return obs, -0.5, True, False, info

            correct = self._check_submission(ans)
            if correct:
                obs = "Submission received. Correct. Success."
                return obs, 1.0, True, False, info
            else:
                obs = "Submission received. Incorrect. Wrong answer."
                return obs, -1.0, True, False, info

        # Execute tool
        try:
            result_text = self._execute_tool(tool, args)
            self.steps_taken += 1
            obs = f"OK: {tool} executed. Result: {result_text}"
            # Not terminal; success only on submit
            return obs, 0.0, False, False, info
        except ValueError as e:
            # Protocol violations are non-terminal to allow recovery
            obs = f"Protocol error: {str(e)}"
            return obs, -0.1, False, False, info
        except Exception as e:
            obs = f"Execution error: {str(e)}"
            return obs, -0.2, False, False, info

    def _execute_tool(self, tool: str, args: Dict[str, Any]) -> str:
        if tool == "load":
            name = args.get("name")
            if name not in self.datasets and name not in self.saved_frames:
                # Allow alias sales_QX -> filter by quarter on sales
                if isinstance(name, str) and name.startswith("sales_Q") and name[-1:].isdigit():
                    q = int(name[-1])
                    base = list(self.datasets["sales"])
                    self.current_data = [r for r in base if r["quarter"] == q]
                    return f"Loaded {len(self.current_data)} records from alias {name}"
                raise ValueError(f"dataset '{name}' not found")
            if name in self.datasets:
                self.current_data = list(self.datasets[name])
                return f"Loaded {len(self.current_data)} rows from '{name}'"
            # saved frame
            self.current_data = list(self.saved_frames[name])
            return f"Loaded {len(self.current_data)} rows from snapshot '{name}'"

        if tool == "select":
            self._require_loaded()
            cols = args.get("columns")
            if not isinstance(cols, list) or not cols:
                raise ValueError("columns must be a non-empty list")
            sel = []
            for r in self.current_data:
                sel.append({c: r.get(c, None) for c in cols})
            self.current_data = sel
            return f"Selected columns {cols}"

        if tool == "filter":
            self._require_loaded()
            col = args.get("column")
            op = args.get("op")
            val = args.get("value")
            if col is None or op is None:
                raise ValueError("filter requires 'column' and 'op'")
            def match(x):
                if op == "==": return x == val
                if op == "!=": return x != val
                if op == ">": return (x is not None and val is not None and x > val)
                if op == ">=": return (x is not None and val is not None and x >= val)
                if op == "<": return (x is not None and val is not None and x < val)
                if op == "<=": return (x is not None and val is not None and x <= val)
                if op == "in":
                    if isinstance(val, list): return x in val
                    return False
                if op == "contains":
                    return isinstance(x, str) and isinstance(val, str) and (val in x)
                if op == "startswith":
                    return isinstance(x, str) and isinstance(val, str) and x.startswith(val)
                if op == "endswith":
                    return isinstance(x, str) and isinstance(val, str) and x.endswith(val)
                raise ValueError(f"unsupported op '{op}'")
            before = len(self.current_data)
            self.current_data = [r for r in self.current_data if match(r.get(col))]
            return f"Filtered {before} -> {len(self.current_data)} rows on {col} {op} {val}"

        if tool == "groupby_aggregate":
            self._require_loaded()
            by = args.get("by")
            agg = args.get("agg")
            target = args.get("target")
            if not by or not isinstance(by, list):
                raise ValueError("'by' must be a non-empty list")
            if agg not in ("sum", "count", "avg", "min", "max"):
                raise ValueError("agg must be one of sum/count/avg/min/max")
            groups: Dict[tuple, List[Dict[str, Any]]] = {}
            for r in self.current_data:
                key = tuple(r.get(c) for c in by)
                groups.setdefault(key, []).append(r)
            out = []
            for k, rows in groups.items():
                if agg == "count":
                    val = len(rows)
                else:
                    vals = [float(r.get(target, 0) or 0) for r in rows]
                    if agg == "sum": val = sum(vals)
                    elif agg == "avg": val = (sum(vals) / len(vals)) if vals else 0.0
                    elif agg == "min": val = min(vals) if vals else 0.0
                    else: val = max(vals) if vals else 0.0
                row = {by[i]: k[i] for i in range(len(by))}
                row[f"{agg}_{target if target else 'count'}"] = round(val, 2) if isinstance(val, float) else val
                out.append(row)
            self.current_data = out
            return f"Grouped by {by} with {agg} on {target}. Rows: {len(out)}"

        if tool == "sort":
            self._require_loaded()
            by = args.get("by")
            ascending = args.get("ascending", True)
            if by is None:
                raise ValueError("sort requires 'by'")
            self.current_data.sort(key=lambda r: (r.get(by) is None, r.get(by)), reverse=not ascending)
            return f"Sorted by {by} ascending={bool(ascending)}"

        if tool == "head":
            self._require_loaded()
            n = int(args.get("n", 5))
            self.current_data = self.current_data[:n]
            return f"Truncated to first {n} rows"

        if tool == "unique":
            self._require_loaded()
            col = args.get("column")
            if col is None:
                raise ValueError("unique requires 'column'")
            seen = []
            for r in self.current_data:
                v = r.get(col)
                if v not in seen:
                    seen.append(v)
            self.last_result = seen
            return f"Unique values in {col}: {seen}"

        if tool == "compute_stat":
            self._require_loaded()
            stat = args.get("stat")
            col = args.get("column")
            if stat not in ("sum", "count", "avg", "min", "max"):
                raise ValueError("stat must be one of sum/count/avg/min/max")
            if stat == "count":
                val = len(self.current_data)
            else:
                vals = [float(r.get(col, 0) or 0) for r in self.current_data]
                if stat == "sum":
                    val = sum(vals)
                elif stat == "avg":
                    val = sum(vals) / len(vals) if vals else 0.0
                elif stat == "min":
                    val = min(vals) if vals else 0.0
                else:
                    val = max(vals) if vals else 0.0
            self.last_result = round(val, 2) if isinstance(val, float) else val
            return f"{stat}({col if col else ''}) = {self.last_result}"

        if tool == "save_as":
            self._require_loaded()
            name = args.get("name")
            if not name or not isinstance(name, str):
                raise ValueError("save_as requires 'name'")
            self.saved_frames[name] = list(self.current_data)
            return f"Saved current data as '{name}'"

        if tool == "switch_to":
            name = args.get("name")
            if name not in self.saved_frames:
                raise ValueError(f"snapshot '{name}' not found")
            self.current_data = list(self.saved_frames[name])
            return f"Switched to snapshot '{name}' with {len(self.current_data)} rows"

        if tool == "join_with":
            self._require_loaded()
            name = args.get("name")
            key = args.get("on")
            how = args.get("how", "inner")
            if name not in self.saved_frames:
                raise ValueError(f"snapshot '{name}' not found")
            if key is None:
                raise ValueError("join_with requires 'on'")
            if how not in ("inner", "left"):
                raise ValueError("how must be 'inner' or 'left'")
            right = self.saved_frames[name]
            # Build index on right
            idx: Dict[Any, List[Dict[str, Any]]] = {}
            for rr in right:
                idx.setdefault(rr.get(key), []).append(rr)
            joined = []
            for l in self.current_data:
                lk = l.get(key)
                matches = idx.get(lk, [])
                if matches:
                    for m in matches:
                        row = dict(l)
                        for k2, v2 in m.items():
                            if k2 in row:
                                row[f"{name}.{k2}"] = v2
                            else:
                                row[k2] = v2
                        joined.append(row)
                elif how == "left":
                    joined.append(dict(l))
            self.current_data = joined
            return f"Joined with '{name}' on '{key}' using {how}. Rows: {len(joined)}"

        if tool == "describe":
            self._require_loaded()
            # Compute schema (keys and simple types)
            keys = {}
            for r in self.current_data[:20]:
                for k, v in r.items():
                    keys[k] = type(v).__name__
            self.last_result = keys
            return f"Schema: {keys}"

        if tool == "reset_data":
            self.current_data = None
            return "Cleared working data"

        raise ValueError(f"unexpected tool '{tool}'")

    def _require_loaded(self) -> None:
        if self.current_data is None:
            raise ValueError("no working data loaded; call load(name=...) first")

    def _check_submission(self, answer: Any) -> bool:
        ref = self.task["reference"]
        typ = self.task["answer_type"]
        if typ == "number":
            # Accept numeric within 0.01 tolerance
            try:
                ans_val = float(answer)
                ref_val = float(ref)
                return abs(ans_val - ref_val) <= 0.01
            except Exception:
                return False
        if typ == "string":
            try:
                ans_str = str(answer).strip().strip("'").strip('"')
                return ans_str == str(ref)
            except Exception:
                return False
        if typ == "csv_names":
            # Compare as normalized lists (order matters for this task)
            if answer is None:
                return ref == ""
            if isinstance(answer, str):
                ans_list = [x.strip() for x in answer.split(",") if x.strip()]
            elif isinstance(answer, list):
                ans_list = [str(x).strip() for x in answer]
            else:
                ans_list = [str(answer).strip()]
            ref_list = [x.strip() for x in str(ref).split(",") if x.strip()]
            return ans_list == ref_list
        return False

    def sample_random_action(self) -> str:
        # Heuristic sampling based on current state
        if self.current_data is None:
            ds = random.choice(list(self.datasets.keys()))
            return f"\\boxed{{load(name='{ds}')}}"
        # Random valid next tool
        options = ["describe", "head", "sort", "unique", "save_as", "compute_stat"]
        tool = random.choice(options)
        if tool == "describe":
            return "\\boxed{describe()}"
        if tool == "head":
            return "\\boxed{head(n=5)}"
        if tool == "sort":
            # guess a numeric col
            sample = self.current_data[0] if self.current_data else {}
            cols = [k for k, v in sample.items() if isinstance(v, (int, float))]
            col = cols[0] if cols else list(sample.keys())[0] if sample else "sale_id"
            return f"\\boxed{{sort(by='{col}', ascending=False)}}"
        if tool == "unique":
            sample = self.current_data[0] if self.current_data else {}
            cols = list(sample.keys())
            col = cols[0] if cols else "category"
            return f"\\boxed{{unique(column='{col}')}}"
        if tool == "save_as":
            name = f"view{random.randint(1,9)}"
            return f"\\boxed{{save_as(name='{name}')}}"
        if tool == "compute_stat":
            sample = self.current_data[0] if self.current_data else {}
            cols = [k for k, v in sample.items() if isinstance(v, (int, float))]
            col = cols[0] if cols else None
            stat = random.choice(["count", "sum", "avg"])
            if stat == "count":
                return "\\boxed{compute_stat(stat='count', column='ignored')}"
            if col:
                return f"\\boxed{{compute_stat(stat='{stat}', column='{col}')}}"
            return "\\boxed{compute_stat(stat='count', column='ignored')}"
        return "\\boxed{describe()}"


class WorkbenchMaestroEnvWithFeedback(WorkbenchMaestroEnv):
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
            error_detail["issue"] = "missing_or_bad_boxed_format"
            hint = "Wrap your call like \\boxed{tool(arg=value)} with valid parameters."
        elif "unsupported action" in text or "unknown tool" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_tool"
            hint = "Check the tool list in the instructions and use an available tool."
        elif "protocol error" in text:
            error_type = "ProtocolViolation"
            if "submission before" in text:
                error_detail["violation"] = "early_submission"
                need = getattr(self, "task", {}).get("required_steps", None)
                have = getattr(self, "steps_taken", None)
                hint = f"Run at least {need} valid tool calls before submit; start with load(name='...')."
            elif "no working data loaded" in text:
                error_detail["violation"] = "missing_load"
                hint = "Call load(name='dataset') first, e.g., \\boxed{load(name='sales')}."
            elif "snapshot" in text and "not found" in text:
                error_detail["violation"] = "missing_snapshot"
                hint = "Create a snapshot via \\boxed{save_as(name='X')} before switching or joining."
            else:
                error_detail["violation"] = "other_protocol"
                hint = "Check parameter names and prerequisites for the tool."
        elif "execution error" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "runtime_error"
            hint = "Verify arg types and values; use describe() to inspect schema."
        elif "timeout" in text or "turn limit" in text:
            error_type = "Timeout"
            error_detail["outcome"] = "episode_truncated"
            hint = "Plan your tool sequence more efficiently; submit before the turn limit."
        elif "submission received. incorrect" in text or "wrong answer" in text:
            error_type = "WrongDecision"
            error_detail["got"] = action
            try:
                error_detail["expected"] = self.task.get("reference")
            except Exception:
                pass
            # Domain-aware hint
            t = self.task.get("type")
            if t == "sum_sales":
                hint = "Ensure you filtered sales by all given constraints before computing sum(amount)."
            elif t == "top_employees":
                hint = "Sort by performance_score desc after filtering department and hire_year, then take the top N names."
            elif t == "inventory_deficit":
                hint = "Filter by category and count rows where stock < reorder_level."
            elif t == "dept_max_sales_via_join":
                hint = "Join sales with employees on rep_id=emp_id, group by department, sum amount, then choose max."
        elif "submission received. correct" in text or "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["steps_taken"] = getattr(self, "steps_taken", None)
            diagnostic["required_steps"] = getattr(self, "task", {}).get("required_steps", None)
            diagnostic["working_loaded"] = self.current_data is not None
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        hint = "Start by loading a relevant dataset: \\boxed{load(name='sales')} or \\boxed{load(name='employees')}."
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": hint if self.feedback_level >= 2 else None,
            "turn": 0,
            "steps_taken": 0,
            "required_steps": self.task.get("required_steps"),
            "working_loaded": False,
        }
        return obs, info