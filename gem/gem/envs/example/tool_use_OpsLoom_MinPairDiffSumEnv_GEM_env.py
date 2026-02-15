from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List

class OpsLoomEnv(Env):
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
        self.tools = {
            "load": {
                "description": "Load a dataset into the working context.",
                "params": {"dataset": "name of dataset"},
                "returns": "Sets current working data.",
                "example": "\\boxed{load dataset=orders}",
            },
            "filter": {
                "description": "Filter rows by a condition (>, >=, <, <=, ==, in).",
                "params": {"condition": "e.g., price>50 or region==north or product_id in [1,2,3]"},
                "returns": "Reduces rows.",
                "example": "\\boxed{filter condition=price>50}",
            },
            "map_product": {
                "description": "Create a new column by multiplying two existing columns.",
                "params": {"new": "new column name", "a": "col A", "b": "col B"},
                "returns": "Adds new numeric column.",
                "example": "\\boxed{map_product new=revenue a=price b=quantity}",
            },
            "map_scale": {
                "description": "Create a new column by scaling an existing numeric column.",
                "params": {"new": "new column name", "source": "column", "by": "float multiplier"},
                "returns": "Adds new numeric column.",
                "example": "\\boxed{map_scale new=weighted source=revenue by=0.9}",
            },
            "join": {
                "description": "Left-join current data with another dataset on a key.",
                "params": {"right": "dataset name", "key": "join key column"},
                "returns": "Adds columns from right dataset.",
                "example": "\\boxed{join right=reviews key=product_id}",
            },
            "groupby": {
                "description": "Group rows by 'by' and aggregate 'target' with 'agg' (sum, mean, max, min).",
                "params": {"by": "group column", "target": "numeric column", "agg": "aggregation"},
                "returns": "Replaces data with aggregated rows: {'group': key, 'value': agg}.",
                "example": "\\boxed{groupby by=customer_id target=revenue agg=sum}",
            },
            "compute_stat": {
                "description": "Compute a scalar (sum, mean, max, min) over a numeric column.",
                "params": {"col": "numeric column", "stat": "sum|mean|max|min"},
                "returns": "Scalar numeric result.",
                "example": "\\boxed{compute_stat col=revenue stat=sum}",
            },
            "sort": {
                "description": "Sort rows by a column ascending or descending.",
                "params": {"col": "column", "order": "asc|desc"},
                "returns": "Reorders rows.",
                "example": "\\boxed{sort col=price order=desc}",
            },
            "select": {
                "description": "Keep only the listed columns.",
                "params": {"cols": "comma-separated list of columns"},
                "returns": "Drops other columns.",
                "example": "\\boxed{select cols=customer_id,revenue}",
            },
            "preview": {
                "description": "Show first few rows.",
                "params": {},
                "returns": "Text preview.",
                "example": "\\boxed{preview}",
            },
            "inspect_state": {
                "description": "Show row count and column names.",
                "params": {},
                "returns": "State summary.",
                "example": "\\boxed{inspect_state}",
            },
            "show_task": {
                "description": "Show the hidden pipeline steps needed to obtain the target.",
                "params": {},
                "returns": "Task specification.",
                "example": "\\boxed{show_task}",
            },
            "status": {
                "description": "Show steps taken and required.",
                "params": {},
                "returns": "Status text.",
                "example": "\\boxed{status}",
            },
            "submit": {
                "description": "Submit final numeric answer.",
                "params": {"value": "numeric answer"},
                "returns": "Terminal evaluation.",
                "example": "\\boxed{submit value=123.45}",
            },
        }

        base_rows = 30 + self.complexity * 10
        num_customers = 10 + self.complexity * 2
        num_products = 20 + self.complexity * 3

        regions = ["north", "south", "east", "west"]
        segments = ["A", "B", "C"]
        categories = ["home", "tech", "food", "books", "fashion"]

        self.datasets: Dict[str, List[Dict[str, Any]]] = {}
        orders = []
        for i in range(base_rows):
            customer_id = random.randint(1, num_customers)
            product_id = random.randint(1, num_products)
            price = round(random.uniform(5, 200), 2)
            quantity = random.randint(1, 6)
            discount = round(random.uniform(0.0, 0.3), 2)
            region = random.choice(regions)
            orders.append({
                "order_id": i + 1,
                "customer_id": customer_id,
                "product_id": product_id,
                "price": price,
                "quantity": quantity,
                "discount": discount,
                "region": region,
            })
        self.datasets["orders"] = orders

        customers = []
        for cid in range(1, num_customers + 1):
            customers.append({
                "customer_id": cid,
                "segment": random.choice(segments),
                "age": random.randint(18, 75),
                "region": random.choice(regions),
            })
        self.datasets["customers"] = customers

        reviews = []
        for pid in range(1, num_products + 1):
            reviews.append({
                "product_id": pid,
                "rating": round(random.uniform(1.0, 5.0), 2),
                "votes": random.randint(0, 400),
                "category": random.choice(categories),
            })
        self.datasets["reviews"] = reviews

        inventory = []
        for pid in range(1, num_products + 1):
            inventory.append({
                "product_id": pid,
                "stock": random.randint(0, 500),
                "cost": round(random.uniform(1.0, 120.0), 2),
                "category": random.choice(categories),
            })
        self.datasets["inventory"] = inventory

        self.execution_state: Dict[str, Any] = {}

    def _get_instructions(self) -> str:
        tool_list = ", ".join(sorted(self.tools.keys()))
        return (
            "You are orchestrating tools to compute a final numeric metric from hidden pipeline steps.\n"
            "Use tools by replying with \\boxed{...}. First load a dataset, then transform, join, group, and compute.\n"
            f"Available tools: {tool_list}\n"
            "Syntax examples:\n"
            " - \\boxed{load dataset=orders}\n"
            " - \\boxed{map_product new=revenue a=price b=quantity}\n"
            " - \\boxed{filter condition=price>50}\n"
            " - \\boxed{join right=reviews key=product_id}\n"
            " - \\boxed{groupby by=customer_id target=revenue agg=sum}\n"
            " - \\boxed{compute_stat col=revenue stat=sum}\n"
            " - \\boxed{preview}, \\boxed{inspect_state}, \\boxed{show_task}, \\boxed{status}\n"
            "Submit your final numeric with \\boxed{submit value=...}\n"
            "Follow prerequisites: load before transforms; ensure referenced columns exist."
        )

    def get_task_suffix(self) -> str:
        loaded = self.execution_state.get("loaded_dataset", None)
        cols = self.execution_state.get("columns", None)
        rows = self.execution_state.get("row_count", None)
        state = f"Loaded={loaded if loaded else 'None'}; Rows={rows if rows is not None else 'N/A'}; Cols={cols if cols else 'N/A'}"
        return (
            f"[Task] Compute the target scalar by executing a pipeline. "
            f"Required steps (typical): {self.task.get('required_steps')} to reach a correct compute_stat before submit. "
            f"Turns: {self.turn_count}/{self.max_turns}. State: {state}. "
            "Use \\boxed{tool arg=value ...}. Submit with \\boxed{submit value=NUMBER}."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        required_steps = random.randint(self.min_required_steps, self.max_required_steps)
        self.task = self._generate_task_requiring_n_steps(required_steps)
        self.turn_count = 0
        self.steps_taken = 0
        self.current_data: Optional[List[Dict[str, Any]]] = None
        self.execution_state = {
            "loaded_dataset": None,
            "columns": None,
            "row_count": None,
            "joined_with": None,
            "grouped_by": None,
            "last_stat": None,
        }
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_task_requiring_n_steps(self, required_steps: int) -> Dict[str, Any]:
        plan: List[Dict[str, Any]] = []
        plan.append({"tool": "load", "dataset": "orders"})
        has_revenue = False
        has_weighted = False
        joined = False
        grouped = False

        steps_before_compute = max(0, required_steps - 1)
        # Build coherent operations
        while len(plan) < steps_before_compute:
            # Try deterministic useful operations first, respecting prerequisites
            if not has_revenue and len(plan) < steps_before_compute:
                plan.append({"tool": "map_product", "new": "revenue", "a": "price", "b": "quantity"})
                has_revenue = True
                continue
            # Optional filter
            if random.random() < 0.6 and len(plan) < steps_before_compute:
                threshold = round(random.uniform(20, 120), 2)
                plan.append({"tool": "filter", "condition": f"price>{threshold}"})
                continue
            # Optional join with reviews to enable weighted revenue
            if not joined and random.random() < 0.5 and len(plan) < steps_before_compute:
                plan.append({"tool": "join", "right": "reviews", "key": "product_id"})
                joined = True
                continue
            # Weighted revenue requires revenue and rating
            if joined and has_revenue and not has_weighted and len(plan) < steps_before_compute:
                plan.append({"tool": "map_product", "new": "weighted_revenue", "a": "revenue", "b": "rating"})
                has_weighted = True
                continue
            # Groupby aggregation
            if not grouped and random.random() < 0.7 and len(plan) < steps_before_compute:
                target_col = "weighted_revenue" if has_weighted else ("revenue" if has_revenue else "price")
                plan.append({"tool": "groupby", "by": "customer_id", "target": target_col, "agg": "sum"})
                grouped = True
                continue
            # Padding with sort that doesn't affect numeric result
            if len(plan) < steps_before_compute:
                plan.append({"tool": "sort", "col": "price", "order": random.choice(["asc", "desc"])})

        # Decide compute_stat column
        compute_col = "weighted_revenue" if has_weighted else ("revenue" if has_revenue else "price")
        plan.append({"tool": "compute_stat", "col": "value" if grouped else compute_col, "stat": random.choice(["sum", "mean"])})

        # Compute target answer by simulating plan
        target_answer = self._simulate_plan(plan)
        description_lines = []
        for i, step in enumerate(plan, 1):
            description_lines.append(f"{i}. {step}")
        return {
            "required_steps": required_steps,
            "plan": plan,
            "task_spec": "Pipeline steps:\n" + "\n".join(description_lines),
            "target_answer": target_answer,
        }

    def _simulate_plan(self, plan: List[Dict[str, Any]]) -> float:
        data = None
        grouped = False
        for step in plan:
            tool = step["tool"]
            if tool == "load":
                ds = step["dataset"]
                data = [dict(row) for row in self.datasets[ds]]
            elif tool == "map_product":
                new = step["new"]; a = step["a"]; b = step["b"]
                for row in data:
                    av = row.get(a); bv = row.get(b)
                    row[new] = (av if av is not None else 0) * (bv if bv is not None else 0)
            elif tool == "map_scale":
                new = step["new"]; src = step["source"]; by = float(step["by"])
                for row in data:
                    sv = row.get(src, 0)
                    row[new] = sv * by
            elif tool == "filter":
                cond = step["condition"]
                data = self._apply_filter_condition(data, cond)
            elif tool == "join":
                right = step["right"]; key = step["key"]
                right_data = self.datasets[right]
                index = {}
                for r in right_data:
                    index[r[key]] = r
                new_data = []
                for row in data:
                    joined_row = dict(row)
                    rv = index.get(row.get(key))
                    if rv:
                        for k, v in rv.items():
                            if k != key:
                                joined_row[k] = v
                    new_data.append(joined_row)
                data = new_data
            elif tool == "groupby":
                by = step["by"]; target = step["target"]; agg = step["agg"]
                groups: Dict[Any, List[float]] = {}
                for row in data:
                    key = row.get(by)
                    val = float(row.get(target, 0))
                    groups.setdefault(key, []).append(val)
                out = []
                for g, vals in groups.items():
                    if agg == "sum":
                        v = sum(vals)
                    elif agg == "mean":
                        v = (sum(vals) / max(len(vals), 1)) if vals else 0.0
                    elif agg == "max":
                        v = max(vals) if vals else 0.0
                    else:
                        v = min(vals) if vals else 0.0
                    out.append({"group": g, "value": v})
                data = out
                grouped = True
            elif tool == "sort":
                col = step["col"]; order = step["order"]
                rev = order == "desc"
                data = sorted(data, key=lambda r: (r.get(col, 0),), reverse=rev)
            elif tool == "compute_stat":
                col = step["col"]; stat = step["stat"]
                vals = [float(r.get(col, 0)) for r in data]
                if stat == "sum":
                    return round(sum(vals), 6)
                elif stat == "mean":
                    return round(sum(vals) / max(len(vals), 1), 6)
                elif stat == "max":
                    return round(max(vals) if vals else 0.0, 6)
                else:
                    return round(min(vals) if vals else 0.0, 6)
        return 0.0

    def _get_columns(self, data: Optional[List[Dict[str, Any]]]) -> Optional[List[str]]:
        if data is None or not data:
            return None
        cols = set()
        for row in data:
            for k in row.keys():
                cols.add(k)
        return sorted(list(cols))

    def _apply_filter_condition(self, data: List[Dict[str, Any]], condition: str) -> List[Dict[str, Any]]:
        condition = condition.strip()
        # in [a,b,c]
        m_in = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*in\s*\[(.*?)\]\s*$", condition)
        if m_in:
            col = m_in.group(1)
            vals_raw = m_in.group(2)
            vals = [self._auto_cast(v.strip()) for v in vals_raw.split(",") if v.strip() != ""]
            return [row for row in data if row.get(col) in vals]

        comparators = [">=", "<=", "==", ">", "<"]
        for comp in comparators:
            parts = condition.split(comp)
            if len(parts) == 2:
                left = parts[0].strip()
                right = self._auto_cast(parts[1].strip())
                if comp == ">":
                    return [row for row in data if self._get_num(row.get(left)) > self._get_num(right)]
                if comp == ">=":
                    return [row for row in data if self._get_num(row.get(left)) >= self._get_num(right)]
                if comp == "<":
                    return [row for row in data if self._get_num(row.get(left)) < self._get_num(right)]
                if comp == "<=":
                    return [row for row in data if self._get_num(row.get(left)) <= self._get_num(right)]
                if comp == "==":
                    return [row for row in data if row.get(left) == right]
        return data

    def _get_num(self, v: Any) -> float:
        try:
            return float(v)
        except Exception:
            return 0.0

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1

        parsed = self._parse_action(action)
        if parsed is None:
            obs = "Invalid action format. Actions must be in \\boxed{...}."
            info = {"suffix": self.get_task_suffix()}
            return obs, LanguageGameReward.format_error_reward, True, False, info

        tool, args = parsed
        if tool not in self.tools:
            obs = f"Unsupported tool '{tool}'."
            info = {"suffix": self.get_task_suffix()}
            return obs, 0.0, True, False, info  # Fixed: was -0.5

        terminated = False
        truncated = False
        reward = 0.0
        info: Dict[str, Any] = {}

        try:
            if tool == "submit":
                val = args.get("value", None)
                if val is None:
                    obs = "Protocol violation: submit requires 'value'."
                    info["suffix"] = self.get_task_suffix()
                    return obs, 0.0, True, False, info  # Fixed: was -0.2, should end episode
                user_val = float(val)
                target = float(self.task["target_answer"])
                if abs(user_val - target) <= 1e-6:
                    obs = f"Submission accepted. Success. Your value={user_val}, Target={target}."
                    reward = 1.0
                else:
                    obs = f"Submission rejected. WrongDecision. Your value={user_val}, Target={target}."
                    reward = 0.0  # Fixed: was -1.0, failures should be 0.0
                terminated = True
            else:
                obs = self._execute_tool(tool, args)
        except Exception as e:
            obs = f"Execution error: {str(e)}"
            reward = 0.0  # Fixed: was -0.2, failures should be 0.0
            terminated = True  # Execution errors should end episode

        info["suffix"] = self.get_task_suffix()

        if not terminated and self.turn_count >= self.max_turns:
            terminated = True
            truncated = True
            obs = f"Timeout: reached max turns ({self.turn_count}/{self.max_turns})."
            reward = 0.0

        return obs, reward, terminated, truncated, info

    def _execute_tool(self, tool: str, args: Dict[str, Any]) -> str:
        count_step = False
        if tool in {"load", "filter", "map_product", "map_scale", "join", "groupby", "compute_stat", "sort", "select"}:
            count_step = True

        if tool == "load":
            dataset = args.get("dataset", None)
            if dataset not in self.datasets:
                raise ValueError(f"Unknown dataset '{dataset}'.")
            self.current_data = [dict(row) for row in self.datasets[dataset]]
            self.execution_state["loaded_dataset"] = dataset
            self.execution_state["joined_with"] = None
            self.execution_state["grouped_by"] = None

        elif tool == "filter":
            if self.current_data is None:
                return "Protocol violation: no data loaded. Call \\boxed{load dataset=...} first."
            condition = args.get("condition", None)
            if not condition:
                return "Protocol violation: filter requires 'condition'."
            self.current_data = self._apply_filter_condition(self.current_data, str(condition))

        elif tool == "map_product":
            if self.current_data is None:
                return "Protocol violation: no data loaded. Call \\boxed{load dataset=...} first."
            new = args.get("new"); a = args.get("a"); b = args.get("b")
            if not new or not a or not b:
                return "Protocol violation: map_product requires new, a, b."
            for row in self.current_data:
                av = row.get(a, 0)
                bv = row.get(b, 0)
                row[new] = self._get_num(av) * self._get_num(bv)

        elif tool == "map_scale":
            if self.current_data is None:
                return "Protocol violation: no data loaded."
            new = args.get("new"); src = args.get("source"); by = args.get("by")
            if not new or not src or by is None:
                return "Protocol violation: map_scale requires new, source, by."
            factor = float(by)
            for row in self.current_data:
                row[new] = self._get_num(row.get(src, 0)) * factor

        elif tool == "join":
            if self.current_data is None:
                return "Protocol violation: no data loaded."
            right = args.get("right"); key = args.get("key")
            if right not in self.datasets or not key:
                return "Protocol violation: join requires valid right and key."
            right_data = self.datasets[right]
            index = {}
            for r in right_data:
                index[r.get(key)] = r
            merged = []
            for row in self.current_data:
                new_row = dict(row)
                rv = index.get(row.get(key))
                if rv:
                    for k, v in rv.items():
                        if k != key:
                            new_row[k] = v
                merged.append(new_row)
            self.current_data = merged
            self.execution_state["joined_with"] = right

        elif tool == "groupby":
            if self.current_data is None:
                return "Protocol violation: no data loaded."
            by = args.get("by"); target = args.get("target"); agg = args.get("agg")
            if not by or not target or agg not in {"sum", "mean", "max", "min"}:
                return "Protocol violation: groupby requires by, target, agg in {sum, mean, max, min}."
            groups: Dict[Any, List[float]] = {}
            for row in self.current_data:
                key = row.get(by)
                val = self._get_num(row.get(target, 0))
                groups.setdefault(key, []).append(val)
            out = []
            for g, vals in groups.items():
                if agg == "sum":
                    v = sum(vals)
                elif agg == "mean":
                    v = sum(vals) / max(len(vals), 1) if vals else 0.0
                elif agg == "max":
                    v = max(vals) if vals else 0.0
                else:
                    v = min(vals) if vals else 0.0
                out.append({"group": g, "value": v})
            self.current_data = out
            self.execution_state["grouped_by"] = by

        elif tool == "compute_stat":
            if self.current_data is None:
                return "Protocol violation: no data loaded."
            col = args.get("col"); stat = args.get("stat")
            if not col or stat not in {"sum", "mean", "max", "min"}:
                return "Protocol violation: compute_stat requires col and stat in {sum, mean, max, min}."
            vals = [self._get_num(r.get(col, 0)) for r in self.current_data]
            if stat == "sum":
                result = sum(vals)
            elif stat == "mean":
                result = sum(vals) / max(len(vals), 1) if vals else 0.0
            elif stat == "max":
                result = max(vals) if vals else 0.0
            else:
                result = min(vals) if vals else 0.0
            result = round(result, 6)
            self.execution_state["last_stat"] = {"col": col, "stat": stat, "value": result}

        elif tool == "sort":
            if self.current_data is None:
                return "Protocol violation: no data loaded."
            col = args.get("col"); order = args.get("order", "asc")
            rev = str(order).lower() == "desc"
            self.current_data = sorted(self.current_data, key=lambda r: (r.get(col, 0),), reverse=rev)

        elif tool == "select":
            if self.current_data is None:
                return "Protocol violation: no data loaded."
            cols_raw = args.get("cols")
            if not cols_raw:
                return "Protocol violation: select requires 'cols'."
            if isinstance(cols_raw, str):
                cols = [c.strip() for c in cols_raw.split(",") if c.strip()]
            else:
                cols = cols_raw
            pruned = []
            for row in self.current_data:
                pruned.append({k: row.get(k) for k in cols})
            self.current_data = pruned

        elif tool == "preview":
            if self.current_data is None:
                return "No data loaded. Use \\boxed{load dataset=...}."
            sample = self.current_data[:5]
            rows_str = "\n".join([str(r) for r in sample]) if sample else "(empty)"
            self.execution_state["columns"] = self._get_columns(self.current_data)
            self.execution_state["row_count"] = len(self.current_data)
            return f"Preview (first {len(sample)} rows):\n{rows_str}"

        elif tool == "inspect_state":
            cols = self._get_columns(self.current_data)
            rows = len(self.current_data) if self.current_data is not None else 0
            self.execution_state["columns"] = cols
            self.execution_state["row_count"] = rows
            return f"State: rows={rows}, columns={cols if cols else []}"

        elif tool == "show_task":
            return f"Task specification:\n{self.task['task_spec']}\nExpected steps until compute_stat: {len(self.task['plan'])}"

        elif tool == "status":
            return f"Status: steps_taken={self.steps_taken}, required~={self.task['required_steps']}, turns={self.turn_count}/{self.max_turns}"

        if count_step:
            self.steps_taken += 1

        self.execution_state["columns"] = self._get_columns(self.current_data)
        self.execution_state["row_count"] = len(self.current_data) if self.current_data is not None else None

        if tool == "compute_stat":
            st = self.execution_state.get("last_stat", {})
            return f"Computed {st.get('stat')} of {st.get('col')} -> ScalarResult={st.get('value')}"

        if tool == "load":
            return f"Loaded dataset '{self.execution_state['loaded_dataset']}' with {len(self.current_data)} rows."
        if tool == "filter":
            return f"Filter applied. Rows now: {len(self.current_data)}."
        if tool == "map_product":
            return "map_product applied. Columns now: " + str(self.execution_state["columns"])
        if tool == "map_scale":
            return "map_scale applied. Columns now: " + str(self.execution_state["columns"])
        if tool == "join":
            return f"Joined with '{self.execution_state['joined_with']}'. Columns now: {self.execution_state['columns']}"
        if tool == "groupby":
            return f"Groupby applied on '{self.execution_state['grouped_by']}'. Rows: {len(self.current_data)} (groups)."
        if tool == "sort":
            return "Sort applied."
        if tool == "select":
            return "Select applied. Columns now: " + str(self.execution_state["columns"])

        return "OK."

    def _parse_action(self, action: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        m = re.search(r"\\boxed\{(.*)\}", action.strip(), flags=re.DOTALL)
        if not m:
            return None
        content = m.group(1).strip()
        if not content:
            return None
        # Tokenization: tool and key=value pairs
        parts = content.split()
        tool = parts[0].strip().lower()
        args: Dict[str, Any] = {}
        for kv in parts[1:]:
            if "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            k = k.strip()
            v = v.strip()
            if v.startswith("[") and v.endswith("]"):
                inner = v[1:-1]
                arr = [self._auto_cast(s.strip()) for s in inner.split(",") if s.strip() != ""]
                args[k] = arr
            elif "," in v and k == "cols":
                arr = [s.strip() for s in v.split(",") if s.strip() != ""]
                args[k] = arr
            else:
                args[k] = self._auto_cast(v)
        return tool, args

    def _auto_cast(self, s: Any) -> Any:
        if not isinstance(s, str):
            return s
        try:
            if "." in s:
                return float(s)
            return int(s)
        except Exception:
            return s

    def sample_random_action(self) -> str:
        if self.execution_state.get("loaded_dataset") is None:
            return "\\boxed{load dataset=orders}"
        # Randomly pick an observation or simple transform
        choices = [
            "\\boxed{preview}",
            "\\boxed{inspect_state}",
            "\\boxed{filter condition=price>50}",
            "\\boxed{map_product new=revenue a=price b=quantity}",
            "\\boxed{compute_stat col=price stat=sum}",
            "\\boxed{status}",
        ]
        return random.choice(choices)


class OpsLoomEnvWithFeedback(OpsLoomEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        text = obs.lower()
        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint: Optional[str] = None

        if "invalid action format" in text or "must be in \\boxed" in text:
            error_type = "FormatError"
            error_detail["issue"] = "missing_boxed_format"
            hint = "Use \\boxed{tool arg=value} exactly. For example: \\boxed{load dataset=orders}."

        elif "unsupported tool" in text:
            error_type = "UnsupportedAction"
            error_detail["tool"] = "unknown"
            hint = "Check available tools via instructions. Try \\boxed{show_task} or \\boxed{status}."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "no data loaded" in text:
                error_detail["violation"] = "missing_load"
                hint = "Call \\boxed{load dataset=orders} before transforms."
            elif "requires" in text:
                error_detail["violation"] = "missing_parameters"
                hint = "Supply required parameters (see tool example)."
            else:
                error_detail["violation"] = "general"
                hint = "Verify prerequisites and parameter names."

        elif "execution error" in text:
            error_type = "ProtocolViolation"
            error_detail["exception"] = obs
            hint = "Verify dataset names and columns; use \\boxed{inspect_state} to see current columns."

        elif "submission rejected" in text or "wrongdecision" in text:
            error_type = "WrongDecision"
            error_detail["expected"] = self.task.get("target_answer")
            try:
                m = re.search(r"value=([0-9\.\-]+)", obs)
                if m:
                    error_detail["got"] = float(m.group(1))
            except Exception:
                error_detail["got"] = None
            hint = "Use \\boxed{show_task} to follow the recipe, then \\boxed{compute_stat col=... stat=...} and submit that scalar."

        elif "timeout" in text:
            error_type = "Timeout"
            error_detail["outcome"] = "max_turns_reached"
            hint = "Act faster: begin with \\boxed{load dataset=orders}, then follow \\boxed{show_task}."

        elif "submission accepted" in text or "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "loaded_dataset": self.execution_state.get("loaded_dataset"),
                "joined_with": self.execution_state.get("joined_with"),
                "grouped_by": self.execution_state.get("grouped_by"),
                "steps_taken": getattr(self, "steps_taken", None),
                "required_steps": self.task.get("required_steps"),
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
            "hint": "Start by \\boxed{show_task} to see the pipeline or \\boxed{load dataset=orders}.",
            "turn": 0,
            "state": {
                "loaded_dataset": None,
                "steps_taken": 0,
                "required_steps": self.task.get("required_steps"),
            },
        }
        return obs, info