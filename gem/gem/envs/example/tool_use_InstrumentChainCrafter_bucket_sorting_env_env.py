from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List

class InstrumentChainCrafterEnv(Env):
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

    def _init_database(self):
        self.tools = {
            "load_table": {
                "description": "Load a named table into the workspace and select it as active.",
                "parameters": [{"name": "name", "type": "string"}],
                "returns": "Sets current table, adds to workspace."
            },
            "select_table": {
                "description": "Select an already loaded table as the active table.",
                "parameters": [{"name": "name", "type": "string"}],
                "returns": "Sets current table."
            },
            "join_tables": {
                "description": "Inner join two loaded tables on a common key, set result as active.",
                "parameters": [{"name": "left", "type": "string"},
                               {"name": "right", "type": "string"},
                               {"name": "on", "type": "string"},
                               {"name": "out", "type": "string"}],
                "returns": "Creates a new table in workspace and selects it."
            },
            "filter_rows": {
                "description": "Filter active table using a single condition (e.g., region=North, quantity>2).",
                "parameters": [{"name": "condition", "type": "string"}],
                "returns": "Mutates the active table to filtered rows."
            },
            "compute_column": {
                "description": "Create/update a column from an arithmetic expression using row fields and constants.",
                "parameters": [{"name": "name", "type": "string"},
                               {"name": "expr", "type": "string"}],
                "returns": "Adds/updates column for each row."
            },
            "aggregate": {
                "description": "Group by a column and aggregate a target with op (sum, count, avg, min, max).",
                "parameters": [{"name": "by", "type": "string"},
                               {"name": "op", "type": "string"},
                               {"name": "target", "type": "string"},
                               {"name": "out_name", "type": "string"}],
                "returns": "Replaces active table with grouped aggregate rows."
            },
            "sort_rows": {
                "description": "Sort the active table by a column ascending or descending.",
                "parameters": [{"name": "by", "type": "string"},
                               {"name": "order", "type": "string"}],
                "returns": "Mutates active table ordering."
            },
            "select_columns": {
                "description": "Project the active table to a subset of columns.",
                "parameters": [{"name": "cols", "type": "string"}],
                "returns": "Mutates active table columns."
            },
            "sample_rows": {
                "description": "Keep the first n rows of the active table.",
                "parameters": [{"name": "n", "type": "int"}],
                "returns": "Mutates active table rows."
            },
            "describe_table": {
                "description": "Describe the active table schema and row count.",
                "parameters": [],
                "returns": "Textual summary, does not change state."
            },
            "save_table": {
                "description": "Save the current active table as a named workspace table.",
                "parameters": [{"name": "name", "type": "string"}],
                "returns": "Adds/updates workspace entry."
            },
            "submit_answer": {
                "description": "Submit the final answer. Allowed only after enough steps.",
                "parameters": [{"name": "field", "type": "string"},
                               {"name": "value", "type": "string"}],
                "returns": "Terminal evaluation."
            },
        }
        categories = ["Gadgets", "Home", "Outdoors", "Office"]
        suppliers = ["Acme", "Nimbus", "Zenith", "Polar"]
        regions = ["North", "South", "East", "West"]
        tiers = ["Bronze", "Silver", "Gold"]
        num_products = 12 + (self.complexity - 1) * 3
        num_customers = 15 + (self.complexity - 1) * 4
        num_orders = 30 + (self.complexity - 1) * 6
        self.base_catalog = {
            "products": [],
            "customers": [],
            "orders": []
        }
        for i in range(num_products):
            pid = f"P{i+1:03d}"
            self.base_catalog["products"].append({
                "product_id": pid,
                "name": f"Prod{i+1}",
                "category": random.choice(categories),
                "supplier": random.choice(suppliers),
                "price": round(random.uniform(8.0, 120.0), 2)
            })
        for i in range(num_customers):
            cid = f"C{i+1:03d}"
            self.base_catalog["customers"].append({
                "customer_id": cid,
                "name": f"Cust{i+1}",
                "tier": random.choice(tiers),
                "region": random.choice(regions)
            })
        # Dates in 2023
        months = [f"2023-{m:02d}" for m in range(1, 13)]
        for i in range(num_orders):
            prod = random.choice(self.base_catalog["products"])
            cust = random.choice(self.base_catalog["customers"])
            date_month = random.choice(months)
            day = random.randint(1, 28)
            date = f"{date_month}-{day:02d}"
            qty = random.randint(1, 6)
            price = round(prod["price"] * random.uniform(0.9, 1.2), 2)
            self.base_catalog["orders"].append({
                "order_id": f"O{i+1:04d}",
                "customer_id": cust["customer_id"],
                "product_id": prod["product_id"],
                "quantity": qty,
                "price": price,
                "date": date,
                "region": cust["region"]
            })

    def _generate_task_requiring_n_steps(self, required_steps: int) -> Dict[str, Any]:
        regions = ["North", "South", "East", "West"]
        categories = list(set([p["category"] for p in self.base_catalog["products"]]))
        pick_region = random.choice(regions)
        pick_category = random.choice(categories)
        months_sorted = sorted(list(set([o["date"][:7] for o in self.base_catalog["orders"]])))
        if len(months_sorted) >= 3:
            start_month = random.choice(months_sorted[:-2])
            end_month = random.choice(months_sorted[months_sorted.index(start_month)+1:])
        else:
            start_month = months_sorted[0]
            end_month = months_sorted[-1]
        template_type = "top_product_by_revenue"
        if required_steps <= 3:
            template_type = "count_customers_with_orders"
        elif required_steps <= 6:
            template_type = random.choice(["top_product_by_revenue", "avg_order_value_gold"])
        else:
            template_type = random.choice(["top_product_by_revenue", "avg_order_value_gold", "count_customers_with_orders"])

        expected = None
        if template_type == "top_product_by_revenue":
            # Filter orders by region and date range, join with products (for category), compute revenue and aggregate by product
            def in_range(d):
                return start_month <= d[:7] <= end_month
            orders = [o for o in self.base_catalog["orders"] if o["region"] == pick_region and in_range(o["date"])]
            # join products to filter category
            prod_by_id = {p["product_id"]: p for p in self.base_catalog["products"]}
            orders = [o for o in orders if prod_by_id.get(o["product_id"], {}).get("category") == pick_category]
            revenue_by_prod: Dict[str, float] = {}
            for o in orders:
                rev = o["quantity"] * o["price"]
                revenue_by_prod[o["product_id"]] = revenue_by_prod.get(o["product_id"], 0.0) + rev
            if revenue_by_prod:
                expected = max(revenue_by_prod.items(), key=lambda kv: kv[1])[0]
            else:
                expected = "NONE"
            task = {
                "type": template_type,
                "region": pick_region,
                "category": pick_category,
                "start_month": start_month,
                "end_month": end_month,
                "expected": expected,
            }
        elif template_type == "avg_order_value_gold":
            def is_gold(cust_id):
                c = next((c for c in self.base_catalog["customers"] if c["customer_id"] == cust_id), None)
                return c is not None and c["tier"] == "Gold"
            # Filter orders by product category and gold customers
            prod_by_id = {p["product_id"]: p for p in self.base_catalog["products"]}
            selected_orders = [o for o in self.base_catalog["orders"] if is_gold(o["customer_id"]) and prod_by_id.get(o["product_id"], {}).get("category") == pick_category]
            values = [o["quantity"] * o["price"] for o in selected_orders]
            expected_val = round(sum(values) / len(values), 2) if values else 0.0
            task = {
                "type": template_type,
                "category": pick_category,
                "expected": expected_val,
            }
        else:
            # count customers having more than K orders in region
            pick_region = pick_region
            orders = [o for o in self.base_catalog["orders"] if o["region"] == pick_region]
            counts: Dict[str, int] = {}
            for o in orders:
                counts[o["customer_id"]] = counts.get(o["customer_id"], 0) + 1
            possible_k = [1, 2, 3, 4]
            k = random.choice(possible_k)
            expected_ct = sum(1 for v in counts.values() if v > k)
            task = {
                "type": template_type,
                "region": pick_region,
                "k": k,
                "expected": expected_ct,
            }

        gate = required_steps  # minimum number of tool calls required before submission allowed
        task["required_steps_before_submit"] = gate
        return task

    def _dataset_tables(self) -> Dict[str, List[Dict[str, Any]]]:
        products = self.base_catalog["products"]
        customers = self.base_catalog["customers"]
        orders = self.base_catalog["orders"]
        return {"products": products, "customers": customers, "orders": orders}

    def _get_instructions(self) -> str:
        tool_list = "\n".join([f"- {name}: {meta['description']}" for name, meta in self.tools.items()])
        fmt = "Use \\boxed{tool: arg1=value; arg2=value} with semicolon-separated args. One tool per turn."
        return (
            "You are assembling a data transformation workflow using provided tools.\n"
            "Goal: Solve the stated objective by orchestrating tool calls, then submit your final answer.\n"
            "Rules:\n"
            "- Maintain state: load/select tables, join/filter/compute/aggregate, and inspect results.\n"
            "- You must perform a minimum number of tool calls before using submit_answer.\n"
            "- Submissions ending the episode: correct (success) or incorrect (failure).\n"
            "- Format errors or unknown tools terminate immediately with penalty.\n"
            "Available tools:\n" + tool_list + "\n"
            "Action format:\n" + fmt
        )

    def get_task_suffix(self) -> str:
        gate = self.task["required_steps_before_submit"]
        base_info = self._dataset_tables()
        tables_summary = ", ".join([f"{name}({len(rows)} rows)" for name, rows in base_info.items()])
        cols = {
            "products": ["product_id", "name", "category", "supplier", "price"],
            "customers": ["customer_id", "name", "tier", "region"],
            "orders": ["order_id", "customer_id", "product_id", "quantity", "price", "date", "region"],
        }
        cols_text = "; ".join([f"{k}: {', '.join(v)}" for k, v in cols.items()])
        if self.task["type"] == "top_product_by_revenue":
            obj = (
                f"Objective: Find the product_id with highest total revenue in region={self.task['region']} "
                f"for months [{self.task['start_month']}..{self.task['end_month']}] restricted to category={self.task['category']}.\n"
                "Revenue is quantity * price at order level, aggregated per product."
            )
            submit_fmt = "Submit with \\boxed{submit_answer: field=product_id; value=PXYZ}."
        elif self.task["type"] == "avg_order_value_gold":
            obj = (
                f"Objective: Compute the average order value (quantity*price) for Gold customers restricted to products in category={self.task['category']}."
            )
            submit_fmt = "Submit with \\boxed{submit_answer: field=avg_value; value=NUMBER} using 2 decimal precision."
        else:
            obj = (
                f"Objective: Count customers in region={self.task['region']} who have strictly more than k={self.task['k']} orders."
            )
            submit_fmt = "Submit with \\boxed{submit_answer: field=count; value=INTEGER}."
        return (
            f"Workspace tables: {tables_summary}\n"
            f"Columns: {cols_text}\n"
            f"{obj}\n"
            f"Submission gate: At least {gate} tool calls before submit_answer is allowed.\n"
            f"{submit_fmt}\n"
            "Reminder: One tool call per turn in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        required_steps = random.randint(self.min_required_steps, self.max_required_steps)
        self.task = self._generate_task_requiring_n_steps(required_steps)
        self.turn_count = 0
        self.steps_taken = 0
        self.workspace_tables: Dict[str, List[Dict[str, Any]]] = {}
        self.current_table_name: Optional[str] = None
        self.current_table: Optional[List[Dict[str, Any]]] = None
        self.provenance: List[str] = []
        self.simulated_tables = self._dataset_tables()
        instructions = self._get_instructions()
        info = {"suffix": self.get_task_suffix()}
        return instructions, info

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        info: Dict[str, Any] = {"suffix": self.get_task_suffix()}
        if self.turn_count >= self.max_turns:
            obs = "TIMEOUT: Maximum turns reached. Episode terminated."
            return obs, -0.2, True, True, info

        parsed = self._parse_action(action)
        if not parsed:
            obs = "INVALID ACTION FORMAT: Use \\boxed{tool: arg1=value; arg2=value}."
            return obs, LanguageGameReward.format_error_reward, True, False, info

        tool_name, args = parsed
        if tool_name not in self.tools:
            obs = f"UNSUPPORTED TOOL: {tool_name}. Episode terminated."
            return obs, -0.5, True, False, info

        if tool_name == "submit_answer":
            gate = self.task["required_steps_before_submit"]
            if self.steps_taken < gate:
                remaining = gate - self.steps_taken
                obs = f"PROTOCOL VIOLATION: submit_answer not allowed yet. Need {remaining} more tool call(s) before submission."
                return obs, -0.1, False, False, info
            field = args.get("field")
            value = args.get("value")
            outcome_obs, reward = self._evaluate_submission(field, value)
            terminated = True
            return outcome_obs, reward, terminated, False, info

        try:
            result_text, counted = self._execute_tool(tool_name, args)
            if counted:
                self.steps_taken += 1
                self.provenance.append(tool_name)
            gate = self.task["required_steps_before_submit"]
            progress = f"Progress: {self.steps_taken}/{gate} required tool calls before submission allowed."
            obs = f"Tool {tool_name} executed.\n{result_text}\n{progress}"
            return obs, 0.0, False, False, info
        except Exception as e:
            obs = f"EXECUTION ERROR: {str(e)}"
            return obs, -0.25, False, False, info

    def _evaluate_submission(self, field: Optional[str], value: Optional[str]) -> Tuple[str, float]:
        if self.task["type"] == "top_product_by_revenue":
            if field != "product_id":
                return "FAILURE: Wrong field. Expected field=product_id.", -0.4
            expected = self.task["expected"]
            if expected == "NONE":
                ok = (value == "NONE")
            else:
                ok = (value == expected)
            if ok:
                return f"SUCCESS: Correct product_id submitted ({value}).", 1.0
            else:
                return f"FAILURE: Incorrect product_id ({value}).", -0.5
        elif self.task["type"] == "avg_order_value_gold":
            if field != "avg_value":
                return "FAILURE: Wrong field. Expected field=avg_value.", -0.4
            try:
                submitted = round(float(value), 2)
            except:
                return "FAILURE: avg_value must be a number with 2 decimals.", -0.5
            expected = round(float(self.task["expected"]), 2)
            if abs(submitted - expected) < 1e-9:
                return f"SUCCESS: Correct average value ({submitted}).", 1.0
            else:
                return f"FAILURE: Incorrect average value ({submitted}).", -0.5
        else:
            if field != "count":
                return "FAILURE: Wrong field. Expected field=count.", -0.4
            try:
                submitted = int(value)
            except:
                return "FAILURE: count must be an integer.", -0.5
            expected = int(self.task["expected"])
            if submitted == expected:
                return f"SUCCESS: Correct count ({submitted}).", 1.0
            else:
                return f"FAILURE: Incorrect count ({submitted}).", -0.5

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Tuple[str, bool]:
        counted = tool_name not in ("describe_table",)
        if tool_name == "load_table":
            name = args.get("name")
            if name not in self.simulated_tables:
                raise ValueError(f"Table '{name}' not found.")
            self.workspace_tables[name] = [dict(r) for r in self.simulated_tables[name]]
            self.current_table_name = name
            self.current_table = self.workspace_tables[name]
            return f"Loaded table '{name}' with {len(self.current_table)} rows.", counted

        if tool_name == "select_table":
            name = args.get("name")
            if name not in self.workspace_tables:
                raise ValueError(f"Table '{name}' not loaded in workspace.")
            self.current_table_name = name
            self.current_table = self.workspace_tables[name]
            return f"Selected active table '{name}'.", counted

        if tool_name == "join_tables":
            left = args.get("left")
            right = args.get("right")
            on = args.get("on")
            out = args.get("out") or f"join_{left}_{right}"
            if left not in self.workspace_tables or right not in self.workspace_tables:
                raise ValueError("Both left and right tables must be loaded.")
            left_rows = self.workspace_tables[left]
            right_rows = self.workspace_tables[right]
            idx: Dict[Any, List[Dict[str, Any]]] = {}
            for rr in right_rows:
                key = rr.get(on)
                idx.setdefault(key, []).append(rr)
            joined: List[Dict[str, Any]] = []
            for lr in left_rows:
                key = lr.get(on)
                if key in idx:
                    for rr in idx[key]:
                        merged = dict(lr)
                        for k, v in rr.items():
                            if k in merged and k != on:
                                merged[f"{right}_{k}"] = v
                            else:
                                merged[k] = v
                        joined.append(merged)
            self.workspace_tables[out] = joined
            self.current_table_name = out
            self.current_table = joined
            return f"Joined '{left}' and '{right}' on '{on}', result '{out}' with {len(joined)} rows.", counted

        if tool_name == "filter_rows":
            if self.current_table is None:
                raise ValueError("No active table. Call load_table or select_table first.")
            cond = args.get("condition")
            col, op, val = self._parse_condition(cond)
            filtered = []
            for r in self.current_table:
                rv = r.get(col)
                cmp = self._compare(rv, op, val)
                if cmp:
                    filtered.append(r)
            self.current_table[:] = filtered
            return f"Filtered rows with condition '{cond}'. Remaining rows: {len(filtered)}.", counted

        if tool_name == "compute_column":
            if self.current_table is None:
                raise ValueError("No active table. Call load_table or select_table first.")
            name = args.get("name")
            expr = args.get("expr")
            tokens = self._tokenize_expr(expr)
            for r in self.current_table:
                r[name] = self._eval_expr(tokens, r)
            return f"Computed column '{name}' from expr '{expr}'.", counted

        if tool_name == "aggregate":
            if self.current_table is None:
                raise ValueError("No active table. Call load_table or select_table first.")
            by = args.get("by")
            op = args.get("op")
            target = args.get("target")
            out_name = args.get("out_name") or f"{op}_{target}"
            groups: Dict[Any, List[Dict[str, Any]]] = {}
            for r in self.current_table:
                key = r.get(by)
                groups.setdefault(key, []).append(r)
            result = []
            for k, rows in groups.items():
                vals = [self._to_float_safe(row.get(target)) for row in rows]
                if op == "sum":
                    agg = sum(vals)
                elif op == "count":
                    agg = len(vals)
                elif op == "avg":
                    agg = sum(vals) / len(vals) if vals else 0.0
                elif op == "min":
                    agg = min(vals) if vals else 0.0
                elif op == "max":
                    agg = max(vals) if vals else 0.0
                else:
                    raise ValueError(f"Unsupported aggregate op '{op}'.")
                result.append({by: k, out_name: round(agg, 2) if isinstance(agg, float) else agg})
            self.current_table = result
            self.current_table_name = f"agg_{by}"
            self.workspace_tables[self.current_table_name] = result
            return f"Aggregated by '{by}' with op '{op}' on '{target}'. Rows: {len(result)}.", counted

        if tool_name == "sort_rows":
            if self.current_table is None:
                raise ValueError("No active table. Call load_table or select_table first.")
            by = args.get("by")
            order = args.get("order", "asc").lower()
            reverse = order == "desc"
            self.current_table.sort(key=lambda r: r.get(by, 0), reverse=reverse)
            return f"Sorted rows by '{by}' in {order} order.", counted

        if tool_name == "select_columns":
            if self.current_table is None:
                raise ValueError("No active table. Call load_table or select_table first.")
            cols = [c.strip() for c in (args.get("cols") or "").split(",") if c.strip()]
            new_rows = []
            for r in self.current_table:
                new_rows.append({c: r.get(c) for c in cols})
            self.current_table[:] = new_rows
            return f"Selected columns: {', '.join(cols)}.", counted

        if tool_name == "sample_rows":
            if self.current_table is None:
                raise ValueError("No active table. Call load_table or select_table first.")
            n = int(args.get("n"))
            self.current_table[:] = self.current_table[:n]
            return f"Sampled first {n} rows.", counted

        if tool_name == "describe_table":
            if self.current_table is None:
                raise ValueError("No active table. Call load_table or select_table first.")
            if len(self.current_table) == 0:
                cols = []
            else:
                cols = list(self.current_table[0].keys())
            return f"Active table '{self.current_table_name}' has {len(self.current_table)} rows and columns: {', '.join(cols)}.", counted

        if tool_name == "save_table":
            if self.current_table is None:
                raise ValueError("No active table. Call load_table or select_table first.")
            name = args.get("name")
            self.workspace_tables[name] = [dict(r) for r in self.current_table]
            return f"Saved active table as '{name}'.", counted

        raise ValueError(f"Tool '{tool_name}' not implemented.")

    def _tokenize_expr(self, expr: str) -> List[str]:
        # Simple tokenizer for + - * / and identifiers/constants
        return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+\.\d+|\d+|[\+\-\*/\(\)]", expr)

    def _eval_expr(self, tokens: List[str], row: Dict[str, Any]) -> float:
        # Shunting-yard is overkill; simple left-to-right with * and / precedence
        def val(tok):
            if re.fullmatch(r"\d+\.\d+|\d+", tok):
                return float(tok)
            return self._to_float_safe(row.get(tok, 0))
        # Convert to RPN using a simple precedence rule
        out = []
        ops = []
        prec = {"+": 1, "-": 1, "*": 2, "/": 2}
        for t in tokens:
            if t in prec:
                while ops and ops[-1] in prec and prec[ops[-1]] >= prec[t]:
                    out.append(ops.pop())
                ops.append(t)
            elif t == "(":
                ops.append(t)
            elif t == ")":
                while ops and ops[-1] != "(":
                    out.append(ops.pop())
                if ops and ops[-1] == "(":
                    ops.pop()
            else:
                out.append(t)
        while ops:
            out.append(ops.pop())
        stack = []
        for t in out:
            if t in prec:
                b = stack.pop()
                a = stack.pop()
                if t == "+":
                    stack.append(a + b)
                elif t == "-":
                    stack.append(a - b)
                elif t == "*":
                    stack.append(a * b)
                elif t == "/":
                    stack.append(0.0 if b == 0 else a / b)
            else:
                stack.append(val(t))
        return stack[0] if stack else 0.0

    def _parse_condition(self, cond: str) -> Tuple[str, str, Any]:
        m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(==|=|!=|>=|<=|>|<)\s*(.+?)\s*$", cond or "")
        if not m:
            raise ValueError(f"Invalid condition syntax '{cond}'.")
        col, op, val_raw = m.groups()
        val = self._parse_value(val_raw)
        if op == "=":
            op = "=="
        return col, op, val

    def _compare(self, rv: Any, op: str, val: Any) -> bool:
        if op == "==":
            return rv == val
        if op == "!=":
            return rv != val
        try:
            a = rv
            b = val
            return {
                ">": lambda x, y: x > y,
                "<": lambda x, y: x < y,
                ">=": lambda x, y: x >= y,
                "<=": lambda x, y: x <= y,
            }[op](a, b)
        except Exception:
            return False

    def _to_float_safe(self, v: Any) -> float:
        try:
            return float(v)
        except Exception:
            return 0.0

    def _parse_action(self, action: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        if not isinstance(action, str):
            return None
        m = re.search(r"\\boxed\{(.*)\}", action, flags=re.DOTALL)
        if not m:
            return None
        content = m.group(1).strip()
        if ":" not in content:
            return None
        tool, arg_str = content.split(":", 1)
        tool = tool.strip()
        args = {}
        parts = [p.strip() for p in re.split(r"[;,\n]", arg_str) if p.strip()]
        for p in parts:
            if "=" not in p:
                continue
            k, v = p.split("=", 1)
            k = k.strip()
            v = v.strip()
            args[k] = self._parse_value(v)
        return tool, args

    def _parse_value(self, v: str) -> Any:
        v = v.strip()
        if re.fullmatch(r"\".*\"|'.*'", v):
            return v[1:-1]
        if re.fullmatch(r"\d+\.\d+", v):
            return float(v)
        if re.fullmatch(r"\d+", v):
            return int(v)
        if v.lower() in ("true", "false"):
            return v.lower() == "true"
        return v

    def sample_random_action(self) -> str:
        candidates = [
            "\\boxed{load_table: name=orders}",
            "\\boxed{select_table: name=orders}",
            "\\boxed{filter_rows: condition=region=North}",
            "\\boxed{compute_column: name=revenue; expr=quantity*price}",
            "\\boxed{aggregate: by=product_id; op=sum; target=revenue; out_name=total_revenue}",
            "\\boxed{sort_rows: by=total_revenue; order=desc}",
            "\\boxed{submit_answer: field=product_id; value=P001}",
        ]
        return random.choice(candidates)


class InstrumentChainCrafterEnvWithFeedback(InstrumentChainCrafterEnv):
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
            hint = "Use \\boxed{tool: arg=value; ...} with a single tool per turn."

        elif "unsupported tool" in text:
            error_type = "UnsupportedAction"
            error_detail["tool"] = self._extract_unknown_tool(obs)
            hint = "Choose a tool from the provided list. Try 'load_table' or 'select_table' first."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            remaining = self._extract_remaining_steps(obs)
            error_detail["remaining_steps"] = remaining
            hint = f"Perform at least {remaining} more tool call(s) before submitting. Load, join, filter, compute, and aggregate as needed."

        elif "execution error" in text:
            error_type = "ExecutionError"
            error_detail["message"] = obs
            hint = "Check preconditions: ensure an active table exists, correct column names, and valid condition syntax."

        elif "failure" in text:
            error_type = "WrongDecision"
            error_detail["expected_type"] = self.task.get("type")
            error_detail["expected_field"] = self._expected_field_for_task()
            if self.feedback_level >= 2:
                hint = self._task_hint()

        elif "timeout" in text:
            error_type = "Timeout"
            error_detail["outcome"] = "max_turns_reached"
            hint = "Start earlier with load_table and iterate towards aggregation; avoid unnecessary steps."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = "Good job."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["steps_taken"] = getattr(self, "steps_taken", None)
            diagnostic["required_steps_before_submit"] = self.task.get("required_steps_before_submit")
            diagnostic["active_table"] = getattr(self, "current_table_name", None)
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        first_hint = "Begin by loading a relevant table: \\boxed{load_table: name=orders} or \\boxed{load_table: name=products}."
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": first_hint if self.feedback_level >= 2 else None,
            "turn": 0,
            "steps_taken": 0,
            "required_steps_before_submit": self.task.get("required_steps_before_submit"),
            "active_table": None,
        }
        return obs, info

    def _extract_unknown_tool(self, obs: str) -> Optional[str]:
        m = re.search(r"UNSUPPORTED TOOL: ([A-Za-z_][A-Za-z0-9_]*)", obs)
        return m.group(1) if m else None

    def _extract_remaining_steps(self, obs: str) -> Optional[int]:
        m = re.search(r"Need (\d+) more tool call\(s\)", obs)
        return int(m.group(1)) if m else None

    def _expected_field_for_task(self) -> Optional[str]:
        t = self.task.get("type")
        if t == "top_product_by_revenue":
            return "product_id"
        if t == "avg_order_value_gold":
            return "avg_value"
        if t == "count_customers_with_orders":
            return "count"
        return None

    def _task_hint(self) -> str:
        t = self.task.get("type")
        if t == "top_product_by_revenue":
            return "Load orders and products; filter orders by region and month range; join on product_id to restrict category; compute revenue=quantity*price; aggregate sum by product_id; sort desc and submit the top product_id."
        if t == "avg_order_value_gold":
            return "Load orders and customers; filter customers to Gold; join orders to customers on customer_id; restrict products to the category via join or filter; compute revenue; aggregate avg over all orders and submit avg_value."
        return "Load orders; filter by region; aggregate count per customer_id; count entries with count>k and submit the integer."