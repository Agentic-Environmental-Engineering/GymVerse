from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List, Union

class UtilityLoomEnv(Env):
    def __init__(self, complexity: int = 1, max_turns: Optional[int] = 100, **_):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.max_turns = max_turns if max_turns is not None else 100
        self.min_required_steps = self.complexity
        self.max_required_steps = self.complexity * 2
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.sources: Dict[str, List[Dict[str, Any]]] = {}
        self._init_database()
        self.reset()

    def _init_database(self):
        base_orders = 20 + (self.complexity - 1) * 5
        base_products = max(5, 8 + self.complexity)
        base_customers = max(8, 12 + self.complexity)

        random.seed(12345 + self.complexity)
        categories = ["Gadgets", "Home", "Outdoor", "Books", "Office", "Toys"]
        regions = ["North", "South", "East", "West"]
        tiers = ["Bronze", "Silver", "Gold"]

        product_ids = list(range(1000, 1000 + base_products))
        customer_ids = list(range(5000, 5000 + base_customers))
        products = []
        for pid in product_ids:
            p = {
                "product_id": pid,
                "category": random.choice(categories),
                "unit_price": round(random.uniform(5, 200), 2),
            }
            products.append(p)

        customers = []
        for cid in customer_ids:
            c = {
                "customer_id": cid,
                "region": random.choice(regions),
                "tier": random.choice(tiers),
            }
            customers.append(c)

        orders = []
        for i in range(base_orders):
            pid = random.choice(product_ids)
            prod = next(x for x in products if x["product_id"] == pid)
            qty = random.randint(1, 12)
            unit_price = prod["unit_price"]
            amount = round(qty * unit_price, 2)
            cid = random.choice(customer_ids)
            order = {
                "order_id": 100000 + i,
                "product_id": pid,
                "customer_id": cid,
                "quantity": qty,
                "unit_price": unit_price,
                "amount": amount,
                "date": f"2025-01-{random.randint(1,28):02d}",
            }
            orders.append(order)

        self.sources = {
            "orders": orders,
            "products": products,
            "customers": customers,
        }

        self.tools = {
            "load": {
                "description": "Load a source dataset into the working set.",
                "parameters": [{"name": "source", "type": "string"}],
                "returns": "Sets current dataset; returns row count.",
            },
            "schema": {
                "description": "Show the current dataset columns.",
                "parameters": [],
                "returns": "List of column names.",
            },
            "peek": {
                "description": "Preview first n rows of the current dataset.",
                "parameters": [{"name": "n", "type": "int"}],
                "returns": "First n rows.",
            },
            "count": {
                "description": "Count rows in the current dataset.",
                "parameters": [],
                "returns": "Integer row count.",
            },
            "select": {
                "description": "Keep only the specified columns in the current dataset.",
                "parameters": [{"name": "columns", "type": "list[str]"}],
                "returns": "Updates current dataset schema.",
            },
            "filter": {
                "description": "Filter rows by a simple condition: field op value (op in ==,!=,>,<,>=,<=).",
                "parameters": [{"name": "condition", "type": "string"}],
                "returns": "Updates current dataset by filtering.",
            },
            "map": {
                "description": "Create new_field by expression: field*coef or field+const (numeric).",
                "parameters": [
                    {"name": "new_field", "type": "string"},
                    {"name": "expr", "type": "string"},
                ],
                "returns": "Adds/updates a numeric column.",
            },
            "join": {
                "description": "Join current dataset with another source on key mapping.",
                "parameters": [
                    {"name": "other_source", "type": "string"},
                    {"name": "on", "type": "list[str]"},
                    {"name": "how", "type": "string"},
                ],
                "returns": "Updates current dataset with join result.",
            },
            "aggregate": {
                "description": "Compute metric over field: metric in sum,avg,min,max.",
                "parameters": [
                    {"name": "metric", "type": "string"},
                    {"name": "field", "type": "string"},
                ],
                "returns": "Numeric result stored as last_result.",
            },
            "help": {
                "description": "List available tools.",
                "parameters": [],
                "returns": "Textual tool descriptions.",
            },
            "task": {
                "description": "Repeat the current task description.",
                "parameters": [],
                "returns": "Task description.",
            },
        }

        self.user_current_ds: Optional[List[Dict[str, Any]]] = None
        self.user_loaded_source_name: Optional[str] = None
        self.last_result: Optional[Union[int, float]] = None
        self.execution_state: Dict[str, Any] = {}

    def _get_instructions(self) -> str:
        lines = []
        lines.append("You are operating a small tool suite to solve a data task.")
        lines.append("Use tools via \\boxed{...} commands. Categories:")
        lines.append("- query:help | query:task")
        lines.append("- tool:load(source=NAME)")
        lines.append("- tool:schema() | tool:peek(n=3) | tool:count()")
        lines.append("- tool:select(columns=[col1,col2])")
        lines.append("- tool:filter(condition=\"field op value\")")
        lines.append("- tool:map(new_field=name, expr=\"field*coef\" or \"field+const\")")
        lines.append("- tool:join(other_source=NAME, on=[this_key,other_key], how=inner)")
        lines.append("- tool:aggregate(metric=sum|avg|min|max, field=NAME)")
        lines.append("When you have the final numeric answer, submit with \\boxed{final:submit(value=NUMBER)}")
        return "\n".join(lines)

    def get_task_suffix(self) -> str:
        loaded = "yes" if self.user_current_ds is not None else "no"
        turns_left = self.max_turns - self.turn_count
        return f"State: loaded={loaded}, steps_used={self.steps_taken}, required_steps≈{self.required_steps}, turns_left={turns_left}. Respond using \\boxed{...}."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self.turn_count = 0
        self.steps_taken = 0
        self.user_current_ds = None
        self.user_loaded_source_name = None
        self.last_result = None
        self.execution_state = {}

        self.required_steps = random.randint(self.min_required_steps, self.max_required_steps)
        self.task_spec = self._generate_task_requiring_n_steps(self.required_steps)

        obs = self._get_instructions() + "\n\n" + self.task_spec["text"]
        info = {"suffix": self.get_task_suffix()}
        return obs, info

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False
        info: Dict[str, Any] = {"suffix": self.get_task_suffix()}

        parsed = self._parse_action(action)
        if parsed is None:
            obs = "Invalid action format. Use \\boxed{...} with a valid command."
            reward = float(LanguageGameReward.format_error_reward)
            terminated = True
            return obs, reward, terminated, truncated, info

        kind = parsed.get("kind")
        if kind == "query":
            q = parsed.get("name")
            if q == "help":
                obs = self._obs_help()
                reward = 0.0
            elif q == "task":
                obs = self.task_spec["text"]
                reward = 0.0
            else:
                obs = f"Unsupported action: query:{q}"
                reward = 0.0  # Fixed: was -1.0, should be 0.0 for failures
                terminated = True
            return obs, reward, terminated, truncated, info

        if kind == "final":
            value = parsed.get("value")
            try:
                v = float(value)
            except Exception:
                obs = "Final submission error: value must be numeric."
                reward = 0.0  # Fixed: was -0.5, should be 0.0 for failures
                terminated = True
                return obs, reward, terminated, truncated, info

            correct = abs(v - float(self.task_spec["target_value"])) <= 1e-6
            if correct:
                obs = f"Success: Correct final value submitted ({v})."
                reward = 1.0
            else:
                obs = f"Wrong final answer: submitted {v}, expected {self.task_spec['target_value']}."
                reward = 0.0  # Fixed: was -0.5, should be 0.0 for failures
            terminated = True
            return obs, reward, terminated, truncated, info

        if kind == "tool":
            name = parsed.get("name")
            args = parsed.get("args", {})
            if name not in self.tools:
                obs = f"Unsupported action: tool:{name}"
                reward = 0.0  # Fixed: was -1.0, should be 0.0 for failures
                terminated = True
                return obs, reward, terminated, truncated, info
            try:
                result_text, step_count_increment = self._execute_tool(name, args)
                if step_count_increment:
                    self.steps_taken += 1
                obs = result_text
                reward = 0.0
            except ValueError as ve:
                obs = f"Protocol violation: {ve}"
                reward = 0.0  # Fixed: was -0.1, should be 0.0 for failures
                terminated = True  # Protocol violations should end episode
            except Exception as e:
                obs = f"Execution error: {e}"
                reward = 0.0  # Fixed: was -0.2, should be 0.0 for failures
                terminated = True  # Execution errors should end episode
        else:
            obs = f"Unsupported action: {kind}"
            reward = 0.0  # Fixed: was -1.0, should be 0.0 for failures
            terminated = True

        if not terminated and self.turn_count >= self.max_turns:
            truncated = True
            terminated = True
            obs = "Timeout: maximum turns reached before submission."
            reward = 0.0

        info["suffix"] = self.get_task_suffix()
        return obs, reward, terminated, truncated, info

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not isinstance(action, str):
            return None
        m = re.search(r"\\boxed\{(.*)\}\s*$", action.strip(), flags=re.DOTALL)
        if not m:
            return None
        inner = m.group(1).strip()

        if inner.startswith("query:"):
            name = inner.split(":", 1)[1].strip()
            return {"kind": "query", "name": name}

        if inner.startswith("final:submit"):
            vm = re.search(r"final:submit\s*\(\s*value\s*=\s*([^)]+)\)", inner)
            if not vm:
                return {"kind": "final", "value": None}
            val = vm.group(1).strip()
            val = val.strip("\"'")
            return {"kind": "final", "value": val}

        if inner.startswith("tool:"):
            tm = re.match(r"tool:([a-zA-Z_][a-zA-Z0-9_]*)\s*(\((.*)\))?$", inner, flags=re.DOTALL)
            if not tm:
                return None
            name = tm.group(1)
            args_str = tm.group(3) if tm.group(3) else ""
            args = self._parse_args(args_str)
            return {"kind": "tool", "name": name, "args": args}

        return None

    def _parse_args(self, args_str: str) -> Dict[str, Any]:
        args: Dict[str, Any] = {}
        if not args_str.strip():
            return args
        parts = self._split_args(args_str)
        for p in parts:
            if "=" not in p:
                continue
            k, v = p.split("=", 1)
            key = k.strip()
            val = v.strip()
            parsed_val = self._parse_value(val)
            args[key] = parsed_val
        return args

    def _split_args(self, s: str) -> List[str]:
        parts = []
        depth = 0
        buf = ""
        in_str = False
        quote_char = ""
        for ch in s:
            if in_str:
                buf += ch
                if ch == quote_char:
                    in_str = False
                continue
            if ch in ("'", '"'):
                in_str = True
                quote_char = ch
                buf += ch
                continue
            if ch == "[":
                depth += 1
                buf += ch
                continue
            if ch == "]":
                depth -= 1
                buf += ch
                continue
            if ch == "," and depth == 0:
                parts.append(buf.strip())
                buf = ""
                continue
            buf += ch
        if buf.strip():
            parts.append(buf.strip())
        return parts

    def _parse_value(self, v: str) -> Any:
        v = v.strip()
        if v.startswith("[") and v.endswith("]"):
            inner = v[1:-1].strip()
            if not inner:
                return []
            items = self._split_args(inner)
            return [self._parse_value(x) for x in items]
        if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
            return v[1:-1]
        if re.fullmatch(r"-?\d+\.\d+", v):
            return float(v)
        if re.fullmatch(r"-?\d+", v):
            return int(v)
        return v

    def sample_random_action(self) -> str:
        choices = [
            r"\boxed{query:help}",
            r"\boxed{tool:load(source=orders)}",
            r"\boxed{tool:schema()}",
            r"\boxed{tool:peek(n=3)}",
        ]
        return random.choice(choices)

    def _obs_help(self) -> str:
        lines = ["Available tools:"]
        for name, spec in self.tools.items():
            if name in ["help", "task"]:
                continue
            lines.append(f"- {name}: {spec['description']}")
        lines.append("Queries: query:help, query:task")
        lines.append("Submit: final:submit(value=NUMBER)")
        return "\n".join(lines)

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> Tuple[str, bool]:
        increment = False
        if name == "help":
            return self._obs_help(), False
        if name == "task":
            return self.task_spec["text"], False

        if name == "load":
            src = args.get("source")
            if src not in self.sources:
                raise ValueError(f"unknown source '{src}'. Known: {list(self.sources.keys())}")
            self.user_loaded_source_name = src
            # deep copy rows
            self.user_current_ds = [dict(r) for r in self.sources[src]]
            increment = True
            return f"Loaded '{src}'. Rows={len(self.user_current_ds)}.", increment

        if name in ["schema", "peek", "count", "select", "filter", "map", "join", "aggregate"]:
            if self.user_current_ds is None and name != "aggregate":
                raise ValueError("no data loaded. Call tool:load(source=...) first.")

        if name == "schema":
            cols = list(self.user_current_ds[0].keys()) if self.user_current_ds else []
            return f"Schema: {cols}", False

        if name == "peek":
            n = int(args.get("n", 3))
            rows = self.user_current_ds[: max(0, n)] if self.user_current_ds else []
            return f"Peek first {n} rows: {rows}", False

        if name == "count":
            c = len(self.user_current_ds) if self.user_current_ds else 0
            return f"Count: {c}", False

        if name == "select":
            cols = args.get("columns")
            if not isinstance(cols, list) or not cols:
                raise ValueError("select requires columns=[col1,...]")
            new_ds = []
            for r in self.user_current_ds:
                new_ds.append({k: r.get(k) for k in cols})
            self.user_current_ds = new_ds
            increment = True
            return f"Selected columns: {cols}. New schema={list(self.user_current_ds[0].keys()) if self.user_current_ds else cols}.", increment

        if name == "filter":
            cond = args.get("condition")
            if not isinstance(cond, str) or not cond.strip():
                raise ValueError("filter requires condition string")
            field, op, val = self._parse_condition(cond)
            if self.user_current_ds and field not in self.user_current_ds[0]:
                raise ValueError(f"unknown field '{field}' in condition")
            filtered = []
            for r in self.user_current_ds:
                rv = r.get(field)
                if self._eval_condition(rv, op, val):
                    filtered.append(r)
            self.user_current_ds = filtered
            increment = True
            return f"Filter applied: {field} {op} {val}. Rows now={len(filtered)}.", increment

        if name == "map":
            nf = args.get("new_field")
            expr = args.get("expr")
            if not nf or not isinstance(expr, str):
                raise ValueError("map requires new_field and expr")
            m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*([\*\+])\s*([\-]?\d+(\.\d+)?)\s*$", expr)
            if not m:
                raise ValueError("expr must be 'field*coef' or 'field+const'")
            fld = m.group(1)
            op = m.group(2)
            coef = float(m.group(3))
            if self.user_current_ds and fld not in self.user_current_ds[0]:
                raise ValueError(f"unknown field '{fld}' in expr")
            new_ds = []
            for r in self.user_current_ds:
                basev = r.get(fld)
                if not isinstance(basev, (int, float)):
                    raise ValueError(f"field '{fld}' is not numeric in data")
                if op == "*":
                    nv = basev * coef
                else:
                    nv = basev + coef
                nr = dict(r)
                nr[nf] = round(float(nv), 6)
                new_ds.append(nr)
            self.user_current_ds = new_ds
            increment = True
            return f"Mapped '{nf}' from expr '{expr}'.", increment

        if name == "join":
            other = args.get("other_source")
            on = args.get("on")
            how = args.get("how", "inner")
            if other not in self.sources:
                raise ValueError(f"unknown other_source '{other}'")
            if not isinstance(on, list) or len(on) != 2:
                raise ValueError("join requires on=[this_key,other_key]")
            if how != "inner":
                raise ValueError("only inner join is supported")
            this_key, other_key = on[0], on[1]
            other_rows = self.sources[other]
            index = {}
            for r in other_rows:
                index.setdefault(r.get(other_key), []).append(r)
            joined = []
            for r in self.user_current_ds:
                val = r.get(this_key)
                matches = index.get(val, [])
                for mrow in matches:
                    jr = dict(r)
                    for k, v in mrow.items():
                        if k in jr:
                            jr[f"{other}_{k}"] = v
                        else:
                            jr[k] = v
                    joined.append(jr)
            self.user_current_ds = joined
            increment = True
            return f"Joined with '{other}' on {this_key}={other_key}. Rows now={len(joined)}.", increment

        if name == "aggregate":
            metric = args.get("metric")
            field = args.get("field")
            ds = self.user_current_ds
            if ds is None:
                raise ValueError("no data loaded. Call tool:load(source=...) first.")
            if not ds:
                vals = []
            else:
                if field not in ds[0]:
                    raise ValueError(f"unknown field '{field}' for aggregate")
                vals = [r.get(field) for r in ds if isinstance(r.get(field), (int, float))]
            if metric == "sum":
                res = sum(vals) if vals else 0.0
            elif metric == "avg":
                res = (sum(vals) / len(vals)) if vals else 0.0
            elif metric == "min":
                res = min(vals) if vals else 0.0
            elif metric == "max":
                res = max(vals) if vals else 0.0
            else:
                raise ValueError("metric must be one of sum,avg,min,max")
            self.last_result = round(float(res), 6)
            increment = True
            return f"Aggregate {metric}({field}) = {self.last_result}", increment

        raise ValueError(f"tool '{name}' not implemented")

    def _parse_condition(self, cond: str) -> Tuple[str, str, Any]:
        m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(==|!=|>=|<=|>|<)\s*(.+)\s*$", cond)
        if not m:
            raise ValueError("condition must be 'field op value'")
        field = m.group(1)
        op = m.group(2)
        raw = m.group(3).strip()
        if (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
            val: Any = raw[1:-1]
        elif re.fullmatch(r"-?\d+\.\d+", raw):
            val = float(raw)
        elif re.fullmatch(r"-?\d+", raw):
            val = int(raw)
        else:
            # treat as bare string token
            val = raw
        return field, op, val

    def _eval_condition(self, rv: Any, op: str, val: Any) -> bool:
        if isinstance(rv, (int, float)) and isinstance(val, (int, float)):
            a = float(rv)
            b = float(val)
        else:
            a = rv
            b = val
        if op == "==":
            return a == b
        if op == "!=":
            return a != b
        if op == ">":
            try:
                return a > b
            except Exception:
                return False
        if op == "<":
            try:
                return a < b
            except Exception:
                return False
        if op == ">=":
            try:
                return a >= b
            except Exception:
                return False
        if op == "<=":
            try:
                return a <= b
            except Exception:
                return False
        return False

    def _generate_task_requiring_n_steps(self, steps: int) -> Dict[str, Any]:
        rng = random.Random(999 + self.complexity + steps)

        need_join_products = rng.random() < 0.5 or self.complexity >= 4
        need_join_customers = rng.random() < 0.5 or self.complexity >= 5

        filters = []
        if rng.random() < 0.7:
            qty_thr = rng.randint(2, 6)
            filters.append(("quantity", ">=", qty_thr))
        if rng.random() < 0.5:
            price_thr = round(rng.uniform(10, 60), 2)
            filters.append(("unit_price", ">", price_thr))
        if need_join_customers and rng.random() < 0.8:
            region = rng.choice(["North", "South", "East", "West"])
            filters.append(("region", "==", region))
        if need_join_products and rng.random() < 0.6:
            category = rng.choice(["Gadgets", "Home", "Outdoor", "Books", "Office", "Toys"])
            # randomly choose equality or inequality to vary
            if rng.random() < 0.5:
                filters.append(("category", "==", category))
            else:
                filters.append(("category", "!=", category))

        make_revenue = rng.random() < 0.5 or self.complexity >= 6
        agg_field = "revenue" if make_revenue else rng.choice(["amount", "quantity", "unit_price"])
        metric = rng.choice(["sum", "avg", "max", "min"])

        pipeline = []
        pipeline.append(("load", {"source": "orders"}))
        if need_join_products:
            pipeline.append(("join", {"other_source": "products", "on": ["product_id", "product_id"], "how": "inner"}))
        if need_join_customers:
            pipeline.append(("join", {"other_source": "customers", "on": ["customer_id", "customer_id"], "how": "inner"}))
        for f in filters:
            field, op, val = f
            if isinstance(val, str):
                val_repr = f"'{val}'"
            else:
                val_repr = str(val)
            cond = f"{field} {op} {val_repr}"
            pipeline.append(("filter", {"condition": cond}))
        if make_revenue:
            pipeline.append(("map", {"new_field": "revenue", "expr": "amount*1"}))

        # pad or trim to target length before aggregate
        core_len = len(pipeline) + 1  # +1 for aggregate
        if core_len < steps:
            # Add selects to pad
            pad_columns_options = [
                ["order_id", "amount"],
                ["order_id", "quantity", "unit_price"],
                ["product_id", "amount"],
                ["customer_id", "quantity", "amount"],
            ]
            while len(pipeline) + 1 < steps:
                cols = rng.choice(pad_columns_options)
                # ensure selected columns exist after joins and maps
                pipeline.append(("select", {"columns": cols}))
        elif core_len > steps:
            # Remove some filters if too long
            while len(pipeline) + 1 > steps and any(step[0] == "filter" for step in pipeline):
                idx = next(i for i, s in enumerate(pipeline) if s[0] == "filter")
                pipeline.pop(idx)
            while len(pipeline) + 1 > steps and len(pipeline) > 1:
                pipeline.pop(-1)

        pipeline.append(("aggregate", {"metric": metric, "field": agg_field}))

        # Compute target by executing pipeline internally
        ds = [dict(r) for r in self.sources["orders"]]
        for op, args in pipeline:
            if op == "load":
                ds = [dict(r) for r in self.sources[args["source"]]]
            elif op == "join":
                other = self.sources[args["other_source"]]
                this_key, other_key = args["on"]
                idx = {}
                for r in other:
                    idx.setdefault(r.get(other_key), []).append(r)
                joined = []
                for r in ds:
                    v = r.get(this_key)
                    matches = idx.get(v, [])
                    for mrow in matches:
                        jr = dict(r)
                        for k, val in mrow.items():
                            if k in jr:
                                jr[f"{args['other_source']}_{k}"] = val
                            else:
                                jr[k] = val
                        joined.append(jr)
                ds = joined
            elif op == "filter":
                field, aop, aval = self._parse_condition(args["condition"])
                ds = [r for r in ds if self._eval_condition(r.get(field), aop, aval)]
            elif op == "map":
                m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*([\*\+])\s*([\-]?\d+(\.\d+)?)\s*$", args["expr"])
                fld = m.group(1)
                oper = m.group(2)
                c = float(m.group(3))
                new_field = args["new_field"]
                new_ds = []
                for r in ds:
                    basev = r.get(fld)
                    if not isinstance(basev, (int, float)):
                        continue
                    nv = basev * c if oper == "*" else basev + c
                    nr = dict(r)
                    nr[new_field] = round(float(nv), 6)
                    new_ds.append(nr)
                ds = new_ds
            elif op == "select":
                cols = args["columns"]
                ds = [{k: r.get(k) for k in cols} for r in ds]
            elif op == "aggregate":
                vals = [r.get(args["field"]) for r in ds if isinstance(r.get(args["field"]), (int, float))]
                if args["metric"] == "sum":
                    res = sum(vals) if vals else 0.0
                elif args["metric"] == "avg":
                    res = (sum(vals) / len(vals)) if vals else 0.0
                elif args["metric"] == "min":
                    res = min(vals) if vals else 0.0
                else:
                    res = max(vals) if vals else 0.0
                target_value = round(float(res), 6)
            else:
                pass

        description_lines = []
        description_lines.append("Task: Compute a single numeric value using the tools.")
        description_lines.append("Starting point: the 'orders' source.")
        if need_join_products:
            description_lines.append("- You will need product attributes (e.g., category). Consider joining with 'products'.")
        if need_join_customers:
            description_lines.append("- You will need customer attributes (e.g., region). Consider joining with 'customers'.")
        for f in filters:
            field, op, val = f
            vdesc = f"'{val}'" if isinstance(val, str) else str(val)
            description_lines.append(f"- Apply filter: {field} {op} {vdesc}.")
        if make_revenue:
            description_lines.append("- Create a 'revenue' field; revenue can be amount*1 (already equivalent).")
        description_lines.append(f"- Then compute {metric} of {agg_field}.")
        description_lines.append("Submit with \\boxed{final:submit(value=...)}")

        return {
            "pipeline": pipeline,
            "text": "\n".join(description_lines),
            "target_value": target_value,
        }


class UtilityLoomEnvWithFeedback(UtilityLoomEnv):
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
            hint = "Wrap your command like \\boxed{tool:load(source=orders)}."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["message"] = obs
            hint = "Check query:help to list valid tools and syntax."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "no data loaded" in text:
                error_detail["violation"] = "no_data_loaded"
                hint = "Begin with \\boxed{tool:load(source=orders)}."
            elif "unknown field" in text:
                error_detail["violation"] = "unknown_field"
                hint = "Inspect columns using \\boxed{tool:schema()} before filtering or aggregating."
            else:
                error_detail["violation"] = "invalid_parameters"
                hint = "Review tool arguments and use query:help if unsure."

        elif "execution error" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "runtime_error"
            hint = "Simplify step; check schema and use peek/count to validate assumptions."

        elif "wrong final answer" in text:
            error_type = "WrongDecision"
            try:
                got_match = re.search(r"submitted ([\-\d\.eE]+)", obs)
                got = float(got_match.group(1)) if got_match else None
            except Exception:
                got = None
            error_detail["got"] = got
            error_detail["expected_hint"] = "hidden"
            hint = "Recheck that you joined the needed table(s) and applied all filters before aggregating."

        elif "timeout" in text:
            error_type = "Timeout"
            error_detail["outcome"] = "max_turns_reached"
            hint = "Plan the sequence: load -> required joins -> filters -> aggregate -> submit."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "loaded": self.user_current_ds is not None,
                "steps_taken": getattr(self, "steps_taken", None),
                "required_steps": getattr(self, "required_steps", None),
                "turns_left": self.max_turns - getattr(self, "turn_count", 0),
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
            "hint": "Start by loading 'orders' with \\boxed{tool:load(source=orders)} or ask \\boxed{query:help}.",
            "turn": 0,
            "state": {
                "loaded": False,
                "steps_taken": 0,
                "required_steps": getattr(self, "required_steps", None),
                "turns_left": self.max_turns,
            },
        }
        return obs, info