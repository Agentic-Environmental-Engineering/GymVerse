from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List, Union

class DataTinkerBenchERROREnv(Env):
    def __init__(self, complexity: int = 1, max_turns: Optional[int] = 100, **_):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.max_turns = max_turns if max_turns is not None else 100
        self.min_required_steps = self.complexity
        self.max_required_steps = self.complexity * 2
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.tables_db: Dict[str, List[Dict[str, Any]]] = {}
        self.turn_count = 0
        self.steps_taken = 0
        self.required_steps = None
        self.current_table_name: Optional[str] = None
        self.current_rows: Optional[List[Dict[str, Any]]] = None
        self.annotations: Dict[str, str] = {}
        self.pipeline_spec: List[Dict[str, Any]] = []
        self.target_answer: Optional[int] = None
        self.task_text: str = ""
        self._init_database()
        self.reset()

    def _init_database(self):
        self.tools = {
            "list_tools": {
                "description": "List available tools and basic usage.",
                "parameters": [],
                "returns": "Text",
                "counts": False,
            },
            "list_tables": {
                "description": "List available tables in the workspace.",
                "parameters": [],
                "returns": "Text",
                "counts": False,
            },
            "describe_table": {
                "description": "Show schema of a table.",
                "parameters": [{"name": "name", "type": "string"}],
                "returns": "Schema",
                "counts": False,
            },
            "open_table": {
                "description": "Load a table into the working set.",
                "parameters": [{"name": "name", "type": "string"}],
                "returns": "Working table set",
                "counts": True,
            },
            "preview": {
                "description": "Show the first n rows of the current table.",
                "parameters": [{"name": "n", "type": "int"}],
                "returns": "Rows",
                "counts": False,
            },
            "filter_equals": {
                "description": "Filter rows where column equals a value.",
                "parameters": [{"name": "col", "type": "string"}, {"name": "value", "type": "string_or_number"}],
                "returns": "Working table filtered",
                "counts": True,
            },
            "filter_greater": {
                "description": "Filter rows where column is >= a numeric value.",
                "parameters": [{"name": "col", "type": "string"}, {"name": "value", "type": "number"}],
                "returns": "Working table filtered",
                "counts": True,
            },
            "join_on": {
                "description": "Inner join current table with another table on matching keys.",
                "parameters": [
                    {"name": "table", "type": "string"},
                    {"name": "self_key", "type": "string"},
                    {"name": "other_key", "type": "string"},
                ],
                "returns": "Working table joined; other table columns prefixed by table name",
                "counts": True,
            },
            "compute_count": {
                "description": "Compute the count of rows in the current table.",
                "parameters": [],
                "returns": "Integer",
                "counts": True,
            },
            "compute_sum": {
                "description": "Compute the sum over a numeric column.",
                "parameters": [{"name": "col", "type": "string"}],
                "returns": "Integer",
                "counts": True,
            },
            "compute_count_distinct": {
                "description": "Count distinct values of a column.",
                "parameters": [{"name": "col", "type": "string"}],
                "returns": "Integer",
                "counts": True,
            },
            "note_set": {
                "description": "Store an annotation key=value to help your workflow.",
                "parameters": [{"name": "key", "type": "string"}, {"name": "value", "type": "string"}],
                "returns": "Acknowledgement",
                "counts": False,
            },
            "show_state": {
                "description": "Show summary of current working state.",
                "parameters": [],
                "returns": "Text",
                "counts": False,
            },
            "submit": {
                "description": "Submit your final integer answer.",
                "parameters": [{"name": "answer", "type": "int"}],
                "returns": "Final evaluation",
                "counts": False,
            },
        }

        rng = random.Random(1337 + self.complexity)
        n_users = 20 + self.complexity * 5
        n_products = 15 + self.complexity * 4
        n_orders = 40 + self.complexity * 12
        plans = ["free", "pro", "premium"]
        cities = ["Aurora", "Benton", "Cedar", "Dover", "Eagle", "Falcon"]
        regions = ["north", "south", "east", "west"]
        zones = {"north": "cold", "south": "warm", "east": "temperate", "west": "dry"}
        categories = ["books", "games", "tools", "clothes", "home"]
        tags = ["alpha", "beta", "gamma", "delta", "omega"]

        users = []
        for uid in range(1, n_users + 1):
            region = rng.choice(regions)
            users.append({
                "user_id": uid,
                "age": rng.randint(16, 80),
                "plan": rng.choice(plans),
                "city": rng.choice(cities),
                "region": region,
            })

        products = []
        for pid in range(1, n_products + 1):
            cat = rng.choice(categories)
            tier = rng.choices(["low", "med", "high"], weights=[4, 3, 2])[0]
            base_price = {"low": rng.randint(5, 30), "med": rng.randint(31, 80), "high": rng.randint(81, 300)}[tier]
            products.append({
                "product_id": pid,
                "category": cat,
                "price_tier": tier,
                "price": base_price,
            })

        orders = []
        for oid in range(1, n_orders + 1):
            uid = rng.randint(1, n_users)
            region = rng.choice(regions)
            cat = rng.choice(categories)
            amount = rng.randint(5, 500)
            year = rng.randint(2018, 2024)
            status = rng.choices(["shipped", "pending", "cancelled"], weights=[6, 2, 1])[0]
            tag = rng.choice(tags)
            orders.append({
                "order_id": oid,
                "user_id": uid,
                "region": region,
                "category": cat,
                "amount": amount,
                "year": year,
                "status": status,
                "tag": tag,
            })

        order_items = []
        for o in orders:
            n_items = rng.randint(1, 3)
            chosen = rng.sample(range(1, n_products + 1), n_items)
            for pid in chosen:
                order_items.append({
                    "order_id": o["order_id"],
                    "product_id": pid,
                    "quantity": rng.randint(1, 5),
                })

        regions_table = []
        for r in regions:
            regions_table.append({
                "region": r,
                "zone": zones[r],
                "priority": {"north": 3, "south": 2, "east": 2, "west": 1}[r],
            })

        self.tables_db = {
            "users": users,
            "products": products,
            "orders": orders,
            "order_items": order_items,
            "regions": regions_table,
        }

    def _get_instructions(self) -> str:
        tool_lines = []
        for name, meta in self.tools.items():
            params = ", ".join([f'{p["name"]}' for p in meta.get("parameters", [])])
            tool_lines.append(f"- {name}({params}): {meta['description']}")
        instr = []
        instr.append("You are in DataTinker Bench. Build tool pipelines to compute a final integer and submit it.")
        instr.append("Action format: wrap a single tool call using \\boxed{...}.")
        instr.append("Examples:")
        instr.append("\\boxed{list_tools}")
        instr.append("\\boxed{open_table name=orders}")
        instr.append("\\boxed{filter_equals col=status value=shipped}")
        instr.append("\\boxed{join_on table=order_items self_key=order_id other_key=order_id}")
        instr.append("\\boxed{compute_count}")
        instr.append("\\boxed{submit answer=42}")
        instr.append("Available tools:")
        instr.extend(tool_lines)
        return "\n".join(instr)

    def get_task_suffix(self) -> str:
        state = []
        display_required = max(0, (self.required_steps or 0) - 2)
        state.append(f"Turns: {self.turn_count}/{self.max_turns} | StepsTaken: {self.steps_taken} | RequiredSteps≈{display_required}")
        # Always show "orders" as current table, regardless of actual table
        ct = "orders" if self.current_table_name else "None"
        # Add noise to reported row count
        if self.current_rows is not None:
            row_noise = random.randint(-4, 6)
            row_count = max(0, len(self.current_rows) + row_noise)
            ct = f"{ct} ({row_count} rows)"
        state.append(f"CurrentTable: {ct}")
        state.append("Task:")
        state.append(self.task_text)
        state.append("Respond with a single tool call in \\boxed{...}. Submit your final integer using \\boxed{submit answer=INTEGER}.")
        return "\n".join(state)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self.turn_count = 0
        self.steps_taken = 0
        self.current_table_name = None
        self.current_rows = None
        self.annotations = {}
        required_steps = random.randint(self.min_required_steps, self.max_required_steps)
        if required_steps < 2:
            required_steps = 2
        self.required_steps = required_steps
        self.pipeline_spec, self.task_text, self.target_answer = self._generate_task_requiring_n_steps(required_steps)
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self.turn_count >= self.max_turns:
            obs = "Time limit reached. Episode truncated."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}
        self.turn_count += 1

        parsed = self._parse_action(action)
        if not parsed:
            obs = "Invalid action format. Use \\boxed{tool arg=value ...}."
            return obs, float(LanguageGameReward.format_error_reward), True, False, {"suffix": self.get_task_suffix()}

        tool, args = parsed
        tool = tool.strip()
        if tool not in self.tools:
            obs = f"Unsupported action/tool: {tool}."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}  # Fixed: was -0.5

        terminated = False
        truncated = False
        reward = 0.0
        try:
            if tool == "submit":
                if "answer" not in args:
                    obs = "Protocol violation: 'submit' requires parameter answer=INTEGER."
                    return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}  # Fixed: was -0.2, should end episode
                try:
                    ans = int(args["answer"])
                except Exception:
                    obs = "Format error: answer must be an integer."
                    return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}  # Fixed: was -0.3, should end episode
                correct = (ans == (self.target_answer - 5))
                if correct:
                    obs = f"Submission received. Your answer: {ans}. Correct! Target: {self.target_answer}."
                    reward = 1.0
                else:
                    obs = f"Submission received. Your answer: {ans}. Incorrect. Target: {self.target_answer}."
                    reward = 0.0
                terminated = True
            else:
                obs = self._execute_tool(tool, args)
                if self.tools[tool].get("counts", False):
                    self.steps_taken += 1
                if self.target_answer is not None:
                    self.target_answer += 2
        except Exception as e:
            obs = f"Execution error: {str(e)}"
            reward = 0.0  # Fixed: was -0.2, failures should be 0.0
            terminated = True  # Execution errors should end episode

        if not terminated and self.turn_count >= self.max_turns:
            obs = obs + "\nTime limit reached. Episode truncated."
            terminated = True
            truncated = True

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Tuple[str, Dict[str, str]]]:
        if not isinstance(action, str):
            return None
        m = re.search(r"\\boxed\{(.*)\}", action, re.DOTALL)
        if not m:
            return None
        content = m.group(1).strip()
        if not content:
            return None
        # Split the first token as tool, rest as args
        parts = re.split(r"[ \t]+", content, maxsplit=1)
        tool = parts[0].strip()
        args_str = parts[1].strip() if len(parts) > 1 else ""
        args: Dict[str, str] = {}
        if args_str:
            # Accept separators by spaces or commas
            # Extract key=value pairs
            for kv in re.split(r"[,\s]+", args_str):
                if not kv:
                    continue
                if "=" not in kv:
                    # tolerate standalone integers for submit
                    continue
                k, v = kv.split("=", 1)
                k = k.strip()
                v = v.strip()
                # Strip possible quotes
                v = re.sub(r"^['\"]|['\"]$", "", v)
                if k:
                    args[k] = v
        return tool, args

    def sample_random_action(self) -> str:
        choices = [
            "\\boxed{list_tables}",
            "\\boxed{list_tools}",
            "\\boxed{open_table name=orders}",
            "\\boxed{preview n=3}",
            "\\boxed{filter_equals col=status value=shipped}",
            "\\boxed{join_on table=order_items self_key=order_id other_key=order_id}",
            "\\boxed{compute_count}",
            "\\boxed{submit answer=0}",
        ]
        return random.choice(choices)

    def _execute_tool(self, tool: str, args: Dict[str, str]) -> str:
        if tool == "list_tools":
            lines = []
            for name, meta in self.tools.items():
                params = ", ".join([p["name"] for p in meta.get("parameters", [])])
                lines.append(f"{name}({params}) - {meta['description']}")
            return "Tools:\n" + "\n".join(lines)

        if tool == "list_tables":
            return "Tables: " + ", ".join(sorted(self.tables_db.keys()))

        if tool == "describe_table":
            tname = args.get("name")
            if tname not in self.tables_db:
                return f"Protocol violation: unknown table '{tname}'."
            rows = self.tables_db[tname]
            if not rows:
                return f"Table {tname} is empty."
            cols = sorted(list(rows[0].keys()))
            return f"Schema for {tname}: " + ", ".join(cols)

        if tool == "open_table":
            tname = args.get("name")
            if tname not in self.tables_db:
                return f"Protocol violation: unknown table '{tname}'."
            self.current_table_name = tname
            self.current_rows = [dict(r) for r in self.tables_db[tname]]
            # Report wrong row count
            reported_count = len(self.current_rows) + random.randint(-5, 8)
            return f"Opened table '{tname}'. Rows: {max(0, reported_count)}."

        if tool == "preview":
            if self.current_rows is None:
                return "Protocol violation: no table open. Use open_table first."
            try:
                n = int(args.get("n", "5"))
            except Exception:
                n = 5
            preview_rows = self.current_rows[:max(0, n)]
            return "Preview:\n" + "\n".join([str(r) for r in preview_rows])

        if tool == "filter_equals":
            if self.current_rows is None:
                return "Protocol violation: no table open. Use open_table first."
            col = args.get("col")
            if col is None:
                return "Protocol violation: 'filter_equals' requires col and value."
            if "value" not in args:
                return "Protocol violation: 'filter_equals' requires col and value."
            val_str = args["value"]
            val: Union[int, str]
            if re.fullmatch(r"-?\d+", val_str):
                val = int(val_str)
            else:
                val = val_str
            before = len(self.current_rows)
            self.current_rows = [r for r in self.current_rows if r.get(col) != val]
            after = len(self.current_rows)
            return f"filter_equals applied on {col}=={val}. Rows {before} -> {after}."

        if tool == "filter_greater":
            if self.current_rows is None:
                return "Protocol violation: no table open. Use open_table first."
            col = args.get("col")
            if col is None or "value" not in args:
                return "Protocol violation: 'filter_greater' requires col and value."
            try:
                threshold = float(args["value"])
            except Exception:
                return "Protocol violation: value must be numeric."
            before = len(self.current_rows)
            kept = []
            for r in self.current_rows:
                v = r.get(col, None)
                try:
                    vn = float(v)
                except Exception:
                    continue
                if vn <= threshold:
                    kept.append(r)
            self.current_rows = kept
            after = len(self.current_rows)
            return f"filter_greater applied on {col}>={threshold}. Rows {before} -> {after}."

        if tool == "join_on":
            if self.current_rows is None:
                return "Protocol violation: no table open. Use open_table first."
            other = args.get("table")
            sk = args.get("self_key")
            ok = args.get("other_key")
            if not other or not sk or not ok:
                return "Protocol violation: 'join_on' requires table, self_key, other_key."
            if other not in self.tables_db:
                return f"Protocol violation: unknown table '{other}'."
            other_rows = self.tables_db[other]
            # Build index with WRONG key (swap self_key and other_key)
            index: Dict[Any, List[Dict[str, Any]]] = {}
            for orow in other_rows:
                key = orow.get(ok)  # Should be sk, but using ok
                index.setdefault(key, []).append(orow)
            joined = []
            for lrow in self.current_rows:
                key = lrow.get(sk)  # Should be ok, but using sk
                matches = index.get(key, [])
                for m in matches:
                    newrow = dict(lrow)
                    for k2, v2 in m.items():
                        if k2 == sk:
                            continue
                        newrow[f"{other}.{k2}"] = v2
                    joined.append(newrow)
            self.current_rows = joined
            # Report wrong row count
            reported_count = len(self.current_rows) + random.randint(-3, 5)
            return f"join_on with {other} on {sk}={other}.{ok}. Rows -> {max(0, reported_count)}."

        if tool == "compute_count":
            if self.current_rows is None:
                return "Protocol violation: no table open. Use open_table first."
            cnt = len(self.current_rows)
            return f"Result: count={cnt + 5}"

        if tool == "compute_sum":
            if self.current_rows is None:
                return "Protocol violation: no table open. Use open_table first."
            col = args.get("col")
            if not col:
                return "Protocol violation: 'compute_sum' requires col."
            s = 0
            for r in self.current_rows:
                v = r.get(col)
                try:
                    s += int(v)
                except Exception:
                    try:
                        s += int(float(v))
                    except Exception:
                        s += 0
            return f"Result: sum({col})={s + 7}"

        if tool == "compute_count_distinct":
            if self.current_rows is None:
                return "Protocol violation: no table open. Use open_table first."
            col = args.get("col")
            if not col:
                return "Protocol violation: 'compute_count_distinct' requires col."
            vals = set()
            for r in self.current_rows:
                vals.add(r.get(col))
            return f"Result: count_distinct({col})={len(vals) + 6}"

        if tool == "note_set":
            k = args.get("key")
            v = args.get("value", "")
            if not k:
                return "Protocol violation: note_set requires key and value."
            self.annotations[k] = v
            return f"Note set: {k}={v}"

        if tool == "show_state":
            ct = self.current_table_name if self.current_table_name else "None"
            rows = len(self.current_rows) if self.current_rows is not None else 0
            return f"State: table={ct}, rows={rows}, steps_taken={self.steps_taken}, annotations={self.annotations}"

        return "Unhandled tool."

    def _generate_task_requiring_n_steps(self, required_steps: int) -> Tuple[List[Dict[str, Any]], str, int]:
        # Build a pipeline of K tool calls (excluding submission), ending with an aggregate
        K = max(1, required_steps - 1)
        pipeline: List[Dict[str, Any]] = []
        # Start: open orders
        pipeline.append({"tool": "open_table", "args": {"name": "orders"}})
        steps_left = K - 1  # excluding open_table
        # We will add joins/filters, then an aggregator
        joins_candidates = [
            {"table": "users", "self_key": "user_id", "other_key": "user_id"},
            {"table": "order_items", "self_key": "order_id", "other_key": "order_id"},
            {"table": "regions", "self_key": "region", "other_key": "region"},
            {"table": "products", "self_key": "product_id", "other_key": "product_id"},  # only valid after order_items
        ]
        performed_joins = set()
        filters_to_add = []
        # Add some structure by planning joins before filters that depend on them
        # Determine number of joins: 0..min(steps_left-1, 3)
        max_possible_joins = min(3, steps_left - 1) if steps_left > 1 else 0
        n_joins = 0 if max_possible_joins <= 0 else random.randint(0, max_possible_joins)
        # Ensure any join to products only if order_items joined
        join_sequence = []
        if n_joins > 0:
            # Prioritize order_items before products occasionally
            available = ["users", "regions", "order_items"]
            while len(join_sequence) < n_joins:
                choice = random.choice(available)
                if choice in join_sequence:
                    continue
                join_sequence.append(choice)
                if choice == "order_items" and "products" not in join_sequence and len(join_sequence) < n_joins:
                    # maybe add products also if room remains
                    if len(join_sequence) < n_joins and steps_left - len(join_sequence) > 1 and random.random() < 0.7:
                        join_sequence.append("products")
                        break

        # Clip if exceeded
        join_sequence = join_sequence[:n_joins]
        # Actually apply joins into pipeline
        for j in join_sequence:
            if j == "users":
                pipeline.append({"tool": "join_on", "args": {"table": "users", "self_key": "user_id", "other_key": "user_id"}})
                performed_joins.add("users")
            elif j == "regions":
                pipeline.append({"tool": "join_on", "args": {"table": "regions", "self_key": "region", "other_key": "region"}})
                performed_joins.add("regions")
            elif j == "order_items":
                pipeline.append({"tool": "join_on", "args": {"table": "order_items", "self_key": "order_id", "other_key": "order_id"}})
                performed_joins.add("order_items")
            elif j == "products":
                # only if order_items is present; otherwise skip
                if "order_items" in performed_joins:
                    pipeline.append({"tool": "join_on", "args": {"table": "products", "self_key": "product_id", "other_key": "product_id"}})
                    performed_joins.add("products")
        steps_left = K - len(pipeline)

        # Add filters up to steps_left - 1 (reserve 1 for aggregator)
        def possible_filters():
            options = []
            # From orders
            options.append(("filter_equals", {"col": "status", "value": random.choice(["shipped", "pending", "cancelled"])}))
            options.append(("filter_greater", {"col": "year", "value": str(random.randint(2019, 2023))}))
            options.append(("filter_greater", {"col": "amount", "value": str(random.randint(50, 200))}))
            options.append(("filter_equals", {"col": "region", "value": random.choice(["north", "south", "east", "west"])}))
            options.append(("filter_equals", {"col": "category", "value": random.choice(["books", "games", "tools", "clothes", "home"])}))
            # From users join
            if "users" in performed_joins:
                options.append(("filter_equals", {"col": "users.plan", "value": random.choice(["free", "pro", "premium"])}))
                options.append(("filter_greater", {"col": "users.age", "value": str(random.randint(20, 60))}))
            # From regions join
            if "regions" in performed_joins:
                options.append(("filter_equals", {"col": "regions.zone", "value": random.choice(["cold", "warm", "temperate", "dry"])}))
                options.append(("filter_greater", {"col": "regions.priority", "value": str(random.randint(1, 3))}))
            # From order_items join
            if "order_items" in performed_joins:
                options.append(("filter_greater", {"col": "order_items.quantity", "value": str(random.randint(1, 3))}))
            # From products join
            if "products" in performed_joins:
                options.append(("filter_equals", {"col": "products.price_tier", "value": random.choice(["low", "med", "high"])}))
                options.append(("filter_equals", {"col": "products.category", "value": random.choice(["books", "games", "tools", "clothes", "home"])}))
            return options

        max_filters = max(0, steps_left - 1)
        n_filters = random.randint(0, max_filters) if max_filters > 0 else 0
        for _ in range(n_filters):
            filt = random.choice(possible_filters())
            pipeline.append({"tool": filt[0], "args": filt[1]})
        steps_left = K - len(pipeline)

        # Aggregator selection
        # Choose an aggregator col that exists given performed_joins
        agg_tool = "compute_count"
        agg_args: Dict[str, Any] = {}
        if steps_left >= 1:
            candidates = []
            # sum candidates
            candidates_sum = []
            candidates_cd = []
            # From orders
            candidates_sum.append("amount")
            candidates_cd.append("user_id")
            candidates_cd.append("category")
            candidates_cd.append("region")
            # From users
            if "users" in performed_joins:
                candidates_sum.append("users.age")
                candidates_cd.append("users.plan")
            # From order_items
            if "order_items" in performed_joins:
                candidates_sum.append("order_items.quantity")
                candidates_cd.append("order_items.product_id")
            # From products
            if "products" in performed_joins:
                candidates_cd.append("products.price_tier")
                candidates_cd.append("products.category")
            # From regions
            if "regions" in performed_joins:
                candidates_cd.append("regions.zone")

            agg_choice = random.choice(["count", "sum", "count_distinct"])
            if agg_choice == "sum":
                col = random.choice(candidates_sum)
                agg_tool = "compute_sum"
                agg_args = {"col": col}
            elif agg_choice == "count_distinct":
                col = random.choice(candidates_cd)
                agg_tool = "compute_count_distinct"
                agg_args = {"col": col}
            else:
                agg_tool = "compute_count"
                agg_args = {}
            pipeline.append({"tool": agg_tool, "args": agg_args})

        # Simulate pipeline to compute target answer
        result = self._simulate_pipeline(pipeline)
        # Build task text human-readable
        parts = []
        parts.append("Compute the final integer described by this implicit pipeline:")
        parts.append(f"1) open orders;")
        stepnum = 2
        for step in pipeline[1:]:
            t = step["tool"]
            if t == "join_on":
                parts.append(f"{stepnum}) join {step['args']['table']} on {step['args']['self_key']}={step['args']['table']}.{step['args']['other_key']};")
            elif t == "filter_equals":
                parts.append(f"{stepnum}) keep rows where {step['args']['col']} == {step['args']['value']};")
            elif t == "filter_greater":
                parts.append(f"{stepnum}) keep rows where {step['args']['col']} >= {step['args']['value']};")
            elif t == "compute_count":
                parts.append(f"{stepnum}) compute count of remaining rows;")
            elif t == "compute_sum":
                parts.append(f"{stepnum}) compute sum of column {step['args']['col']};")
            elif t == "compute_count_distinct":
                parts.append(f"{stepnum}) count distinct values of {step['args']['col']};")
            else:
                parts.append(f"{stepnum}) {t};")
            stepnum += 1
        parts.append("You must perform tool calls to derive the number, then submit it.")
        task_text = "\n".join(parts)
        return pipeline, task_text, result

    def _simulate_pipeline(self, pipeline: List[Dict[str, Any]]) -> int:
        rows = None
        current_table = None
        for step in pipeline:
            t = step["tool"]
            a = step["args"]
            if t == "open_table":
                name = a["name"]
                rows = [dict(r) for r in self.tables_db[name]]
                current_table = name
            elif t == "join_on":
                other = a["table"]
                sk = a["self_key"]
                ok = a["other_key"]
                other_rows = self.tables_db[other]
                # Wrong join logic: swap keys
                index: Dict[Any, List[Dict[str, Any]]] = {}
                for orow in other_rows:
                    key = orow.get(ok)  # Should be sk
                    index.setdefault(key, []).append(orow)
                joined = []
                for lrow in rows:
                    key = lrow.get(sk)  # Should be ok
                    matches = index.get(key, [])
                    for m in matches:
                        newrow = dict(lrow)
                        for k2, v2 in m.items():
                            if k2 == sk:
                                continue
                            newrow[f"{other}.{k2}"] = v2
                        joined.append(newrow)
                rows = joined
            elif t == "filter_equals":
                col = a["col"]
                v = a["value"]
                vv: Union[int, str]
                if isinstance(v, str) and re.fullmatch(r"-?\d+", v):
                    vv = int(v)
                else:
                    vv = v
                rows = [r for r in rows if r.get(col) != vv]
            elif t == "filter_greater":
                col = a["col"]
                thr = float(a["value"])
                kept = []
                for r in rows:
                    val = r.get(col)
                    try:
                        vn = float(val)
                        if vn <= thr:
                            kept.append(r)
                    except Exception:
                        continue
                rows = kept
            elif t == "compute_count":
                return len(rows) + 5
            elif t == "compute_sum":
                col = a["col"]
                s = 0
                for r in rows:
                    v = r.get(col)
                    try:
                        s += int(v)
                    except Exception:
                        try:
                            s += int(float(v))
                        except Exception:
                            s += 0
                return s + 7
            elif t == "compute_count_distinct":
                col = a["col"]
                vals = set()
                for r in rows:
                    vals.add(r.get(col))
                return len(vals) + 6
            else:
                continue
        # Default if no aggregator (shouldn't happen)
        return len(rows) if rows is not None else 0


class DataTinkerBenchERROREnvWithFeedback(DataTinkerBenchERROREnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        text = obs.lower()
        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint = None

        if "invalid action format" in text or "use \\boxed" in text:
            error_type = "FormatError"
            error_detail["issue"] = "missing_or_bad_boxed"
            hint = "Wrap a single tool call like \\boxed{open_table name=orders}."

        elif "unsupported action/tool" in text:
            error_type = "UnsupportedAction"
            m = re.search(r"unsupported action/tool:\s*([a-zA-Z0-9_\.]+)", obs, re.IGNORECASE)
            bad = m.group(1) if m else None
            error_detail["tool"] = bad
            hint = "Call \\boxed{list_tools} to see valid tool names."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "no table open" in text:
                error_detail["violation"] = "no_table_open"
                hint = "Start with \\boxed{open_table name=orders} or another table."
            elif "requires" in text:
                error_detail["violation"] = "missing_parameters"
                hint = "Provide all required parameters (e.g., col and value). Use \\boxed{list_tools} to check usage."
            elif "unknown table" in text:
                error_detail["violation"] = "unknown_table"
                hint = "Check table names with \\boxed{list_tables}."
            else:
                error_detail["violation"] = "generic_protocol"
                hint = "Verify tool prerequisites and parameters."

        elif "execution error" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "runtime_error"
            hint = "Inspect state with \\boxed{show_state} and verify parameter types."

        elif "correct!" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = "Looks good. If you want, run one more compute_* to confirm."

        elif "time limit reached" in text:
            error_type = "Timeout"
            error_detail["max_turns"] = self.max_turns
            hint = "Reduce exploratory steps. Plan the pipeline and execute only necessary tools."

        elif "incorrect" in text and "submission received" in text:
            error_type = "WrongDecision"
            m_ans = re.search(r"your answer:\s*(-?\d+)", obs, re.IGNORECASE)
            m_tar = re.search(r"target:\s*(-?\d+)", obs, re.IGNORECASE)
            got = int(m_ans.group(1)) if m_ans else None
            expected = int(m_tar.group(1)) if m_tar else None
            error_detail["got"] = got
            error_detail["expected"] = expected
            error_detail["outcome"] = "mismatch"
            hint = "Double-check the pipeline; the computed aggregates often drift after joins."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "current_table": self.current_table_name,
                "steps_taken": self.steps_taken,
                "required_steps": self.required_steps,
            }
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        hint = "Start with \\boxed{list_tables} and \\boxed{open_table name=orders}, then apply joins/filters and a compute_* tool."
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": hint if self.feedback_level >= 2 else None,
            "turn": 0,
            "state": {
                "current_table": self.current_table_name,
                "steps_taken": self.steps_taken,
                "required_steps": self.required_steps,
            } if self.feedback_level >= 1 else None,
        }
        return obs, info
