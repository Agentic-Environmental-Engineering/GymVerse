from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List, Union


class ToolCrafterRelayEnv(Env):
    def __init__(self, complexity: int = 1, max_turns: Optional[int] = 100, **_):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.max_turns = max_turns if max_turns is not None else 100
        self.min_required_steps = self.complexity
        self.max_required_steps = self.complexity * 2
        self._init_database()
        self.reset()

    def _init_database(self):
        random.seed()
        base_users = 8 + self.complexity * 3
        base_products = 10 + self.complexity * 4
        base_orders = 30 + self.complexity * 10
        self.cities = ["Aurora", "Belltown", "Cedar", "Duneshore", "Elmvale", "Fairoaks"]
        self.segments = ["consumer", "small_business", "enterprise"]
        self.categories = ["books", "electronics", "home", "toys", "outdoors", "beauty", "grocery"]
        self.statuses = ["shipped", "pending", "returned", "cancelled"]

        self.tables = {}
        users = []
        for uid in range(1, base_users + 1):
            users.append({
                "id": uid,
                "city": random.choice(self.cities),
                "segment": random.choice(self.segments),
            })
        products = []
        for pid in range(1, base_products + 1):
            products.append({
                "id": pid,
                "category": random.choice(self.categories),
                "price": round(random.uniform(5.0, 500.0), 2),
            })
        orders = []
        for oid in range(1, base_orders + 1):
            orders.append({
                "order_id": oid,
                "user_id": random.randint(1, len(users)),
                "product_id": random.randint(1, len(products)),
                "quantity": random.randint(1, 5),
                "status": random.choices(self.statuses, weights=[0.6, 0.2, 0.1, 0.1])[0],
            })
        self.tables["users"] = users
        self.tables["products"] = products
        self.tables["orders"] = orders

        self.tool_catalog = {
            "open_session": {"params": [], "desc": "Initialize a work session. Must be first."},
            "help": {"params": [], "desc": "List tools and usage."},
            "list_sources": {"params": [], "desc": "List available tables."},
            "inspect_schema": {"params": ["table"], "desc": "Show schema of a table."},
            "load": {"params": ["table"], "desc": "Load a table into current workspace."},
            "filter": {"params": ["condition"], "desc": "Filter rows by simple condition e.g., city=='Aurora' or price>100."},
            "add_field": {"params": ["name", "expr"], "desc": "Add computed field by expr (supports +,-,*,/ of fields/constants)."},
            "join": {"params": ["table", "on"], "desc": "Inner join current data with table using on='left_key=right_key'."},
            "groupby": {"params": ["by", "agg", "field"], "desc": "Aggregate by group. agg in {sum,avg,count}. If agg=count, field='*'."},
            "sort": {"params": ["field", "order"], "desc": "Sort by field, order in {asc,desc}."},
            "head": {"params": ["n"], "desc": "Keep first n rows."},
            "select": {"params": ["columns"], "desc": "Keep comma-separated columns."},
            "distinct_count": {"params": ["field"], "desc": "Replace with one-row table: distinct_count of field."},
            "count_rows": {"params": [], "desc": "Replace with one-row table: row_count."},
            "extract_value": {"params": ["field"], "desc": "Extract field from first row into last_extracted_value."},
            "compute_metric": {"params": ["name", "price_field", "weight_field"], "desc": "Compute named metric; supports weighted_avg_price."},
            "submit_answer": {"params": ["value"], "desc": "Submit final answer after meeting minimum steps."},
        }
        self.counted_tools = {
            "open_session", "list_sources", "inspect_schema", "load", "filter", "add_field", "join",
            "groupby", "sort", "head", "select", "distinct_count", "count_rows", "extract_value", "compute_metric"
        }

    def _get_instructions(self) -> str:
        tools_list = []
        for t, meta in self.tool_catalog.items():
            params = meta["params"]
            sig = f"{t}" + ("" if not params else " " + ",".join([f"{p}=..." for p in params]))
            tools_list.append(f"- {sig}: {meta['desc']}")
        tools_text = "\n".join(tools_list)
        lines = []
        lines.append("You are in a data workbench. Use tools to transform hidden tables and derive the requested answer.")
        lines.append("Protocol:")
        lines.append("1) Start with open_session. 2) Load tables. 3) Apply filters, joins, aggregations. 4) Extract or compute the final value. 5) Submit with submit_answer.")
        lines.append("Action format: use \\boxed{tool arg=value, arg2=value2}. Strings may be quoted. Example: \\boxed{filter condition=city=='Aurora'}")
        lines.append("Available tools:")
        lines.append(tools_text)
        return "\n".join(lines)

    def get_task_suffix(self) -> str:
        status = []
        status.append(f"Turn: {self.turn_count}/{self.max_turns}")
        status.append(f"Counted steps: {self.steps_taken} (min required before submission: {self.required_steps_min})")
        status.append(f"Session open: {self.session_open}")
        status.append(f"Current table loaded: {self.current_table if self.current_table else 'None'}")
        status.append(f"Task: {self.task_description}")
        status.append("Respond with one tool call per turn using \\boxed{...}.")
        return "\n".join(status)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self.turn_count = 0
        self.steps_taken = 0
        self.session_open = False
        self.current_data: Optional[List[Dict[str, Any]]] = None
        self.current_table: Optional[str] = None
        self.last_extracted_value: Optional[Union[str, float, int]] = None
        self.required_steps_min = random.randint(self.min_required_steps, self.max_required_steps)
        self._generate_task_requiring_n_steps(self.required_steps_min)
        obs = self._get_instructions()
        info = {"suffix": self.get_task_suffix()}
        return obs, info

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self.turn_count is None:
            self.turn_count = 0
        self.turn_count += 1

        info = {"suffix": self.get_task_suffix()}
        parsed = self._parse_action(action)
        if parsed is None:
            obs = "Invalid action format. Use \\boxed{tool arg=value, ...}."
            reward = LanguageGameReward.format_error_reward
            return obs, reward, True, False, info

        tool, args = parsed
        if tool not in self.tool_catalog:
            obs = f"Unknown tool '{tool}'. Episode terminated."
            reward = LanguageGameReward.format_error_reward
            return obs, reward, True, False, info

        if tool != "open_session" and not self.session_open and tool not in {"help"}:
            obs = "Protocol violation: open_session must be called before other tools."
            reward = 0.0
            if self.turn_count >= self.max_turns:
                return obs + " Timeout.", 0.0, True, True, info
            return obs, reward, False, False, info

        try:
            result_text, counted_flag, terminal_flag, success_flag, wrong_flag = self._execute_tool(tool, args)
        except Exception as e:
            obs = f"Execution error: {e}"
            reward = 0.0
            if self.turn_count >= self.max_turns:
                return obs + " Timeout.", 0.0, True, True, info
            return obs, reward, False, False, info

        if counted_flag:
            self.steps_taken += 1

        if terminal_flag:
            if success_flag:
                obs = result_text + " Success."
                reward = 1.0
                return obs, reward, True, False, info
            if wrong_flag:
                obs = result_text + " Final answer incorrect."
                reward = 0.0
                return obs, reward, True, False, info
            obs = result_text
            reward = 0.0
            return obs, reward, True, False, info

        if self.turn_count >= self.max_turns:
            obs = result_text + " Timeout."
            return obs, 0.0, True, True, info

        return result_text, 0.0, False, False, info

    def _parse_action(self, action: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        m = re.search(r"\\boxed\{(.*)\}", action.strip(), flags=re.DOTALL)
        if not m:
            return None
        content = m.group(1).strip()
        if not content:
            return None
        parts = re.split(r"\s+", content, maxsplit=1)
        tool = parts[0].strip()
        args_str = parts[1].strip() if len(parts) > 1 else ""
        args: Dict[str, Any] = {}
        if args_str:
            tokens = self._split_args(args_str)
            for tok in tokens:
                if "=" not in tok:
                    # positional ignored; enforce key=value
                    continue
                k, v = tok.split("=", 1)
                k = k.strip()
                v = v.strip()
                if (len(v) >= 2 and ((v[0] == v[-1] == "'") or (v[0] == v[-1] == '"'))):
                    v = v[1:-1]
                else:
                    if re.fullmatch(r"-?\d+\.\d+", v):
                        try:
                            v = float(v)
                        except:
                            pass
                    elif re.fullmatch(r"-?\d+", v):
                        try:
                            v = int(v)
                        except:
                            pass
                    else:
                        lv = v.lower()
                        if lv == "true":
                            v = True
                        elif lv == "false":
                            v = False
                args[k] = v
        return tool, args

    def _split_args(self, s: str) -> List[str]:
        res = []
        buf = []
        in_quote = False
        quote_char = ""
        for ch in s:
            if ch in "\"'":
                if not in_quote:
                    in_quote = True
                    quote_char = ch
                    buf.append(ch)
                else:
                    if ch == quote_char:
                        in_quote = False
                        buf.append(ch)
                    else:
                        buf.append(ch)
            elif ch == "," and not in_quote:
                tok = "".join(buf).strip()
                if tok:
                    res.append(tok)
                buf = []
            else:
                buf.append(ch)
        tok = "".join(buf).strip()
        if tok:
            res.append(tok)
        return res

    def sample_random_action(self) -> str:
        examples = [
            r"\boxed{open_session}",
            r"\boxed{list_sources}",
            r"\boxed{inspect_schema table='orders'}",
            r"\boxed{load table='orders'}",
            r"\boxed{filter condition=status=='shipped'}",
            r"\boxed{join table='products', on='product_id=id'}",
            r"\boxed{groupby by='category', agg='sum', field='revenue'}",
            r"\boxed{compute_metric name='weighted_avg_price', price_field='price', weight_field='quantity'}",
            r"\boxed{submit_answer value=42}",
        ]
        return random.choice(examples)

    def _generate_task_requiring_n_steps(self, required_steps: int):
        # Pick a template and parameters that exist in data
        users = self.tables["users"]
        products = self.tables["products"]
        orders = self.tables["orders"]

        template = random.choice(["top_category_city", "users_with_status", "weighted_avg_segment_category"])
        if template == "top_category_city":
            city = random.choice(self.cities)
            self.task_description = f"Return the category with the highest total revenue for shipped orders placed by users in city '{city}'. "\
                                    f"Revenue is price*quantity. Break ties by lexicographically smallest category."
            self.target_answer = self._compute_top_category_city(city)
        elif template == "users_with_status":
            status = random.choice(self.statuses)
            self.task_description = f"Return the count of distinct users who have at least one order with status '{status}'."
            self.target_answer = self._compute_users_with_status(status)
        else:
            cat = random.choice(self.categories)
            seg = random.choice(self.segments)
            self.task_description = f"Return the weighted average price of products in category '{cat}' purchased by users in segment '{seg}'. "\
                                    f"Weight by quantity; compute as sum(price*quantity)/sum(quantity), rounded to 3 decimals."
            self.target_answer = self._compute_weighted_avg(cat, seg)

        # Provide a plan hint; actual sequence is flexible, but submission requires minimum counted steps
        self.plan_hint = "Hint: open_session -> load orders -> filter/join -> add_field/groupby or compute_metric -> extract_value -> submit_answer"
        self.required_steps_min = required_steps

    def _compute_top_category_city(self, city: str) -> str:
        users_by_city = {u["id"] for u in self.tables["users"] if u["city"] == city}
        rev_by_cat: Dict[str, float] = {}
        pid_to_product = {p["id"]: p for p in self.tables["products"]}
        for o in self.tables["orders"]:
            if o["status"] != "shipped":
                continue
            if o["user_id"] not in users_by_city:
                continue
            prod = pid_to_product.get(o["product_id"])
            if not prod:
                continue
            revenue = prod["price"] * o["quantity"]
            rev_by_cat[prod["category"]] = rev_by_cat.get(prod["category"], 0.0) + revenue
        if not rev_by_cat:
            # If none match, define answer deterministically as "none"
            return "none"
        max_rev = max(rev_by_cat.values())
        candidates = [c for c, v in rev_by_cat.items() if abs(v - max_rev) < 1e-9]
        return sorted(candidates)[0]

    def _compute_users_with_status(self, status: str) -> int:
        has_status = set()
        for o in self.tables["orders"]:
            if o["status"] == status:
                has_status.add(o["user_id"])
        return len(has_status)

    def _compute_weighted_avg(self, category: str, segment: str) -> float:
        pid_to_product = {p["id"]: p for p in self.tables["products"]}
        uid_to_user = {u["id"]: u for u in self.tables["users"]}
        total_w = 0
        total_pxw = 0.0
        for o in self.tables["orders"]:
            u = uid_to_user.get(o["user_id"])
            if not u or u["segment"] != segment:
                continue
            p = pid_to_product.get(o["product_id"])
            if not p or p["category"] != category:
                continue
            total_w += o["quantity"]
            total_pxw += p["price"] * o["quantity"]
        if total_w == 0:
            return 0.0
        return round(total_pxw / total_w, 3)

    def _execute_tool(self, tool: str, args: Dict[str, Any]):
        counted = tool in self.counted_tools
        terminal = False
        success = False
        wrong = False

        if tool == "open_session":
            if self.session_open:
                return "Session already open.", counted, False, False, False
            self.session_open = True
            return "Session opened.", counted, False, False, False

        if tool == "help":
            lines = ["Tools:"]
            for t, meta in self.tool_catalog.items():
                lines.append(f"{t} params={meta['params']}: {meta['desc']}")
            return "\n".join(lines), False, False, False, False

        if tool == "list_sources":
            return "Sources: " + ", ".join(sorted(self.tables.keys())), counted, False, False, False

        if tool == "inspect_schema":
            table = args.get("table")
            if table not in self.tables:
                return f"Protocol violation: unknown table '{table}'.", False, False, False, False
            schema = sorted(list(self.tables[table][0].keys())) if self.tables[table] else []
            return f"Schema for {table}: " + ", ".join(schema), counted, False, False, False

        if tool == "load":
            table = args.get("table")
            if table not in self.tables:
                return f"Protocol violation: unknown table '{table}'.", False, False, False, False
            self.current_table = table
            self.current_data = [dict(row) for row in self.tables[table]]
            return f"Loaded {len(self.current_data)} rows from {table}.", counted, False, False, False

        if tool == "filter":
            if self.current_data is None:
                return "Protocol violation: no data loaded.", False, False, False, False
            cond = args.get("condition")
            if not isinstance(cond, str) or not cond:
                return "Protocol violation: filter requires condition.", False, False, False, False
            self.current_data = [r for r in self.current_data if self._eval_condition(r, cond)]
            return f"Filter applied. Rows: {len(self.current_data)}.", counted, False, False, False

        if tool == "add_field":
            if self.current_data is None:
                return "Protocol violation: no data loaded.", False, False, False, False
            name = args.get("name")
            expr = args.get("expr")
            if not name or not expr:
                return "Protocol violation: add_field needs name and expr.", False, False, False, False
            for r in self.current_data:
                r[name] = self._eval_expr(r, expr)
            return f"Field '{name}' added.", counted, False, False, False

        if tool == "join":
            if self.current_data is None:
                return "Protocol violation: no data loaded.", False, False, False, False
            table = args.get("table")
            on = args.get("on")
            if table not in self.tables or not on or "=" not in on:
                return "Protocol violation: join requires valid table and on='left=right'.", False, False, False, False
            left_key, right_key = [s.strip() for s in on.split("=", 1)]
            right_rows = self.tables[table]
            index = {}
            for rr in right_rows:
                keyv = rr.get(right_key)
                if keyv is not None:
                    index.setdefault(keyv, []).append(rr)
            joined = []
            for l in self.current_data:
                lk = l.get(left_key)
                if lk in index:
                    for rr in index[lk]:
                        newrow = dict(l)
                        for k, v in rr.items():
                            if k in newrow:
                                newrow[f"{table}.{k}"] = v
                            else:
                                newrow[k] = v
                        joined.append(newrow)
            self.current_data = joined
            return f"Join completed with table '{table}'. Rows: {len(self.current_data)}.", counted, False, False, False

        if tool == "groupby":
            if self.current_data is None:
                return "Protocol violation: no data loaded.", False, False, False, False
            by = args.get("by")
            agg = args.get("agg")
            field = args.get("field")
            if not by or not agg or not field:
                return "Protocol violation: groupby requires by, agg, field.", False, False, False, False
            groups: Dict[Any, List[Dict[str, Any]]] = {}
            for r in self.current_data:
                groups.setdefault(r.get(by), []).append(r)
            rows = []
            if agg == "count" and field == "*":
                for k, lst in groups.items():
                    rows.append({by: k, "count": len(lst)})
            elif agg in {"sum", "avg"}:
                for k, lst in groups.items():
                    vals = [self._safe_get_num(x.get(field)) for x in lst]
                    s = sum(vals)
                    if agg == "avg":
                        v = (s / len(vals)) if len(vals) else 0.0
                        rows.append({by: k, f"avg_{field}": v})
                    else:
                        rows.append({by: k, f"sum_{field}": s})
            else:
                return "Protocol violation: unsupported aggregator.", False, False, False, False
            self.current_data = rows
            return f"Groupby applied. Rows: {len(self.current_data)}.", counted, False, False, False

        if tool == "sort":
            if self.current_data is None:
                return "Protocol violation: no data loaded.", False, False, False, False
            field = args.get("field")
            order = str(args.get("order", "asc")).lower()
            reverse = (order == "desc")
            try:
                self.current_data.sort(key=lambda r: (r.get(field) is None, r.get(field)), reverse=reverse)
            except TypeError:
                # Fallback: cast to str
                self.current_data.sort(key=lambda r: str(r.get(field)), reverse=reverse)
            return f"Sorted by {field} {order}.", counted, False, False, False

        if tool == "head":
            if self.current_data is None:
                return "Protocol violation: no data loaded.", False, False, False, False
            n = int(args.get("n", 1))
            self.current_data = self.current_data[:max(0, n)]
            return f"Head kept first {n} rows.", counted, False, False, False

        if tool == "select":
            if self.current_data is None:
                return "Protocol violation: no data loaded.", False, False, False, False
            cols = args.get("columns")
            if not isinstance(cols, str) or not cols:
                return "Protocol violation: select needs columns='a,b,c'.", False, False, False, False
            keep = [c.strip() for c in cols.split(",")]
            new = []
            for r in self.current_data:
                new.append({k: v for k, v in r.items() if k in keep})
            self.current_data = new
            return f"Selected columns: {', '.join(keep)}.", counted, False, False, False

        if tool == "distinct_count":
            if self.current_data is None:
                return "Protocol violation: no data loaded.", False, False, False, False
            field = args.get("field")
            vals = set(r.get(field) for r in self.current_data)
            self.current_data = [{"distinct_count": len(vals)}]
            return f"Distinct count of {field} computed: {len(vals)}.", counted, False, False, False

        if tool == "count_rows":
            if self.current_data is None:
                return "Protocol violation: no data loaded.", False, False, False, False
            cnt = len(self.current_data)
            self.current_data = [{"row_count": cnt}]
            self.last_extracted_value = cnt
            return f"Row count: {cnt}.", counted, False, False, False

        if tool == "extract_value":
            if self.current_data is None or not self.current_data:
                return "Protocol violation: no data available to extract.", False, False, False, False
            field = args.get("field")
            if field not in self.current_data[0]:
                return f"Protocol violation: field '{field}' not in data.", False, False, False, False
            self.last_extracted_value = self.current_data[0][field]
            return f"Extracted value: {self.last_extracted_value}.", counted, False, False, False

        if tool == "compute_metric":
            name = args.get("name")
            if name == "weighted_avg_price":
                if self.current_data is None:
                    return "Protocol violation: no data loaded.", False, False, False, False
                pf = args.get("price_field", "price")
                wf = args.get("weight_field", "quantity")
                total_w = 0
                total_pxw = 0.0
                for r in self.current_data:
                    w = self._safe_get_num(r.get(wf))
                    p = self._safe_get_num(r.get(pf))
                    total_w += int(w)
                    total_pxw += p * w
                val = round((total_pxw / total_w) if total_w else 0.0, 3)
                self.last_extracted_value = val
                self.current_data = [{"metric": val}]
                return f"Computed weighted_avg_price: {val}.", counted, False, False, False
            return "Protocol violation: unsupported metric.", False, False, False, False

        if tool == "submit_answer":
            # Enforce minimum counted steps before submission
            if self.steps_taken < self.required_steps_min:
                return f"Submission rejected: only {self.steps_taken} counted steps taken, need at least {self.required_steps_min}.", False, False, False, False
            value = args.get("value")
            if value is None and self.last_extracted_value is not None:
                value = self.last_extracted_value
            # Compare to target
            if isinstance(self.target_answer, float):
                try:
                    vnum = float(value)
                except:
                    wrong = True
                    return f"Submitted value not numeric: {value}.", False, True, False, True
                ok = abs(vnum - float(self.target_answer)) <= 1e-3
                if ok:
                    success = True
                    return f"Answer accepted: {vnum}.", False, True, True, False
                else:
                    wrong = True
                    return f"Answer wrong: expected ~{self.target_answer}, got {vnum}.", False, True, False, True
            else:
                # string compare, case-sensitive
                sval = str(value)
                ok = (sval == str(self.target_answer))
                if ok:
                    success = True
                    return f"Answer accepted: {sval}.", False, True, True, False
                else:
                    wrong = True
                    return f"Answer wrong: expected '{self.target_answer}', got '{sval}'.", False, True, False, True

        return "No-op.", False, False, False, False

    def _eval_condition(self, row: Dict[str, Any], cond: str) -> bool:
        m = re.match(r"\s*([A-Za-z0-9_.]+)\s*(==|!=|>=|<=|>|<|contains)\s*(.+)\s*$", cond)
        if not m:
            return False
        field, op, val_raw = m.groups()
        val = val_raw.strip()
        if (len(val) >= 2 and ((val[0] == val[-1] == "'") or (val[0] == val[-1] == '"'))):
            val = val[1:-1]
        else:
            if re.fullmatch(r"-?\d+\.\d+", val):
                try:
                    val = float(val)
                except:
                    pass
            elif re.fullmatch(r"-?\d+", val):
                try:
                    val = int(val)
                except:
                    pass
        rv = row.get(field)
        if op == "contains":
            return str(val) in str(rv)
        try:
            if op == "==":
                return rv == val
            if op == "!=":
                return rv != val
            if op == ">":
                return self._safe_get_num(rv) > self._safe_get_num(val)
            if op == "<":
                return self._safe_get_num(rv) < self._safe_get_num(val)
            if op == ">=":
                return self._safe_get_num(rv) >= self._safe_get_num(val)
            if op == "<=":
                return self._safe_get_num(rv) <= self._safe_get_num(val)
        except:
            return False
        return False

    def _eval_expr(self, row: Dict[str, Any], expr: str) -> float:
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_.]*|\d+\.\d+|\d+|[\+\-\*/]", expr)
        values: List[Union[float, str]] = []
        for t in tokens:
            if t in "+-*/":
                values.append(t)
            elif re.fullmatch(r"\d+\.\d+", t) or re.fullmatch(r"\d+", t):
                values.append(float(t))
            else:
                values.append(self._safe_get_num(row.get(t)))
        # left-to-right without precedence for simplicity, but apply * and / precedence lightly
        # Implement standard precedence: first pass for * and /
        def reduce_ops(vals, ops):
            i = 0
            out = []
            while i < len(vals):
                v = vals[i]
                if isinstance(v, str) and v in ops:
                    # operator, apply to last out value and next vals[i+1]
                    op = v
                    a = out.pop()
                    b = vals[i+1]
                    if op == "*":
                        out.append(a * b)
                    else:
                        out.append(a / b if b != 0 else 0.0)
                    i += 2
                else:
                    out.append(v)
                    i += 1
            return out
        vals = values[:]
        vals = reduce_ops(vals, {"*", "/"})
        # now left-to-right + and -
        res = None
        op = None
        for v in vals:
            if isinstance(v, str) and v in {"+", "-"}:
                op = v
            else:
                if res is None:
                    res = float(v)
                else:
                    res = res + v if op == "+" else res - v
        return float(res if res is not None else 0.0)

    def _safe_get_num(self, v) -> float:
        try:
            if isinstance(v, bool):
                return float(int(v))
            return float(v)
        except:
            return 0.0


class ToolCrafterRelayEnvWithFeedback(ToolCrafterRelayEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        text = obs.lower()
        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint = None

        if "invalid action format" in text or "unknown tool" in text:
            error_type = "FormatError"
            if "invalid action format" in text:
                error_detail["issue"] = "missing_or_bad_boxed"
                hint = "Use \\boxed{tool arg=value, ...} with a known tool."
            else:
                error_detail["issue"] = "unknown_tool"
                hint = "Use a listed tool from help or instructions."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "open_session" in text:
                error_detail["violation"] = "session_not_open"
                hint = "Call \\boxed{open_session} before other tools."
            elif "no data loaded" in text:
                error_detail["violation"] = "data_not_loaded"
                hint = "Load a table first, e.g., \\boxed{load table='orders'}."
            elif "unknown table" in text:
                error_detail["violation"] = "unknown_table"
                hint = "Check table names with \\boxed{list_sources}."
            else:
                error_detail["violation"] = "other_protocol"
                hint = "Re-check tool parameters and required prerequisites."

        elif "submission rejected" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "early_submission"
            error_detail["steps_taken"] = getattr(self, "steps_taken", None)
            error_detail["required_min"] = getattr(self, "required_steps_min", None)
            hint = "Perform more counted steps (e.g., filter/join/groupby) before submitting."

        elif "execution error" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "runtime_exception"
            hint = "Verify parameter types and values. Inspect schema or sources if needed."

        elif "timeout" in text:
            error_type = "Timeout"
            hint = "Plan your sequence to fit within max turns and avoid retries."

        elif "final answer incorrect" in text or "answer wrong" in text:
            error_type = "WrongDecision"
            got_match = re.search(r"got ['\"]?([^'\"]+)['\"]?", obs)
            if got_match:
                error_detail["got"] = got_match.group(1)
            if isinstance(getattr(self, "target_answer", None), (int, float)):
                error_detail["expected_type"] = "numeric"
            else:
                error_detail["expected_type"] = "string"
            hint = "Double-check filters and joins. Inspect intermediate rows with sort/head and ensure correct aggregation."

        elif "success" in text or "answer accepted" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "steps_taken": getattr(self, "steps_taken", None),
                "required_steps_min": getattr(self, "required_steps_min", None),
                "session_open": getattr(self, "session_open", None),
                "current_table": getattr(self, "current_table", None),
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
            "hint": "Start with \\boxed{open_session}, then \\boxed{list_sources} or \\boxed{load table='orders'}.",
            "turn": 0,
            "state": {
                "steps_taken": 0,
                "required_steps_min": getattr(self, "required_steps_min", None),
                "session_open": False,
                "current_table": None,
            },
        }
        return obs, info