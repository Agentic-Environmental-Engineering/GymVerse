from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List, Union


class ToolchainConductorEnv(Env):
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
        # Tool catalog
        # Tools must be executable and stateful
        self.tools = {
            "compute_metric": {
                "description": "Compute a simple metric on a single dataset without joins. Args: dataset(str), op(str in [count_rows, sum, count_unique]), field(optional str), where(optional dict eq filters). Returns number.",
                "params": ["dataset", "op", "field", "where"]
            },
            "load": {
                "description": "Load a dataset into the current workspace. Args: source(str).",
                "params": ["source"]
            },
            "concat": {
                "description": "Append rows from another dataset into current data. Args: source(str).",
                "params": ["source"]
            },
            "join": {
                "description": "Join current data with another dataset. Args: source(str), left_on(str), right_on(str), how(optional str in [inner,left]). Right fields are prefixed by source name.",
                "params": ["source", "left_on", "right_on", "how"]
            },
            "filter": {
                "description": "Filter rows by a condition. Args: field(str), op(str in [eq,gt,gte,lt,lte,contains]), value",
                "params": ["field", "op", "value"]
            },
            "derive": {
                "description": "Create a new field by combining two fields. Args: new_field(str), op(str in [product,sum,ratio]), a(str field), b(str field or numeric literal).",
                "params": ["new_field", "op", "a", "b"]
            },
            "sum": {
                "description": "Compute sum of a numeric field over current data. Args: field(str). Stores result in last_result.",
                "params": ["field"]
            },
            "count_rows": {
                "description": "Count number of rows in current data. No args. Stores result in last_result.",
                "params": []
            },
            "count_unique": {
                "description": "Count unique values of a field. Args: field(str). Stores result in last_result.",
                "params": ["field"]
            },
            "describe": {
                "description": "Describe current data schema and size. No args.",
                "params": []
            },
            "preview": {
                "description": "Preview first n rows. Args: n(int).",
                "params": ["n"]
            },
            "answer": {
                "description": "Submit final numeric answer. Args: value(number).",
                "params": ["value"]
            },
        }

        # Synthetic datasets
        # Categories, brands, regions scale with complexity
        base_categories = ["Gadget", "Home", "Food", "Book", "Sport", "Beauty", "Toy", "Garden"]
        base_brands = ["Axiom", "Nimbus", "Orion", "Pioneer", "Quanta", "Riviera", "Solace", "Trident", "Umbra"]
        base_regions = ["North", "South", "East", "West", "Central", "Coastal", "Highland", "Metro"]

        cat_count = min(len(base_categories), 4 + self.complexity // 2)
        brand_count = min(len(base_brands), 5 + self.complexity)
        region_count = min(len(base_regions), 3 + self.complexity)

        self.categories = random.sample(base_categories, cat_count)
        self.brands = random.sample(base_brands, brand_count)
        self.regions = random.sample(base_regions, region_count)

        # Products
        prod_count = 20 + self.complexity * 10
        self.products = []
        for i in range(prod_count):
            pid = f"P{i+1:03d}"
            brand = random.choice(self.brands)
            category = random.choice(self.categories)
            price = round(random.uniform(5, 300), 2)
            self.products.append({"id": pid, "brand": brand, "category": category, "price": price})

        # Taxes per region with multiplier = 1 + rate
        self.taxes = []
        for r in self.regions:
            rate = round(random.uniform(0.02, 0.18), 3)
            self.taxes.append({"region": r, "rate": rate, "multiplier": round(1.0 + rate, 3)})

        # Discounts per brand with multiplier = 1 - discount_rate
        # Some brands may have no discount -> multiplier 1.0
        self.discounts = []
        for b in self.brands:
            if random.random() < 0.6:
                drate = round(random.uniform(0.02, 0.25), 3)
                self.discounts.append({"brand": b, "discount_rate": drate, "multiplier": round(1.0 - drate, 3)})
        # Ensure at least one discount exists
        if not self.discounts:
            b = random.choice(self.brands)
            drate = round(random.uniform(0.05, 0.2), 3)
            self.discounts.append({"brand": b, "discount_rate": drate, "multiplier": round(1.0 - drate, 3)})

        # Transactions split into two parts to support concat
        # Each transaction references product_id, region, quantity
        def gen_transactions(n_rows: int, start_id: int) -> List[Dict[str, Any]]:
            rows = []
            for i in range(n_rows):
                tid = start_id + i
                prod = random.choice(self.products)
                region = random.choice(self.regions)
                qty = random.randint(1, 8 + self.complexity)
                rows.append({
                    "trans_id": f"T{tid:05d}",
                    "product_id": prod["id"],
                    "region": region,
                    "quantity": qty
                })
            return rows

        n1 = 60 + self.complexity * 20
        n2 = 40 + self.complexity * 15
        self.transactions_A = gen_transactions(n1, 1)
        self.transactions_B = gen_transactions(n2, 1 + n1)

        # Assemble datasets registry
        self.datasets = {
            "products": self.products,
            "taxes": self.taxes,
            "discounts": self.discounts,
            "transactions_A": self.transactions_A,
            "transactions_B": self.transactions_B,
        }

        # Execution state
        self.current_data = None
        self.current_data_name = None
        self.last_result: Optional[float] = None

        # Task state
        self.task: Dict[str, Any] = {}
        self.turn_count = 0
        self.steps_taken = 0
        self.required_steps = None

    def _get_instructions(self) -> str:
        tool_list = ", ".join(sorted(self.tools.keys()))
        lines = []
        lines.append("You are orchestrating tools to compute a requested analytic from simulated datasets.")
        lines.append("Datasets: products(id, brand, category, price), taxes(region, rate, multiplier), discounts(brand, discount_rate, multiplier), transactions_A(trans_id, product_id, region, quantity), transactions_B(trans_id, product_id, region, quantity).")
        lines.append(f"Available tools: {tool_list}.")
        lines.append("Usage format: use \\boxed{tool_name(arg1=value1, arg2='text', ...)} exactly once per turn.")
        lines.append("Examples:")
        lines.append("\\boxed{load(source='transactions_A')}")
        lines.append("\\boxed{join(source='products', left_on='product_id', right_on='id', how='inner')}")
        lines.append("\\boxed{filter(field='region', op='eq', value='North')}")
        lines.append("\\boxed{derive(new_field='revenue', op='product', a='quantity', b='products.price')}")
        lines.append("\\boxed{sum(field='revenue')}")
        lines.append("\\boxed{answer(value=12345.67)}")
        lines.append("")
        lines.append(f"Task: {self.task.get('description','')}")
        return "\n".join(lines)

    def get_task_suffix(self) -> str:
        cur = "none"
        rows = 0
        if isinstance(self.current_data, list):
            cur = self.current_data_name or "anonymous"
            rows = len(self.current_data)
        suffix = []
        suffix.append(f"Turn {self.turn_count}/{self.max_turns} | Tool calls used: {self.steps_taken} | Required steps range: [{self.min_required_steps}, {self.max_required_steps}]")
        suffix.append(f"Current data: {cur}, rows={rows}")
        suffix.append("Respond using \\boxed{...} with a single tool call.")
        return "\n".join(suffix)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self.turn_count = 0
        self.steps_taken = 0
        self.current_data = None
        self.current_data_name = None
        self.last_result = None

        self.required_steps = random.randint(self.min_required_steps, self.max_required_steps)
        self.task = self._generate_task_requiring_n_steps(self.required_steps)
        obs = self._get_instructions()
        info = {"suffix": self.get_task_suffix()}
        return obs, info

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        info = {}
        # Timeout check first
        if self.turn_count > self.max_turns:
            obs = "Timeout: maximum turns reached."
            return obs, float(LanguageGameReward.fail_reward), True, True, {"suffix": self.get_task_suffix()}

        parsed = self._parse_action(action)
        if parsed is None:
            obs = "Invalid action format. Use \\boxed{tool_name(...)} exactly."
            return obs, float(LanguageGameReward.format_error_reward), True, False, {"suffix": self.get_task_suffix()}

        tool_name, args = parsed
        if tool_name not in self.tools:
            obs = f"Unknown tool: {tool_name}"
            return obs, float(LanguageGameReward.fail_reward), True, False, {"suffix": self.get_task_suffix()}

        # Execute tool
        try:
            result_text, is_terminal, success = self._execute_tool(tool_name, args)
            # Count only valid execution attempts as steps
            self.steps_taken += 1
            if is_terminal:
                if success:
                    reward = float(LanguageGameReward.success_reward)
                    obs = f"{result_text}\nSuccess: correct answer."
                else:
                    reward = float(LanguageGameReward.fail_reward)
                    obs = f"{result_text}\nFailure: incorrect answer."
                return obs, reward, True, False, {"suffix": self.get_task_suffix()}
            else:
                # Continue
                obs = result_text
                # Check timeout after action
                if self.turn_count >= self.max_turns:
                    obs += "\nTimeout: maximum turns reached."
                    return obs, float(LanguageGameReward.fail_reward), True, True, {"suffix": self.get_task_suffix()}
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
        except Exception as e:
            obs = f"Execution error: {e}"
            # Protocol errors do not terminate; user can recover
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        if not isinstance(action, str):
            return None
        m = re.search(r"\\boxed\{(.+)\}\s*$", action.strip(), flags=re.DOTALL)
        if not m:
            return None
        inner = m.group(1).strip()
        # Expect form: tool(args) or tool()
        m2 = re.match(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\((.*)\))?\s*$", inner, flags=re.DOTALL)
        if not m2:
            return None
        tool = m2.group(1)
        args_str = m2.group(2)
        args: Dict[str, Any] = {}
        if args_str is None or args_str.strip() == "":
            return tool, args

        # Simple key=value parser supporting strings, numbers, booleans, and lists of strings
        tokens = self._split_args(args_str)
        for tok in tokens:
            if "=" not in tok:
                continue
            k, v = tok.split("=", 1)
            key = k.strip()
            val = v.strip()
            parsed_val: Any
            # String literal
            if (len(val) >= 2 and ((val[0] == "'" and val[-1] == "'") or (val[0] == '"' and val[-1] == '"'))):
                parsed_val = val[1:-1]
            elif re.match(r"^\-?\d+\.\d+$", val):
                parsed_val = float(val)
            elif re.match(r"^\-?\d+$", val):
                parsed_val = int(val)
            elif val.lower() in ["true", "false"]:
                parsed_val = True if val.lower() == "true" else False
            elif val.startswith("[") and val.endswith("]"):
                inner_list = val[1:-1].strip()
                if inner_list == "":
                    parsed_val = []
                else:
                    parts = self._split_list(inner_list)
                    lst = []
                    for p in parts:
                        p = p.strip()
                        if (len(p) >= 2 and ((p[0] == "'" and p[-1] == "'") or (p[0] == '"' and p[-1] == '"'))):
                            lst.append(p[1:-1])
                        elif re.match(r"^\-?\d+\.\d+$", p):
                            lst.append(float(p))
                        elif re.match(r"^\-?\d+$", p):
                            lst.append(int(p))
                        else:
                            lst.append(p)
                    parsed_val = lst
            else:
                parsed_val = val
            args[key] = parsed_val
        return tool, args

    def _split_args(self, s: str) -> List[str]:
        parts = []
        depth = 0
        current = []
        in_string = False
        string_char = None
        for ch in s:
            if in_string:
                current.append(ch)
                if ch == string_char:
                    in_string = False
                    string_char = None
                continue
            if ch in ("'", '"'):
                in_string = True
                string_char = ch
                current.append(ch)
            elif ch == "(" or ch == "[":
                depth += 1
                current.append(ch)
            elif ch == ")" or ch == "]":
                depth -= 1
                current.append(ch)
            elif ch == "," and depth == 0:
                parts.append("".join(current).strip())
                current = []
            else:
                current.append(ch)
        if current:
            parts.append("".join(current).strip())
        return parts

    def _split_list(self, s: str) -> List[str]:
        return self._split_args(s)

    def sample_random_action(self) -> str:
        tool = random.choice(list(self.tools.keys()))
        if tool == "load":
            src = random.choice(list(self.datasets.keys()))
            return f"\\boxed{{load(source='{src}')}}"
        if tool == "compute_metric":
            ds = random.choice(list(self.datasets.keys()))
            op = random.choice(["count_rows", "sum", "count_unique"])
            if op == "sum":
                field = "price" if ds == "products" else "quantity"
                return f"\\boxed{{compute_metric(dataset='{ds}', op='sum', field='{field}')}}"
            if op == "count_unique":
                field = "brand" if ds == "products" else "region"
                return f"\\boxed{{compute_metric(dataset='{ds}', op='count_unique', field='{field}')}}"
            return f"\\boxed{{compute_metric(dataset='{ds}', op='count_rows')}}"
        if tool == "join":
            return "\\boxed{join(source='products', left_on='product_id', right_on='id', how='inner')}"
        if tool == "filter":
            return "\\boxed{filter(field='region', op='eq', value='North')}"
        if tool == "derive":
            return "\\boxed{derive(new_field='revenue', op='product', a='quantity', b='products.price')}"
        if tool == "sum":
            return "\\boxed{sum(field='quantity')}"
        if tool == "count_rows":
            return "\\boxed{count_rows()}"
        if tool == "count_unique":
            return "\\boxed{count_unique(field='product_id')}"
        if tool == "describe":
            return "\\boxed{describe()}"
        if tool == "preview":
            return "\\boxed{preview(n=3)}"
        if tool == "concat":
            return "\\boxed{concat(source='transactions_B')}"
        if tool == "answer":
            return "\\boxed{answer(value=0)}"
        return "\\boxed{describe()}"

    def _generate_task_requiring_n_steps(self, required_steps: int) -> Dict[str, Any]:
        # Two families: Simple metrics (low complexity) and Revenue with joins (higher complexity)
        # Keep sampling a feasible configuration whose minimal plan length falls in range
        for _ in range(100):
            if required_steps <= 3:
                # Simple metric tasks solvable with: compute_metric + answer = 2 steps
                ds = random.choice(["products", "transactions_A", "transactions_B"])
                op = random.choice(["count_rows", "count_unique", "sum"])
                if op == "count_unique":
                    field = "brand" if ds == "products" else "region"
                elif op == "sum":
                    field = "price" if ds == "products" else "quantity"
                else:
                    field = None

                where = None
                if self.complexity >= 2 and random.random() < 0.4:
                    # one simple eq filter
                    if ds == "products":
                        where = {"category": random.choice(self.categories)}
                    else:
                        where = {"region": random.choice(self.regions)}
                desc = f"Return the {op} of {field if field else 'rows'} in dataset {ds}"
                if where:
                    k = list(where.keys())[0]
                    desc += f" where {k} == '{where[k]}'"
                desc += ". Submit with answer(value=...)."

                value = self._compute_simple_metric(ds, op, field, where)
                return {
                    "type": "simple_metric",
                    "description": desc,
                    "dataset": ds,
                    "op": op,
                    "field": field,
                    "where": where,
                    "reference_answer": value,
                }
            else:
                # Revenue task with joins and optional unions/discounts/taxes and filters
                use_union = random.random() < 0.5
                need_join_products = True
                use_taxes = random.random() < 0.5
                use_discounts = random.random() < 0.5
                # At least category and region filters; optionally add brand filter
                filters = [{"field": "products.category", "op": "eq", "value": random.choice(self.categories)},
                           {"field": "region", "op": "eq", "value": random.choice(self.regions)}]
                if random.random() < 0.5:
                    filters.append({"field": "products.brand", "op": "eq", "value": random.choice(self.brands)})

                # Minimal plan length estimation
                base = 1  # load
                if use_union:
                    base += 1  # concat
                base += 1  # join products
                joins_more = 0
                if use_taxes:
                    base += 1
                    joins_more += 1
                if use_discounts:
                    base += 1
                    joins_more += 1
                base += len(filters)  # filter steps
                # derivations: price*tax_multiplier*(discount_multiplier) then * quantity
                derive_steps = 1 + (1 if use_taxes else 0) + (1 if use_discounts else 0)
                base += derive_steps
                base += 1  # sum
                base += 1  # answer

                # Adjust toggles to approach required range
                if base < self.min_required_steps:
                    # add a harmless extra filter by quantity threshold
                    thr = random.randint(1, 3)
                    filters.append({"field": "quantity", "op": "gte", "value": thr})
                    base += 1
                if base <= self.max_required_steps:
                    # Build description
                    cat = [f for f in filters if f["field"] == "products.category"][0]["value"]
                    reg = [f for f in filters if f["field"] == "region"][0]["value"]
                    desc_parts = [f"Compute total revenue from transactions in region '{reg}' for category '{cat}'"]
                    extra = [f for f in filters if f["field"] not in ("products.category", "region")]
                    for e in extra:
                        desc_parts.append(f"and {e['field']} == '{e['value']}'" if e["op"] == "eq" else f"and {e['field']} {e['op']} {e['value']}")
                    if use_union:
                        desc_parts.append("(union parts A and B)")
                    if use_taxes:
                        desc_parts.append("(apply taxes)")
                    if use_discounts:
                        desc_parts.append("(apply discounts)")
                    desc = " ".join(desc_parts) + ". Submit with answer(value=...)."

                    ref = self._compute_revenue_reference(
                        use_union=use_union,
                        use_taxes=use_taxes,
                        use_discounts=use_discounts,
                        filters=filters
                    )
                    return {
                        "type": "revenue",
                        "description": desc,
                        "use_union": use_union,
                        "use_taxes": use_taxes,
                        "use_discounts": use_discounts,
                        "filters": filters,
                        "reference_answer": ref,
                    }

        # Fallback to simplest possible task
        value = self._compute_simple_metric("products", "count_rows", None, None)
        return {
            "type": "simple_metric",
            "description": "Return the count of rows in dataset products. Submit with answer(value=...).",
            "dataset": "products",
            "op": "count_rows",
            "field": None,
            "where": None,
            "reference_answer": value,
        }

    def _compute_simple_metric(self, ds: str, op: str, field: Optional[str], where: Optional[Dict[str, Any]]) -> float:
        data = self.datasets[ds]
        rows = data
        if where:
            k, v = list(where.items())[0]
            rows = [r for r in rows if r.get(k) == v]
        if op == "count_rows":
            return float(len(rows))
        if op == "sum":
            if field is None:
                return 0.0
            total = 0.0
            for r in rows:
                val = r.get(field)
                if isinstance(val, (int, float)):
                    total += float(val)
            return float(round(total, 6))
        if op == "count_unique":
            if field is None:
                return 0.0
            s = set()
            for r in rows:
                s.add(r.get(field))
            return float(len(s))
        return 0.0

    def _compute_revenue_reference(self, use_union: bool, use_taxes: bool, use_discounts: bool, filters: List[Dict[str, Any]]) -> float:
        # Start with transactions
        trans = list(self.transactions_A)
        if use_union:
            trans = trans + list(self.transactions_B)
        # Indexes for joins
        prod_idx = {p["id"]: p for p in self.products}
        taxes_idx = {t["region"]: t for t in self.taxes}
        disc_idx = {d["brand"]: d for d in self.discounts}
        total = 0.0
        for t in trans:
            p = prod_idx.get(t["product_id"])
            if not p:
                continue
            rec = {
                "trans_id": t["trans_id"],
                "product_id": t["product_id"],
                "region": t["region"],
                "quantity": t["quantity"],
                "products.brand": p["brand"],
                "products.category": p["category"],
                "products.price": p["price"],
            }
            if use_taxes:
                tx = taxes_idx.get(t["region"])
                if not tx:
                    continue
                rec["taxes.multiplier"] = tx["multiplier"]
            if use_discounts:
                mul = 1.0
                if p["brand"] in disc_idx:
                    mul = disc_idx[p["brand"]]["multiplier"]
                rec["discounts.multiplier"] = mul

            # Apply filters
            passed = True
            for f in filters:
                field = f["field"]; op = f["op"]; val = f["value"]
                rv = rec.get(field)
                if op == "eq":
                    if rv != val:
                        passed = False; break
                elif op == "gte":
                    try:
                        if float(rv) < float(val):
                            passed = False; break
                    except:
                        passed = False; break
                elif op == "gt":
                    try:
                        if float(rv) <= float(val):
                            passed = False; break
                    except:
                        passed = False; break
                elif op == "lte":
                    try:
                        if float(rv) > float(val):
                            passed = False; break
                    except:
                        passed = False; break
                elif op == "lt":
                    try:
                        if float(rv) >= float(val):
                            passed = False; break
                    except:
                        passed = False; break
                elif op == "contains":
                    if not (isinstance(rv, str) and isinstance(val, str) and (val in rv)):
                        passed = False; break
                else:
                    passed = False; break
            if not passed:
                continue

            # Compute adjusted price
            price = rec["products.price"]
            adjusted = price
            if use_taxes:
                adjusted *= rec.get("taxes.multiplier", 1.0)
            if use_discounts:
                adjusted *= rec.get("discounts.multiplier", 1.0)
            revenue = adjusted * t["quantity"]
            total += revenue
        return float(round(total, 6))

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Tuple[str, bool, bool]:
        # Returns: (observation, terminal, success)
        if tool_name == "compute_metric":
            ds = args.get("dataset")
            op = args.get("op")
            field = args.get("field")
            where = args.get("where")
            if ds not in self.datasets:
                raise ValueError(f"Protocol violation: dataset not found '{ds}'.")
            if op not in ("count_rows", "sum", "count_unique"):
                raise ValueError("Protocol violation: unsupported op for compute_metric.")
            if where is not None and not isinstance(where, dict):
                raise ValueError("Protocol violation: 'where' must be a dict of equality condition.")
            value = self._compute_simple_metric(ds, op, field, where)
            self.last_result = value
            return f"compute_metric executed on {ds}. Result: {value}", False, False

        if tool_name == "load":
            src = args.get("source")
            if src not in self.datasets:
                raise ValueError(f"Protocol violation: unknown source '{src}'.")
            self.current_data = [dict(r) for r in self.datasets[src]]
            self.current_data_name = src
            return f"Loaded dataset '{src}' with {len(self.current_data)} rows.", False, False

        if tool_name == "concat":
            if self.current_data is None:
                raise ValueError("Protocol violation: no data loaded. Call load() first.")
            src = args.get("source")
            if src not in self.datasets:
                raise ValueError(f"Protocol violation: unknown source '{src}'.")
            add = self.datasets[src]
            self.current_data.extend([dict(r) for r in add])
            return f"Concatenated dataset '{src}'. Current rows: {len(self.current_data)}.", False, False

        if tool_name == "join":
            if self.current_data is None:
                raise ValueError("Protocol violation: no data loaded. Call load() first.")
            src = args.get("source")
            left_on = args.get("left_on")
            right_on = args.get("right_on")
            how = args.get("how", "inner")
            if src not in self.datasets:
                raise ValueError(f"Protocol violation: unknown source '{src}'.")
            data_right = self.datasets[src]
            idx = {}
            for r in data_right:
                key = r.get(right_on)
                if key not in idx:
                    idx[key] = []
                idx[key].append(r)
            out = []
            for left in self.current_data:
                lk = left.get(left_on)
                matched = idx.get(lk, [])
                if matched:
                    for r in matched:
                        new_row = dict(left)
                        for k, v in r.items():
                            new_row[f"{src}.{k}"] = v
                        out.append(new_row)
                elif how == "left":
                    new_row = dict(left)
                    for k in data_right[0].keys():
                        new_row[f"{src}.{k}"] = None
                    out.append(new_row)
            self.current_data = out
            return f"Joined with '{src}' on {left_on}={right_on} ({how}). Rows now: {len(self.current_data)}.", False, False

        if tool_name == "filter":
            if self.current_data is None:
                raise ValueError("Protocol violation: no data loaded. Call load() first.")
            field = args.get("field")
            op = args.get("op")
            value = args.get("value")
            if field is None or op is None:
                raise ValueError("Protocol violation: filter requires field and op.")
            def cmp(rv, op, val):
                if op == "eq":
                    return rv == val
                if op == "contains":
                    return isinstance(rv, str) and isinstance(val, str) and (val in rv)
                try:
                    rvf = float(rv)
                    vf = float(val)
                except:
                    return False
                if op == "gt":
                    return rvf > vf
                if op == "gte":
                    return rvf >= vf
                if op == "lt":
                    return rvf < vf
                if op == "lte":
                    return rvf <= vf
                return False
            before = len(self.current_data)
            self.current_data = [r for r in self.current_data if cmp(r.get(field), op, value)]
            return f"Filter applied: {field} {op} {value}. Rows: {before} -> {len(self.current_data)}.", False, False

        if tool_name == "derive":
            if self.current_data is None:
                raise ValueError("Protocol violation: no data loaded. Call load() first.")
            new_field = args.get("new_field")
            op = args.get("op")
            a = args.get("a")
            b = args.get("b")
            if new_field is None or op not in ("product", "sum", "ratio") or a is None or b is None:
                raise ValueError("Protocol violation: derive requires new_field, op in [product,sum,ratio], a, b.")
            # b can be numeric literal or field
            for row in self.current_data:
                av = row.get(a)
                bv = None
                if isinstance(b, (int, float)):
                    bv = b
                else:
                    # try numeric literal in string
                    try:
                        bv = float(b) if isinstance(b, str) and re.match(r"^\-?\d+(\.\d+)?$", b) else row.get(b)
                    except:
                        bv = row.get(b)
                if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
                    if op == "product":
                        row[new_field] = av * bv
                    elif op == "sum":
                        row[new_field] = av + bv
                    elif op == "ratio":
                        row[new_field] = (av / bv) if bv != 0 else 0.0
                else:
                    row[new_field] = None
            return f"Derived field '{new_field}' using {op} of {a} and {b}.", False, False

        if tool_name == "sum":
            if self.current_data is None:
                raise ValueError("Protocol violation: no data loaded. Call load() first.")
            field = args.get("field")
            if field is None:
                raise ValueError("Protocol violation: sum requires field.")
            total = 0.0
            for r in self.current_data:
                v = r.get(field)
                if isinstance(v, (int, float)):
                    total += float(v)
            self.last_result = float(round(total, 6))
            return f"Sum computed for field '{field}': {self.last_result}", False, False

        if tool_name == "count_rows":
            if self.current_data is None:
                raise ValueError("Protocol violation: no data loaded. Call load() first.")
            self.last_result = float(len(self.current_data))
            return f"Row count: {int(self.last_result)}", False, False

        if tool_name == "count_unique":
            if self.current_data is None:
                raise ValueError("Protocol violation: no data loaded. Call load() first.")
            field = args.get("field")
            if field is None:
                raise ValueError("Protocol violation: count_unique requires field.")
            s = set()
            for r in self.current_data:
                s.add(r.get(field))
            self.last_result = float(len(s))
            return f"Unique count for '{field}': {int(self.last_result)}", False, False

        if tool_name == "describe":
            if self.current_data is None:
                return "No data loaded.", False, False
            # Collect schema keys
            keys = set()
            for r in self.current_data[:5]:
                for k in r.keys():
                    keys.add(k)
            return f"Current data '{self.current_data_name}': rows={len(self.current_data)}, fields={sorted(list(keys))[:12]}", False, False

        if tool_name == "preview":
            if self.current_data is None:
                raise ValueError("Protocol violation: no data loaded. Call load() first.")
            n = args.get("n", 5)
            try:
                n = int(n)
            except:
                n = 5
            sample = self.current_data[:max(0, n)]
            return f"Preview first {len(sample)} rows: {sample}", False, False

        if tool_name == "answer":
            value = args.get("value")
            if not isinstance(value, (int, float)):
                raise ValueError("Protocol violation: answer requires numeric value.")
            target = float(self.task.get("reference_answer", 0.0))
            ok = abs(float(value) - target) <= max(1e-6, 1e-6 * max(1.0, abs(target)))
            return f"Answer submitted: {value}. Target: {target}", True, ok

        raise ValueError("UnsupportedAction")



class ToolchainConductorEnvWithFeedback(ToolchainConductorEnv):
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
            hint = "Wrap exactly one tool call inside \\boxed{...} with named arguments."

        elif "unknown tool" in text:
            error_type = "UnsupportedAction"
            m = re.search(r"unknown tool:\s*([a-z0-9_]+)", obs, flags=re.I)
            error_detail["tool"] = m.group(1) if m else None
            hint = "Use only the listed tools in the instructions."

        elif "timeout" in text:
            error_type = "Timeout"
            error_detail["reason"] = "max_turns_reached"
            hint = "Plan your steps. Prioritize loading data, required joins, filters, derivations, then aggregate and answer."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            # Identify common violations
            if "no data loaded" in text:
                error_detail["violation"] = "missing_load"
                hint = "Call load(source=...) before join/filter/aggregate."
            elif "dataset not found" in text:
                error_detail["violation"] = "bad_dataset_name"
                hint = "Use a valid dataset: products, taxes, discounts, transactions_A, transactions_B."
            elif "requires field" in text:
                error_detail["violation"] = "missing_arg_or_field"
                hint = "Double-check required arguments and that you created needed fields via derive()."
            else:
                error_detail["violation"] = "general_protocol_error"
                hint = "Verify tool arguments and call order."

        elif "execution error" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "runtime_error"
            hint = "Check that referenced fields exist and argument types are correct."

        elif "failure: incorrect answer" in text:
            error_type = "WrongDecision"
            error_detail["expected"] = "numeric value close to the hidden target"
            m = re.search(r"answer submitted:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?:[^0-9eE]|$)", obs, flags=re.I)
            got_str = m.group(1) if m else None
            try:
                error_detail["got"] = float(got_str) if got_str is not None else None
            except Exception:
                error_detail["got"] = None
            hint = "Recompute using tools: load data, union if needed, join products (and taxes/discounts), apply filters, derive revenue, sum, then submit."

        elif "success: correct answer" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "steps_taken": getattr(self, "steps_taken", None),
                "required_range": [self.min_required_steps, self.max_required_steps],
                "current_data": getattr(self, "current_data_name", None),
            }
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info = info or {}
        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        hint = "For simple tasks, consider compute_metric(...), then answer(value=...). For revenue tasks, start with load(...)."
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": hint if self.feedback_level >= 2 else None,
            "turn": 0,
            "state": {
                "steps_taken": 0,
                "required_range": [self.min_required_steps, self.max_required_steps],
                "current_data": None,
            } if self.feedback_level >= 1 else None,
        }
        return obs, info
