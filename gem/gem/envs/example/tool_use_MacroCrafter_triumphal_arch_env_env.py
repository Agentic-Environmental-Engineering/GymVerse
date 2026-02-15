from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List, Callable, Union

class MacroCrafterEnv(Env):
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
            "load": {
                "description": "Load a data source into a named table. Args: source(str), name(optional str). Sets active table.",
                "parameters": ["source", "name"],
                "returns": "Table loaded, set active"
            },
            "save_as": {
                "description": "Save the active table under a new name. Args: name(str). Sets active table to that name.",
                "parameters": ["name"],
                "returns": "Table saved"
            },
            "switch_active": {
                "description": "Switch active table. Args: name(str).",
                "parameters": ["name"],
                "returns": "Active table switched"
            },
            "preview": {
                "description": "Show first n rows. Args: n(int, optional).",
                "parameters": ["n"],
                "returns": "Text preview"
            },
            "select": {
                "description": "Keep only specified columns. Args: columns(list of str as comma-separated or bracket).",
                "parameters": ["columns"],
                "returns": "Columns reduced"
            },
            "filter": {
                "description": "Filter rows by simple conditions. Args: condition(str). Supports AND, ==, !=, >, >=, <, <=, contains, startswith, endswith, in [a|b], notin [a|b].",
                "parameters": ["condition"],
                "returns": "Rows filtered"
            },
            "derive": {
                "description": "Add a column from expression. Args: new(str), expr(str). Supports: col +/-/*// number or col; concat(col,'-',col).",
                "parameters": ["new", "expr"],
                "returns": "Column added"
            },
            "aggregate": {
                "description": "Aggregate values. Args: op(sum|avg|max|min|count), column(optional), by(optional str). Without by returns scalar; with by returns table with (by, value).",
                "parameters": ["op", "column", "by"],
                "returns": "Scalar or table"
            },
            "groupby_aggregate": {
                "description": "Alias of aggregate with explicit grouping. Args: by(str), column(str), op(str).",
                "parameters": ["by", "column", "op"],
                "returns": "Table"
            },
            "join": {
                "description": "Join active table with another table. Args: other(str), on(str), how(optional='inner').",
                "parameters": ["other", "on", "how"],
                "returns": "Joined table"
            },
            "sort_by": {
                "description": "Sort active table. Args: column(str), descending(optional bool).",
                "parameters": ["column", "descending"],
                "returns": "Sorted table"
            },
            "slice_rows": {
                "description": "Keep top n rows. Args: n(int).",
                "parameters": ["n"],
                "returns": "Sliced table"
            },
            "count_rows": {
                "description": "Return row count as scalar.",
                "parameters": [],
                "returns": "Scalar"
            },
            "unique_values": {
                "description": "Count unique values for a column. Args: column(str). Returns scalar.",
                "parameters": ["column"],
                "returns": "Scalar"
            },
            "regex_extract": {
                "description": "Extract regex group into new column. Args: column(str), pattern(str), into(str).",
                "parameters": ["column", "pattern", "into"],
                "returns": "Column added"
            },
            "submit_answer": {
                "description": "Submit final answer. Args: value(str|number). Ends episode.",
                "parameters": ["value"],
                "returns": "Evaluation"
            },
        }

        # Simulated data sources
        rnd = random.Random(1337 + self.complexity)
        num_products = 12 + self.complexity * 2
        num_customers = 30 + self.complexity * 5
        num_sales = 120 + self.complexity * 30

        categories = ["Electronics", "Home", "Sports", "Books", "Toys", "Beauty"]
        brands = ["Acme", "ZenTek", "Nova", "Quanta", "Polar", "Nimbus"]
        regions = ["North", "South", "East", "West"]
        channels = ["web", "store"]
        subcats = {
            "Electronics": ["Phone", "Laptop", "Audio"],
            "Home": ["Kitchen", "Decor", "Furniture"],
            "Sports": ["Outdoor", "Fitness", "Team"],
            "Books": ["Fiction", "Nonfiction", "Comics"],
            "Toys": ["STEM", "Board", "Plush"],
            "Beauty": ["Skincare", "Makeup", "Hair"]
        }

        products = []
        for i in range(1, num_products + 1):
            cat = rnd.choice(categories)
            products.append({
                "product_id": f"P{i:03d}",
                "category": cat,
                "subcategory": rnd.choice(subcats[cat]),
                "brand": rnd.choice(brands),
            })

        customers = []
        domains = ["gmail.com", "yahoo.com", "proton.me", "outlook.com"]
        area_codes = {"North": "212", "South": "404", "East": "617", "West": "415"}
        for i in range(1, num_customers + 1):
            region = rnd.choice(regions)
            email = f"user{i}@{rnd.choice(domains)}"
            phone = f"({area_codes[region]})-{rnd.randint(100,999)}-{rnd.randint(1000,9999)}"
            customers.append({
                "customer_id": f"C{i:03d}",
                "name": f"Name{i}",
                "email": email,
                "age": rnd.randint(18, 75),
                "signup_date": f"20{rnd.randint(20,24)}-{rnd.randint(1,12):02d}-{rnd.randint(1,28):02d}",
                "phone": phone,
                "region": region,
            })

        sales = []
        for i in range(1, num_sales + 1):
            prod = rnd.choice(products)
            cust = rnd.choice(customers)
            qty = rnd.randint(1, 5)
            price = rnd.choice([9.99, 14.99, 24.99, 49.99, 99.99, 199.99])
            date = f"202{rnd.randint(1,4)}-{rnd.randint(1,12):02d}-{rnd.randint(1,28):02d}"
            sales.append({
                "order_id": f"S{i:05d}",
                "customer_id": cust["customer_id"],
                "product_id": prod["product_id"],
                "date": date,
                "quantity": qty,
                "price": float(price),
                "region": cust["region"],
                "channel": rnd.choice(channels),
                "returned": rnd.random() < 0.08,
                "promo_code": rnd.choice(["", "SAVE", "WELCOME", "VIP", ""])  # sparse
            })

        regions_map = [{"region": r, "area_code": area_codes[r]} for r in regions]

        self.sources = {
            "sales.csv": sales,
            "products.csv": products,
            "customers.json": customers,
            "regions.csv": regions_map,
        }

        self.tables: Dict[str, List[Dict[str, Any]]] = {}
        self.active_table: Optional[str] = None
        self.last_scalar: Optional[Union[int, float, str]] = None
        self.turn_count: int = 0
        self.steps_taken: int = 0
        self.required_steps: int = 0
        self.task: Dict[str, Any] = {}
        self.goal_value: Any = None

    def _get_instructions(self) -> str:
        lines = []
        lines.append("You are in a data workbench. Use tools to transform data and compute the requested scalar, then submit it.")
        lines.append("Call tools with the exact format: \\boxed{tool_name(param1=value1, param2=\"text\", ...)}")
        lines.append("Finish by calling: \\boxed{submit_answer(value=...)}")
        lines.append("Available tools:")
        for name, meta in self.tools.items():
            lines.append(f"- {name}: {meta['description']}")
        lines.append("Notes:")
        lines.append("- Columns and table names are case-sensitive strings.")
        lines.append("- filter condition supports simple AND with operators: ==, !=, >, >=, <, <=, contains, startswith, endswith, in [a|b], notin [a|b].")
        lines.append("- aggregate without 'by' yields a scalar. With 'by', it produces a table with columns [by, value].")
        lines.append("Return strictly one boxed call per turn.")
        return "\n".join(lines)

    def get_task_suffix(self) -> str:
        available = ", ".join(sorted(self.tables.keys())) if self.tables else "None"
        active = self.active_table if self.active_table else "None"
        return (
            f"Task: {self.task.get('description','')}\n"
            f"Turns: {self.turn_count}/{self.max_turns} | Steps taken: {self.steps_taken} | Suggested steps: {self.required_steps}\n"
            f"Active table: {active}\n"
            f"Available tables: {available}\n"
            f"Respond with a single tool call formatted as \\boxed{{tool_name(param=value, ...)}}"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self.tables = {}
        self.active_table = None
        self.last_scalar = None
        self.turn_count = 0
        self.steps_taken = 0
        self.required_steps = random.randint(self.min_required_steps, self.max_required_steps)
        self.task = self._generate_task_requiring_n_steps(self.required_steps)
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_task_requiring_n_steps(self, required_steps: int) -> Dict[str, Any]:
        # Choose a template based on required_steps
        # Tiers: simple (2-4), medium (4-7), complex (7-12+)
        def compute_revenue(row):
            return float(row.get("quantity", 0)) * float(row.get("price", 0.0))

        # Helper to compute ground truth safely
        def eval_template(template: str) -> Optional[Dict[str, Any]]:
            rnd = random
            if template == "avg_order_value_by_region_month":
                region = rnd.choice(["North", "South", "East", "West"])
                month = rnd.randint(1, 12)
                # Compute from self.sources["sales.csv"]
                rows = [r for r in self.sources["sales.csv"] if r["region"] == region and int(r["date"][5:7]) == month and not r["returned"]]
                if not rows:
                    return None
                values = [compute_revenue(r) for r in rows]
                ans = sum(values) / len(values)
                desc = f"Compute the average order revenue (quantity*price) for region '{region}' in month {month} (1-12), excluding returned orders. Submit the numeric value."
                return {"type": "scalar_float", "answer": round(ans, 4), "description": desc, "min_steps": 4}
            if template == "total_revenue_category_channel":
                cat = random.choice(list({p["category"] for p in self.sources["products.csv"]}))
                channel = random.choice(["web", "store"])
                # join sales with products
                prod_by_id = {p["product_id"]: p for p in self.sources["products.csv"]}
                rows = [r for r in self.sources["sales.csv"] if not r["returned"] and r["channel"] == channel and prod_by_id.get(r["product_id"], {}).get("category") == cat]
                if not rows:
                    return None
                ans = sum(compute_revenue(r) for r in rows)
                desc = f"Compute total revenue (sum of quantity*price) for category '{cat}' and channel '{channel}', excluding returned orders."
                return {"type": "scalar_float", "answer": round(ans, 4), "description": desc, "min_steps": 5}
            if template == "top_category_by_revenue":
                prod_by_id = {p["product_id"]: p for p in self.sources["products.csv"]}
                totals: Dict[str, float] = {}
                for r in self.sources["sales.csv"]:
                    if r["returned"]:
                        continue
                    cat = prod_by_id.get(r["product_id"], {}).get("category")
                    if not cat:
                        continue
                    totals[cat] = totals.get(cat, 0.0) + compute_revenue(r)
                if not totals:
                    return None
                top_cat = max(totals.items(), key=lambda x: x[1])[0]
                desc = "Find the single category with the highest total revenue (quantity*price), excluding returned orders. Submit the category name."
                return {"type": "scalar_str", "answer": top_cat, "description": desc, "min_steps": 6}
            if template == "unique_customers_brand_min_age":
                brand = random.choice(list({p["brand"] for p in self.sources["products.csv"]}))
                min_age = random.randint(20, 50)
                prod_by_id = {p["product_id"]: p for p in self.sources["products.csv"]}
                cust_by_id = {c["customer_id"]: c for c in self.sources["customers.json"]}
                custs = set()
                for r in self.sources["sales.csv"]:
                    if prod_by_id.get(r["product_id"], {}).get("brand") == brand:
                        if cust_by_id.get(r["customer_id"], {}).get("age", 0) >= min_age:
                            custs.add(r["customer_id"])
                if not custs:
                    return None
                desc = f"Count unique customers who bought brand '{brand}' and are age >= {min_age}. Submit the integer count."
                return {"type": "scalar_int", "answer": int(len(custs)), "description": desc, "min_steps": 7}
            if template == "gmail_customers_min_orders":
                k = random.randint(2, 4)
                # orders per customer
                counts: Dict[str, int] = {}
                for r in self.sources["sales.csv"]:
                    counts[r["customer_id"]] = counts.get(r["customer_id"], 0) + 1
                cust_by_id = {c["customer_id"]: c for c in self.sources["customers.json"]}
                cands = [cid for cid, cnt in counts.items() if cnt >= k and cust_by_id.get(cid, {}).get("email", "").endswith("@gmail.com")]
                if not cands:
                    return None
                desc = f"Count customers with email domain 'gmail.com' who have at least {k} orders. Submit the integer count."
                return {"type": "scalar_int", "answer": int(len(cands)), "description": desc, "min_steps": 8}
            return None

        # Pick template fitting required_steps
        templates_ordered = [
            ("avg_order_value_by_region_month", (2, 4)),
            ("total_revenue_category_channel", (4, 7)),
            ("top_category_by_revenue", (5, 8)),
            ("unique_customers_brand_min_age", (6, 10)),
            ("gmail_customers_min_orders", (7, 12)),
        ]
        # Filter by bracket preference but keep fallback variety
        candidates = [t for t, (lo, hi) in templates_ordered if lo <= required_steps <= hi]
        if not candidates:
            candidates = [t for t, _ in templates_ordered]
        # Try compute until feasible
        for _ in range(50):
            template = random.choice(candidates)
            out = eval_template(template)
            if out is not None:
                self.goal_value = out["answer"]
                out["suggested_steps"] = max(out["min_steps"], required_steps)
                return out
        # Fallback guaranteed simple
        out = eval_template("avg_order_value_by_region_month")
        if out is None:
            # degenerate fallback to zero target to avoid impossibility
            out = {"type": "scalar_int", "answer": 0, "description": "Submit 0.", "min_steps": 1, "suggested_steps": 1}
            self.goal_value = 0
        else:
            self.goal_value = out["answer"]
        return out

    def _parse_action(self, action: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        if not isinstance(action, str):
            return None
        m = re.search(r"\\boxed\{(.*)\}\s*$", action.strip())
        if not m:
            return None
        inner = m.group(1).strip()
        fm = re.match(r"^([A-Za-z_]\w*)\s*\(\s*(.*?)\s*\)\s*$", inner)
        if not fm:
            return None
        tool = fm.group(1)
        args_str = fm.group(2).strip()
        args: Dict[str, Any] = {}
        if args_str == "":
            return tool, args
        # split by commas not inside quotes or brackets
        parts = []
        buf = ""
        depth = 0
        in_s = False
        in_d = False
        for ch in args_str:
            if ch == "'" and not in_d:
                in_s = not in_s
            elif ch == '"' and not in_s:
                in_d = not in_d
            elif ch in "([{" and not in_s and not in_d:
                depth += 1
            elif ch in ")]}" and not in_s and not in_d:
                depth -= 1
            if ch == "," and depth == 0 and not in_s and not in_d:
                parts.append(buf.strip())
                buf = ""
            else:
                buf += ch
        if buf.strip():
            parts.append(buf.strip())
        for p in parts:
            if "=" not in p:
                return None
            k, v = p.split("=", 1)
            k = k.strip()
            v = v.strip()
            args[k] = self._parse_value(v)
        return tool, args

    def _parse_value(self, v: str) -> Any:
        if len(v) >= 2 and ((v[0] == v[-1] == "'") or (v[0] == v[-1] == '"')):
            return v[1:-1]
        if v.lower() in ["true", "false"]:
            return v.lower() == "true"
        # list format e.g., [a|b|c] or ["a","b"]
        if v.startswith("[") and v.endswith("]"):
            body = v[1:-1].strip()
            if "|" in body and "," not in body:
                items = [self._parse_value(x.strip()) for x in body.split("|")]
            else:
                # split by comma
                items = [self._parse_value(x.strip()) for x in body.split(",") if x.strip()]
            return items
        # number
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
        return v

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)
        if not parsed:
            obs = "Invalid action format. Use \\boxed{tool_name(param=value,...)}"
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
        tool, args = parsed
        if tool not in self.tools:
            obs = f"Unsupported tool '{tool}'."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        try:
            result_text, scalar = self._execute_tool(tool, args)
            self.steps_taken += 1
            self.last_scalar = scalar if scalar is not None else self.last_scalar
            terminated = False
            truncated = False
            reward = 0.0
            if tool == "submit_answer":
                terminated = True
                # reward handled in _execute_tool by including correctness in result_text
                if "SUCCESS" in result_text:
                    reward = 1.0
                else:
                    reward = 0.0
            if self.turn_count >= self.max_turns and not terminated:
                terminated = True
                truncated = True
                result_text += "\nTimeout reached."
                reward = 0.0
            return result_text, reward, terminated, truncated, {"suffix": self.get_task_suffix()}
        except ValueError as e:
            obs = f"Protocol error: {str(e)}"
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
        except Exception as e:
            obs = f"Execution error: {str(e)}"
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

    def _ensure_active(self):
        if not self.active_table or self.active_table not in self.tables:
            raise ValueError("No active table. Load or switch to a table first.")

    def _copy_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [dict(r) for r in rows]

    def _execute_tool(self, tool: str, args: Dict[str, Any]) -> Tuple[str, Optional[Union[int, float, str]]]:
        if tool == "load":
            source = args.get("source")
            if source not in self.sources:
                raise ValueError(f"Source not found: {source}")
            name = args.get("name", source)
            self.tables[name] = self._copy_rows(self.sources[source])
            self.active_table = name
            cols = list(self.tables[name][0].keys()) if self.tables[name] else []
            return f"Loaded '{source}' as table '{name}' with {len(self.tables[name])} rows. Columns: {cols}", None

        if tool == "save_as":
            self._ensure_active()
            name = args.get("name")
            if not name or not isinstance(name, str):
                raise ValueError("Missing or invalid name.")
            self.tables[name] = self._copy_rows(self.tables[self.active_table])
            self.active_table = name
            return f"Saved current table as '{name}'.", None

        if tool == "switch_active":
            name = args.get("name")
            if name not in self.tables:
                raise ValueError(f"Table not found: {name}")
            self.active_table = name
            return f"Active table switched to '{name}'.", None

        if tool == "preview":
            self._ensure_active()
            n = int(args.get("n", 5))
            rows = self.tables[self.active_table][:max(0, n)]
            return f"Preview first {n} rows:\n{rows}", None

        if tool == "select":
            self._ensure_active()
            cols_arg = args.get("columns")
            if isinstance(cols_arg, str):
                cols = [c.strip() for c in cols_arg.split(",")]
            elif isinstance(cols_arg, list):
                cols = [str(c) for c in cols_arg]
            else:
                raise ValueError("columns must be list or comma-separated string.")
            new_rows = []
            for r in self.tables[self.active_table]:
                new_rows.append({c: r.get(c) for c in cols})
            self.tables[self.active_table] = new_rows
            return f"Selected columns: {cols}", None

        if tool == "filter":
            self._ensure_active()
            cond = args.get("condition")
            if not isinstance(cond, str) or not cond.strip():
                raise ValueError("Missing condition.")
            pred = self._compile_condition(cond)
            rows = [r for r in self.tables[self.active_table] if pred(r)]
            self.tables[self.active_table] = rows
            return f"Filtered rows; now {len(rows)} rows remain.", None

        if tool == "derive":
            self._ensure_active()
            new = args.get("new")
            expr = args.get("expr")
            if not new or not expr:
                raise ValueError("Missing new or expr.")
            for r in self.tables[self.active_table]:
                r[new] = self._eval_expr(expr, r)
            return f"Derived column '{new}' using expr '{expr}'.", None

        if tool == "aggregate":
            self._ensure_active()
            op = str(args.get("op"))
            column = args.get("column")
            by = args.get("by")
            rows = self.tables[self.active_table]
            if by:
                # group
                groups: Dict[Any, List[Dict[str, Any]]] = {}
                for r in rows:
                    k = r.get(by)
                    groups.setdefault(k, []).append(r)
                out = []
                for k, sub in groups.items():
                    val = self._agg_values(op, sub, column)
                    out.append({by: k, "value": val})
                self.tables[self.active_table] = out
                return f"Grouped by '{by}' with op '{op}' on column '{column}'. {len(out)} groups.", None
            else:
                val = self._agg_values(op, rows, column)
                return f"Aggregate result: {val}", val

        if tool == "groupby_aggregate":
            self._ensure_active()
            by = args.get("by")
            column = args.get("column")
            op = str(args.get("op"))
            rows = self.tables[self.active_table]
            groups: Dict[Any, List[Dict[str, Any]]] = {}
            for r in rows:
                k = r.get(by)
                groups.setdefault(k, []).append(r)
            out = []
            for k, sub in groups.items():
                val = self._agg_values(op, sub, column)
                out.append({by: k, "value": val})
            self.tables[self.active_table] = out
            return f"Groupby aggregate produced {len(out)} rows.", None

        if tool == "join":
            self._ensure_active()
            other = args.get("other")
            on = args.get("on")
            how = args.get("how", "inner")
            if other not in self.tables:
                raise ValueError(f"Table not found: {other}")
            if not on:
                raise ValueError("Missing 'on' key.")
            left = self.tables[self.active_table]
            right = self.tables[other]
            index: Dict[Any, List[Dict[str, Any]]] = {}
            for r in right:
                k = r.get(on)
                index.setdefault(k, []).append(r)
            joined = []
            for l in left:
                k = l.get(on)
                matches = index.get(k, [])
                if matches:
                    for rr in matches:
                        merged = dict(l)
                        for key, val in rr.items():
                            if key in merged:
                                merged[f"{other}.{key}"] = val
                            else:
                                merged[key] = val
                        joined.append(merged)
                elif how == "left":
                    joined.append(dict(l))
            self.tables[self.active_table] = joined
            return f"Joined with '{other}' on '{on}' (how={how}). Rows now {len(joined)}.", None

        if tool == "sort_by":
            self._ensure_active()
            col = args.get("column")
            desc = bool(args.get("descending", False))
            rows = self.tables[self.active_table]
            try:
                rows.sort(key=lambda r: (r.get(col) is None, r.get(col)), reverse=desc)
            except Exception:
                # Fallback to string sort
                rows.sort(key=lambda r: str(r.get(col)), reverse=desc)
            return f"Sorted by '{col}' descending={desc}.", None

        if tool == "slice_rows":
            self._ensure_active()
            n = int(args.get("n", 0))
            self.tables[self.active_table] = self.tables[self.active_table][:max(0, n)]
            return f"Sliced to top {n} rows.", None

        if tool == "count_rows":
            self._ensure_active()
            cnt = len(self.tables[self.active_table])
            return f"Row count: {cnt}", cnt

        if tool == "unique_values":
            self._ensure_active()
            col = args.get("column")
            uniq = len({r.get(col) for r in self.tables[self.active_table]})
            return f"Unique count for '{col}': {uniq}", uniq

        if tool == "regex_extract":
            self._ensure_active()
            col = args.get("column")
            pattern = args.get("pattern")
            into = args.get("into")
            try:
                rgx = re.compile(pattern)
            except re.error:
                raise ValueError("Invalid regex pattern.")
            for r in self.tables[self.active_table]:
                val = r.get(col, "")
                m = rgx.search(str(val))
                r[into] = m.group(1) if (m and m.groups()) else (m.group(0) if m else None)
            return f"Extracted regex '{pattern}' from '{col}' into '{into}'.", None

        if tool == "submit_answer":
            val = args.get("value")
            verdict, msg = self._evaluate_submission(val)
            return msg, None

        raise ValueError(f"Unhandled tool: {tool}")

    def _compile_condition(self, cond: str) -> Callable[[Dict[str, Any]], bool]:
        parts = [p.strip() for p in re.split(r"\band\b", cond)]
        testers: List[Callable[[Dict[str, Any]], bool]] = []
        for p in parts:
            testers.append(self._parse_simple_condition(p))
        def pred(row):
            return all(t(row) for t in testers)
        return pred

    def _parse_simple_condition(self, s: str) -> Callable[[Dict[str, Any]], bool]:
        s_strip = s.strip()
        # in/notin
        m = re.match(r"^([A-Za-z_]\w*)\s+(in|notin)\s+\[(.*)\]\s*$", s_strip)
        if m:
            col, op, body = m.group(1), m.group(2), m.group(3)
            values = [self._parse_value(x.strip()) for x in body.split("|")]
            if op == "in":
                return lambda r: r.get(col) in values
            else:
                return lambda r: r.get(col) not in values
        # contains/startswith/endswith
        m = re.match(r"^([A-Za-z_]\w*)\s+(contains|startswith|endswith)\s+(.+)$", s_strip)
        if m:
            col, op, val = m.group(1), m.group(2), m.group(3)
            val = self._parse_value(val)
            if op == "contains":
                return lambda r: val in str(r.get(col, ""))
            if op == "startswith":
                return lambda r: str(r.get(col, "")).startswith(str(val))
            if op == "endswith":
                return lambda r: str(r.get(col, "")).endswith(str(val))
        # comparisons
        m = re.match(r"^([A-Za-z_]\w*)\s*(==|!=|>=|<=|>|<)\s*(.+)$", s_strip)
        if m:
            col, op, val = m.group(1), m.group(2), self._parse_value(m.group(3))
            def cast(x):
                try:
                    if isinstance(val, (int, float)):
                        return float(x)
                    return x
                except Exception:
                    return x
            if op == "==":
                return lambda r: cast(r.get(col)) == val
            if op == "!=":
                return lambda r: cast(r.get(col)) != val
            if op == ">":
                return lambda r: cast(r.get(col, 0)) > val
            if op == "<":
                return lambda r: cast(r.get(col, 0)) < val
            if op == ">=":
                return lambda r: cast(r.get(col, 0)) >= val
            if op == "<=":
                return lambda r: cast(r.get(col, 0)) <= val
        # boolean shorthand like "returned == false" already handled, else always False
        return lambda r: False

    def _eval_expr(self, expr: str, row: Dict[str, Any]) -> Any:
        # support "col1 + 2", "col1 * col2", "concat(col1,'-',col2)"
        expr = expr.strip()
        m = re.match(r"^concat\((.*)\)$", expr)
        if m:
            inner = m.group(1)
            # split by commas not in quotes
            parts = []
            buf = ""
            in_s = False
            in_d = False
            for ch in inner:
                if ch == "'" and not in_d:
                    in_s = not in_s
                elif ch == '"' and not in_s:
                    in_d = not in_d
                if ch == "," and not in_s and not in_d:
                    parts.append(buf.strip())
                    buf = ""
                else:
                    buf += ch
            if buf.strip():
                parts.append(buf.strip())
            vals = []
            for p in parts:
                v = self._parse_value(p)
                if isinstance(v, str) and p.strip() not in ["'"+v+"'", '"'+v+'"']:
                    # treat as column if not quoted
                    v = row.get(v)
                elif not isinstance(v, str):
                    # if numeric or other, if corresponds to column name then fetch
                    if isinstance(v, (int, float)):
                        pass
                vals.append("" if v is None else str(v))
            return "".join(vals)

        m = re.match(r"^\s*([A-Za-z_]\w*)\s*([\+\-\*/])\s*([A-Za-z_]\w*|-?\d+(\.\d+)?)\s*$", expr)
        if m:
            left = m.group(1)
            op = m.group(2)
            right = m.group(3)
            lv = row.get(left, 0)
            rv = row.get(right, None) if re.match(r"^[A-Za-z_]\w*$", right) else float(right)
            try:
                lvf = float(lv)
                rvf = float(rv if rv is not None else 0)
                if op == "+":
                    return lvf + rvf
                if op == "-":
                    return lvf - rvf
                if op == "*":
                    return lvf * rvf
                if op == "/":
                    return lvf / rvf if rvf != 0 else 0.0
            except Exception:
                return None
        return None

    def _agg_values(self, op: str, rows: List[Dict[str, Any]], column: Optional[str]) -> Union[int, float]:
        op = op.lower()
        if op == "count":
            return len(rows)
        vals: List[float] = []
        if column is None:
            # For count only
            return len(rows)
        for r in rows:
            v = r.get(column)
            try:
                vals.append(float(v))
            except Exception:
                pass
        if not vals:
            return 0 if op in ["count"] else 0.0
        if op == "sum":
            return sum(vals)
        if op == "avg":
            return sum(vals) / len(vals)
        if op == "max":
            return max(vals)
        if op == "min":
            return min(vals)
        return 0.0

    def _evaluate_submission(self, submitted: Any) -> Tuple[bool, str]:
        exp = self.goal_value
        # Coerce types
        ok = False
        if isinstance(exp, (int, float)):
            try:
                if isinstance(submitted, str) and re.fullmatch(r"-?\d+(\.\d+)?", submitted.strip()):
                    val = float(submitted)
                elif isinstance(submitted, (int, float)):
                    val = float(submitted)
                else:
                    return False, f"Submission invalid type. Expected number close to {exp}."
                # tolerance
                ok = abs(val - float(exp)) <= max(1e-6, abs(float(exp)) * 1e-6)
            except Exception:
                ok = False
        elif isinstance(exp, str):
            if isinstance(submitted, str):
                ok = (submitted == exp)
            else:
                ok = False
        else:
            ok = (submitted == exp)
        if ok:
            return True, f"SUCCESS: Correct answer."
        else:
            return False, f"FINAL: Incorrect answer."

    def sample_random_action(self) -> str:
        choices = [
            r'\boxed{load(source="sales.csv")}',
            r'\boxed{preview(n=3)}',
            r'\boxed{filter(condition="returned == false")}',
            r'\boxed{derive(new="revenue", expr="quantity*price")}',
            r'\boxed{aggregate(op="sum", column="revenue")}',
            r'\boxed{submit_answer(value=0)}',
        ]
        return random.choice(choices)


class MacroCrafterEnvWithFeedback(MacroCrafterEnv):
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
            hint = "Wrap exactly one tool call like \\boxed{tool(arg=value)}."

        elif "unsupported tool" in text:
            error_type = "UnsupportedAction"
            m = re.search(r"unsupported tool '([^']+)'", obs, re.IGNORECASE)
            error_detail["tool"] = m.group(1) if m else None
            hint = "Use only listed tools. Call \\boxed{preview(n=3)} or \\boxed{load(source=\"...\")} to begin."

        elif "protocol error" in text:
            error_type = "ProtocolViolation"
            if "no active table" in text:
                error_detail["violation"] = "no_active_table"
                hint = "Load a source first with \\boxed{load(source=\"sales.csv\")} or switch_active to an existing table."
            elif "source not found" in text:
                error_detail["violation"] = "bad_source"
                hint = "Load one of: sales.csv, products.csv, customers.json, regions.csv."
            elif "table not found" in text:
                error_detail["violation"] = "unknown_table"
                hint = "Ensure you saved or loaded the table name before referencing it."
            else:
                error_detail["violation"] = "other"
                hint = "Check tool prerequisites and parameter names."

        elif "execution error" in text:
            error_type = "WrongDecision"
            error_detail["issue"] = "runtime_failure"
            hint = "Inspect columns via \\boxed{preview(n=5)} and adjust parameters (e.g., correct column names)."

        elif "final: incorrect answer" in text:
            error_type = "WrongDecision"
            error_detail["expected_type"] = type(self.goal_value).__name__
            error_detail["suggested_steps"] = self.required_steps
            # High-level hint based on task description
            desc = self.task.get("description", "").lower()
            if "revenue" in desc and "category" in desc:
                hint = "Join sales with products, exclude returned, derive revenue=quantity*price, then aggregate by the requested filter."
            elif "highest total revenue" in desc:
                hint = "Derive revenue, exclude returns, join with products, group by category, sum, sort desc, slice 1."
            elif "gmail.com" in desc:
                hint = "Aggregate order counts by customer, join customers, filter email domain via regex_extract, then count rows."
            else:
                hint = "Preview data, derive necessary columns, apply filters, then aggregate or count as required."

        elif "timeout reached" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Plan the pipeline first. Use preview to inspect, then apply minimal steps to reach the scalar."

        elif "success: correct answer" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "active_table": getattr(self, "active_table", None),
                "available_tables": list(getattr(self, "tables", {}).keys()),
                "steps_taken": getattr(self, "steps_taken", None),
                "suggested_steps": getattr(self, "required_steps", None),
            }
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint
        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        first_hint = "Start by loading a relevant source, e.g., \\boxed{load(source=\"sales.csv\")} then \\boxed{preview(n=5)}."
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": first_hint if self.feedback_level >= 2 else None,
            "turn": 0,
            "state": {
                "active_table": self.active_table,
                "available_tables": list(self.tables.keys()),
                "steps_taken": self.steps_taken,
                "suggested_steps": self.required_steps,
            } if self.feedback_level >= 1 else None
        }
        return obs, info