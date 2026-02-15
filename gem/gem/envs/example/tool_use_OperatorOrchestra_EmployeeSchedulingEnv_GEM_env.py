from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List, Union


class OperatorOrchestraEnv(Env):
    def __init__(self, complexity: int = 1, max_turns: Optional[int] = 100, **_):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.max_turns = max_turns if max_turns is not None else 100

        self.min_required_steps = self.complexity
        self.max_required_steps = self.complexity * 2

        self.tools = {}
        self.tables = {}
        self.current_table: Optional[List[Dict[str, Any]]] = None
        self.current_table_name: Optional[str] = None
        self.execution_state: Dict[str, Any] = {}
        self.task: Dict[str, Any] = {}
        self.turn_count = 0
        self.steps_taken = 0

        self._init_database()
        self.reset()

    def _init_database(self):
        random.seed()
        num_products = 12 + 4 * self.complexity
        num_sales = 60 + 25 * self.complexity
        num_users = 30 + 20 * self.complexity
        num_events = 120 + 40 * self.complexity

        categories = ["Gadgets", "Home", "Sports", "Books", "Garden", "Toys"][: min(3 + self.complexity // 3, 6)]
        suppliers = ["Acme", "Globex", "Innotech", "Umbrella", "Stark", "Wayne"][: min(3 + self.complexity // 2, 6)]
        countries = ["US", "CA", "UK", "DE", "FR", "IN", "JP"][: min(3 + self.complexity // 2, 7)]
        actions = ["view", "cart", "purchase", "return", "wishlist"][: min(3 + (self.complexity + 1) // 3, 5)]
        months = [f"2024-{m:02d}" for m in range(1, 1 + min(9, 2 + self.complexity))]

        products = []
        for i in range(num_products):
            pid = f"P{i+1:03d}"
            cat = random.choice(categories)
            sup = random.choice(suppliers)
            unit_price = round(random.uniform(5, 250), 2)
            products.append(
                {
                    "product_id": pid,
                    "category": cat,
                    "supplier": sup,
                    "unit_price": unit_price,
                    "name": f"Prod-{cat[:2]}-{i+1}",
                }
            )

        sales = []
        for i in range(num_sales):
            pr = random.choice(products)
            q = random.randint(1, 8)
            dtm = random.choice(months)
            dtd = random.randint(1, 28)
            date = f"{dtm}-{dtd:02d}"
            # unit_price stored per sale as snapshot of product price
            sales.append(
                {
                    "sale_id": f"S{i+1:05d}",
                    "product_id": pr["product_id"],
                    "quantity": q,
                    "unit_price": pr["unit_price"],
                    "date": date,
                }
            )

        inventory = []
        for pr in products:
            stock = random.randint(0, 300)
            inventory.append(
                {"product_id": pr["product_id"], "stock": stock}
            )

        users = []
        for i in range(num_users):
            uid = f"U{i+1:04d}"
            users.append({"user_id": uid, "country": random.choice(countries)})

        events = []
        for i in range(num_events):
            u = random.choice(users)
            pr = random.choice(products)
            act = random.choice(actions)
            dtm = random.choice(months)
            dtd = random.randint(1, 28)
            ts = f"{dtm}-{dtd:02d}"
            events.append(
                {
                    "event_id": f"E{i+1:05d}",
                    "user_id": u["user_id"],
                    "action": act,
                    "product_id": pr["product_id"] if random.random() < 0.8 else None,
                    "ts": ts,
                }
            )

        self.tables = {
            "products": products,
            "sales": sales,
            "inventory": inventory,
            "users": users,
            "events": events,
        }

        self.tools = {
            "load_table": {
                "description": "Load a named table into the working context",
                "parameters": {"name": "string"},
                "returns": "Sets the current working table",
                "example": r"\boxed{load_table name=sales}",
            },
            "select": {
                "description": "Keep only specific columns (comma-separated)",
                "parameters": {"columns": "string list (comma-separated)"},
                "returns": "Reduces columns of current table",
                "example": r"\boxed{select columns=product_id,quantity,unit_price}",
            },
            "filter": {
                "description": "Filter rows by a simple condition like col==val, col>num, col~substr",
                "parameters": {"where": "simple predicate"},
                "returns": "Filters current table",
                "example": r"\boxed{filter where=date~2024-05}",
            },
            "add_column": {
                "description": "Add new numeric column from expression like col1*col2, col1*number, col1+col2, col1-number",
                "parameters": {"name": "string", "expr": "simple arithmetic expression"},
                "returns": "Adds a numeric column",
                "example": r"\boxed{add_column name=revenue expr=quantity*unit_price}",
            },
            "group_agg": {
                "description": "Aggregate by a key: by=col; agg=func:target. funcs=sum,avg,max,min,count,nunique",
                "parameters": {"by": "string", "agg": "func:target"},
                "returns": "Aggregated table",
                "example": r"\boxed{group_agg by=category; agg=sum:revenue}",
            },
            "join": {
                "description": "Join current table (left) to a right table on equality. inner join only.",
                "parameters": {"right": "string", "on": "leftcol=rightcol"},
                "returns": "Joined table",
                "example": r"\boxed{join right=products; on=product_id=product_id}",
            },
            "sort": {
                "description": "Sort rows by a column",
                "parameters": {"by": "string", "order": "asc|desc"},
                "returns": "Reorders current table",
                "example": r"\boxed{sort by=sum_revenue; order=desc}",
            },
            "distinct": {
                "description": "Drop duplicate rows by a column (keep first occurrence)",
                "parameters": {"col": "string"},
                "returns": "Deduplicated table",
                "example": r"\boxed{distinct col=user_id}",
            },
            "compute_stat": {
                "description": "Compute scalar from table: count_rows or sum:col/avg:col/max:col/min:col",
                "parameters": {"stat": "string"},
                "returns": "Scalar result saved as last_stat_result",
                "example": r"\boxed{compute_stat stat=count_rows}",
            },
            "head": {
                "description": "Preview first n rows (non-destructive)",
                "parameters": {"n": "int"},
                "returns": "Prints preview; table unchanged",
                "example": r"\boxed{head n=3}",
            },
            "help": {
                "description": "Show available tools",
                "parameters": {},
                "returns": "Tool list",
                "example": r"\boxed{help}",
            },
            "SUBMIT": {
                "description": "Submit final answer: answer=value",
                "parameters": {"answer": "string or number"},
                "returns": "Terminates episode with success/failure",
                "example": r"\boxed{SUBMIT answer=Gadgets}",
            },
        }

    def _generate_task_requiring_n_steps(self, required_steps: int) -> Dict[str, Any]:
        # Choose task type based on required steps
        kinds = []
        if required_steps <= 4:
            kinds.append("avg_stock_supplier")
        if 3 <= required_steps <= 8:
            kinds.append("unique_users_action_country")
        kinds.append("top_category_revenue_month")

        kind = random.choice(kinds)

        if kind == "avg_stock_supplier":
            # pick supplier that exists
            suppliers = list({p["supplier"] for p in self.tables["products"]})
            supplier = random.choice(suppliers)
            # compute true answer
            prod_ids = [p["product_id"] for p in self.tables["products"] if p["supplier"] == supplier]
            stocks = [r["stock"] for r in self.tables["inventory"] if r["product_id"] in prod_ids]
            answer = round(sum(stocks) / len(stocks), 4) if stocks else 0.0
            expected_steps = 4  # load inventory -> join products -> filter supplier -> compute_stat avg:stock
            return {
                "kind": kind,
                "params": {"supplier": supplier},
                "answer": answer,
                "answer_type": "number",
                "expected_steps": expected_steps,
                "prompt": f"Compute the average inventory stock for products supplied by '{supplier}'. Submit the numeric average.",
            }

        if kind == "unique_users_action_country":
            # choose action and country that yield non-zero events
            # build counts
            # join events with users by user_id, then count unique users per (action, country)
            user_country = {u["user_id"]: u["country"] for u in self.tables["users"]}
            counts: Dict[Tuple[str, str], set] = {}
            for e in self.tables["events"]:
                c = user_country.get(e["user_id"])
                if c is None:
                    continue
                key = (e["action"], c)
                counts.setdefault(key, set()).add(e["user_id"])
            nonzero = [(k, len(v)) for k, v in counts.items() if len(v) > 0]
            if not nonzero:
                # fallback to all users
                action = random.choice(list({e["action"] for e in self.tables["events"]}))
                country = random.choice(list({u["country"] for u in self.tables["users"]}))
                answer = 0
            else:
                k, val = random.choice(nonzero)
                action, country = k
                answer = val
            expected_steps = 6  # load events -> join users -> filter action -> filter country -> distinct user_id -> compute_stat count_rows
            return {
                "kind": kind,
                "params": {"action": action, "country": country},
                "answer": int(answer),
                "answer_type": "number",
                "expected_steps": expected_steps,
                "prompt": f"Count unique users with action '{action}' in country '{country}'. Submit the integer count.",
            }

        # default: top_category_revenue_month
        # choose month with max sales
        months = list({s["date"][:7] for s in self.tables["sales"]})
        month = random.choice(months) if months else "2024-01"
        # compute revenue per category in that month
        price = {p["product_id"]: p["unit_price"] for p in self.tables["products"]}
        category_of = {p["product_id"]: p["category"] for p in self.tables["products"]}
        cat_rev: Dict[str, float] = {}
        for s in self.tables["sales"]:
            if s["date"].startswith(month):
                pid = s["product_id"]
                qty = s["quantity"]
                u = s.get("unit_price", price.get(pid, 0.0))
                cat = category_of.get(pid, "Unknown")
                cat_rev[cat] = cat_rev.get(cat, 0.0) + (qty * u)
        if not cat_rev:
            # fallback to any category
            for s in self.tables["sales"]:
                pid = s["product_id"]
                qty = s["quantity"]
                u = s.get("unit_price", price.get(pid, 0.0))
                cat = category_of.get(pid, "Unknown")
                cat_rev[cat] = cat_rev.get(cat, 0.0) + (qty * u)
        answer_cat = max(cat_rev.items(), key=lambda x: x[1])[0] if cat_rev else "Unknown"
        expected_steps = 6  # load sales -> join products -> add revenue -> filter month -> group_agg by category sum:revenue -> sort desc
        return {
            "kind": "top_category_revenue_month",
            "params": {"month": month},
            "answer": str(answer_cat),
            "answer_type": "string",
            "expected_steps": expected_steps,
            "prompt": f"Find the category with the highest total revenue in {month}. Submit the category name.",
        }

    def _get_instructions(self) -> str:
        tool_list = "\n".join(
            [
                f"- {name}: {spec['description']}"
                for name, spec in self.tools.items()
                if name != "SUBMIT"
            ]
        )
        usage_hint = "Actions must be enclosed exactly as: \\boxed{tool arg=value; arg2=value}. Values may be unquoted (no spaces) or quoted with single quotes."
        examples = [
            r"\boxed{load_table name=sales}",
            r"\boxed{join right=products; on=product_id=product_id}",
            r"\boxed{add_column name=revenue expr=quantity*unit_price}",
            r"\boxed{filter where=date~2024-05}",
            r"\boxed{group_agg by=category; agg=sum:revenue}",
            r"\boxed{sort by=sum_revenue; order=desc}",
            r"\boxed{head n=3}",
            r"\boxed{compute_stat stat=count_rows}",
            r"\boxed{SUBMIT answer=Gadgets}",
        ]
        return (
            "You are orchestrating data tools to solve an analytics task.\n"
            "Goal: produce a single correct final answer via SUBMIT.\n"
            "Available tools:\n"
            f"{tool_list}\n"
            "Terminal action:\n- SUBMIT: finalize with your answer.\n"
            f"{usage_hint}\n"
            "Examples:\n"
            + "\n".join(examples)
        )

    def get_task_suffix(self) -> str:
        current = "None loaded"
        if self.current_table is not None and self.current_table_name is not None:
            cols = list(self.current_table[0].keys()) if self.current_table else []
            current = f"{self.current_table_name} rows={len(self.current_table)} cols={cols}"
        tools_brief = ", ".join(sorted([k for k in self.tools.keys() if k != "SUBMIT"]))
        fmt = "Format your action as a single \\boxed{...} with tool and args; one action per turn."
        progress = f"Turn {self.turn_count}/{self.max_turns} | Steps used: {self.steps_taken}"
        task_text = self.task.get("prompt", "")
        expected = self.task.get("expected_steps", "?")
        hint = f"Typical steps needed: {expected} (may vary)."
        return (
            f"Task: {task_text}\n"
            f"Current: {current}\n"
            f"Tools: {tools_brief}\n"
            f"{progress}\n"
            f"{fmt}\n"
            f"{hint}"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self.current_table = None
        self.current_table_name = None
        self.execution_state = {"last_stat_result": None}
        self.turn_count = 0
        self.steps_taken = 0

        required_steps = random.randint(self.min_required_steps, self.max_required_steps)
        self.task = self._generate_task_requiring_n_steps(required_steps)

        obs = self._get_instructions()
        info = {"suffix": self.get_task_suffix()}
        return obs, info

    def _parse_action(self, action: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        if not isinstance(action, str):
            return None
        m = re.search(r"\\boxed\{(.*)\}\s*$", action.strip(), flags=re.DOTALL)
        if not m:
            return None
        inside = m.group(1).strip()
        if not inside:
            return None
        # Split tool and args
        parts = inside.split(None, 1)
        tool = parts[0].strip()
        arg_str = parts[1].strip() if len(parts) > 1 else ""
        # Parse key=value pairs separated by ; or , (ignore extra spaces)
        args: Dict[str, Any] = {}
        if arg_str:
            # Normalize separators to ;
            tmp = re.sub(r"\s*,\s*", ";", arg_str)
            # Now split by ;
            chunks = [c.strip() for c in tmp.split(";") if c.strip()]
            for ch in chunks:
                if "=" not in ch:
                    # tolerate single flag like help
                    continue
                k, v = ch.split("=", 1)
                key = k.strip()
                val = v.strip()
                # strip optional quotes
                if (val.startswith("'") and val.endswith("'")) or (val.startswith('"') and val.endswith('"')):
                    val = val[1:-1]
                # Try list via comma inside value (only for columns list)
                if key in ("columns",):
                    partsv = [p.strip() for p in val.split(",") if p.strip()]
                    args[key] = partsv
                    continue
                # Try int
                if re.fullmatch(r"-?\d+", val):
                    args[key] = int(val)
                    continue
                # Try float
                if re.fullmatch(r"-?\d+\.\d+", val):
                    try:
                        args[key] = float(val)
                        continue
                    except:
                        pass
                args[key] = val
        return tool, args

    def _ensure_loaded(self) -> Optional[str]:
        if self.current_table is None:
            return "ERROR: Protocol violation - no table loaded. Use load_table first."
        return None

    def _exec_load_table(self, args: Dict[str, Any]) -> str:
        name = args.get("name")
        if name not in self.tables:
            return f"ERROR: Unknown table '{name}'. Available: {list(self.tables.keys())}"
        # deep-ish copy (shallow per-row ok)
        self.current_table = [dict(row) for row in self.tables[name]]
        self.current_table_name = name
        return f"Loaded table '{name}' with {len(self.current_table)} rows."

    def _exec_select(self, args: Dict[str, Any]) -> str:
        err = self._ensure_loaded()
        if err:
            return err
        cols = args.get("columns")
        if not cols or not isinstance(cols, list):
            return "ERROR: Protocol violation - select requires columns list."
        new = []
        for row in self.current_table:
            new.append({c: row.get(c, None) for c in cols})
        self.current_table = new
        return f"Selected columns {cols}. Now cols={cols}."

    def _parse_condition(self, cond: str):
        # supports: col==val, col!=val, col>num, col>=num, col<num, col<=num, col~substr
        cond = cond.strip()
        m = re.match(r"^(\w+)\s*(==|!=|>=|<=|>|<|~)\s*(.+)$", cond)
        if not m:
            return None, None, None
        col, op, val_raw = m.group(1), m.group(2), m.group(3).strip()
        if (val_raw.startswith("'") and val_raw.endswith("'")) or (val_raw.startswith('"') and val_raw.endswith('"')):
            val = val_raw[1:-1]
        else:
            if re.fullmatch(r"-?\d+", val_raw):
                val = int(val_raw)
            elif re.fullmatch(r"-?\d+\.\d+", val_raw):
                try:
                    val = float(val_raw)
                except:
                    val = val_raw
            else:
                val = val_raw
        return col, op, val

    def _exec_filter(self, args: Dict[str, Any]) -> str:
        err = self._ensure_loaded()
        if err:
            return err
        cond = args.get("where")
        if not cond or not isinstance(cond, str):
            return "ERROR: Protocol violation - filter requires where=condition."
        col, op, val = self._parse_condition(cond)
        if col is None:
            return "ERROR: Unsupported condition syntax."
        out = []
        for row in self.current_table:
            rv = row.get(col)
            keep = False
            try:
                if op == "==":
                    keep = rv == val
                elif op == "!=":
                    keep = rv != val
                elif op == ">":
                    keep = (rv is not None) and (float(rv) > float(val))
                elif op == "<":
                    keep = (rv is not None) and (float(rv) < float(val))
                elif op == ">=":
                    keep = (rv is not None) and (float(rv) >= float(val))
                elif op == "<=":
                    keep = (rv is not None) and (float(rv) <= float(val))
                elif op == "~":
                    keep = (rv is not None) and (str(val) in str(rv))
            except:
                keep = False
            if keep:
                out.append(row)
        self.current_table = out
        return f"Filtered with '{cond}'. Rows now: {len(self.current_table)}."

    def _exec_add_column(self, args: Dict[str, Any]) -> str:
        err = self._ensure_loaded()
        if err:
            return err
        name = args.get("name")
        expr = args.get("expr")
        if not name or not expr:
            return "ERROR: Protocol violation - add_column requires name and expr."
        # parse simple patterns: a*b, a*number, a+b, a-number
        m = re.match(r"^\s*(\w+)\s*([\*\+\-])\s*([\w\.\-]+)\s*$", str(expr))
        if not m:
            return "ERROR: Unsupported expression syntax."
        left, op, right = m.group(1), m.group(2), m.group(3)
        right_is_num = re.fullmatch(r"-?\d+(\.\d+)?", right) is not None
        right_num = float(right) if right_is_num else None
        cnt = 0
        for row in self.current_table:
            lv = row.get(left, 0)
            rv = right_num if right_is_num else row.get(right, 0)
            try:
                lf = float(lv)
                rf = float(rv)
                if op == "*":
                    row[name] = lf * rf
                elif op == "+":
                    row[name] = lf + rf
                elif op == "-":
                    row[name] = lf - rf
                cnt += 1
            except:
                row[name] = None
        return f"Added column '{name}' using expr '{expr}' to {cnt} rows."

    def _exec_group_agg(self, args: Dict[str, Any]) -> str:
        err = self._ensure_loaded()
        if err:
            return err
        by = args.get("by")
        agg = args.get("agg")
        if not by or not agg or ":" not in str(agg):
            return "ERROR: Protocol violation - group_agg requires by and agg=func:target."
        func, target = agg.split(":", 1)
        func = func.strip()
        target = target.strip()
        groups: Dict[Any, List[Dict[str, Any]]] = {}
        for row in self.current_table:
            key = row.get(by)
            groups.setdefault(key, []).append(row)
        result = []
        for key, rows in groups.items():
            vals = [r.get(target) for r in rows if r.get(target) is not None]
            numeric_vals = []
            for v in vals:
                try:
                    numeric_vals.append(float(v))
                except:
                    pass
            out_row = {by: key}
            if func == "sum":
                out_row[f"sum_{target}"] = sum(numeric_vals)
            elif func == "avg":
                out_row[f"avg_{target}"] = (sum(numeric_vals) / len(numeric_vals)) if numeric_vals else 0.0
            elif func == "max":
                out_row[f"max_{target}"] = max(numeric_vals) if numeric_vals else None
            elif func == "min":
                out_row[f"min_{target}"] = min(numeric_vals) if numeric_vals else None
            elif func == "count":
                out_row[f"count_{target}"] = len(rows)
            elif func == "nunique":
                out_row[f"nunique_{target}"] = len(set(vals))
            else:
                return f"ERROR: Unsupported aggregate '{func}'."
            result.append(out_row)
        self.current_table = result
        return f"Aggregated by '{by}' using {func}:{target}. Rows now: {len(self.current_table)}."

    def _exec_join(self, args: Dict[str, Any]) -> str:
        err = self._ensure_loaded()
        if err:
            return err
        right = args.get("right")
        on = args.get("on")
        if not right or right not in self.tables:
            return "ERROR: Protocol violation - join requires valid right table."
        if not on or "=" not in str(on):
            return "ERROR: Protocol violation - join requires on=leftcol=rightcol."
        leftcol, rightcol = on.split("=", 1)
        leftcol = leftcol.strip()
        rightcol = rightcol.strip()
        right_rows = self.tables[right]
        index: Dict[Any, List[Dict[str, Any]]] = {}
        for rr in right_rows:
            key = rr.get(rightcol)
            index.setdefault(key, []).append(rr)
        joined = []
        for l in self.current_table:
            key = l.get(leftcol)
            matches = index.get(key, [])
            for r in matches:
                combined = dict(l)
                for rk, rv in r.items():
                    if rk == rightcol and leftcol == rightcol:
                        continue
                    if rk in combined:
                        combined[f"r_{rk}"] = rv
                    else:
                        combined[rk] = rv
                joined.append(combined)
        self.current_table = joined
        return f"Inner joined with '{right}' on {leftcol}={rightcol}. Rows now: {len(self.current_table)}."

    def _exec_sort(self, args: Dict[str, Any]) -> str:
        err = self._ensure_loaded()
        if err:
            return err
        by = args.get("by")
        order = str(args.get("order", "asc")).lower()
        if not by:
            return "ERROR: Protocol violation - sort requires by=column."
        reverse = order == "desc"
        try:
            self.current_table.sort(key=lambda r: (r.get(by) is None, r.get(by)), reverse=reverse)
        except TypeError:
            # try to coerce to float
            def kf(r):
                v = r.get(by)
                try:
                    return float(v)
                except:
                    return float("inf")
            self.current_table.sort(key=kf, reverse=reverse)
        return f"Sorted by {by} order={order}."

    def _exec_distinct(self, args: Dict[str, Any]) -> str:
        err = self._ensure_loaded()
        if err:
            return err
        col = args.get("col")
        if not col:
            return "ERROR: Protocol violation - distinct requires col."
        seen = set()
        out = []
        for row in self.current_table:
            key = row.get(col)
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
        self.current_table = out
        return f"Distinct by {col}. Rows now: {len(out)}."

    def _exec_compute_stat(self, args: Dict[str, Any]) -> str:
        err = self._ensure_loaded()
        if err:
            return err
        stat = args.get("stat")
        if not stat:
            return "ERROR: Protocol violation - compute_stat requires stat."
        if stat == "count_rows":
            val = len(self.current_table)
            self.execution_state["last_stat_result"] = val
            return f"Computed count_rows={val}."
        if ":" in str(stat):
            func, col = stat.split(":", 1)
            func = func.strip()
            col = col.strip()
            vals = []
            for r in self.current_table:
                v = r.get(col)
                try:
                    vals.append(float(v))
                except:
                    pass
            if func == "sum":
                val = sum(vals)
            elif func == "avg":
                val = sum(vals) / len(vals) if vals else 0.0
            elif func == "max":
                val = max(vals) if vals else None
            elif func == "min":
                val = min(vals) if vals else None
            else:
                return f"ERROR: Unsupported stat function '{func}'."
            self.execution_state["last_stat_result"] = val
            return f"Computed {func}:{col}={val}."
        return "ERROR: Unsupported stat syntax."

    def _exec_head(self, args: Dict[str, Any]) -> str:
        err = self._ensure_loaded()
        if err:
            return err
        n = int(args.get("n", 3)) if isinstance(args.get("n", None), int) else 3
        preview = self.current_table[: max(0, n)]
        return f"Preview first {n} rows: {preview}"

    def _exec_help(self) -> str:
        names = sorted(self.tools.keys())
        return f"Tools: {names}"

    def _execute_tool(self, tool: str, args: Dict[str, Any]) -> Tuple[str, bool, float, bool]:
        # returns (message, ok, penalty, protocol_violation_flag)
        if tool == "load_table":
            return self._exec_load_table(args), True, 0.0, False
        if tool == "select":
            msg = self._exec_select(args)
            return (msg, not msg.startswith("ERROR:"), -0.2 if msg.startswith("ERROR:") else 0.0, msg.startswith("ERROR: Protocol violation"))
        if tool == "filter":
            msg = self._exec_filter(args)
            return (msg, not msg.startswith("ERROR:"), -0.2 if msg.startswith("ERROR:") else 0.0, msg.startswith("ERROR: Protocol violation"))
        if tool == "add_column":
            msg = self._exec_add_column(args)
            return (msg, not msg.startswith("ERROR:"), -0.2 if msg.startswith("ERROR:") else 0.0, msg.startswith("ERROR: Protocol violation"))
        if tool == "group_agg":
            msg = self._exec_group_agg(args)
            return (msg, not msg.startswith("ERROR:"), -0.2 if msg.startswith("ERROR:") else 0.0, msg.startswith("ERROR: Protocol violation"))
        if tool == "join":
            msg = self._exec_join(args)
            return (msg, not msg.startswith("ERROR:"), -0.2 if msg.startswith("ERROR:") else 0.0, msg.startswith("ERROR: Protocol violation"))
        if tool == "sort":
            msg = self._exec_sort(args)
            return (msg, not msg.startswith("ERROR:"), -0.2 if msg.startswith("ERROR:") else 0.0, msg.startswith("ERROR: Protocol violation"))
        if tool == "distinct":
            msg = self._exec_distinct(args)
            return (msg, not msg.startswith("ERROR:"), -0.2 if msg.startswith("ERROR:") else 0.0, msg.startswith("ERROR: Protocol violation"))
        if tool == "compute_stat":
            msg = self._exec_compute_stat(args)
            return (msg, not msg.startswith("ERROR:"), -0.2 if msg.startswith("ERROR:") else 0.0, msg.startswith("ERROR: Protocol violation"))
        if tool == "head":
            msg = self._exec_head(args)
            return (msg, not msg.startswith("ERROR:"), -0.2 if msg.startswith("ERROR:") else 0.0, msg.startswith("ERROR: Protocol violation"))
        if tool == "help":
            return self._exec_help(), True, 0.0, False
        return (f"ERROR: Unknown tool '{tool}'.", False, -0.5, False)

    def _check_answer(self, submitted: str) -> bool:
        expected = self.task.get("answer")
        atype = self.task.get("answer_type")
        if atype == "number":
            try:
                got = float(submitted)
                exp = float(expected)
                return abs(got - exp) <= 1e-6 or abs(got - exp) <= max(0.001, 1e-3 * max(1.0, abs(exp)))
            except:
                return False
        else:
            return str(submitted).strip().lower() == str(expected).strip().lower()

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        # Timeout check first
        if self.turn_count > self.max_turns:
            obs = "Timeout: maximum turns reached."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        parsed = self._parse_action(action)
        if not parsed:
            obs = "Invalid action format. Use \\boxed{tool arg=value; ...} exactly."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        tool, args = parsed
        tool_lower = tool.strip()

        if tool_lower.upper() == "SUBMIT":
            answer = args.get("answer", None)
            if answer is None:
                obs = "ERROR: Protocol violation - SUBMIT requires answer="
                return obs, -0.5, False, False, {"suffix": self.get_task_suffix()}
            correct = self._check_answer(str(answer))
            if correct:
                obs = f"Submission received. SUCCESS. Final answer is correct: {answer}"
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                exp_show = self.task.get("answer")
                obs = f"Submission received. FAILURE. Final answer incorrect. Expected was '{exp_show}'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        if tool_lower not in self.tools:
            obs = f"ERROR: Unknown tool '{tool_lower}'. Terminating."
            return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}

        msg, ok, penalty, prot = self._execute_tool(tool_lower, args)
        reward = 0.0
        terminated = False
        truncated = False
        if ok:
            self.steps_taken += 1
            obs = f"OK: {msg}"
        else:
            obs = msg
            # Protocol violations do not terminate, others might
            if "Unknown tool" in msg:
                terminated = True
            # keep episode alive for protocol errors; apply penalty
            reward = penalty

        if self.turn_count >= self.max_turns:
            terminated = True
            truncated = True
            obs += " | Timeout reached."

        info = {"suffix": self.get_task_suffix()}
        return obs, reward, terminated, truncated, info

    def sample_random_action(self) -> str:
        samples = [
            r"\boxed{help}",
            r"\boxed{load_table name=sales}",
            r"\boxed{join right=products; on=product_id=product_id}",
            r"\boxed{add_column name=revenue expr=quantity*unit_price}",
            r"\boxed{filter where=date~2024-05}",
            r"\boxed{group_agg by=category; agg=sum:revenue}",
            r"\boxed{sort by=sum_revenue; order=desc}",
            r"\boxed{head n=1}",
            r"\boxed{compute_stat stat=count_rows}",
            r"\boxed{SUBMIT answer=Gadgets}",
        ]
        return random.choice(samples)


class OperatorOrchestraEnvWithFeedback(OperatorOrchestraEnv):
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
            hint = "Wrap exactly one action in \\boxed{...} with a tool and key=value args."

        elif "unknown tool" in text:
            error_type = "UnsupportedAction"
            error_detail["tool"] = "unknown"
            hint = "Call one of the listed tools. Try \\boxed{help} to see available tools."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            # derive a simple detail
            if "no table loaded" in text:
                error_detail["violation"] = "no_table_loaded"
                hint = "Start with \\boxed{load_table name=<table>}."
            elif "requires" in text:
                error_detail["violation"] = "missing_or_bad_arguments"
                hint = "Provide required arguments in key=value form; see examples in instructions."
            else:
                error_detail["violation"] = "bad_sequence_or_args"
                hint = "Check tool preconditions and argument names; use \\boxed{help} for tools list."

        elif "failure. final answer incorrect" in text:
            error_type = "WrongDecision"
            # extract expected if present
            m = re.search(r"expected was '([^']+)'", obs, flags=re.IGNORECASE)
            exp = m.group(1) if m else None
            error_detail["expected"] = exp
            # we can't reliably parse 'got', action isn't included in obs; include best-effort
            error_detail["got"] = None
            # context-aware hint
            kind = self.task.get("kind")
            if kind == "top_category_revenue_month":
                hint = "Compute revenue=quantity*unit_price, group by category for the month, sort desc, then submit top category."
            elif kind == "unique_users_action_country":
                hint = "Join events with users on user_id, filter by action and country, distinct user_id, then compute count_rows."
            elif kind == "avg_stock_supplier":
                hint = "Join inventory with products, filter supplier, then compute avg:stock."
            else:
                hint = "Re-run the necessary aggregations and verify the final value before submitting."

        elif "timeout" in text:
            error_type = "Timeout"
            error_detail["outcome"] = "max_turns_reached"
            hint = "Plan your pipeline up front. Start with load_table, then apply transformations efficiently."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            # state snapshot
            diagnostic["state"] = {
                "current_table": self.current_table_name,
                "rows": len(self.current_table) if self.current_table is not None else None,
                "last_stat_result": self.execution_state.get("last_stat_result"),
                "task_kind": self.task.get("kind"),
                "expected_steps": self.task.get("expected_steps"),
                "steps_taken": self.steps_taken,
            }
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        first_hint = None
        kind = self.task.get("kind")
        if kind == "top_category_revenue_month":
            month = self.task["params"]["month"]
            first_hint = f"Start with \\boxed{{load_table name=sales}}, then join products on product_id for month {month}."
        elif kind == "unique_users_action_country":
            first_hint = "Start with \\boxed{load_table name=events}, then join users on user_id."
        elif kind == "avg_stock_supplier":
            first_hint = "Start with \\boxed{load_table name=inventory}, then join products on product_id."
        else:
            first_hint = "Start by loading a relevant table with \\boxed{load_table name=<table>}."

        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": first_hint,
            "turn": 0,
            "state": {
                "current_table": None,
                "rows": None,
                "last_stat_result": None,
                "task_kind": self.task.get("kind"),
                "expected_steps": self.task.get("expected_steps"),
                "steps_taken": 0,
            },
        }
        return obs, info