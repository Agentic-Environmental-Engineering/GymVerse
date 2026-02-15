from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class ToolchainComposerEnv(Env):
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

        self.tools: Dict[str, Dict[str, Any]] = {}
        self.datasets: Dict[str, Any] = {}
        self.workspace_tables: Dict[str, List[Dict[str, Any]]] = {}
        self.named_results: Dict[str, Any] = {}
        self.handle_counter = 0
        self.task: Dict[str, Any] = {}
        self.turn_count = 0
        self.steps_taken = 0
        self.terminated = False
        self.truncated = False
        self.expected_answer: str = ""

        self._init_database()
        self.reset()

    def _init_database(self):
        self.tools = {
            "load_table": {
                "description": "Load a named dataset into the workspace.",
                "parameters": [
                    {"name": "source", "type": "string", "required": True},
                    {"name": "alias", "type": "string", "required": False},
                ],
                "returns": "handle (string)",
                "example": r"\boxed{tool: load_table, args: {source: products, alias: p}}",
            },
            "select_columns": {
                "description": "Project a table to a subset of columns.",
                "parameters": [
                    {"name": "handle", "type": "string", "required": True},
                    {"name": "columns", "type": "list[string]|csv", "required": True},
                ],
                "returns": "new handle",
                "example": r"\boxed{tool: select_columns, args: {handle: t1, columns: [name, revenue]}}",
            },
            "filter_rows": {
                "description": "Filter rows by a simple condition (col op value). Supported ops: ==, !=, >, <, >=, <=, in.",
                "parameters": [
                    {"name": "handle", "type": "string", "required": True},
                    {"name": "where", "type": "string", "required": True},
                ],
                "returns": "new handle",
                "example": r"\boxed{tool: filter_rows, args: {handle: t2, where: region == 'West'}}",
            },
            "join": {
                "description": "Inner join two tables using a key. on can be 'key' or 'left=right'.",
                "parameters": [
                    {"name": "left", "type": "string", "required": True},
                    {"name": "right", "type": "string", "required": True},
                    {"name": "on", "type": "string", "required": True},
                ],
                "returns": "new handle",
                "example": r"\boxed{tool: join, args: {left: t_items, right: t_products, on: product_id}}",
            },
            "derive_column": {
                "description": "Create a derived numeric column from simple expressions. Supported patterns: a*b, a*b*(1 - c).",
                "parameters": [
                    {"name": "handle", "type": "string", "required": True},
                    {"name": "name", "type": "string", "required": True},
                    {"name": "expr", "type": "string", "required": True},
                ],
                "returns": "new handle",
                "example": r"\boxed{tool: derive_column, args: {handle: t3, name: revenue, expr: price * qty}}",
            },
            "groupby_agg": {
                "description": "Group by columns and aggregate: sum:col, avg:col, max:col, min:col, count.",
                "parameters": [
                    {"name": "handle", "type": "string", "required": True},
                    {"name": "by", "type": "list[string]|csv", "required": True},
                    {"name": "agg", "type": "string", "required": True},
                ],
                "returns": "new handle",
                "example": r"\boxed{tool: groupby_agg, args: {handle: t4, by: [product_name], agg: sum:revenue}}",
            },
            "sort": {
                "description": "Sort rows by a column.",
                "parameters": [
                    {"name": "handle", "type": "string", "required": True},
                    {"name": "by", "type": "string", "required": True},
                    {"name": "order", "type": "string", "required": False},
                ],
                "returns": "new handle",
                "example": r"\boxed{tool: sort, args: {handle: t5, by: revenue, order: desc}}",
            },
            "head": {
                "description": "Preview first n rows (returns same handle).",
                "parameters": [
                    {"name": "handle", "type": "string", "required": True},
                    {"name": "n", "type": "int", "required": False},
                ],
                "returns": "same handle",
                "example": r"\boxed{tool: head, args: {handle: t6, n: 3}}",
            },
            "save_result": {
                "description": "Save a handle by name for later reference.",
                "parameters": [
                    {"name": "handle", "type": "string", "required": True},
                    {"name": "name", "type": "string", "required": True},
                ],
                "returns": "name",
                "example": r"\boxed{tool: save_result, args: {handle: t7, name: top_products}}",
            },
        }

    def _build_datasets(self):
        random.seed()  # keep using global seed set in reset
        regions = ["North", "South", "East", "West"]
        categories = ["Gadget", "Home", "Outdoors", "Books", "Toys"]
        segments = ["consumer", "business", "vip"]

        num_products = 10 + self.complexity * 3
        adjectives = ["Swift", "Bright", "Silent", "Mighty", "Clever", "Compact", "Granite", "Velvet", "Nimbus", "Vertex", "Aurora", "Summit", "Vivid", "Quantum", "Nova", "Echo", "Vertex2", "Polar", "Solar", "Zenith"]
        nouns = ["Widget", "Lamp", "Tent", "Book", "Drone", "Mixer", "Chair", "Bottle", "Speaker", "Camera", "Watch", "Router", "Cleaner", "Grill", "Table", "Backpack", "Scooter", "Board", "Tablet", "Printer"]
        random.shuffle(adjectives)
        random.shuffle(nouns)

        products = []
        for i in range(num_products):
            pid = f"P{i+1:03d}"
            name = f"{adjectives[i % len(adjectives)]} {nouns[i % len(nouns)]}"
            price = random.randint(8, 180)
            cat = random.choice(categories)
            products.append({"product_id": pid, "product_name": name, "category": cat, "price": price})

        num_customers = 20 + self.complexity * 5
        customers = []
        for i in range(num_customers):
            cid = f"C{i+1:03d}"
            seg = random.choices(segments, weights=[5, 3, 2], k=1)[0]
            customers.append({"customer_id": cid, "segment": seg})

        num_items = 120 + self.complexity * 30
        order_items = []
        for i in range(num_items):
            prod = random.choice(products)
            cid = random.choice(customers)
            qty = random.randint(1, 6)
            region = random.choice(regions)
            order_items.append(
                {
                    "order_id": f"O{i+1:04d}",
                    "product_id": prod["product_id"],
                    "qty": qty,
                    "region": region,
                    "customer_id": cid["customer_id"],
                }
            )

        # Promos table: for a subset of product-region pairs
        promos = []
        for prod in random.sample(products, k=max(1, len(products) // 2)):
            region = random.choice(regions)
            discount = random.choice([0.1, 0.2])
            promos.append({"product_id": prod["product_id"], "region": region, "discount": discount})

        self.datasets = {
            "products": products,
            "order_items": order_items,
            "customers": customers,
            "promos": promos,
        }

    def _compute_topk_solution(self, region: str, top_k: int, min_qty: int, segment: Optional[str], use_promos: bool) -> List[str]:
        products = {p["product_id"]: p for p in self.datasets["products"]}
        customers = {c["customer_id"]: c for c in self.datasets["customers"]}
        promo_map = {}
        if use_promos:
            for pr in self.datasets["promos"]:
                promo_map[(pr["product_id"], pr["region"])] = pr["discount"]
        items = self.datasets["order_items"]

        revenue_by_name: Dict[str, float] = {}
        for it in items:
            if it["region"] != region:
                continue
            if it["qty"] < min_qty:
                continue
            if segment is not None:
                cust = customers.get(it["customer_id"])
                if not cust or cust["segment"] != segment:
                    continue
            prod = products.get(it["product_id"])
            if not prod:
                continue
            base = prod["price"] * it["qty"]
            if use_promos:
                d = promo_map.get((it["product_id"], it["region"]), 0.0)
                base = base * (1 - d)
            name = prod["product_name"]
            revenue_by_name[name] = revenue_by_name.get(name, 0.0) + base

        # Sort by revenue desc, then name asc
        sorted_items = sorted(revenue_by_name.items(), key=lambda kv: (-kv[1], kv[0]))
        return [name for name, _ in sorted_items[:top_k]]

    def _generate_task_requiring_n_steps(self, required_steps: int) -> Dict[str, Any]:
        # Build datasets first
        self._build_datasets()

        base_steps = 8  # load products, load items, join, derive revenue, filter region, group, sort, head
        add_qty = False
        add_promos = False
        add_customers = False
        add_tie_sort = False
        top_k = 1

        # Toggle knobs to meet target steps
        planned = base_steps
        if required_steps >= 9:
            add_qty = True
            planned += 1
        if required_steps >= 11:
            add_promos = True
            planned += 2
        if required_steps >= 13:
            add_customers = True
            planned += 3
        if required_steps >= 14:
            add_tie_sort = True
            planned += 1
        if required_steps >= 15:
            top_k = 3
            # head is already counted; leaving planned unchanged

        # If planned exceeds max, trim optional toggles
        while planned > self.max_required_steps:
            if add_tie_sort:
                add_tie_sort = False
                planned -= 1
            elif add_customers:
                add_customers = False
                planned -= 3
            elif add_promos:
                add_promos = False
                planned -= 2
            elif add_qty:
                add_qty = False
                planned -= 1
            else:
                break

        # Choose constraints
        region = random.choice(["North", "South", "East", "West"])
        min_qty = 1 if not add_qty else random.choice([2, 3])
        segment = None
        if add_customers:
            segment = random.choice(["consumer", "business", "vip"])

        # Ensure task is solvable: loop to adjust constraints
        attempt = 0
        solution: List[str] = []
        while attempt < 10:
            solution = self._compute_topk_solution(region, top_k, min_qty, segment, add_promos)
            if len(solution) > 0:
                break
            # Relax: lower min_qty or remove segment or switch region
            if add_qty and min_qty > 1:
                min_qty = 1
            elif segment is not None:
                segment = None
            else:
                region = random.choice(["North", "South", "East", "West"])
            attempt += 1

        # If still empty, relax everything
        if len(solution) == 0:
            add_promos = False
            add_customers = False
            segment = None
            min_qty = 1
            region = random.choice(["North", "South", "East", "West"])
            solution = self._compute_topk_solution(region, top_k, min_qty, segment, add_promos)

        k_effective = min(top_k, max(1, len(solution)))
        solution = solution[:k_effective]

        description_parts = [
            f"Compute the top {k_effective} product name(s) by total revenue in region '{region}'.",
            "Revenue is computed as price * qty."
        ]
        if add_promos:
            description_parts.append("Apply promos: if a (product_id, region) appears in 'promos', multiply revenue by (1 - discount).")
        if add_qty and min_qty > 1:
            description_parts.append(f"Only consider line-items with qty >= {min_qty}.")
        if add_customers and segment:
            description_parts.append(f"Only include customers in segment '{segment}'.")
        description_parts.append("Break ties by product name ascending. Submit the names joined by ', ' in order.")
        description = " ".join(description_parts)

        expected = ", ".join(solution)

        return {
            "type": "TopKRevenue",
            "required_steps": required_steps,
            "planned_steps": planned,
            "constraints": {
                "region": region,
                "min_qty": min_qty,
                "segment": segment,
                "use_promos": add_promos,
                "tie_sort": add_tie_sort,
                "top_k": k_effective,
            },
            "description": description,
            "expected": expected,
        }

    def _get_instructions(self) -> str:
        ds = self.datasets
        ds_info = []
        for name, rows in ds.items():
            if isinstance(rows, list) and rows:
                keys = list(rows[0].keys())
                ds_info.append(f"- {name} (rows={len(rows)}): columns={keys}")
            else:
                ds_info.append(f"- {name}: {type(rows)}")
        tools_info = []
        for name, spec in self.tools.items():
            params = ", ".join([p["name"] for p in spec["parameters"]])
            tools_info.append(f"* {name}({params}): {spec['description']}")

        lines = []
        lines.append("You are composing a toolchain to answer a data question.")
        lines.append(f"Task: {self.task.get('description','')}")
        lines.append("Available datasets:")
        lines.extend(ds_info)
        lines.append("")
        lines.append("Available tools:")
        lines.extend(tools_info)
        lines.append("")
        lines.append("Action protocol:")
        lines.append("- Call tools using: \\boxed{tool: TOOL_NAME, args: {param1: value1, param2: value2}}")
        lines.append("- Submit your final answer using: \\boxed{final: YOUR_ANSWER}")
        lines.append("Notes:")
        lines.append("- Strings can be unquoted words or quoted with single quotes.")
        lines.append("- columns can be [col1, col2] or 'col1, col2'.")
        lines.append("- Invalid action format ends the episode immediately.")
        lines.append(f"- You will likely need between {self.min_required_steps} and {self.max_required_steps} tool calls.")
        return "\n".join(lines)

    def get_task_suffix(self) -> str:
        return (
            f"[State] Turn {self.turn_count}/{self.max_turns} | Tool steps used: {self.steps_taken} | "
            f"Planned pipeline size: {self.task.get('planned_steps')} | "
            "Format your next action as \\boxed{...}"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        required_steps = random.randint(self.min_required_steps, self.max_required_steps)
        self.workspace_tables = {}
        self.named_results = {}
        self.handle_counter = 0
        self.turn_count = 0
        self.steps_taken = 0
        self.terminated = False
        self.truncated = False

        self.task = self._generate_task_requiring_n_steps(required_steps)
        self.expected_answer = self.task["expected"]

        obs = self._get_instructions()
        info = {"suffix": self.get_task_suffix()}
        return obs, info

    def _new_handle(self, alias: Optional[str] = None) -> str:
        if alias:
            # accept only safe alias pattern
            if re.fullmatch(r"[A-Za-z_]\w*", alias) and alias not in self.workspace_tables:
                self.workspace_tables[alias] = []
                return alias
        self.handle_counter += 1
        h = f"t{self.handle_counter}"
        self.workspace_tables[h] = []
        return h

    def _ensure_handle(self, handle: str) -> List[Dict[str, Any]]:
        if handle not in self.workspace_tables:
            raise ValueError(f"No such handle: {handle}")
        return self.workspace_tables[handle]

    def _parse_csv_or_list(self, v: str) -> List[str]:
        v = v.strip()
        if v.startswith("[") and v.endswith("]"):
            inner = v[1:-1].strip()
            if not inner:
                return []
            parts = [p.strip() for p in inner.split(",")]
            return [self._strip_quotes(x) for x in parts]
        # csv
        parts = [p.strip() for p in v.split(",")]
        return [self._strip_quotes(x) for x in parts]

    def _strip_quotes(self, s: str) -> str:
        s = s.strip()
        if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
            return s[1:-1]
        return s

    def _parse_value(self, s: str):
        s = s.strip()
        # list
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1].strip()
            if not inner:
                return []
            parts = [p.strip() for p in inner.split(",")]
            return [self._parse_value(p) for p in parts]
        # quoted string
        if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
            return s[1:-1]
        # number
        if re.fullmatch(r"-?\d+", s):
            return int(s)
        if re.fullmatch(r"-?\d+\.\d+", s):
            return float(s)
        # bare word
        return s

    def _parse_args(self, args_str: str) -> Dict[str, Any]:
        args = {}
        # Split by commas not inside brackets or quotes (simplified heuristic)
        parts = []
        depth = 0
        current = []
        in_quote = None
        for ch in args_str:
            if in_quote:
                current.append(ch)
                if ch == in_quote:
                    in_quote = None
                continue
            if ch in ("'", '"'):
                in_quote = ch
                current.append(ch)
                continue
            if ch == "[":
                depth += 1
                current.append(ch)
                continue
            if ch == "]":
                depth -= 1
                current.append(ch)
                continue
            if ch == "," and depth == 0:
                parts.append("".join(current).strip())
                current = []
            else:
                current.append(ch)
        if current:
            parts.append("".join(current).strip())
        for p in parts:
            if not p:
                continue
            if ":" in p:
                k, v = p.split(":", 1)
            elif "=" in p:
                k, v = p.split("=", 1)
            else:
                continue
            k = k.strip()
            v = v.strip()
            if v.startswith("[") and v.endswith("]"):
                args[k] = self._parse_value(v)
            else:
                args[k] = self._parse_value(v)
        return args

    def _parse_action(self, action: str):
        if not isinstance(action, str):
            return None
        m = re.search(r"\\boxed\{(.*)\}\s*$", action.strip(), re.DOTALL)
        if not m:
            return None
        inner = m.group(1).strip()
        # final
        m_final = re.match(r"final\s*[:=]\s*(.+)$", inner, re.IGNORECASE | re.DOTALL)
        if m_final:
            ans = m_final.group(1).strip()
            ans = self._strip_quotes(ans)
            return ("final", ans, None)
        # tool
        m_tool = re.search(r"tool\s*[:=]\s*([A-Za-z_]\w*)", inner)
        m_args = re.search(r"args\s*[:=]\s*\{(.*)\}\s*$", inner, re.DOTALL)
        if m_tool:
            tool = m_tool.group(1)
            args = {}
            if m_args:
                args = self._parse_args(m_args.group(1))
            return ("tool", tool, args)
        return None

    def _exec_load_table(self, args: Dict[str, Any]) -> Tuple[str, str]:
        source = args.get("source")
        alias = args.get("alias")
        if source not in self.datasets:
            raise ValueError(f"Unknown dataset: {source}")
        handle = self._new_handle(alias)
        self.workspace_tables[handle] = [dict(r) for r in self.datasets[source]]
        return handle, f"Loaded '{source}' into handle {handle} (rows={len(self.workspace_tables[handle])})."

    def _exec_select_columns(self, args: Dict[str, Any]) -> Tuple[str, str]:
        handle = args.get("handle")
        cols = args.get("columns")
        if isinstance(cols, str):
            cols = self._parse_csv_or_list(cols)
        rows = self._ensure_handle(handle)
        new_rows = [{c: r.get(c) for c in cols} for r in rows]
        newh = self._new_handle()
        self.workspace_tables[newh] = new_rows
        return newh, f"Selected columns {cols} from {handle} into {newh}."

    def _eval_condition(self, row: Dict[str, Any], where: str) -> bool:
        # Supports: col op value, op in {==, !=, >, <, >=, <=, in}
        m = re.match(r"\s*(\w+)\s*(==|!=|>=|<=|>|<|in)\s*(.+)\s*$", where)
        if not m:
            raise ValueError(f"Unsupported condition: {where}")
        col, op, val = m.group(1), m.group(2), m.group(3).strip()
        if op == "in":
            vals = self._parse_value(val)
            return row.get(col) in vals
        v = self._parse_value(val)
        rv = row.get(col)
        try:
            if op == "==":
                return rv == v
            if op == "!=":
                return rv != v
            if op == ">":
                return rv > v
            if op == "<":
                return rv < v
            if op == ">=":
                return rv >= v
            if op == "<=":
                return rv <= v
        except Exception:
            return False
        return False

    def _exec_filter_rows(self, args: Dict[str, Any]) -> Tuple[str, str]:
        handle = args.get("handle")
        where = args.get("where")
        rows = self._ensure_handle(handle)
        new_rows = [r for r in rows if self._eval_condition(r, where)]
        newh = self._new_handle()
        self.workspace_tables[newh] = new_rows
        return newh, f"Filtered {handle} -> {newh} with where='{where}' (rows={len(new_rows)})."

    def _exec_join(self, args: Dict[str, Any]) -> Tuple[str, str]:
        left = args.get("left")
        right = args.get("right")
        on = args.get("on")
        lrows = self._ensure_handle(left)
        rrows = self._ensure_handle(right)
        if "=" in str(on):
            lkey, rkey = [x.strip() for x in str(on).split("=", 1)]
        else:
            lkey = rkey = str(on)
        index = {}
        for rr in rrows:
            index.setdefault(rr.get(rkey), []).append(rr)
        out = []
        for lr in lrows:
            key = lr.get(lkey)
            matches = index.get(key, [])
            for rr in matches:
                merged = dict(lr)
                for k, v in rr.items():
                    if k in merged and merged[k] != v:
                        merged[f"{k}_right"] = v
                    else:
                        merged[k] = v
                out.append(merged)
        newh = self._new_handle()
        self.workspace_tables[newh] = out
        return newh, f"Joined {left} ⋈ {right} on {on} -> {newh} (rows={len(out)})."

    def _exec_derive_column(self, args: Dict[str, Any]) -> Tuple[str, str]:
        handle = args.get("handle")
        name = args.get("name")
        expr = str(args.get("expr"))
        rows = self._ensure_handle(handle)

        # Supported: a*b or a*b*(1 - c)
        m1 = re.match(r"\s*(\w+)\s*\*\s*(\w+)\s*$", expr)
        m2 = re.match(r"\s*(\w+)\s*\*\s*(\w+)\s*\*\s*\(\s*1\s*-\s*(\w+)\s*\)\s*$", expr)
        if not (m1 or m2):
            raise ValueError(f"Unsupported expr: {expr}")
        out = []
        for r in rows:
            if m2:
                a, b, c = m2.group(1), m2.group(2), m2.group(3)
                va = r.get(a, 0)
                vb = r.get(b, 0)
                vc = r.get(c, 0)
                try:
                    val = float(va) * float(vb) * (1.0 - float(vc))
                except Exception:
                    val = 0.0
            else:
                a, b = m1.group(1), m1.group(2)
                va = r.get(a, 0)
                vb = r.get(b, 0)
                try:
                    val = float(va) * float(vb)
                except Exception:
                    val = 0.0
            nr = dict(r)
            nr[name] = val
            out.append(nr)
        newh = self._new_handle()
        self.workspace_tables[newh] = out
        return newh, f"Derived column '{name}' with expr '{expr}' -> {newh}."

    def _exec_groupby_agg(self, args: Dict[str, Any]) -> Tuple[str, str]:
        handle = args.get("handle")
        by = args.get("by")
        agg = str(args.get("agg"))
        if isinstance(by, str):
            by = self._parse_csv_or_list(by)
        rows = self._ensure_handle(handle)
        groups = {}
        for r in rows:
            key = tuple(r.get(c) for c in by)
            groups.setdefault(key, []).append(r)
        out = []
        if agg == "count":
            for key, items in groups.items():
                row = {c: key[i] for i, c in enumerate(by)}
                row["count"] = len(items)
                out.append(row)
            agg_desc = "count"
        else:
            m = re.match(r"\s*(sum|avg|max|min)\s*:\s*(\w+)\s*$", agg)
            if not m:
                raise ValueError(f"Unsupported agg: {agg}")
            op, col = m.group(1), m.group(2)
            for key, items in groups.items():
                vals = [float(it.get(col, 0)) for it in items]
                row = {c: key[i] for i, c in enumerate(by)}
                if op == "sum":
                    row[f"sum_{col}"] = sum(vals)
                elif op == "avg":
                    row[f"avg_{col}"] = sum(vals) / len(vals) if vals else 0.0
                elif op == "max":
                    row[f"max_{col}"] = max(vals) if vals else 0.0
                elif op == "min":
                    row[f"min_{col}"] = min(vals) if vals else 0.0
                out.append(row)
            agg_desc = f"{op}:{col}"
        newh = self._new_handle()
        self.workspace_tables[newh] = out
        return newh, f"Grouped by {by} with {agg_desc} -> {newh} (rows={len(out)})."

    def _exec_sort(self, args: Dict[str, Any]) -> Tuple[str, str]:
        handle = args.get("handle")
        by = args.get("by")
        order = str(args.get("order", "asc")).lower()
        rows = self._ensure_handle(handle)
        reverse = order == "desc"
        out = sorted(rows, key=lambda r: (r.get(by),), reverse=reverse)
        newh = self._new_handle()
        self.workspace_tables[newh] = out
        return newh, f"Sorted {handle} by {by} {order} -> {newh}."

    def _exec_head(self, args: Dict[str, Any]) -> Tuple[str, str]:
        handle = args.get("handle")
        n = int(args.get("n", 5))
        rows = self._ensure_handle(handle)
        preview = rows[:max(0, n)]
        return handle, f"Preview first {n} rows of {handle}:\n{preview}"

    def _exec_save_result(self, args: Dict[str, Any]) -> Tuple[str, str]:
        handle = args.get("handle")
        name = args.get("name")
        rows = self._ensure_handle(handle)
        self.named_results[name] = rows
        return handle, f"Saved handle {handle} as '{name}'."

    def _execute_tool(self, tool: str, args: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        if tool == "load_table":
            h, msg = self._exec_load_table(args)
            return f"Tool load_table executed. {msg}", h
        if tool == "select_columns":
            h, msg = self._exec_select_columns(args)
            return f"Tool select_columns executed. {msg}", h
        if tool == "filter_rows":
            h, msg = self._exec_filter_rows(args)
            return f"Tool filter_rows executed. {msg}", h
        if tool == "join":
            h, msg = self._exec_join(args)
            return f"Tool join executed. {msg}", h
        if tool == "derive_column":
            h, msg = self._exec_derive_column(args)
            return f"Tool derive_column executed. {msg}", h
        if tool == "groupby_agg":
            h, msg = self._exec_groupby_agg(args)
            return f"Tool groupby_agg executed. {msg}", h
        if tool == "sort":
            h, msg = self._exec_sort(args)
            return f"Tool sort executed. {msg}", h
        if tool == "head":
            h, msg = self._exec_head(args)
            return f"Tool head executed. {msg}", h
        if tool == "save_result":
            h, msg = self._exec_save_result(args)
            return f"Tool save_result executed. {msg}", h
        raise ValueError(f"Unknown tool: {tool}")

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self.terminated or self.truncated:
            return "Episode already ended.", 0.0, True, self.truncated, {"suffix": self.get_task_suffix()}

        self.turn_count += 1

        parsed = self._parse_action(action)
        if not parsed:
            obs = "Invalid action format. Use \\boxed{tool: NAME, args: {...}} or \\boxed{final: ANSWER}."
            self.terminated = True
            reward = LanguageGameReward.format_error_reward
            info = {"suffix": self.get_task_suffix()}
            return obs, reward, True, False, info

        kind, name, args = parsed

        if kind == "final":
            ans = str(name).strip()
            correct = ans == self.expected_answer
            if correct:
                obs = f"Final answer submitted. Correct. Success.\nExpected: {self.expected_answer}\nGot: {ans}"
                reward = 1.0
            else:
                obs = f"Final answer submitted. Incorrect.\nExpected: {self.expected_answer}\nGot: {ans}"
                reward = 0.0
            self.terminated = True
            return obs, reward, True, False, {"suffix": self.get_task_suffix()}

        # tool execution
        if name not in self.tools:
            obs = f"Unsupported tool '{name}'. See instructions for available tools."
            # Do not terminate on unsupported tool; allow recovery
            info = {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, info

        try:
            out_text, handle = self._execute_tool(name, args or {})
            self.steps_taken += 1
            # Auto hint: how to complete task
            obs = out_text
            # Task completion check happens only on final submission
            terminated = False
            truncated = False
            info = {"suffix": self.get_task_suffix()}
        except Exception as e:
            obs = f"Execution error: {e}"
            terminated = False
            truncated = False
            info = {"suffix": self.get_task_suffix()}
            return obs, 0.0, terminated, truncated, info

        if self.turn_count >= self.max_turns:
            self.truncated = True
            obs = obs + "\nTimeout: reached max turns."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

    def sample_random_action(self) -> str:
        examples = [
            r"\boxed{tool: load_table, args: {source: products, alias: p}}",
            r"\boxed{tool: load_table, args: {source: order_items, alias: oi}}",
            r"\boxed{tool: join, args: {left: oi, right: p, on: product_id}}",
            r"\boxed{tool: derive_column, args: {handle: t1, name: revenue, expr: price * qty}}",
            r"\boxed{tool: filter_rows, args: {handle: t2, where: region == 'West'}}",
            r"\boxed{tool: groupby_agg, args: {handle: t3, by: [product_name], agg: sum:revenue}}",
            r"\boxed{tool: sort, args: {handle: t4, by: sum_revenue, order: desc}}",
            r"\boxed{tool: head, args: {handle: t5, n: 3}}",
            r"\boxed{final: Example Answer}",
        ]
        return random.choice(examples)


class ToolchainComposerEnvWithFeedback(ToolchainComposerEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def _classify(self, text: str) -> Tuple[str, Dict[str, Any], Optional[str]]:
        t = text.lower()
        error_type = "OK"
        detail: Dict[str, Any] = {}
        hint: Optional[str] = None

        if "invalid action format" in t or "use \\boxed" in t:
            error_type = "FormatError"
            detail["issue"] = "missing_or_bad_boxed"
            hint = "Use \\boxed{tool: NAME, args: {...}} for tools or \\boxed{final: ANSWER} to submit."

        elif "unsupported tool" in t:
            error_type = "UnsupportedAction"
            detail["tool"] = "unknown"
            hint = "Call an available tool (see the tools list in the instructions)."

        elif "execution error" in t:
            error_type = "ProtocolViolation"
            # try to extract some common reasons
            if "no such handle" in t:
                detail["violation"] = "missing_handle"
                hint = "Load the required table first using load_table, and use the returned handle."
            elif "unsupported expr" in t:
                detail["violation"] = "bad_expression"
                hint = "Use expr like 'price * qty' or 'price * qty * (1 - discount)'."
            elif "unsupported condition" in t:
                detail["violation"] = "bad_condition"
                hint = "Use where like region == 'West' or qty >= 2 or segment in ['consumer','vip']."
            else:
                detail["violation"] = "runtime_error"
                hint = "Check handles, column names, and parameter spelling."

        elif "timeout: reached max turns" in t:
            error_type = "Timeout"
            detail["outcome"] = "max_turns"
            hint = "Plan the pipeline first. Start by loading relevant tables, then join, derive, group, sort."

        elif "final answer submitted. incorrect" in t:
            error_type = "WrongDecision"
            # Try to extract expected and got
            m_exp = re.search(r"expected:\s*(.+)\s*got:\s*(.+)$", text, re.IGNORECASE | re.DOTALL)
            if m_exp:
                detail["expected"] = m_exp.group(1).strip()
                detail["got"] = m_exp.group(2).strip()
            hint = "Re-check filters (region/qty/segment) and whether promos must be applied before aggregation."

        elif "final answer submitted. correct" in t or "success" in t:
            error_type = "OK"
            detail["outcome"] = "success"
            hint = None

        else:
            error_type = "OK"
            detail["outcome"] = "step"

        return error_type, detail, hint

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        error_type, error_detail, hint = self._classify(obs)

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            error_detail["turn"] = getattr(self, "turn_count", None)
            error_detail["steps_taken"] = getattr(self, "steps_taken", None)
            error_detail["required_steps_range"] = [self.min_required_steps, self.max_required_steps]
            error_detail["planned_steps"] = self.task.get("planned_steps")
            error_detail["constraints"] = self.task.get("constraints")
            diagnostic["error_detail"] = error_detail
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {
                "outcome": "episode_start",
                "turn": 0,
                "steps_taken": 0,
                "required_steps_range": [self.min_required_steps, self.max_required_steps],
                "planned_steps": self.task.get("planned_steps"),
                "constraints": self.task.get("constraints"),
            },
            "hint": "Start by loading the tables mentioned in the task (e.g., products and order_items) using load_table.",
        }
        return obs, info