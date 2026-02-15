from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List

class OpsKitComposerEnv(Env):
    def __init__(self, complexity: int = 1, max_turns: Optional[int] = 100, **_):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.max_turns = max_turns if max_turns is not None else 100
        self.min_required_steps = self.complexity
        self.max_required_steps = self.complexity * 2
        self._init_database()
        self.reset()

    def _init_database(self):
        self.tools = {
            "load": {"params": ["source"], "returns": "Set current_data to dataset source"},
            "load_aux": {"params": ["source"], "returns": "Set aux_data to dataset source"},
            "join_aux": {"params": ["on_self", "on_aux", "how"], "returns": "Join current_data with aux_data by keys"},
            "filter": {"params": ["field", "op", "value"], "returns": "Filter records by condition"},
            "map": {"params": ["field", "op", "operand"], "returns": "Transform field values"},
            "aggregate": {"params": ["group_by", "target_field", "op"], "returns": "Aggregate with optional grouping"},
            "sort": {"params": ["field", "order"], "returns": "Sort data by field asc/desc"},
            "head": {"params": ["n"], "returns": "Take first n records"},
            "select": {"params": ["fields"], "returns": "Keep only listed fields (pipe-separated)"},
            "finalize": {"params": [], "returns": "Submit result for evaluation"},
        }
        base_items = 6 + self.complexity * 3
        categories = ["Books", "Electronics", "Toys", "Home", "Clothing", "Sports", "Garden"]
        departments = ["Retail", "Outlet", "Online", "Warehouse"]
        def make_catalog(name: str, n: int) -> List[Dict[str, Any]]:
            rows = []
            for i in range(n):
                rid = i + 1
                rows.append({
                    "id": rid,
                    "name": f"{name}_Item_{rid}",
                    "category": random.choice(categories),
                    "department": random.choice(departments),
                    "price": round(random.uniform(5, 500), 2),
                    "rating": round(random.uniform(1.0, 5.0), 1),
                    "stock": random.randint(0, 200),
                })
            return rows
        nA = base_items
        nB = base_items + random.randint(0, 4)
        self.datasets = {
            "catalog_A": make_catalog("A", nA),
            "catalog_B": make_catalog("B", nB),
        }
        discounts = []
        for ds in ["catalog_A", "catalog_B"]:
            for r in self.datasets[ds]:
                if random.random() < 0.6:
                    discounts.append({"id": r["id"], "discount": round(random.uniform(0.05, 0.35), 2), "source": ds})
        reviews = []
        for ds in ["catalog_A", "catalog_B"]:
            for r in self.datasets[ds]:
                if random.random() < 0.7:
                    reviews.append({"id": r["id"], "count": random.randint(0, 500), "source": ds})
        self.datasets["discounts"] = discounts
        self.datasets["reviews"] = reviews
        extra_sources = max(0, self.complexity - 4)
        for k in range(extra_sources):
            name = f"aux_{k+1}"
            self.datasets[name] = make_catalog(f"X{k+1}", base_items - 2 + random.randint(0, 5))
        self.turn_count = 0
        self.steps_taken = 0
        self.current_data = None
        self.loaded_source = None
        self.aux_data = None
        self.last_result = None
        self.task = {}

    def _generate_task_requiring_n_steps(self, required_steps: int) -> Dict[str, Any]:
        templates = ["scalar_agg", "top_group_sum", "list_top_n"]
        if self.complexity >= 5:
            templates.append("scalar_agg_with_join")
            templates.append("top_group_sum_with_join")
        template = random.choice(templates)
        base_filters = random.randint(1, min(3, self.complexity))
        use_map = random.random() < 0.5
        use_join = ("with_join" in template)
        source = random.choice([s for s in self.datasets.keys() if s.startswith("catalog")])
        field_choices = ["price", "rating", "stock"]
        agg_ops = ["sum", "avg", "max", "min", "count"]
        target_field = random.choice(field_choices)
        op = random.choice(agg_ops)
        group_by = None
        sort_field = None
        sort_order = "desc"
        head_n = random.randint(1, 3 + self.complexity // 3)
        select_fields = ["id", "name", "price", "rating", "stock", "category", "department"]
        chosen_select = random.sample(select_fields, k=min(len(select_fields), 3 + random.randint(0, 2)))
        filters = []
        for _ in range(base_filters):
            f = random.choice(["price", "rating", "stock", "category", "department"])
            if f in ["price", "rating", "stock"]:
                opf = random.choice(["gt", "ge", "lt", "le"])
                val = round(random.uniform(2, 100), 2) if f == "price" else (round(random.uniform(2.0, 4.5), 1) if f == "rating" else random.randint(10, 150))
            else:
                opf = random.choice(["eq", "in", "contains"])
                if f == "category":
                    val = random.choice(["Books", "Electronics", "Toys", "Home"])
                elif f == "department":
                    val = random.choice(["Retail", "Outlet", "Online", "Warehouse"])
                else:
                    val = random.choice(["Retail", "Online"])
            filters.append({"field": f, "op": opf, "value": val})
        recipe = []
        recipe.append({"tool": "load", "args": {"source": source}})
        if use_join:
            recipe.append({"tool": "load_aux", "args": {"source": "discounts"}})
            recipe.append({"tool": "join_aux", "args": {"on_self": "id", "on_aux": "id", "how": "left"}})
            use_map = True
        if use_map:
            if target_field == "price":
                recipe.append({"tool": "map", "args": {"field": "price", "op": "mul", "operand": 0.9}})
            elif target_field == "rating":
                recipe.append({"tool": "map", "args": {"field": "rating", "op": "add", "operand": 0.1}})
            else:
                recipe.append({"tool": "map", "args": {"field": "stock", "op": "mul", "operand": 1.0}})
        for flt in filters:
            recipe.append({"tool": "filter", "args": flt})
        if template.startswith("scalar_agg"):
            recipe.append({"tool": "aggregate", "args": {"group_by": "", "target_field": target_field, "op": op}})
            goal_type = "scalar_agg"
            sort_field = None
        elif template.startswith("top_group_sum"):
            group_by = random.choice(["category", "department"])
            recipe.append({"tool": "aggregate", "args": {"group_by": group_by, "target_field": target_field, "op": "sum"}})
            sort_field = group_by
            recipe.append({"tool": "sort", "args": {"field": "agg_value", "order": "desc"}})
            recipe.append({"tool": "head", "args": {"n": 1}})
            recipe.append({"tool": "select", "args": {"fields": group_by}})
            goal_type = "top_group"
        else:
            sort_field = random.choice(["price", "rating", "stock"])
            recipe.append({"tool": "sort", "args": {"field": sort_field, "order": "desc"}})
            recipe.append({"tool": "head", "args": {"n": head_n}})
            recipe.append({"tool": "select", "args": {"fields": "|".join(chosen_select)}})
            goal_type = "list_rows"
        while len(recipe) < required_steps:
            add_choice = random.choice(["filter", "sort", "map"])
            if add_choice == "filter":
                recipe.append({"tool": "filter", "args": random.choice(filters)})
            elif add_choice == "sort":
                recipe.append({"tool": "sort", "args": {"field": sort_field or target_field, "order": random.choice(["asc", "desc"])}})
            else:
                recipe.append({"tool": "map", "args": {"field": target_field, "op": "mul", "operand": 1.0}})
        expected_result = self._simulate_recipe(recipe, goal_type)
        description = ""
        if goal_type == "scalar_agg":
            description = f"Compute a single scalar: {op} of '{target_field}' from {source} after applying the listed constraints and any necessary adjustments. Use tools to load, optionally join discounts, filter, and aggregate. Finish with finalize."
        elif goal_type == "top_group":
            description = "Find the top group by total adjusted price after filtering. Group by a categorical field (category or department), sum price, sort descending, take the top group, and finalize."
        else:
            description = "Produce a compact list: sort by a metric, take top N, select relevant fields, and finalize."
        return {
            "objective": description,
            "source": source,
            "recipe": recipe,
            "goal_type": goal_type,
            "expected_result": expected_result,
            "filters": filters,
            "required_steps": required_steps,
        }

    def _simulate_recipe(self, recipe: List[Dict[str, Any]], goal_type: str) -> Any:
        data = None
        aux = None
        def deep_copy(rows):
            return [dict(r) for r in rows] if rows is not None else None
        for step in recipe:
            tool = step["tool"]
            args = step["args"]
            if tool == "load":
                src = args["source"]
                data = deep_copy(self.datasets[src])
            elif tool == "load_aux":
                src = args["source"]
                aux = deep_copy(self.datasets.get(src, []))
            elif tool == "join_aux":
                if data is None or aux is None:
                    return None
                on_self = args["on_self"]; on_aux = args["on_aux"]; how = args.get("how", "left")
                index_aux = {}
                for row in aux:
                    key = row.get(on_aux)
                    index_aux.setdefault(key, []).append(row)
                joined = []
                for r in data:
                    key = r.get(on_self)
                    aux_rows = index_aux.get(key, [])
                    if aux_rows:
                        for a in aux_rows:
                            jr = dict(r)
                            for k, v in a.items():
                                if k != on_aux:
                                    jr[k] = v
                            joined.append(jr)
                    elif how == "left":
                        joined.append(dict(r))
                data = joined
            elif tool == "map":
                if data is None: return None
                field = args["field"]; op = args["op"]; operand = args.get("operand", None)
                mapped = []
                for r in data:
                    val = r.get(field)
                    if op == "mul" and isinstance(val, (int, float)) and operand is not None:
                        r[field] = round(val * float(operand), 4)
                    elif op == "add" and isinstance(val, (int, float)) and operand is not None:
                        r[field] = round(val + float(operand), 4)
                    elif op == "lower" and isinstance(val, str):
                        r[field] = val.lower()
                    elif op == "upper" and isinstance(val, str):
                        r[field] = val.upper()
                    mapped.append(r)
                data = mapped
            elif tool == "filter":
                if data is None: return None
                field = args["field"]; op = args["op"]; val = args["value"]
                def cond(x):
                    xv = x.get(field)
                    if op == "eq":
                        return xv == val
                    if op == "neq":
                        return xv != val
                    if op == "gt":
                        return isinstance(xv, (int, float)) and xv > float(val)
                    if op == "ge":
                        return isinstance(xv, (int, float)) and xv >= float(val)
                    if op == "lt":
                        return isinstance(xv, (int, float)) and xv < float(val)
                    if op == "le":
                        return isinstance(xv, (int, float)) and xv <= float(val)
                    if op == "in":
                        return xv in (val if isinstance(val, list) else [val])
                    if op == "contains":
                        return isinstance(xv, str) and isinstance(val, str) and val.lower() in xv.lower()
                    return False
                data = [r for r in data if cond(r)]
            elif tool == "aggregate":
                if data is None: return None
                group_by = args.get("group_by", "")
                target = args["target_field"]; op = args["op"]
                if not group_by:
                    values = [r.get(target) for r in data if isinstance(r.get(target), (int, float))]
                    if op == "sum":
                        agg = round(sum(values), 4)
                    elif op == "avg":
                        agg = round(sum(values) / (len(values) or 1), 4)
                    elif op == "max":
                        agg = round(max(values) if values else 0.0, 4)
                    elif op == "min":
                        agg = round(min(values) if values else 0.0, 4)
                    elif op == "count":
                        agg = len(values)
                    else:
                        agg = None
                    data = [{"agg_value": agg}]
                else:
                    groups = {}
                    for r in data:
                        gk = r.get(group_by)
                        if gk is None: continue
                        groups.setdefault(gk, []).append(r)
                    out = []
                    for gk, rows in groups.items():
                        vals = [rw.get(target) for rw in rows if isinstance(rw.get(target), (int, float))]
                        if op == "sum":
                            agg = round(sum(vals), 4)
                        elif op == "avg":
                            agg = round(sum(vals) / (len(vals) or 1), 4)
                        elif op == "max":
                            agg = round(max(vals) if vals else 0.0, 4)
                        elif op == "min":
                            agg = round(min(vals) if vals else 0.0, 4)
                        elif op == "count":
                            agg = len(vals)
                        else:
                            agg = None
                        out.append({group_by: gk, "agg_value": agg})
                    data = out
            elif tool == "sort":
                if data is None: return None
                field = args["field"]; order = args.get("order", "desc")
                data = sorted(data, key=lambda r: r.get(field, 0), reverse=(order == "desc"))
            elif tool == "head":
                if data is None: return None
                n = int(args["n"])
                data = data[:max(0, n)]
            elif tool == "select":
                if data is None: return None
                fields = args["fields"]
                fl = fields.split("|") if isinstance(fields, str) else fields
                out = []
                for r in data:
                    out.append({k: r.get(k) for k in fl if k in r})
                data = out
        if goal_type == "scalar_agg":
            if isinstance(data, list) and len(data) == 1 and "agg_value" in data[0]:
                return data[0]["agg_value"]
            return None
        if goal_type == "top_group":
            if isinstance(data, list) and len(data) == 1:
                key = None
                for k in data[0].keys():
                    if k != "agg_value":
                        key = k; break
                return data[0].get(key)
            return None
        if goal_type == "list_rows":
            return data
        return None

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        required_steps = random.randint(self.min_required_steps, self.max_required_steps)
        self.task = self._generate_task_requiring_n_steps(required_steps)
        self.turn_count = 0
        self.steps_taken = 0
        self.current_data = None
        self.loaded_source = None
        self.aux_data = None
        self.last_result = None
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _get_instructions(self) -> str:
        tool_list = ", ".join(sorted(self.tools.keys()))
        return (
            "OpsKit Composer: Orchestrate tools to achieve the objective.\n"
            f"Available tools: {tool_list}\n"
            "One tool call per turn. Respect protocol: load a source before transforming; join requires aux loaded.\n"
            "Argument format: tool_name(param=value,...) with values strings or numbers. For 'select', use pipe-separated fields like fields=id|name|price.\n"
            "Finalize ends the episode and evaluates your current data against the objective.\n"
            "Action format: use \\boxed{tool_name(param=value,...)} exactly."
        )

    def get_task_suffix(self) -> str:
        state = f"State: turns={self.turn_count}/{self.max_turns}, steps_taken={self.steps_taken}, loaded_source={self.loaded_source or 'none'}"
        obj = f"Objective: {self.task.get('objective')} | Base source: {self.task.get('source')} | Required steps target≈{self.task.get('required_steps')}."
        fmt = "Respond with a single boxed action like \\boxed{load(source=catalog_A)}."
        return f"{state}\n{obj}\n{fmt}"

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        m = re.search(r"\\boxed\{(.+?)\}", action, flags=re.DOTALL)
        if not m:
            return None
        content = m.group(1).strip()
        m2 = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\((.*)\)$", content)
        if not m2:
            if re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*\)\s*$", content):
                tool_name = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)", content).group(1)
                return {"tool": tool_name, "args": {}}
            return None
        tool_name = m2.group(1)
        arg_str = m2.group(2).strip()
        args = {}
        if arg_str:
            parts = [p.strip() for p in arg_str.split(",") if p.strip()]
            for p in parts:
                kv = p.split("=", 1)
                if len(kv) != 2:
                    return None
                k = kv[0].strip()
                vraw = kv[1].strip()
                if vraw.startswith("'") and vraw.endswith("'"):
                    v = vraw[1:-1]
                elif vraw.startswith('"') and vraw.endswith('"'):
                    v = vraw[1:-1]
                else:
                    try:
                        if "." in vraw:
                            v = float(vraw)
                        else:
                            v = int(vraw)
                    except:
                        v = vraw
                args[k] = v
        return {"tool": tool_name, "args": args}

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        if tool_name == "load":
            src = args.get("source")
            if src not in self.datasets:
                raise ValueError(f"Source not found: {src}")
            self.current_data = [dict(r) for r in self.datasets[src]]
            self.loaded_source = src
            return f"Loaded {len(self.current_data)} records from {src}"
        if tool_name == "load_aux":
            src = args.get("source")
            if src not in self.datasets:
                raise ValueError(f"Source not found: {src}")
            self.aux_data = [dict(r) for r in self.datasets[src]]
            return f"Aux dataset '{src}' loaded with {len(self.aux_data)} records"
        if tool_name == "join_aux":
            if self.current_data is None:
                raise ValueError("Protocol violation: load a primary source before join_aux.")
            if self.aux_data is None:
                raise ValueError("Protocol violation: load_aux must be called before join_aux.")
            on_self = args.get("on_self"); on_aux = args.get("on_aux"); how = args.get("how", "left")
            idx = {}
            for r in self.aux_data:
                k = r.get(on_aux)
                idx.setdefault(k, []).append(r)
            joined = []
            for r in self.current_data:
                k = r.get(on_self)
                aux_rows = idx.get(k, [])
                if aux_rows:
                    for a in aux_rows:
                        jr = dict(r)
                        for kk, vv in a.items():
                            if kk != on_aux:
                                jr[kk] = vv
                        joined.append(jr)
                elif how == "left":
                    joined.append(dict(r))
            self.current_data = joined
            return f"Joined current_data with aux_data on {on_self}={on_aux}, result {len(self.current_data)} records"
        if tool_name == "filter":
            if self.current_data is None:
                raise ValueError("Protocol violation: load must be called before filter.")
            field = args.get("field"); op = args.get("op"); val = args.get("value")
            def cond(x):
                xv = x.get(field)
                if op == "eq":
                    return xv == val
                if op == "neq":
                    return xv != val
                if op == "gt":
                    return isinstance(xv, (int, float)) and xv > float(val)
                if op == "ge":
                    return isinstance(xv, (int, float)) and xv >= float(val)
                if op == "lt":
                    return isinstance(xv, (int, float)) and xv < float(val)
                if op == "le":
                    return isinstance(xv, (int, float)) and xv <= float(val)
                if op == "in":
                    arr = val if isinstance(val, list) else [val]
                    return xv in arr
                if op == "contains":
                    return isinstance(xv, str) and isinstance(val, str) and val.lower() in xv.lower()
                raise ValueError(f"Unsupported filter op: {op}")
            before = len(self.current_data)
            self.current_data = [r for r in self.current_data if cond(r)]
            return f"Filtered {before} -> {len(self.current_data)} records by {field} {op} {val}"
        if tool_name == "map":
            if self.current_data is None:
                raise ValueError("Protocol violation: load must be called before map.")
            field = args.get("field"); op = args.get("op"); operand = args.get("operand", None)
            out = []
            for r in self.current_data:
                val = r.get(field)
                if op == "mul":
                    if not isinstance(val, (int, float)):
                        raise ValueError("map mul requires numeric field")
                    r[field] = round(val * float(operand), 4)
                elif op == "add":
                    if not isinstance(val, (int, float)):
                        raise ValueError("map add requires numeric field")
                    r[field] = round(val + float(operand), 4)
                elif op == "lower":
                    if not isinstance(val, str):
                        raise ValueError("map lower requires string field")
                    r[field] = val.lower()
                elif op == "upper":
                    if not isinstance(val, str):
                        raise ValueError("map upper requires string field")
                    r[field] = val.upper()
                else:
                    raise ValueError(f"Unsupported map op: {op}")
                out.append(r)
            self.current_data = out
            return f"Mapped field '{field}' with op {op}"
        if tool_name == "aggregate":
            if self.current_data is None:
                raise ValueError("Protocol violation: load must be called before aggregate.")
            group_by = args.get("group_by", "")
            target = args.get("target_field")
            op = args.get("op")
            if not group_by:
                vals = [r.get(target) for r in self.current_data if isinstance(r.get(target), (int, float))]
                if op == "sum":
                    agg = round(sum(vals), 4)
                elif op == "avg":
                    agg = round(sum(vals) / (len(vals) or 1), 4)
                elif op == "max":
                    agg = round(max(vals) if vals else 0.0, 4)
                elif op == "min":
                    agg = round(min(vals) if vals else 0.0, 4)
                elif op == "count":
                    agg = len(vals)
                else:
                    raise ValueError(f"Unsupported aggregate op: {op}")
                self.current_data = [{"agg_value": agg}]
                self.last_result = {"type": "scalar", "value": agg}
                return f"Aggregated (no group) {op} on {target} -> {agg}"
            else:
                groups = {}
                for r in self.current_data:
                    gk = r.get(group_by)
                    if gk is None: continue
                    groups.setdefault(gk, []).append(r)
                out = []
                for gk, rows in groups.items():
                    vals = [rw.get(target) for rw in rows if isinstance(rw.get(target), (int, float))]
                    if op == "sum":
                        agg = round(sum(vals), 4)
                    elif op == "avg":
                        agg = round(sum(vals) / (len(vals) or 1), 4)
                    elif op == "max":
                        agg = round(max(vals) if vals else 0.0, 4)
                    elif op == "min":
                        agg = round(min(vals) if vals else 0.0, 4)
                    elif op == "count":
                        agg = len(vals)
                    else:
                        raise ValueError(f"Unsupported aggregate op: {op}")
                    out.append({group_by: gk, "agg_value": agg})
                self.current_data = out
                self.last_result = {"type": "grouped", "group_by": group_by}
                return f"Aggregated by {group_by} ({op} {target}); groups={len(out)}"
        if tool_name == "sort":
            if self.current_data is None:
                raise ValueError("Protocol violation: load must be called before sort.")
            field = args.get("field"); order = args.get("order", "desc")
            self.current_data = sorted(self.current_data, key=lambda r: r.get(field, 0), reverse=(order == "desc"))
            return f"Sorted by {field} {order}"
        if tool_name == "head":
            if self.current_data is None:
                raise ValueError("Protocol violation: load must be called before head.")
            n = int(args.get("n", 1))
            self.current_data = self.current_data[:max(0, n)]
            return f"Head {n} taken; remaining {len(self.current_data)}"
        if tool_name == "select":
            if self.current_data is None:
                raise ValueError("Protocol violation: load must be called before select.")
            fields = args.get("fields")
            fl = fields.split("|") if isinstance(fields, str) else fields
            self.current_data = [{k: r.get(k) for k in fl if k in r} for r in self.current_data]
            return f"Selected fields: {', '.join(fl)}"
        if tool_name == "finalize":
            goal_type = self.task.get("goal_type")
            expected = self.task.get("expected_result")
            if goal_type == "scalar_agg":
                if isinstance(self.current_data, list) and len(self.current_data) == 1 and "agg_value" in self.current_data[0]:
                    candidate = self.current_data[0]["agg_value"]
                else:
                    candidate = None
            elif goal_type == "top_group":
                if isinstance(self.current_data, list) and len(self.current_data) == 1:
                    keys = [k for k in self.current_data[0].keys() if k != "agg_value"]
                    candidate = self.current_data[0].get(keys[0]) if keys else None
                else:
                    candidate = None
            elif goal_type == "list_rows":
                candidate = self.current_data
            else:
                candidate = None
            self.last_result = candidate
            if candidate == expected:
                return "Finalize: Success. Result matches objective."
            else:
                return f"Finalize: Failure. Result does not match objective. Candidate={candidate}"
        raise ValueError(f"Unknown tool: {tool_name}")

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        info: Dict[str, Any] = {"suffix": self.get_task_suffix()}
        parsed = self._parse_action(action)
        if parsed is None:
            obs = "Invalid action format. Use \\boxed{tool_name(param=value,...)}."
            return obs, LanguageGameReward.format_error_reward, True, False, info
        tool_name = parsed["tool"]
        args = parsed["args"]
        if tool_name not in self.tools:
            obs = f"Unsupported tool: {tool_name}. Episode terminated."
            return obs, -0.5, True, False, info
        try:
            result_text = self._execute_tool(tool_name, args)
            self.steps_taken += 1
            if tool_name == "finalize":
                if "Success" in result_text:
                    obs = f"{result_text}"
                    return obs, 1.0, True, False, info
                else:
                    obs = f"{result_text}"
                    return obs, 0.0, True, False, info
            if self.turn_count >= self.max_turns:
                obs = "Timeout: Reached max turns. Episode truncated."
                return obs, 0.0, True, True, info
            obs = f"Tool {tool_name} executed. {result_text}"
            return obs, 0.0, False, False, info
        except Exception as e:
            obs = f"Execution error: {str(e)}. Episode terminated."
            return obs, -0.5, True, False, info

    def sample_random_action(self) -> str:
        choices = []
        if self.loaded_source is None:
            choices.append("\\boxed{load(source=catalog_A)}")
        else:
            choices.extend([
                "\\boxed{filter(field=price,op=gt,value=100)}",
                "\\boxed{sort(field=price,order=desc)}",
                "\\boxed{head(n=3)}",
                "\\boxed{select(fields=id|name|price)}",
            ])
        return random.choice(choices)


class OpsKitComposerEnvWithFeedback(OpsKitComposerEnv):
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
            hint = "Wrap a single tool call in \\boxed{...} using tool_name(param=value,...)"
        elif "unsupported tool" in text:
            error_type = "UnsupportedAction"
            m = re.search(r"unsupported tool:\s*([a-z_]+)", text)
            error_detail["tool"] = m.group(1) if m else None
            hint = "Choose from the available tools listed in the instructions."
        elif "execution error" in text:
            if "protocol violation" in text:
                error_type = "ProtocolViolation"
                if "load a primary source before join_aux" in text:
                    error_detail["violation"] = "join_without_aux_or_load"
                    hint = "Call load(source=...) then load_aux(source=...) before join_aux."
                elif "load must be called before" in text:
                    error_detail["violation"] = "missing_load"
                    hint = "Start with load(source=...) to set current_data."
                else:
                    error_detail["violation"] = "invalid_sequence"
                    hint = "Ensure preconditions: load -> transform -> finalize."
            else:
                error_type = "UnsupportedAction"
                error_detail["issue"] = "execution_error"
                hint = "Check parameters and tool protocol; adjust argument types and order."
        elif "timeout" in text:
            error_type = "Timeout"
            error_detail["outcome"] = "max_turns_reached"
            hint = "Plan fewer steps; prioritize essential operations and finalize sooner."
        elif "finalize: failure" in text:
            error_type = "WrongDecision"
            error_detail["expected_type"] = self.task.get("goal_type")
            error_detail["candidate"] = getattr(self, "last_result", None)
            hint = "Recheck filters, joins, and aggregation fields; verify group_by and sort orders before finalizing."
        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = "Good job."
        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "loaded_source": getattr(self, "loaded_source", None),
                "steps_taken": getattr(self, "steps_taken", None),
                "goal_type": self.task.get("goal_type"),
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
            "hint": "Begin with load(source=...) using the base source listed, then apply filters or joins as needed.",
            "turn": 0,
        }
        return obs, info