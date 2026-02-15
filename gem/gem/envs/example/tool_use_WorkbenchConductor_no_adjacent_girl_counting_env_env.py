from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional

class WorkbenchConductorEnv(Env):
    def __init__(self, complexity: int = 1, max_turns: Optional[int] = 100, **_):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.max_turns = max_turns if max_turns is not None else 100
        self.min_required_steps = self.complexity
        self.max_required_steps = self.complexity * 2
        self._init_database()
        self.reset()

    def _init_database(self):
        random.seed(42)
        self.currency_rates = {"USD": 1.0, "EUR": 1.1, "JPY": 0.009, "GBP": 1.25}
        categories = ["Gadgets", "Home", "Outdoors", "Books", "Toys", "Kitchen"]
        regions = ["NA", "EU", "JP", "UK"]
        num_products = 12 + (self.complexity - 1) * 6
        products = []
        for i in range(num_products):
            pid = f"P{i+1:03d}"
            products.append({
                "id": pid,
                "name": f"Item-{i+1}",
                "category": random.choice(categories),
                "price": round(random.uniform(5.0, 500.0), 2),
                "currency": random.choice(list(self.currency_rates.keys())),
                "rating": round(random.uniform(1.0, 5.0), 2)
            })
        sales = []
        for p in products:
            t = random.randint(1, 3)
            for _ in range(t):
                sales.append({
                    "product_id": p["id"],
                    "quantity": random.randint(1, 20),
                    "region": random.choice(regions)
                })
        self.datasets = {"products": products, "sales": sales}
        self.tools = {
            "load": {
                "description": "Load a dataset into the active workspace.",
                "parameters": [{"name": "dataset", "type": "string", "choices": list(self.datasets.keys())}],
                "returns": "Active dataset loaded"
            },
            "save_as": {
                "description": "Save the current active dataset under a name.",
                "parameters": [{"name": "name", "type": "string"}],
                "returns": "Dataset saved to workspace"
            },
            "switch_to": {
                "description": "Switch active dataset to a saved workspace dataset.",
                "parameters": [{"name": "name", "type": "string"}],
                "returns": "Active dataset changed"
            },
            "join_with": {
                "description": "Join the active dataset with a saved dataset by matching keys.",
                "parameters": [
                    {"name": "name", "type": "string"},
                    {"name": "on_left", "type": "string"},
                    {"name": "on_right", "type": "string"}
                ],
                "returns": "Active dataset becomes joined dataset"
            },
            "filter": {
                "description": "Filter rows by a condition field/operator/value.",
                "parameters": [
                    {"name": "field", "type": "string"},
                    {"name": "op", "type": "string", "choices": ["eq", "neq", "gt", "lt", "contains"]},
                    {"name": "value", "type": "any"}
                ],
                "returns": "Active dataset filtered"
            },
            "select": {
                "description": "Project the active dataset to a subset of fields.",
                "parameters": [{"name": "fields", "type": "string_list"}],
                "returns": "Active dataset with fewer fields"
            },
            "convert_to_usd": {
                "description": "Convert a numeric field to USD using currency in a currency_field per row.",
                "parameters": [
                    {"name": "field", "type": "string"},
                    {"name": "currency_field", "type": "string"}
                ],
                "returns": "Active dataset updated with field converted to USD"
            },
            "sort_by": {
                "description": "Sort the active dataset by a field.",
                "parameters": [
                    {"name": "field", "type": "string"},
                    {"name": "order", "type": "string", "choices": ["asc", "desc"]}
                ],
                "returns": "Active dataset sorted"
            },
            "take": {
                "description": "Take the first n rows of the active dataset.",
                "parameters": [{"name": "n", "type": "int"}],
                "returns": "Active dataset truncated to n rows"
            },
            "aggregate": {
                "description": "Aggregate a field with an operation.",
                "parameters": [
                    {"name": "op", "type": "string", "choices": ["sum", "avg", "max", "min", "count"]},
                    {"name": "field", "type": "string"}
                ],
                "returns": "Scalar result"
            },
            "count_rows": {
                "description": "Count rows in the active dataset.",
                "parameters": [],
                "returns": "Scalar count"
            },
            "inspect_preview": {
                "description": "Preview first few rows.",
                "parameters": [],
                "returns": "Textual preview"
            },
            "submit_answer": {
                "description": "Submit the final numeric answer.",
                "parameters": [{"name": "value", "type": "number"}],
                "returns": "Ends episode with success/failure"
            }
        }
        self.task = None
        self.active_data = None
        self.named_datasets = {}
        self.last_scalar = None
        self.turn_count = 0
        self.steps_taken = 0

    def _get_instructions(self) -> str:
        tool_list = ", ".join(sorted(self.tools.keys()))
        return (
            "You are orchestrating tools to compute a final numeric answer. "
            "Use one tool per step in the format \\boxed{tool_name param=value param2=value}. "
            "Available tools: " + tool_list + ". "
            "Parameters: strings without spaces or put in quotes, numbers as plain numerals. "
            "Do not explain; only respond with a single boxed tool call each step."
        )

    def get_task_suffix(self) -> str:
        status = "no active dataset" if self.active_data is None else f"{len(self.active_data)} rows, fields: {self._fields_summary(self.active_data)}"
        return (
            f"Objective: {self.task['objective_text']}\n"
            f"Workspace: active={status}; saved={list(self.named_datasets.keys())}\n"
            "Format: respond with \\boxed{tool_name param=value ...} only."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        required_steps = random.randint(self.min_required_steps, self.max_required_steps)
        required_steps = max(2, required_steps)
        self.task = self._generate_task_requiring_n_steps(required_steps)
        self.active_data = None
        self.named_datasets = {}
        self.last_scalar = None
        self.turn_count = 0
        self.steps_taken = 0
        obs = self._get_instructions()
        info = {"suffix": self.get_task_suffix()}
        return obs, info

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        info: Dict[str, Any] = {}
        if self.turn_count > self.max_turns:
            obs = "Timeout: maximum turns reached."
            return obs, 0.0, True, True, info
        parsed = self._parse_action(action)
        if not parsed:
            obs = "Invalid action format: use \\boxed{tool_name param=value ...}."
            return obs, LanguageGameReward.format_error_reward, True, False, info
        tool, args = parsed
        if tool not in self.tools:
            obs = f"Unsupported tool: {tool}."
            return obs, LanguageGameReward.format_error_reward, True, False, info
        try:
            result_text, is_scalar, scalar_value = self._execute_tool(tool, args)
            terminated = False
            truncated = False
            reward = 0.0
            if tool == "submit_answer":
                expected = self.task["expected_answer"]
                val = scalar_value
                ok = self._compare_values(val, expected)
                if ok:
                    obs = f"Success: correct answer! Expected={expected}, submitted={val}."
                    reward = 1.0
                else:
                    obs = f"Wrong answer: expected={expected}, submitted={val}."
                    reward = 0.0
                terminated = True
            else:
                self.steps_taken += 1
                obs = (
                    f"Executed {tool}. {result_text}\n"
                    f"Progress: steps_taken={self.steps_taken}; turns={self.turn_count}/{self.max_turns}\n"
                    f"{self.get_task_suffix()}"
                )
            return obs, reward, terminated, truncated, info
        except ValueError as e:
            obs = f"Protocol violation: {str(e)}."
            return obs, -0.2, True, False, info
        except Exception as e:
            obs = f"Protocol violation: {str(e)}."
            return obs, -0.2, True, False, info

    def _parse_action(self, action: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        m = re.search(r"\\boxed\{(.+?)\}", action, flags=re.DOTALL)
        if not m:
            return None
        content = m.group(1).strip()
        if not content:
            return None
        parts = content.split()
        tool = parts[0]
        params = " ".join(parts[1:])
        args: Dict[str, Any] = {}
        if params:
            kvs = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=('[^']*'|\"[^\"]*\"|[^\s]+)", params)
            for k, v in kvs:
                v = v.strip()
                if (v.startswith("'") and v.endswith("'")) or (v.startswith("\"") and v.endswith("\"")):
                    v_clean = v[1:-1]
                else:
                    if re.fullmatch(r"-?\d+\.\d+", v):
                        v_clean = float(v)
                    elif re.fullmatch(r"-?\d+", v):
                        v_clean = int(v)
                    elif "," in v:
                        v_clean = [s.strip() for s in v.split(",")]
                    else:
                        v_clean = v
                args[k] = v_clean
        return tool, args

    def sample_random_action(self) -> str:
        choices = [
            "\\boxed{load dataset=products}",
            "\\boxed{filter field=category op=eq value='Gadgets'}",
            "\\boxed{convert_to_usd field=price currency_field=currency}",
            "\\boxed{aggregate op=avg field=price}",
            "\\boxed{submit_answer value=123.45}"
        ]
        return random.choice(choices)

    def _execute_tool(self, tool: str, args: Dict[str, Any]) -> Tuple[str, bool, Optional[float]]:
        if tool == "load":
            dataset = args.get("dataset")
            if dataset not in self.datasets:
                raise ValueError(f"dataset '{dataset}' not found")
            self.active_data = [dict(row) for row in self.datasets[dataset]]
            return f"Loaded '{dataset}' with {len(self.active_data)} rows.", False, None
        if tool == "save_as":
            if self.active_data is None:
                raise ValueError("no active dataset to save")
            name = str(args.get("name"))
            if not name:
                raise ValueError("missing 'name'")
            self.named_datasets[name] = [dict(row) for row in self.active_data]
            return f"Saved active dataset as '{name}'.", False, None
        if tool == "switch_to":
            name = str(args.get("name"))
            if name not in self.named_datasets:
                raise ValueError(f"saved dataset '{name}' not found")
            self.active_data = [dict(row) for row in self.named_datasets[name]]
            return f"Switched to saved dataset '{name}'.", False, None
        if tool == "join_with":
            if self.active_data is None:
                raise ValueError("no active dataset to join")
            name = str(args.get("name"))
            if name not in self.named_datasets:
                raise ValueError(f"saved dataset '{name}' not found")
            left_key = str(args.get("on_left"))
            right_key = str(args.get("on_right"))
            right_data = self.named_datasets[name]
            index = {}
            for r in right_data:
                rk = r.get(right_key)
                index.setdefault(rk, []).append(r)
            joined = []
            for l in self.active_data:
                lk = l.get(left_key)
                if lk in index:
                    for r in index[lk]:
                        merged = dict(l)
                        for k, v in r.items():
                            merged[f"{name}.{k}"] = v
                        joined.append(merged)
            self.active_data = joined
            return f"Joined with '{name}' on {left_key}={right_key}. Rows={len(joined)}.", False, None
        if tool == "filter":
            if self.active_data is None:
                raise ValueError("no active dataset to filter")
            field = str(args.get("field"))
            op = str(args.get("op"))
            value = args.get("value")
            if op not in ["eq", "neq", "gt", "lt", "contains"]:
                raise ValueError("unsupported op")
            def ok(row):
                rv = row.get(field, None)
                if op == "eq":
                    return rv == value
                if op == "neq":
                    return rv != value
                if op == "gt":
                    try:
                        return float(rv) > float(value)
                    except:
                        return False
                if op == "lt":
                    try:
                        return float(rv) < float(value)
                    except:
                        return False
                if op == "contains":
                    return (isinstance(rv, str) and isinstance(value, str) and value in rv)
                return False
            filtered = [r for r in self.active_data if ok(r)]
            self.active_data = filtered
            return f"Filtered to {len(filtered)} rows on {field} {op} {value}.", False, None
        if tool == "select":
            if self.active_data is None:
                raise ValueError("no active dataset to select from")
            fields = args.get("fields")
            if isinstance(fields, str):
                flist = [s.strip() for s in fields.split(",")]
            else:
                flist = list(fields) if fields else []
            projected = []
            for r in self.active_data:
                projected.append({k: r.get(k) for k in flist})
            self.active_data = projected
            return f"Selected fields {flist}. Rows={len(projected)}.", False, None
        if tool == "convert_to_usd":
            if self.active_data is None:
                raise ValueError("no active dataset to convert")
            fld = str(args.get("field"))
            cfield = str(args.get("currency_field"))
            converted = 0
            for r in self.active_data:
                cur = r.get(cfield)
                rate = self.currency_rates.get(cur)
                if rate is None:
                    raise ValueError(f"currency '{cur}' not supported")
                try:
                    val = float(r.get(fld))
                except:
                    raise ValueError(f"field '{fld}' not numeric")
                r[fld] = round(val * rate, 4)
                r[cfield] = "USD"
                converted += 1
            return f"Converted field '{fld}' to USD for {converted} rows.", False, None
        if tool == "sort_by":
            if self.active_data is None:
                raise ValueError("no active dataset to sort")
            fld = str(args.get("field"))
            order = str(args.get("order"))
            rev = order == "desc"
            try:
                self.active_data = sorted(self.active_data, key=lambda r: r.get(fld), reverse=rev)
            except Exception:
                raise ValueError("sorting failed; check field and types")
            return f"Sorted by {fld} {order}.", False, None
        if tool == "take":
            if self.active_data is None:
                raise ValueError("no active dataset to take from")
            n = int(args.get("n"))
            self.active_data = self.active_data[:max(0, n)]
            return f"Took first {n} rows.", False, None
        if tool == "aggregate":
            if self.active_data is None:
                raise ValueError("no active dataset to aggregate")
            op = str(args.get("op"))
            fld = str(args.get("field"))
            if op not in ["sum", "avg", "max", "min", "count"]:
                raise ValueError("unsupported aggregate op")
            vals = []
            if op == "count":
                res = len(self.active_data)
            else:
                for r in self.active_data:
                    v = r.get(fld)
                    if isinstance(v, (int, float)):
                        vals.append(float(v))
                if not vals:
                    raise ValueError("no numeric values for aggregate")
                if op == "sum":
                    res = sum(vals)
                elif op == "avg":
                    res = sum(vals) / len(vals)
                elif op == "max":
                    res = max(vals)
                elif op == "min":
                    res = min(vals)
                else:
                    res = 0.0
            self.last_scalar = round(float(res), 6)
            return f"Aggregate {op}({fld}) = {self.last_scalar}.", True, self.last_scalar
        if tool == "count_rows":
            if self.active_data is None:
                raise ValueError("no active dataset")
            cnt = len(self.active_data)
            self.last_scalar = float(cnt)
            return f"Row count = {cnt}.", True, self.last_scalar
        if tool == "inspect_preview":
            if self.active_data is None:
                raise ValueError("no active dataset")
            preview = self.active_data[:3]
            return f"Preview: {preview}", False, None
        if tool == "submit_answer":
            value = args.get("value")
            try:
                val = float(value)
            except:
                raise ValueError("submitted value must be numeric")
            return f"Submitted {val}.", True, val
        raise ValueError("unknown tool")

    def _fields_summary(self, data):
        if not data:
            return "none"
        return ", ".join(sorted(list(data[0].keys())))

    def _compare_values(self, a: float, b: float) -> bool:
        if a is None or b is None:
            return False
        return abs(float(a) - float(b)) <= 1e-6

    def _generate_task_requiring_n_steps(self, required_steps: int) -> Dict[str, Any]:
        # Ensure at least 2 steps including submit
        rs = max(2, required_steps)
        # Choose template by complexity
        if rs <= 5:
            # Simple average USD price for a category with optional filters/sorts to hit step count
            cat = random.choice(list({p["category"] for p in self.datasets["products"]}))
            threshold = round(random.uniform(2.0, 4.5), 2)
            k = random.randint(3, 7)
            plan = [{"tool": "load", "args": {"dataset": "products"}},
                    {"tool": "convert_to_usd", "args": {"field": "price", "currency_field": "currency"}},
                    {"tool": "filter", "args": {"field": "category", "op": "eq", "value": cat}}]
            optional = [
                {"tool": "filter", "args": {"field": "rating", "op": "gt", "value": threshold}},
                {"tool": "sort_by", "args": {"field": "rating", "order": "desc"}},
                {"tool": "take", "args": {"n": k}},
            ]
            idx = 0
            while len(plan) + 2 < rs and idx < len(optional):
                plan.append(optional[idx])
                idx += 1
            plan.append({"tool": "aggregate", "args": {"op": "avg", "field": "price"}})
            expected = self._compute_expected_answer(plan)
            objective_text = f"Compute the average USD price of products in category '{cat}'"
            if any(step["tool"] == "filter" and step["args"].get("field") == "rating" for step in plan):
                objective_text += f" with rating > {threshold}"
            if any(step["tool"] == "sort_by" for step in plan):
                objective_text += " among the top-rated selection"
            return {
                "required_steps": rs,
                "plan": plan,
                "expected_answer": expected,
                "objective_text": objective_text + ". Submit the numeric average."
            }
        else:
            # Join sales and products, filter by region, sum quantity for a category, using USD conversion pre-join
            cat = random.choice(list({p["category"] for p in self.datasets["products"]}))
            region = random.choice(["NA", "EU", "JP", "UK"])
            plan = [{"tool": "load", "args": {"dataset": "sales"}},
                    {"tool": "save_as", "args": {"name": "sales"}},
                    {"tool": "load", "args": {"dataset": "products"}},
                    {"tool": "filter", "args": {"field": "category", "op": "eq", "value": cat}},
                    {"tool": "convert_to_usd", "args": {"field": "price", "currency_field": "currency"}},
                    {"tool": "join_with", "args": {"name": "sales", "on_left": "id", "on_right": "product_id"}},
                    {"tool": "filter", "args": {"field": "sales.region", "op": "eq", "value": region}}]
            optional = [
                {"tool": "select", "args": {"fields": ["id", "name", "category", "sales.quantity"]}},
                {"tool": "sort_by", "args": {"field": "sales.quantity", "order": "desc"}},
                {"tool": "take", "args": {"n": random.randint(5, 10)}},
            ]
            idx = 0
            while len(plan) + 2 < rs and idx < len(optional):
                plan.append(optional[idx])
                idx += 1
            plan.append({"tool": "aggregate", "args": {"op": "sum", "field": "sales.quantity"}})
            expected = self._compute_expected_answer(plan)
            objective_text = f"Sum the quantity of sales in region '{region}' for products in category '{cat}' after joining sales and products"
            return {
                "required_steps": rs,
                "plan": plan,
                "expected_answer": expected,
                "objective_text": objective_text + ". Submit the numeric sum."
            }

    def _compute_expected_answer(self, plan: list) -> float:
        active = None
        saved: Dict[str, list] = {}
        last_val: Optional[float] = None
        for step in plan:
            t = step["tool"]
            a = step["args"]
            if t == "load":
                ds = a["dataset"]
                active = [dict(r) for r in self.datasets[ds]]
            elif t == "save_as":
                if active is None:
                    raise ValueError("plan invalid: save without active")
                saved[a["name"]] = [dict(r) for r in active]
            elif t == "switch_to":
                active = [dict(r) for r in saved[a["name"]]]
            elif t == "join_with":
                if active is None:
                    raise ValueError("plan invalid: join without active")
                name = a["name"]
                left_key = a["on_left"]
                right_key = a["on_right"]
                right = saved[name]
                idx = {}
                for r in right:
                    rk = r.get(right_key)
                    idx.setdefault(rk, []).append(r)
                joined = []
                for l in active:
                    lk = l.get(left_key)
                    if lk in idx:
                        for r in idx[lk]:
                            merged = dict(l)
                            for k, v in r.items():
                                merged[f"{name}.{k}"] = v
                            joined.append(merged)
                active = joined
            elif t == "filter":
                field = a["field"]
                op = a["op"]
                value = a["value"]
                def ok(row):
                    rv = row.get(field)
                    if op == "eq":
                        return rv == value
                    if op == "neq":
                        return rv != value
                    if op == "gt":
                        try:
                            return float(rv) > float(value)
                        except:
                            return False
                    if op == "lt":
                        try:
                            return float(rv) < float(value)
                        except:
                            return False
                    if op == "contains":
                        return isinstance(rv, str) and isinstance(value, str) and value in rv
                    return False
                active = [r for r in active if ok(r)]
            elif t == "select":
                fields = a["fields"]
                if isinstance(fields, str):
                    flist = [s.strip() for s in fields.split(",")]
                else:
                    flist = list(fields)
                active = [{k: r.get(k) for k in flist} for r in active]
            elif t == "convert_to_usd":
                fld = a["field"]
                cfield = a["currency_field"]
                for r in active:
                    rate = self.currency_rates.get(r.get(cfield))
                    r[fld] = round(float(r.get(fld)) * rate, 4)
                    r[cfield] = "USD"
            elif t == "sort_by":
                fld = a["field"]
                order = a["order"]
                active = sorted(active, key=lambda r: r.get(fld), reverse=(order == "desc"))
            elif t == "take":
                n = a["n"]
                active = active[:max(0, n)]
            elif t == "aggregate":
                op = a["op"]
                fld = a["field"]
                if op == "count":
                    last_val = float(len(active))
                else:
                    vals = []
                    for r in active:
                        v = r.get(fld)
                        if isinstance(v, (int, float)):
                            vals.append(float(v))
                    if not vals:
                        last_val = 0.0
                    else:
                        if op == "sum":
                            last_val = sum(vals)
                        elif op == "avg":
                            last_val = sum(vals) / len(vals)
                        elif op == "max":
                            last_val = max(vals)
                        elif op == "min":
                            last_val = min(vals)
                        else:
                            last_val = 0.0
            elif t == "count_rows":
                last_val = float(len(active))
            elif t == "inspect_preview":
                pass
        return round(float(last_val if last_val is not None else 0.0), 6)


class WorkbenchConductorEnvWithFeedback(WorkbenchConductorEnv):
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
            error_detail["issue"] = "missing_or_malformed_boxed"
            hint = "Respond with exactly one tool call, e.g., \\boxed{load dataset=products}"
        elif "unsupported tool" in text:
            error_type = "UnsupportedAction"
            m = re.search(r"unsupported tool: ([a-z_]+)", obs, flags=re.IGNORECASE)
            error_detail["tool"] = m.group(1) if m else None
            hint = "Choose a valid tool: " + ", ".join(sorted(self.tools.keys()))
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "no active dataset" in text:
                error_detail["violation"] = "missing_load"
                hint = "Start with \\boxed{load dataset=...} before filtering or aggregating."
            elif "saved dataset" in text and "not found" in text:
                error_detail["violation"] = "missing_save_as"
                hint = "Use \\boxed{save_as name=...} before joining or switching."
            else:
                error_detail["violation"] = "invalid_parameters_or_state"
                hint = "Check parameter names and required preconditions for the chosen tool."
        elif "timeout" in text:
            error_type = "Timeout"
            error_detail["outcome"] = "max_turns_reached"
            hint = "Plan fewer steps or prioritize the key operations before submitting."
        elif "wrong answer" in text:
            error_type = "WrongDecision"
            exp = self.task["expected_answer"]
            got_m = re.search(r"submitted=([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", obs)
            got = float(got_m.group(1)) if got_m else None
            error_detail["expected"] = exp
            error_detail["got"] = got
            hint = "Use aggregate tools on the correctly filtered/joined data, then submit that scalar."
        elif "success: correct answer" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None
        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["workspace"] = {
                "active_rows": None if self.active_data is None else len(self.active_data),
                "saved": list(self.named_datasets.keys())
            }
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint
        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        first_hint = "Begin with \\boxed{load dataset=products} or \\boxed{load dataset=sales} depending on the objective."
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": first_hint if self.feedback_level >= 2 else None,
            "turn": 0,
            "workspace": {"active_rows": None, "saved": []}
        }
        return obs, info
