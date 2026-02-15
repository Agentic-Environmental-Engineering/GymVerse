from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List, Union

class ToolLatticeEnv(Env):
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
                "description": "Load a base table into the current workspace.",
                "parameters": ["source"],
                "returns": "current_data updated",
            },
            "load_aux": {
                "description": "Load an auxiliary table for joining.",
                "parameters": ["source"],
                "returns": "aux_data updated",
            },
            "filter_rows": {
                "description": "Filter current_data with a condition like field==value or field>number.",
                "parameters": ["expr"],
                "returns": "current_data filtered",
            },
            "select_columns": {
                "description": "Project current_data onto a subset of columns.",
                "parameters": ["cols"],
                "returns": "current_data projected",
            },
            "join_on": {
                "description": "Join current_data with aux_data on keys.",
                "parameters": ["key_current", "key_aux", "how"],
                "returns": "current_data joined",
            },
            "distinct_by": {
                "description": "Remove duplicate rows by key tuple.",
                "parameters": ["fields"],
                "returns": "current_data deduplicated",
            },
            "sort_by": {
                "description": "Sort current_data by a field.",
                "parameters": ["field", "order"],
                "returns": "current_data sorted",
            },
            "compute_metric": {
                "description": "Compute metric over current_data: count/sum/avg/unique_count.",
                "parameters": ["metric", "field"],
                "returns": "result_value updated",
            },
            "submit_answer": {
                "description": "Submit the final numeric answer for evaluation.",
                "parameters": ["value"],
                "returns": "terminates episode",
            },
            "preview": {
                "description": "Preview head count and a few sample rows.",
                "parameters": [],
                "returns": "text",
            },
        }
        base_users = 30 + self.complexity * 10
        base_sales = 80 + self.complexity * 30
        base_inventory = 40 + self.complexity * 10
        base_events = 60 + self.complexity * 20

        regions = ["north", "south", "east", "west", "central"][:min(5, 2 + self.complexity // 3)]
        tiers = ["free", "silver", "gold", "platinum"][:min(4, 2 + self.complexity // 4)]
        categories = ["books", "electronics", "grocery", "toys", "beauty", "sports"][:min(6, 3 + self.complexity // 2)]
        products = [f"prod_{i}" for i in range(10 + self.complexity * 5)]
        days = list(range(1, 31))

        self.tables = {}
        users = []
        for i in range(base_users):
            users.append({
                "id": i + 1,
                "active": random.choice([True, False, True]),
                "region": random.choice(regions),
                "tier": random.choice(tiers),
                "age": random.randint(18, 70),
            })
        self.tables["users"] = users

        sales = []
        for i in range(base_sales):
            uid = random.choice(users)["id"]
            cat = random.choice(categories)
            prod = random.choice(products)
            amt = round(random.uniform(5, 250), 2)
            tax = round(amt * random.uniform(0.05, 0.2), 2)
            refund = random.choice([False, False, False, True])
            sales.append({
                "id": i + 1,
                "user_id": uid,
                "product": prod,
                "category": cat,
                "amount": amt,
                "tax": tax,
                "date": random.choice(days),
                "refund": refund,
            })
        self.tables["sales"] = sales

        inventory = []
        for p in products:
            inventory.append({
                "product": p,
                "category": random.choice(categories),
                "in_stock": random.choice([True, True, False]),
                "price": round(random.uniform(5, 300), 2),
            })
        self.tables["inventory"] = inventory

        events = []
        for i in range(base_events):
            uid = random.choice(users)["id"]
            events.append({
                "user_id": uid,
                "event_type": random.choice(["login", "purchase", "browse", "support"]),
                "timestamp": random.choice(days),
            })
        self.tables["events"] = events

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self.turn_count = 0
        self.steps_taken = 0
        self.required_steps = random.randint(self.min_required_steps, self.max_required_steps)
        self.current_data: Optional[List[Dict[str, Any]]] = None
        self.aux_data: Optional[List[Dict[str, Any]]] = None
        self.result_value: Optional[float] = None
        self.blueprint = self._generate_task_requiring_n_steps(self.required_steps)
        self.target_value = self._compute_target_from_blueprint(self.blueprint)
        obs = self._get_instructions()
        info = {"suffix": self.get_task_suffix()}
        return obs, info

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        if self.turn_count > self.max_turns:
            obs = f"Timeout: Max turns reached. Finalization not received. The hidden target value was {self._format_number(self.target_value)}."
            return obs, 0.0, True, True, {"turn": self.turn_count}

        parsed = self._parse_action(action)
        if parsed is None:
            return "ERROR: Invalid action format. Use \\boxed{tool|param=value;...}.", LanguageGameReward.format_error_reward, True, False, {"turn": self.turn_count}

        tool_name, args = parsed
        if tool_name not in self.tools:
            obs = f"ERROR: Unsupported tool '{tool_name}'."
            return obs, -0.25, True, False, {"turn": self.turn_count}

        try:
            if tool_name == "submit_answer":
                val_raw = args.get("value", None)
                if val_raw is None:
                    return "ERROR: Protocol violation: submit_answer requires 'value'.", -0.25, True, False, {"turn": self.turn_count}
                submitted = self._coerce_number(val_raw)
                if submitted is None:
                    return "ERROR: Invalid submission value; must be numeric.", -0.25, True, False, {"turn": self.turn_count}
                if self.steps_taken < self.required_steps:
                    obs = f"ERROR: Protocol violation: submitted before meeting minimum step requirement ({self.steps_taken}/{self.required_steps})."
                    return obs, -0.2, True, False, {"turn": self.turn_count}
                correct = self._numbers_equal(submitted, self.target_value)
                if correct:
                    return f"SUCCESS: Correct answer {self._format_number(submitted)}. Steps taken: {self.steps_taken}.", 1.0, True, False, {"turn": self.turn_count}
                else:
                    return f"Final answer incorrect. Submitted {self._format_number(submitted)}; expected {self._format_number(self.target_value)}.", 0.0, True, False, {"turn": self.turn_count}

            result_text = self._execute_tool(tool_name, args)
            self.steps_taken += 1
            obs = f"OK: {tool_name} executed. {result_text} Steps: {self.steps_taken}/{self.required_steps}."
            return obs, 0.0, False, False, {"turn": self.turn_count}
        except Exception as e:
            obs = f"ERROR: Execution error: {str(e)}."
            return obs, -0.5, False, False, {"turn": self.turn_count}

    def _get_instructions(self) -> str:
        lines = []
        lines.append("You must orchestrate tools to compute the specified metric from the provided datasets and then submit the final numeric answer.")
        lines.append(f"Minimum valid tool calls required before submission: {self.required_steps}.")
        lines.append("Action format: use \\boxed{tool|param1=value1;param2=value2}. Values with spaces must be quoted, e.g., 'north'.")
        lines.append("Available tools:")
        for name in sorted(self.tools.keys()):
            lines.append(f"- {name}: {self.tools[name]['description']}")
        return "\n".join(lines)

    def get_task_suffix(self) -> str:
        desc = self._describe_blueprint(self.blueprint)
        preview = self._dataset_preview()
        fmt = "Remember to use \\boxed{tool|param=value;...}."
        return f"\nTASK:\n{desc}\n\nDATA PREVIEW:\n{preview}\n\nFORMAT:\n{fmt}"

    def _parse_action(self, action: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        m = re.search(r"\\boxed\{(.*)\}", action.strip(), flags=re.DOTALL)
        if not m:
            return None
        inner = m.group(1).strip()
        if "|" not in inner:
            return None
        tool_name, param_str = inner.split("|", 1)
        tool_name = tool_name.strip()
        args: Dict[str, Any] = {}
        if param_str.strip() == "":
            return (tool_name, args)
        parts = [p for p in param_str.split(";") if p.strip() != ""]
        for p in parts:
            if "=" not in p:
                return None
            k, v = p.split("=", 1)
            k = k.strip()
            v = v.strip()
            v_clean = self._strip_quotes(v)
            if v_clean.lower() in ["true", "false"]:
                args[k] = v_clean.lower() == "true"
            else:
                num = self._coerce_number(v_clean)
                args[k] = num if num is not None else v_clean
        return (tool_name, args)

    def sample_random_action(self) -> str:
        candidates = [
            "\\boxed{load_table|source=sales}",
            "\\boxed{filter_rows|expr=category=='books'}",
            "\\boxed{compute_metric|metric=count;field=id}",
            "\\boxed{submit_answer|value=42}",
            "\\boxed{preview|}",
        ]
        return random.choice(candidates)

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        if tool_name == "load_table":
            src = args.get("source")
            if src not in self.tables:
                raise ValueError(f"Unknown table '{src}'.")
            self.current_data = [dict(row) for row in self.tables[src]]
            return f"Loaded '{src}' with {len(self.current_data)} rows."

        if tool_name == "load_aux":
            src = args.get("source")
            if src not in self.tables:
                raise ValueError(f"Unknown aux table '{src}'.")
            self.aux_data = [dict(row) for row in self.tables[src]]
            return f"Loaded aux '{src}' with {len(self.aux_data)} rows."

        if tool_name == "filter_rows":
            expr = args.get("expr")
            if self.current_data is None:
                raise ValueError("No data loaded. Call load_table first.")
            if not expr:
                raise ValueError("filter_rows requires 'expr'.")
            filtered = self._apply_condition(self.current_data, expr)
            self.current_data = filtered
            return f"Filtered to {len(self.current_data)} rows."

        if tool_name == "select_columns":
            cols = args.get("cols")
            if self.current_data is None:
                raise ValueError("No data loaded. Call load_table first.")
            if cols is None:
                raise ValueError("select_columns requires 'cols'.")
            col_list = [c.strip() for c in (cols.split(",") if isinstance(cols, str) else []) if c.strip() != ""]
            proj = []
            for row in self.current_data:
                proj.append({c: row.get(c) for c in col_list})
            self.current_data = proj
            return f"Selected columns {col_list}. Current rows: {len(self.current_data)}."

        if tool_name == "join_on":
            if self.current_data is None:
                raise ValueError("No data loaded. Call load_table first.")
            if self.aux_data is None:
                raise ValueError("No aux loaded. Call load_aux first.")
            k_curr = args.get("key_current")
            k_aux = args.get("key_aux")
            how = args.get("how", "inner")
            if not k_curr or not k_aux:
                raise ValueError("join_on requires 'key_current' and 'key_aux'.")
            index = {}
            for r in self.aux_data:
                index.setdefault(r.get(k_aux), []).append(r)
            joined = []
            if str(how).lower() == "left":
                for a in self.current_data:
                    matches = index.get(a.get(k_curr), [])
                    if matches:
                        for m in matches:
                            merged = dict(a)
                            for kk, vv in m.items():
                                merged[f"aux.{kk}"] = vv
                            joined.append(merged)
                    else:
                        joined.append(dict(a))
            else:
                for a in self.current_data:
                    matches = index.get(a.get(k_curr), [])
                    for m in matches:
                        merged = dict(a)
                        for kk, vv in m.items():
                            merged[f"aux.{kk}"] = vv
                        joined.append(merged)
            self.current_data = joined
            return f"Joined with aux on {k_curr}={k_aux} ({how}). Rows: {len(self.current_data)}."

        if tool_name == "distinct_by":
            if self.current_data is None:
                raise ValueError("No data loaded. Call load_table first.")
            fields = args.get("fields")
            if not fields:
                raise ValueError("distinct_by requires 'fields'.")
            flist = [f.strip() for f in (fields.split(",") if isinstance(fields, str) else []) if f.strip() != ""]
            seen = set()
            out = []
            for row in self.current_data:
                key = tuple(row.get(f) for f in flist)
                if key not in seen:
                    seen.add(key)
                    out.append(row)
            self.current_data = out
            return f"Distinct by {flist}. Rows: {len(self.current_data)}."

        if tool_name == "sort_by":
            if self.current_data is None:
                raise ValueError("No data loaded. Call load_table first.")
            field = args.get("field")
            order = str(args.get("order", "asc")).lower()
            if not field:
                raise ValueError("sort_by requires 'field'.")
            self.current_data.sort(key=lambda r: (r.get(field) is None, r.get(field)))
            if order == "desc":
                self.current_data.reverse()
            return f"Sorted by {field} ({order}). Rows: {len(self.current_data)}."

        if tool_name == "compute_metric":
            if self.current_data is None:
                raise ValueError("No data loaded. Call load_table first.")
            metric = str(args.get("metric", "count")).lower()
            field = args.get("field")
            if metric == "count":
                self.result_value = float(len(self.current_data))
            elif metric in ["sum", "avg", "unique_count"]:
                if not field:
                    raise ValueError(f"compute_metric '{metric}' requires 'field'.")
                vals = [row.get(field) for row in self.current_data]
                if metric == "sum":
                    nums = [self._coerce_number(v) for v in vals]
                    nums = [n for n in nums if n is not None]
                    self.result_value = float(sum(nums))
                elif metric == "avg":
                    nums = [self._coerce_number(v) for v in vals]
                    nums = [n for n in nums if n is not None]
                    self.result_value = float(sum(nums) / len(nums)) if nums else 0.0
                else:
                    uniq = set(vals)
                    self.result_value = float(len(uniq))
            else:
                raise ValueError(f"Unsupported metric '{metric}'.")
            return f"Computed metric '{metric}' -> {self._format_number(self.result_value)}."
        if tool_name == "preview":
            if self.current_data is None:
                return "No data loaded."
            sample = self.current_data[:3]
            return f"Rows={len(self.current_data)}; sample={sample}."
        raise ValueError(f"Tool not executable: {tool_name}")

    def _generate_task_requiring_n_steps(self, n: int) -> Dict[str, Any]:
        use_join = self.complexity >= 3 and random.choice([True, False, True])
        base = "sales" if random.random() < 0.7 else "events"
        filters = []

        if base == "sales":
            filters.append({"field": "refund", "op": "==", "value": False})
            if self.complexity >= 2:
                filters.append({"field": "category", "op": "==", "value": random.choice([r for r in set([row["category"] for row in self.tables["sales"]])])})
            if self.complexity >= 4:
                day_low = random.randint(1, 15)
                day_high = random.randint(day_low, 30)
                filters.append({"field": "date", "op": ">=", "value": day_low})
                filters.append({"field": "date", "op": "<=", "value": day_high})
        else:
            filters.append({"field": "event_type", "op": "==", "value": random.choice(["login", "purchase", "browse", "support"])})
            if self.complexity >= 2:
                filters.append({"field": "timestamp", "op": ">=", "value": random.randint(1, 20)})

        join = None
        if use_join:
            if base == "sales":
                join = {"aux": "users", "key_current": "user_id", "key_aux": "id", "how": random.choice(["inner", "left"])}
            else:
                join = {"aux": "users", "key_current": "user_id", "key_aux": "id", "how": random.choice(["inner", "left"])}
            if self.complexity >= 3:
                filters.append({"field": "aux.active", "op": "==", "value": True})
            if self.complexity >= 5:
                filters.append({"field": "aux.region", "op": "==", "value": random.choice([r for r in set([row["region"] for row in self.tables["users"]])])})

        metric = random.choice(["count", "sum", "avg"])
        metric_field = None
        if metric in ["sum", "avg"]:
            metric_field = "amount" if base == "sales" else "timestamp"

        neutral_ops = 0
        pipeline_core = 1 + len(filters) + (2 if join else 0) + 1 + 1  # load + filters + (load_aux+join) + compute_metric + submit
        if pipeline_core < n:
            neutral_ops = n - pipeline_core

        return {
            "base_table": base,
            "filters": filters,
            "join": join,
            "metric": metric,
            "metric_field": metric_field,
            "neutral_ops": neutral_ops,
            "required_steps": n,
        }

    def _compute_target_from_blueprint(self, bp: Dict[str, Any]) -> float:
        data = [dict(r) for r in self.tables[bp["base_table"]]]
        if bp.get("join"):
            aux = [dict(r) for r in self.tables[bp["join"]["aux"]]]
            idx = {}
            for r in aux:
                idx.setdefault(r.get(bp["join"]["key_aux"]), []).append(r)
            merged = []
            for a in data:
                matches = idx.get(a.get(bp["join"]["key_current"]), [])
                if bp["join"]["how"] == "left":
                    if matches:
                        for m in matches:
                            merged.append(self._merge_rows(a, m))
                    else:
                        merged.append(dict(a))
                else:
                    for m in matches:
                        merged.append(self._merge_rows(a, m))
            data = merged
        for cond in bp["filters"]:
            data = self._apply_condition(data, self._cond_to_expr(cond))
        metric = bp["metric"]
        field = bp.get("metric_field")
        if metric == "count":
            return float(len(data))
        vals = [self._coerce_number(row.get(field)) for row in data]
        vals = [v for v in vals if v is not None]
        if metric == "sum":
            return float(sum(vals))
        if metric == "avg":
            return float(sum(vals) / len(vals)) if vals else 0.0
        return 0.0

    def _describe_blueprint(self, bp: Dict[str, Any]) -> str:
        parts = []
        parts.append(f"Goal: Compute {bp['metric'].upper()}" + (f" of '{bp['metric_field']}'" if bp['metric_field'] else "") + f" from '{bp['base_table']}'.")
        if bp.get("join"):
            parts.append(f"Join with '{bp['join']['aux']}' on {bp['join']['key_current']}={bp['join']['key_aux']} ({bp['join']['how']}).")
        if bp["filters"]:
            parts.append("Apply filters:")
            for c in bp["filters"]:
                parts.append(f"- {c['field']} {c['op']} {repr(c['value'])}")
        if bp["neutral_ops"] > 0:
            parts.append(f"Include {bp['neutral_ops']} neutral operations (e.g., select_columns or sort_by) before submission.")
        parts.append(f"Minimum tool calls before submission: {bp['required_steps']}.")
        return "\n".join(parts)

    def _dataset_preview(self) -> str:
        previews = []
        for name in ["users", "sales", "inventory", "events"]:
            t = self.tables[name]
            previews.append(f"{name}: rows={len(t)}; sample={t[:2]}")
        return "\n".join(previews)

    def _apply_condition(self, data: List[Dict[str, Any]], expr: str) -> List[Dict[str, Any]]:
        m = re.match(r"(.+?)\s*(==|!=|>=|<=|>|<|in|notin)\s*(.+)", expr)
        if not m:
            raise ValueError(f"Unsupported filter expr '{expr}'.")
        field, op, raw = m.group(1).strip(), m.group(2), m.group(3).strip()
        val = self._strip_quotes(raw)
        comp = val
        comp_num = self._coerce_number(val)
        def get_field(row, name):
            if name.startswith("aux."):
                return row.get(name)
            return row.get(name)
        out = []
        for r in data:
            lhs = get_field(r, field)
            keep = False
            if op == "==":
                keep = lhs == (comp_num if comp_num is not None else comp if comp not in ["true", "false"] else (comp.lower() == "true"))
            elif op == "!=":
                keep = lhs != (comp_num if comp_num is not None else comp if comp not in ["true", "false"] else (comp.lower() == "true"))
            elif op in [">", "<", ">=", "<="]:
                ln = self._coerce_number(lhs)
                rn = comp_num
                if ln is None or rn is None:
                    keep = False
                else:
                    if op == ">": keep = ln > rn
                    elif op == "<": keep = ln < rn
                    elif op == ">=": keep = ln >= rn
                    elif op == "<=": keep = ln <= rn
            elif op == "in":
                vals = [v.strip() for v in val.split(",")]
                keep = str(lhs) in vals
            elif op == "notin":
                vals = [v.strip() for v in val.split(",")]
                keep = str(lhs) not in vals
            if keep:
                out.append(r)
        return out

    def _merge_rows(self, base: Dict[str, Any], aux: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base)
        for kk, vv in aux.items():
            merged[f"aux.{kk}"] = vv
        return merged

    def _cond_to_expr(self, cond: Dict[str, Any]) -> str:
        v = cond["value"]
        if isinstance(v, str):
            return f"{cond['field']} {cond['op']} '{v}'"
        if isinstance(v, bool):
            return f"{cond['field']} {cond['op']} {str(v)}"
        return f"{cond['field']} {cond['op']} {v}"

    def _strip_quotes(self, s: str) -> str:
        s = s.strip()
        if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
            return s[1:-1]
        return s

    def _coerce_number(self, v: Any) -> Optional[float]:
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                if "." in v:
                    return float(v)
                return float(int(v))
            except:
                return None
        return None

    def _numbers_equal(self, a: float, b: float) -> bool:
        return abs(a - b) <= 1e-6

    def _format_number(self, v: Union[int, float]) -> str:
        if abs(float(v) - int(v)) < 1e-6:
            return str(int(v))
        return f"{float(v):.6f}"


class ToolLatticeEnvWithFeedback(ToolLatticeEnv):
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
            hint = "Use \\boxed{tool|param=value;...} and include required parameters."

        elif "unsupported tool" in text:
            error_type = "UnsupportedAction"
            m = re.search(r"unsupported tool '([^']+)'", obs, flags=re.IGNORECASE)
            if m:
                error_detail["tool"] = m.group(1)
            hint = "Choose from available tools listed in the instructions."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "submitted before meeting minimum" in text:
                error_detail["violation"] = "early_submission"
                error_detail["steps_taken"] = getattr(self, "steps_taken", None)
                error_detail["required_steps"] = getattr(self, "required_steps", None)
                hint = "Execute more tools relevant to the task (filters, joins, previews) before submit_answer."
            elif "no data loaded" in text:
                error_detail["violation"] = "missing_load"
                hint = "Start with load_table|source=<table>."
            elif "no aux loaded" in text:
                error_detail["violation"] = "missing_aux"
                hint = "Call load_aux|source=<table> before join_on."
            else:
                error_detail["violation"] = "general_protocol_error"
                hint = "Ensure prerequisites (load, aux load, computed metric) are satisfied."

        elif "execution error" in text:
            error_type = "ExecutionError"
            m = re.search(r"execution error: (.+)\.", obs, flags=re.IGNORECASE)
            if m:
                error_detail["message"] = m.group(1)
            hint = "Check parameter names and values; preview data to verify fields."

        elif "final answer incorrect" in text:
            error_type = "WrongDecision"
            got_m = re.search(r"submitted ([\d\.\-]+)", obs, flags=re.IGNORECASE)
            exp_m = re.search(r"expected ([\d\.\-]+)", obs, flags=re.IGNORECASE)
            if got_m:
                error_detail["got"] = got_m.group(1)
            if exp_m:
                error_detail["expected"] = exp_m.group(1)
            hint = "Re-examine filters and joins; recompute the metric using compute_metric and preview intermediate results."

        elif "timeout: max turns" in text:
            error_type = "Timeout"
            hint = "Plan your sequence and submit_answer before reaching max turns."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = "Good job."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            error_detail["turn"] = getattr(self, "turn_count", None)
            error_detail["steps_taken"] = getattr(self, "steps_taken", None)
            error_detail["required_steps"] = getattr(self, "required_steps", None)
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
                "required_steps": getattr(self, "required_steps", None),
            },
            "hint": "Start with load_table|source=<base_table> as described, preview, then apply filters and join if needed.",
        }
        return obs, info