from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List

class OpsForgeOrchestratorEnv(Env):
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
            "open_dataset": {
                "description": "Load a dataset into the active view.",
                "parameters": [{"name": "name", "type": "string"}],
                "returns": "Sets current view to the dataset rows."
            },
            "select_columns": {
                "description": "Restrict the active view to certain columns.",
                "parameters": [{"name": "columns", "type": "list[str]"}],
                "returns": "Updates current view with the selected columns."
            },
            "filter_rows": {
                "description": "Filter rows by a condition: col op value; ops: >,<,>=,<=,==,!=,contains.",
                "parameters": [{"name": "condition", "type": "string"}],
                "returns": "Updates current view after filtering."
            },
            "join_with": {
                "description": "Inner-join the active view with another dataset on a key.",
                "parameters": [{"name": "name", "type": "string"}, {"name": "on", "type": "string"}],
                "returns": "Updates current view with joined rows."
            },
            "compute_metric": {
                "description": "Compute a metric over a column: count, sum, mean, max, min, unique_count.",
                "parameters": [{"name": "metric", "type": "string"}, {"name": "column", "type": "string"}],
                "returns": "Stores metric in execution state."
            },
            "sort_by": {
                "description": "Sort the active view by a column in asc or desc order.",
                "parameters": [{"name": "column", "type": "string"}, {"name": "order", "type": "string"}],
                "returns": "Updates current view ordering."
            },
            "limit_rows": {
                "description": "Limit the number of rows in the active view.",
                "parameters": [{"name": "n", "type": "int"}],
                "returns": "Truncates the active view."
            },
            "save_view": {
                "description": "Save the current view under an alias.",
                "parameters": [{"name": "alias", "type": "string"}],
                "returns": "Stores the view for later."
            },
            "switch_view": {
                "description": "Switch to a previously saved view.",
                "parameters": [{"name": "alias", "type": "string"}],
                "returns": "Makes the aliased view active."
            },
            "submit_report": {
                "description": "Submit the final report for evaluation.",
                "parameters": [{"name": "title", "type": "string"}],
                "returns": "Triggers evaluation of requirements."
            },
        }
        base_rows = 20 + self.complexity * 5
        regions = ["North", "South", "East", "West"]
        if self.complexity >= 7:
            regions += ["Central"]
        items = ["Widget", "Gadget", "Doodad", "Thingamajig"]
        if self.complexity >= 5:
            items += ["Contraption"]
        def gen_sales(name):
            rows = []
            for i in range(base_rows):
                cid = random.randint(1000, 1020 + self.complexity)
                item = random.choice(items)
                qty = random.randint(1, 8 + self.complexity // 2)
                price = round(random.uniform(5, 120), 2)
                region = random.choice(regions)
                rows.append({
                    "sale_id": f"{name}_{i+1}",
                    "customer_id": cid,
                    "item": item,
                    "qty": qty,
                    "price": price,
                    "region": region
                })
            return rows
        sales_names = ["sales_jan", "sales_feb"]
        if self.complexity >= 5:
            sales_names.append("sales_mar")
        self.datasets = {}
        for sn in sales_names:
            self.datasets[sn] = gen_sales(sn)
        cust_rows = []
        segments = ["retail", "enterprise", "vip"]
        for cid in range(1000, 1050 + self.complexity):
            seg = random.choice(segments)
            loyalty = round(random.uniform(0, 1), 3)
            city = random.choice(["Springfield", "Shelbyville", "Ogdenville", "North Haverbrook"])
            cust_rows.append({
                "id": cid,
                "segment": seg,
                "city": city,
                "loyalty_score": loyalty
            })
        self.datasets["customers"] = cust_rows
        inv_rows = []
        categories = ["hardware", "accessory", "premium"]
        for it in items:
            inv_rows.append({
                "item": it,
                "category": random.choice(categories),
                "stock_level": random.randint(0, 300)
            })
        self.datasets["inventory"] = inv_rows
        self.execution_state = {
            "active_view": None,
            "active_dataset": None,
            "saved_views": {},
            "filters": [],
            "joined": None,
            "metrics": {},
            "tools_used": []
        }

    def _get_instructions(self) -> str:
        tool_list = ", ".join(sorted(self.tools.keys()))
        return (
            "You are orchestrating tools to build a data report. Load datasets, transform them, compute metrics, and submit.\n"
            "Your goal: satisfy the task's required dataset, filters, optional join, and metrics before calling submit_report.\n"
            "Protocol:\n"
            "- Use \\boxed{tool|param1=value1;param2=value2} per turn.\n"
            "- Tools available: " + tool_list + "\n"
            "- Conditions for filter_rows: col op value; ops: >,<,>=,<=,==,!=,contains. Strings may be quoted.\n"
            "- compute_metric supports: count,sum,mean,max,min,unique_count.\n"
            "Finish with submit_report|title=... once requirements are met."
        )

    def get_task_suffix(self) -> str:
        remaining_metrics = []
        for m in self.task["metrics_required"]:
            if m not in self.execution_state["metrics"]:
                remaining_metrics.append(m)
        filters_done = list(self.execution_state["filters"])
        join_needed = self.task["join_required"] and (self.execution_state["joined"] is None)
        left = (self.max_turns - self.turn_count)
        rows = len(self.execution_state["active_view"]) if self.execution_state["active_view"] is not None else 0
        return (
            f"Task: {self.task['description']}\n"
            f"Turns left: {left}\n"
            f"Required steps target: {self.task['required_steps']}\n"
            f"Current view rows: {rows}\n"
            f"Applied filters: {filters_done}\n"
            f"Join required: {self.task['join_required']}, joined: {self.execution_state['joined']}\n"
            f"Metrics remaining: {remaining_metrics}\n"
            "Format: \\boxed{tool|param1=value1;param2=value2}. Use strings with quotes if needed."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        required_steps = random.randint(self.min_required_steps, self.max_required_steps)
        self.task = self._generate_task_requiring_n_steps(required_steps)
        self.turn_count = 0
        self.steps_taken = 0
        self.execution_state = {
            "active_view": None,
            "active_dataset": None,
            "saved_views": {},
            "filters": [],
            "joined": None,
            "metrics": {},
            "tools_used": []
        }
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self.turn_count >= self.max_turns:
            obs = "Timeout: Reached max turns. Episode ended."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}
        self.turn_count += 1
        parsed = self._parse_action(action)
        if parsed is None:
            obs = "Invalid action format. Use \\boxed{tool|param=value}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
        tool_name, args = parsed
        if tool_name not in self.tools:
            obs = f"Unknown tool: {tool_name}. Episode terminated."
            return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
        try:
            result_text, protocol_violation = self._execute_tool(tool_name, args)
        except Exception as e:
            obs = f"Execution error: {e}"
            return obs, -0.2, False, False, {"suffix": self.get_task_suffix()}
        if protocol_violation:
            obs = f"Protocol violation: {result_text}"
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
        self.steps_taken += 1
        if tool_name == "submit_report":
            ok, detail = self._evaluate_submission()
            if ok:
                obs = f"Report submitted successfully.\n{detail}"
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Submission failed.\nMissing: {detail}"
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
        obs = f"Tool {tool_name} executed.\nResult: {result_text}"
        return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        m = re.search(r"\\boxed\{(.*)\}\s*$", action.strip())
        if not m:
            return None
        content = m.group(1).strip()
        if "|" not in content:
            return None
        tool, param_str = content.split("|", 1)
        tool = tool.strip()
        args: Dict[str, Any] = {}
        if param_str.strip():
            parts = [p for p in param_str.split(";") if p.strip()]
            for p in parts:
                if "=" not in p:
                    return None
                k, v = p.split("=", 1)
                k = k.strip()
                v = v.strip()
                if v.startswith("[") and v.endswith("]"):
                    inner = v[1:-1]
                    items = []
                    for token in inner.split(","):
                        tv = token.strip()
                        if tv.startswith("'") and tv.endswith("'"):
                            items.append(tv[1:-1])
                        elif tv.startswith('"') and tv.endswith('"'):
                            items.append(tv[1:-1])
                        else:
                            if tv.isdigit():
                                items.append(int(tv))
                            else:
                                try:
                                    items.append(float(tv))
                                except:
                                    items.append(tv)
                    args[k] = items
                else:
                    if v.startswith("'") and v.endswith("'"):
                        args[k] = v[1:-1]
                    elif v.startswith('"') and v.endswith('"'):
                        args[k] = v[1:-1]
                    else:
                        if v.isdigit():
                            args[k] = int(v)
                        else:
                            try:
                                args[k] = float(v)
                            except:
                                args[k] = v
        return tool, args

    def sample_random_action(self) -> str:
        ds = random.choice(list(self.datasets.keys()))
        return f"\\boxed{{open_dataset|name='{ds}'}}"

    def _generate_task_requiring_n_steps(self, required_steps: int) -> Dict[str, Any]:
        base_dataset = random.choice([n for n in self.datasets.keys() if n.startswith("sales_")])
        join_possible = self.complexity >= 4
        join_required = join_possible and random.choice([True, False])
        # compute nf and nm to meet required_steps ~ 1 load + nf filters + nm metrics + join + 1 submit
        max_filters = 3 if self.complexity <= 5 else 4
        nf = random.randint(1, min(max_filters, max(1, required_steps - 2)))
        join_cost = 1 if join_required else 0
        nm = required_steps - (1 + nf + join_cost + 1)
        if nm < 1:
            nm = 1
            if nf > 1 and (1 + nf + nm + join_cost + 1) > required_steps:
                nf -= 1
            if join_required and (1 + nf + nm + join_cost + 1) > self.max_required_steps:
                join_required = False
                join_cost = 0
        metrics = self._choose_metrics(nm, base_dataset)
        filters = self._choose_filters(nf, base_dataset)
        join_spec = None
        if join_required:
            if random.choice([True, False]):
                join_spec = {"name": "customers", "on": "customer_id", "right_key": "id"}
            else:
                join_spec = {"name": "inventory", "on": "item", "right_key": "item"}
        expected_rows_view = self._simulate_pipeline(base_dataset, filters, join_spec)
        expected_metrics = self._compute_expected_metrics(expected_rows_view, metrics)
        desc = (
            f"Open dataset '{base_dataset}'. Apply filters {filters}. "
            + (f"Join with '{join_spec['name']}' on '{join_spec['on']}'. " if join_required else "")
            + f"Compute metrics {metrics} on the resulting view, then submit the report."
        )
        return {
            "required_steps": required_steps,
            "base_dataset": base_dataset,
            "filters_required": filters,
            "join_required": join_required,
            "join_spec": join_spec,
            "metrics_required": metrics,
            "expected_metrics": expected_metrics,
            "description": desc
        }

    def _choose_metrics(self, nm: int, base_dataset: str) -> List[str]:
        metric_pool = ["count(*)", "sum(price)", "mean(qty)", "max(price)", "min(qty)", "unique_count(item)"]
        random.shuffle(metric_pool)
        chosen = metric_pool[:nm]
        return chosen

    def _choose_filters(self, nf: int, base_dataset: str) -> List[str]:
        rows = self.datasets[base_dataset]
        sample_row = random.choice(rows)
        filters = []
        candidates = []
        candidates.append(f"region == '{sample_row['region']}'")
        candidates.append(f"item contains '{sample_row['item'][:4]}'")
        qty_thresh = random.randint(2, 6)
        candidates.append(f"qty >= {qty_thresh}")
        price_thresh = round(random.uniform(10, 60), 2)
        candidates.append(f"price > {price_thresh}")
        random.shuffle(candidates)
        return candidates[:nf]

    def _simulate_pipeline(self, base_dataset: str, filters: List[str], join_spec: Optional[Dict[str, str]]) -> List[Dict[str, Any]]:
        view = [dict(r) for r in self.datasets[base_dataset]]
        for cond in filters:
            view = self._apply_condition(view, cond)
        if join_spec:
            right = self.datasets[join_spec["name"]]
            on = join_spec["on"]
            right_key = join_spec["right_key"]
            view = self._inner_join(view, right, on, right_key)
        return view

    def _compute_expected_metrics(self, view: List[Dict[str, Any]], metrics: List[str]) -> Dict[str, float]:
        out = {}
        for m in metrics:
            if m == "count(*)":
                out[m] = float(len(view))
            elif m == "sum(price)":
                s = sum([r.get("price", 0.0) for r in view])
                out[m] = float(round(s, 6))
            elif m == "mean(qty)":
                if len(view) == 0:
                    out[m] = 0.0
                else:
                    s = sum([r.get("qty", 0.0) for r in view]) / len(view)
                    out[m] = float(round(s, 6))
            elif m == "max(price)":
                out[m] = float(round(max([r.get("price", 0.0) for r in view]) if view else 0.0, 6))
            elif m == "min(qty)":
                out[m] = float(round(min([r.get("qty", 0) for r in view]) if view else 0.0, 6))
            elif m == "unique_count(item)":
                out[m] = float(len(set([r.get("item") for r in view])))
        return out

    def _apply_condition(self, rows: List[Dict[str, Any]], cond: str) -> List[Dict[str, Any]]:
        m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*(>=|<=|==|!=|>|<|contains)\s*(.+)\s*$", cond)
        if not m:
            return rows
        col, op, val_raw = m.group(1), m.group(2), m.group(3).strip()
        if val_raw.startswith("'") and val_raw.endswith("'"):
            val = val_raw[1:-1]
        elif val_raw.startswith('"') and val_raw.endswith('"'):
            val = val_raw[1:-1]
        else:
            try:
                val = float(val_raw)
                if val_raw.isdigit():
                    val = int(val_raw)
            except:
                val = val_raw
        out = []
        for r in rows:
            rv = r.get(col)
            keep = False
            try:
                if op == "contains":
                    if isinstance(rv, str):
                        keep = (str(val) in rv)
                elif op == "==":
                    keep = (rv == val)
                elif op == "!=":
                    keep = (rv != val)
                elif op == ">":
                    keep = (rv is not None and rv > val)  # type: ignore
                elif op == "<":
                    keep = (rv is not None and rv < val)  # type: ignore
                elif op == ">=":
                    keep = (rv is not None and rv >= val)  # type: ignore
                elif op == "<=":
                    keep = (rv is not None and rv <= val)  # type: ignore
            except:
                keep = False
            if keep:
                out.append(r)
        return out

    def _inner_join(self, left: List[Dict[str, Any]], right: List[Dict[str, Any]], left_key: str, right_key: str) -> List[Dict[str, Any]]:
        index: Dict[Any, Dict[str, Any]] = {}
        for r in right:
            rk = r.get(right_key)
            if rk is not None and rk not in index:
                index[rk] = r
        joined = []
        for l in left:
            lk = l.get(left_key)
            if lk in index:
                merged = dict(l)
                rrec = index[lk]
                for k, v in rrec.items():
                    if k in merged:
                        merged[f"{right_key}_{k}"] = v
                    else:
                        merged[k] = v
                joined.append(merged)
        return joined

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Tuple[str, bool]:
        self.execution_state["tools_used"].append(tool_name)
        if tool_name == "open_dataset":
            name = args.get("name")
            if name not in self.datasets:
                return f"Dataset '{name}' not found.", True
            self.execution_state["active_view"] = [dict(r) for r in self.datasets[name]]
            self.execution_state["active_dataset"] = name
            self.execution_state["filters"] = []
            self.execution_state["joined"] = None
            return f"Opened dataset '{name}' with {len(self.execution_state['active_view'])} rows.", False
        if tool_name == "select_columns":
            cols = args.get("columns")
            if self.execution_state["active_view"] is None:
                return "No dataset loaded. Call open_dataset first.", True
            if not isinstance(cols, list) or len(cols) == 0:
                return "columns must be a non-empty list.", True
            new_view = []
            for r in self.execution_state["active_view"]:
                nr = {}
                for c in cols:
                    nr[c] = r.get(c)
                new_view.append(nr)
            self.execution_state["active_view"] = new_view
            return f"Selected columns {cols}. View now has columns {cols}.", False
        if tool_name == "filter_rows":
            cond = args.get("condition")
            if self.execution_state["active_view"] is None:
                return "No dataset loaded. Call open_dataset first.", True
            if not isinstance(cond, str) or not cond.strip():
                return "condition must be a non-empty string.", True
            before = len(self.execution_state["active_view"])
            self.execution_state["active_view"] = self._apply_condition(self.execution_state["active_view"], cond)
            after = len(self.execution_state["active_view"])
            canon = self._canonical_condition(cond)
            self.execution_state["filters"].append(canon)
            return f"Applied filter '{cond}'. Rows: {before} -> {after}.", False
        if tool_name == "join_with":
            name = args.get("name")
            on = args.get("on")
            if self.execution_state["active_view"] is None:
                return "No dataset loaded. Call open_dataset first.", True
            if name not in self.datasets:
                return f"Dataset '{name}' not found.", True
            if not isinstance(on, str) or not on:
                return "join 'on' must be a column name.", True
            right_key = "id" if (name == "customers" and on == "customer_id") else on
            before = len(self.execution_state["active_view"])
            self.execution_state["active_view"] = self._inner_join(self.execution_state["active_view"], self.datasets[name], on, right_key)
            after = len(self.execution_state["active_view"])
            self.execution_state["joined"] = {"name": name, "on": on, "right_key": right_key}
            return f"Joined with '{name}' on '{on}'. Rows: {before} -> {after}.", False
        if tool_name == "compute_metric":
            metric = args.get("metric")
            column = args.get("column")
            if self.execution_state["active_view"] is None:
                return "No dataset loaded. Call open_dataset first.", True
            if not isinstance(metric, str) or not isinstance(column, str):
                return "metric and column must be strings.", True
            mkey = None
            val = None
            rows = self.execution_state["active_view"]
            if metric == "count" and column == "*":
                mkey = "count(*)"
                val = float(len(rows))
            elif metric == "sum":
                s = sum([r.get(column, 0.0) for r in rows])
                val = float(round(s, 6))
                mkey = f"sum({column})"
            elif metric == "mean":
                if len(rows) == 0:
                    val = 0.0
                else:
                    s = sum([r.get(column, 0.0) for r in rows]) / len(rows)
                    val = float(round(s, 6))
                mkey = f"mean({column})"
            elif metric == "max":
                val = float(round(max([r.get(column, 0.0) for r in rows]) if rows else 0.0, 6))
                mkey = f"max({column})"
            elif metric == "min":
                val = float(round(min([r.get(column, 0.0) for r in rows]) if rows else 0.0, 6))
                mkey = f"min({column})"
            elif metric == "unique_count":
                val = float(len(set([r.get(column) for r in rows])))
                mkey = f"unique_count({column})"
            else:
                return "Unsupported metric. Use count,sum,mean,max,min,unique_count.", True
            self.execution_state["metrics"][mkey] = val
            return f"Computed {mkey} = {val}.", False
        if tool_name == "sort_by":
            col = args.get("column")
            order = args.get("order", "asc")
            if self.execution_state["active_view"] is None:
                return "No dataset loaded. Call open_dataset first.", True
            rev = True if str(order).lower() == "desc" else False
            try:
                self.execution_state["active_view"] = sorted(self.execution_state["active_view"], key=lambda r: (r.get(col) is None, r.get(col)), reverse=rev)
            except Exception:
                return "Unable to sort by given column.", True
            return f"Sorted by '{col}' in {order} order.", False
            # other tools
        if tool_name == "limit_rows":
            n = args.get("n")
            if self.execution_state["active_view"] is None:
                return "No dataset loaded. Call open_dataset first.", True
            if not isinstance(n, int) or n < 0:
                return "n must be a non-negative integer.", True
            before = len(self.execution_state["active_view"])
            self.execution_state["active_view"] = self.execution_state["active_view"][:n]
            after = len(self.execution_state["active_view"])
            return f"Limited rows: {before} -> {after}.", False
        if tool_name == "save_view":
            alias = args.get("alias")
            if self.execution_state["active_view"] is None:
                return "No dataset loaded. Call open_dataset first.", True
            if not isinstance(alias, str) or not alias:
                return "alias must be a non-empty string.", True
            self.execution_state["saved_views"][alias] = [dict(r) for r in self.execution_state["active_view"]]
            return f"Saved current view as '{alias}'.", False
        if tool_name == "switch_view":
            alias = args.get("alias")
            if alias not in self.execution_state["saved_views"]:
                return f"No saved view named '{alias}'.", True
            self.execution_state["active_view"] = [dict(r) for r in self.execution_state["saved_views"][alias]]
            return f"Switched to view '{alias}'. Rows: {len(self.execution_state['active_view'])}.", False
        if tool_name == "submit_report":
            title = args.get("title")
            if not isinstance(title, str) or not title.strip():
                return "title must be a non-empty string.", True
            return f"Submitting report '{title}'.", False
        return "Unhandled tool.", True

    def _canonical_condition(self, cond: str) -> str:
        m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*(>=|<=|==|!=|>|<|contains)\s*(.+)\s*$", cond)
        if not m:
            return cond.strip()
        col, op, val_raw = m.group(1), m.group(2), m.group(3).strip()
        val = val_raw.strip()
        return f"{col} {op} {val}"

    def _evaluate_submission(self) -> Tuple[bool, str]:
        missing = []
        if self.execution_state["active_dataset"] != self.task["base_dataset"]:
            missing.append(f"open_dataset name='{self.task['base_dataset']}'")
        # filters check
        required_filters = set([self._canonical_condition(c) for c in self.task["filters_required"]])
        applied_filters = set(self.execution_state["filters"])
        if required_filters - applied_filters:
            missing.append(f"filters {list(required_filters - applied_filters)}")
        # join check
        if self.task["join_required"]:
            got = self.execution_state["joined"]
            need = self.task["join_spec"]
            if not got or got["name"] != need["name"] or got["on"] != need["on"]:
                missing.append(f"join_with name='{need['name']}' on='{need['on']}'")
        # metrics check
        for m in self.task["metrics_required"]:
            if m not in self.execution_state["metrics"]:
                missing.append(f"compute_metric for {m}")
        # value check
        value_mismatches = []
        for m in self.task["metrics_required"]:
            if m in self.execution_state["metrics"]:
                got = self.execution_state["metrics"][m]
                exp = self.task["expected_metrics"].get(m)
                if exp is None:
                    value_mismatches.append(f"{m} expected missing")
                else:
                    if abs(float(got) - float(exp)) > 1e-6:
                        value_mismatches.append(f"{m} got {got} expected {exp}")
        if missing or value_mismatches:
            detail = "; ".join(missing + value_mismatches)
            return False, detail
        detail = "All requirements met."
        return True, detail


class OpsForgeOrchestratorEnvWithFeedback(OpsForgeOrchestratorEnv):
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
            error_detail["issue"] = "missing_boxed_format"
            hint = "Use \\boxed{tool|param=value} with semicolon-separated params."
        elif "unknown tool:" in text:
            error_type = "UnsupportedAction"
            m = re.search(r"unknown tool:\s*([a-z_]+)", text)
            if m:
                error_detail["tool"] = m.group(1)
            hint = "Choose one of the available tools listed in the instructions."
        elif "protocol violation:" in text:
            error_type = "ProtocolViolation"
            detail_msg = obs.split("Protocol violation:")[-1].strip()
            error_detail["violation"] = detail_msg
            if "no dataset loaded" in text:
                hint = "Start by calling \\boxed{open_dataset|name='<dataset>'}."
            elif "columns must be a non-empty list" in text:
                hint = "Provide columns as a list: \\boxed{select_columns|columns=['item','qty']}."
            else:
                hint = "Check tool prerequisites and parameter types before calling."
        elif "execution error:" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = obs
            hint = "Verify parameter values and ensure the active view is set."
        elif "timeout" in text or truncated:
            error_type = "Timeout"
            error_detail["outcome"] = "reached_max_turns"
            hint = "Plan the sequence: open -> filters -> optional join -> metrics -> submit."
        elif "submission failed" in text:
            error_type = "WrongDecision"
            missing = []
            mm = re.search(r"missing:\s*(.*)$", obs, re.IGNORECASE)
            if mm:
                missing_text = mm.group(1)
                error_detail["missing"] = missing_text
                if "open_dataset" in missing_text:
                    hint = "Open the required dataset first."
                elif "filters" in missing_text:
                    hint = "Apply all listed filters using filter_rows|condition=..."
                elif "join_with" in missing_text:
                    hint = "Perform the specified join before computing metrics."
                elif "compute_metric" in missing_text:
                    hint = "Compute each required metric with proper metric+column."
                else:
                    hint = "Cross-check the task description and complete missing steps."
            else:
                hint = "Complete all requirements before submitting."
        elif "report submitted successfully" in text or "all requirements met" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = "Good job. You satisfied all requirements."
        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            state_info = {
                "required_steps": self.task.get("required_steps"),
                "steps_taken": getattr(self, "steps_taken", None),
                "metrics_required": self.task.get("metrics_required"),
            }
            diagnostic["state"] = state_info
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint
        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        first_ds = self.task.get("base_dataset")
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": f"Start by opening the target dataset: \\boxed{{open_dataset|name='{first_ds}'}}",
            "turn": 0,
            "state": {
                "required_steps": self.task.get("required_steps"),
                "metrics_required": self.task.get("metrics_required"),
            },
        }
        return obs, info