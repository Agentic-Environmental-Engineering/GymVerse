from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class OpsMaestroEnv(Env):
    def __init__(self, complexity: int = 1, max_turns: int = 30, **_):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.max_turns = max_turns if max_turns is not None else 30
        self.min_required_steps = self.complexity
        self.max_required_steps = self.complexity * 2
        self._init_database()
        self.reset()

    def _init_database(self):
        self.available_tools = {}
        self.available_tools["load_dataset"] = {
            "description": "Load a dataset into the working context.",
            "parameters": [{"name": "source", "type": "string"}],
            "returns": "records loaded count",
        }
        self.available_tools["filter_rows"] = {
            "description": "Filter rows by a simple condition like field>value or field==value.",
            "parameters": [{"name": "condition", "type": "string"}],
            "returns": "remaining rows count",
        }
        self.available_tools["select_fields"] = {
            "description": "Keep only specified comma-separated fields.",
            "parameters": [{"name": "fields", "type": "string"}],
            "returns": "field selection confirmation",
        }
        self.available_tools["sort_by"] = {
            "description": "Sort rows by field, order is asc or desc.",
            "parameters": [{"name": "field", "type": "string"}, {"name": "order", "type": "string"}],
            "returns": "sorting confirmation",
        }
        self.available_tools["compute_metric"] = {
            "description": "Compute a metric on a field: sum, avg, max, min, count.",
            "parameters": [{"name": "metric", "type": "string"}, {"name": "field", "type": "string"}],
            "returns": "metric value",
        }
        if self.complexity >= 4:
            self.available_tools["group_by"] = {
                "description": "Group rows by a field and aggregate target with a metric (sum/avg/max/min).",
                "parameters": [{"name": "field", "type": "string"}, {"name": "agg", "type": "string"}],
                "returns": "grouped result",
            }
        if self.complexity >= 6:
            self.available_tools["join_dataset"] = {
                "description": "Inner join current data with another dataset on a key.",
                "parameters": [{"name": "source", "type": "string"}, {"name": "on", "type": "string"}],
                "returns": "joined rows count",
            }
        self.available_tools["format_output"] = {
            "description": "Render the current data into CSV or JSON.",
            "parameters": [{"name": "fmt", "type": "string"}],
            "returns": "rendered output size",
        }
        self.available_tools["save_artifact"] = {
            "description": "Save the current rendered output under a name.",
            "parameters": [{"name": "name", "type": "string"}],
            "returns": "artifact saved",
        }
        self.available_tools["submit"] = {
            "description": "Submit the saved artifact for validation.",
            "parameters": [{"name": "name", "type": "string"}],
            "returns": "submission result",
        }

        # Simulated datasets
        rng = random.Random(self.complexity * 101)
        products = ["Widget", "Gizmo", "Doodad", "Gear", "Bolt", "Clamp"]
        regions = ["North", "South", "East", "West"]
        suppliers = ["AlphaCo", "BravoLtd", "CirrusInc", "DeltaCorp"]
        countries = ["US", "UK", "DE", "FR", "IN", "CN", "BR", "CA", "AU", "JP"]

        def gen_sales(n=40):
            data = []
            for i in range(n):
                data.append({
                    "id": i + 1,
                    "region": rng.choice(regions),
                    "product": rng.choice(products),
                    "quantity": rng.randint(1, 25),
                    "price": round(rng.uniform(5.0, 250.0), 2),
                    "date": f"2023-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
                })
            return data

        def gen_inventory(n=40):
            data = []
            for i in range(n):
                prod = rng.choice(products)
                data.append({
                    "id": i + 1,
                    "product": prod,
                    "stock": rng.randint(0, 200),
                    "cost": round(rng.uniform(1.0, 180.0), 2),
                    "supplier": rng.choice(suppliers),
                })
            return data

        def gen_users(n=50):
            data = []
            for i in range(n):
                data.append({
                    "id": i + 1,
                    "name": f"User{i+1}",
                    "country": rng.choice(countries),
                    "age": rng.randint(18, 75),
                    "active": rng.choice([True, False, True]),
                })
            return data

        def gen_orders(n=35):
            data = []
            for i in range(n):
                data.append({
                    "id": i + 1,
                    "product": rng.choice(products),
                    "units": rng.randint(1, 30),
                    "total": round(rng.uniform(10.0, 500.0), 2),
                    "region": rng.choice(regions),
                })
            return data

        self.sources = {
            "sales.csv": gen_sales(60 - 2 * self.complexity),
            "inventory.json": gen_inventory(60 - 2 * self.complexity),
            "users.csv": gen_users(70 - 2 * self.complexity),
        }
        if self.complexity >= 3:
            self.sources["orders.csv"] = gen_orders(50 - 2 * self.complexity)

        self.execution_state = {}
        self.task = {}
        self.turn_count = 0
        self.steps_taken = 0

    def _generate_task_requiring_n_steps(self, n: int) -> Dict[str, Any]:
        # pick dataset
        candidates = list(self.sources.keys())
        base_source = random.choice(candidates)
        base_data = self.sources[base_source]
        fmt = random.choice(["CSV", "JSON"])

        # derive solvable filters
        mandatory_filters: List[str] = []
        if len(base_data) > 0:
            sample = random.choice(base_data)
            options = []
            for k, v in sample.items():
                if isinstance(v, (int, float)):
                    lower = v - (abs(v) * 0.2 if v != 0 else 1)
                    val = max(lower, 0)
                    options.append(f"{k}>={round(val,2)}")
                elif isinstance(v, str):
                    sv = v.replace('"', '')
                    options.append(f'{k}=="{sv}"')
                elif isinstance(v, bool):
                    options.append(f"{k}=={str(v)}")
            random.shuffle(options)
            filters_needed = max(1, min(3, n // 3))
            for i in range(filters_needed):
                if i < len(options):
                    mandatory_filters.append(options[i])

        # selection fields
        fields = list(base_data[0].keys()) if base_data else []
        select_needed = random.choice([True, False, True]) if self.complexity >= 2 else True
        select_fields = random.sample(fields, k=min(len(fields), max(2, min(4, len(fields))))) if fields else []

        # metric requirement
        numeric_fields = [f for f in fields if isinstance(base_data[0].get(f), (int, float))] if base_data else []
        metric_needed = self.complexity >= 2 and len(numeric_fields) > 0
        metric_field = random.choice(numeric_fields) if metric_needed else None
        metric_type = random.choice(["avg", "sum", "max", "min"]) if metric_needed else None

        # sort requirement
        sort_needed = self.complexity >= 3 and len(fields) > 0
        sort_field = random.choice(fields) if sort_needed else None
        sort_order = random.choice(["asc", "desc"]) if sort_needed else None

        # join requirement
        join_needed = self.complexity >= 6 and "product" in fields and "inventory.json" in self.sources
        join_source = "inventory.json" if join_needed else None
        join_key = "product" if join_needed else None

        # artifact save is always needed for consistency
        artifact_name = f"artifact_{random.randint(100,999)}"

        mandatory_steps = []
        mandatory_steps.append({"tool": "load_dataset", "args": {"source": base_source}})
        for cond in mandatory_filters:
            mandatory_steps.append({"tool": "filter_rows", "args": {"condition": cond}})
        if select_needed and select_fields:
            mandatory_steps.append({"tool": "select_fields", "args": {"fields": ",".join(select_fields)}})
        if sort_needed and sort_field and sort_order:
            mandatory_steps.append({"tool": "sort_by", "args": {"field": sort_field, "order": sort_order}})
        if metric_needed and metric_field and metric_type:
            mandatory_steps.append({"tool": "compute_metric", "args": {"metric": metric_type, "field": metric_field}})
        if join_needed and join_source and join_key:
            mandatory_steps.append({"tool": "join_dataset", "args": {"source": join_source, "on": join_key}})
        mandatory_steps.append({"tool": "format_output", "args": {"fmt": fmt}})
        mandatory_steps.append({"tool": "save_artifact", "args": {"name": artifact_name}})

        # ensure total required steps reachable and not exceeding tool diversity
        required_steps = max(self.min_required_steps, min(self.max_required_steps, n))
        # task description
        constraints_text = []
        constraints_text.append(f"Use dataset: {base_source}")
        if mandatory_filters:
            constraints_text.append(f"Apply filters: {', '.join(mandatory_filters)}")
        if select_needed and select_fields:
            constraints_text.append(f"Select fields: {', '.join(select_fields)}")
        if sort_needed and sort_field and sort_order:
            constraints_text.append(f"Sort by {sort_field} ({sort_order})")
        if metric_needed and metric_field and metric_type:
            constraints_text.append(f"Compute metric {metric_type} on {metric_field}")
        if join_needed and join_source and join_key:
            constraints_text.append(f"Join with {join_source} on {join_key}")
        constraints_text.append(f"Format output as {fmt}")
        constraints_text.append(f"Save artifact named {artifact_name}")
        constraints_text.append(f"Perform at least {required_steps} valid tool calls before submit")

        return {
            "base_source": base_source,
            "required_filters": mandatory_filters,
            "select_fields": select_fields if select_needed else [],
            "sort": {"field": sort_field, "order": sort_order} if sort_needed else None,
            "metric": {"type": metric_type, "field": metric_field} if metric_needed else None,
            "join": {"source": join_source, "on": join_key} if join_needed else None,
            "format": fmt,
            "artifact_name": artifact_name,
            "required_steps": required_steps,
            "constraints_text": constraints_text,
            "mandatory_steps": mandatory_steps,
        }

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        required_steps = random.randint(self.min_required_steps, self.max_required_steps)
        self.task = self._generate_task_requiring_n_steps(required_steps)
        self.execution_state = {
            "loaded_source": None,
            "current_data": None,
            "applied_filters": [],
            "selected_fields": None,
            "sorted_by": None,
            "metrics": {},
            "joined_sources": [],
            "last_format": None,
            "rendered": None,
            "artifacts": {},
        }
        self.turn_count = 0
        self.steps_taken = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _get_instructions(self) -> str:
        tools_list = ", ".join(sorted(self.available_tools.keys()))
        constraints = "\n- " + "\n- ".join(self.task["constraints_text"])
        return (
            "Tool Orchestration Task:\n"
            "Plan and execute a sequence of tool calls to produce an artifact that satisfies all constraints.\n"
            f"Available tools: {tools_list}\n"
            "Action format: use \\boxed{tool|param1=value1;param2=value2}. Use submit at the end.\n"
            "Constraints:\n"
            f"{constraints}\n"
            "You must respect prerequisites (load before transform) and perform at least the required number of valid tool calls before submitting."
        )

    def get_task_suffix(self) -> str:
        unmet = []
        if self.execution_state["loaded_source"] != self.task["base_source"]:
            unmet.append("load_dataset")
        for cond in self.task["required_filters"]:
            if cond not in self.execution_state["applied_filters"]:
                unmet.append(f"filter:{cond}")
        if self.task["select_fields"]:
            if self.execution_state["selected_fields"] != self.task["select_fields"]:
                unmet.append("select_fields")
        if self.task["sort"]:
            if self.execution_state["sorted_by"] != (self.task["sort"]["field"], self.task["sort"]["order"]):
                unmet.append("sort_by")
        if self.task["metric"]:
            mt = self.task["metric"]["type"]
            mf = self.task["metric"]["field"]
            key = f"{mt}:{mf}"
            if key not in self.execution_state["metrics"]:
                unmet.append(f"metric:{key}")
        if self.task["join"]:
            j = self.task["join"]
            if not any(js == (j["source"], j["on"]) for js in self.execution_state["joined_sources"]):
                unmet.append("join_dataset")
        if self.execution_state["last_format"] != self.task["format"]:
            unmet.append("format_output")
        if self.task["artifact_name"] not in self.execution_state["artifacts"]:
            unmet.append("save_artifact")

        tools_list = ", ".join(sorted(self.available_tools.keys()))
        return (
            f"Progress: steps_taken={self.steps_taken}/{self.task['required_steps']}; unmet={', '.join(unmet) if unmet else 'none'}.\n"
            f"Use tools with \\boxed{{tool|param=value;...}}. Available: {tools_list}.\n"
            "Submit with \\boxed{submit|name=your_artifact_name} when ready."
        )

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not isinstance(action, str):
            return None
        m = re.search(r"\\boxed\{(.+?)\}", action)
        if not m:
            return None
        content = m.group(1).strip()
        if "|" in content:
            tool, rest = content.split("|", 1)
            tool = tool.strip()
            args = {}
            if rest.strip():
                parts = [p for p in rest.split(";") if p.strip()]
                for p in parts:
                    if "=" in p:
                        k, v = p.split("=", 1)
                        args[k.strip()] = v.strip()
                    else:
                        # allow flags: key (no '=')
                        args[p.strip()] = ""
            return {"tool": tool, "args": args, "raw": content}
        else:
            tool = content.strip()
            return {"tool": tool, "args": {}, "raw": content}

    def sample_random_action(self) -> str:
        src = self.task.get("base_source", random.choice(list(self.sources.keys())))
        return f"\\boxed{{load_dataset|source={src}}}"

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        info: Dict[str, Any] = {"suffix": self.get_task_suffix()}
        parsed = self._parse_action(action)
        if not parsed:
            obs = "Invalid action format: expected \\boxed{tool|param=value;...}"
            return obs, float(LanguageGameReward.format_error_reward), True, False, info

        tool = parsed["tool"]
        args = parsed["args"]

        if tool not in self.available_tools:
            obs = f"Unsupported tool: {tool}. Episode terminated."
            return obs, 0.0, True, False, info

        if tool == "submit":
            name = args.get("name")
            success, details = self._evaluate_submission(name)
            if success:
                obs = f"Submission success: all constraints satisfied. Artifact '{name}' accepted."
                return obs, 1.0, True, False, info
            else:
                obs = f"Submission failed: {details}"
                return obs, 0.0, True, False, info

        try:
            changed, obs_detail = self._execute_tool(tool, args)
            if changed:
                self.steps_taken += 1
            obs = f"Tool {tool} executed. {obs_detail}"
            terminated = False
            truncated = False
        except Exception as e:
            obs = f"Protocol violation or execution error: {str(e)}"
            terminated = False
            truncated = False

        if not terminated and self.turn_count >= self.max_turns:
            obs = "Timeout: reached max turns before submission."
            return obs, 0.0, True, True, info

        info["suffix"] = self.get_task_suffix()
        return obs, 0.0, terminated, truncated, info

    def _evaluate_submission(self, name: Optional[str]) -> Tuple[bool, str]:
        if not name:
            return False, "Missing artifact name in submit."
        if name not in self.execution_state["artifacts"]:
            return False, f"No saved artifact named '{name}'."
        if self.steps_taken < self.task["required_steps"]:
            return False, f"Insufficient tool calls: {self.steps_taken} < {self.task['required_steps']}."
        if self.execution_state["loaded_source"] != self.task["base_source"]:
            return False, f"Wrong or missing dataset: expected {self.task['base_source']}."
        for cond in self.task["required_filters"]:
            if cond not in self.execution_state["applied_filters"]:
                return False, f"Missing required filter: {cond}."
        if self.task["select_fields"]:
            if self.execution_state["selected_fields"] != self.task["select_fields"]:
                return False, f"Incorrect field selection."
        if self.task["sort"]:
            s = self.task["sort"]
            if self.execution_state["sorted_by"] != (s["field"], s["order"]):
                return False, f"Missing required sort by {s['field']} ({s['order']})."
        if self.task["metric"]:
            mt = self.task["metric"]["type"]
            mf = self.task["metric"]["field"]
            key = f"{mt}:{mf}"
            if key not in self.execution_state["metrics"]:
                return False, f"Missing metric {mt} on {mf}."
        if self.task["join"]:
            j = self.task["join"]
            if not any(js == (j["source"], j["on"]) for js in self.execution_state["joined_sources"]):
                return False, f"Missing join with {j['source']} on {j['on']}."
        if self.execution_state["last_format"] != self.task["format"]:
            return False, f"Wrong format: expected {self.task['format']}."
        if self.task["artifact_name"] != name:
            return False, f"Wrong artifact name: expected {self.task['artifact_name']}."
        return True, "OK"

    def _execute_tool(self, tool: str, args: Dict[str, Any]) -> Tuple[bool, str]:
        es = self.execution_state
        if tool == "load_dataset":
            source = args.get("source")
            if source not in self.sources:
                raise ValueError(f"Source not found: {source}")
            es["current_data"] = [dict(r) for r in self.sources[source]]
            es["loaded_source"] = source
            es["applied_filters"] = []
            es["selected_fields"] = None
            es["sorted_by"] = None
            es["metrics"] = {}
            es["joined_sources"] = []
            es["last_format"] = None
            es["rendered"] = None
            return True, f"Loaded {len(es['current_data'])} records from {source}."
        elif tool == "filter_rows":
            if es["current_data"] is None:
                raise ValueError("No dataset loaded. Use load_dataset first.")
            cond_str = args.get("condition")
            if not cond_str:
                raise ValueError("Missing condition.")
            predicate = self._parse_condition(cond_str)
            filtered = [r for r in es["current_data"] if predicate(r)]
            es["current_data"] = filtered
            es["applied_filters"].append(cond_str)
            return True, f"Filter applied '{cond_str}'. Remaining {len(filtered)} records."
        elif tool == "select_fields":
            if es["current_data"] is None:
                raise ValueError("No dataset loaded. Use load_dataset first.")
            fields_arg = args.get("fields", "")
            fields = [f.strip() for f in fields_arg.split(",") if f.strip()]
            if not fields:
                raise ValueError("No fields specified.")
            reduced = []
            for r in es["current_data"]:
                reduced.append({k: r.get(k) for k in fields})
            es["current_data"] = reduced
            es["selected_fields"] = fields
            return True, f"Selected fields: {', '.join(fields)}."
        elif tool == "sort_by":
            if es["current_data"] is None:
                raise ValueError("No dataset loaded. Use load_dataset first.")
            field = args.get("field")
            order = args.get("order", "asc").lower()
            if field is None:
                raise ValueError("Missing sort field.")
            rev = order == "desc"
            es["current_data"].sort(key=lambda x: (x.get(field) is None, x.get(field)), reverse=rev)
            es["sorted_by"] = (field, order)
            return True, f"Sorted by {field} ({order})."
            # compute_metric
        elif tool == "compute_metric":
            if es["current_data"] is None:
                raise ValueError("No dataset loaded. Use load_dataset first.")
            metric = args.get("metric", "").lower()
            field = args.get("field")
            if metric not in ["sum", "avg", "max", "min", "count"]:
                raise ValueError(f"Unsupported metric: {metric}")
            if metric == "count":
                val = len(es["current_data"])
            else:
                values = [r.get(field) for r in es["current_data"] if isinstance(r.get(field), (int, float))]
                if not values:
                    raise ValueError(f"No numeric values for field '{field}'.")
                if metric == "sum":
                    val = float(sum(values))
                elif metric == "avg":
                    val = float(sum(values) / len(values))
                elif metric == "max":
                    val = float(max(values))
                elif metric == "min":
                    val = float(min(values))
                else:
                    val = 0.0
            es["metrics"][f"{metric}:{field}"] = val
            return True, f"Metric {metric} on {field} = {round(val, 4)}."
        elif tool == "group_by":
            if "group_by" not in self.available_tools:
                raise ValueError("Tool not available at current complexity.")
            if es["current_data"] is None:
                raise ValueError("No dataset loaded. Use load_dataset first.")
            field = args.get("field")
            agg = args.get("agg", "")
            if not field or not agg or ":" not in agg:
                raise ValueError("Provide agg as metric:targetField.")
            metric, target = agg.split(":", 1)
            groups: Dict[Any, List[Dict[str, Any]]] = {}
            for r in es["current_data"]:
                key = r.get(field)
                groups.setdefault(key, []).append(r)
            rendered = []
            for gk, rows in groups.items():
                vals = [r.get(target) for r in rows if isinstance(r.get(target), (int, float))]
                if metric == "sum":
                    val = float(sum(vals)) if vals else 0.0
                elif metric == "avg":
                    val = float(sum(vals) / len(vals)) if vals else 0.0
                elif metric == "max":
                    val = float(max(vals)) if vals else 0.0
                elif metric == "min":
                    val = float(min(vals)) if vals else 0.0
                else:
                    raise ValueError(f"Unsupported group metric: {metric}")
                rendered.append({"group": gk, f"{metric}_{target}": val})
            es["current_data"] = rendered
            return True, f"Grouped by {field} with {metric} on {target}. Groups={len(rendered)}."
        elif tool == "join_dataset":
            if "join_dataset" not in self.available_tools:
                raise ValueError("Tool not available at current complexity.")
            if es["current_data"] is None or es["loaded_source"] is None:
                raise ValueError("No dataset loaded. Use load_dataset first.")
            src = args.get("source")
            on = args.get("on")
            if src not in self.sources:
                raise ValueError(f"Join source not found: {src}")
            if not on:
                raise ValueError("Missing join key 'on'.")
            right = self.sources[src]
            index: Dict[Any, Dict[str, Any]] = {}
            for r in right:
                index[r.get(on)] = r
            joined = []
            for l in es["current_data"]:
                key = l.get(on)
                if key in index:
                    merged = dict(l)
                    for rk, rv in index[key].items():
                        if rk not in merged:
                            merged[rk] = rv
                        else:
                            merged[f"{src}.{rk}"] = rv
                    joined.append(merged)
            es["current_data"] = joined
            es["joined_sources"].append((src, on))
            return True, f"Joined with {src} on {on}. Rows={len(joined)}."
        elif tool == "format_output":
            if es["current_data"] is None:
                raise ValueError("No dataset loaded. Use load_dataset first.")
            fmt = args.get("fmt", "").upper()
            if fmt not in ["CSV", "JSON"]:
                raise ValueError(f"Unsupported format: {fmt}")
            rendered = self._render(es["current_data"], fmt)
            es["rendered"] = rendered
            es["last_format"] = fmt
            return True, f"Rendered output in {fmt}. Size={len(rendered)} chars."
        elif tool == "save_artifact":
            name = args.get("name")
            if not name:
                raise ValueError("Missing artifact name.")
            if es["rendered"] is None:
                raise ValueError("Nothing rendered. Use format_output first.")
            es["artifacts"][name] = {
                "format": es["last_format"],
                "content": es["rendered"],
                "source": es["loaded_source"],
                "metrics": dict(es["metrics"]),
                "filters": list(es["applied_filters"]),
                "selected_fields": es["selected_fields"],
                "sorted_by": es["sorted_by"],
                "joined_sources": list(es["joined_sources"]),
            }
            return True, f"Artifact '{name}' saved."
        else:
            raise ValueError(f"Tool not executable: {tool}")

    def _render(self, data: List[Dict[str, Any]], fmt: str) -> str:
        if fmt == "JSON":
            # simple JSON-like rendering without importing json
            def val_to_str(v):
                if isinstance(v, str):
                    return '"' + v.replace('"', '\\"') + '"'
                elif isinstance(v, bool):
                    return "true" if v else "false"
                elif v is None:
                    return "null"
                else:
                    return str(v)
            items = []
            for r in data:
                fields = []
                for k, v in r.items():
                    fields.append(f'"{k}": {val_to_str(v)}')
                items.append("{" + ", ".join(fields) + "}")
            return "[" + ", ".join(items) + "]"
        else:  # CSV
            if not data:
                return ""
            headers = list(data[0].keys())
            lines = [",".join(headers)]
            for r in data:
                row = []
                for h in headers:
                    v = r.get(h)
                    if v is None:
                        row.append("")
                    else:
                        s = str(v)
                        if "," in s or "\n" in s:
                            s = '"' + s.replace('"', '""') + '"'
                        row.append(s)
                lines.append(",".join(row))
            return "\n".join(lines)

    def _parse_condition(self, cond: str):
        cond = cond.strip()
        # operators: >=, <=, ==, !=, >, <, contains
        ops = ["<=", ">=", "==", "!=", ">", "<", " contains "]
        chosen = None
        for op in ["<=", ">=", "==", "!=", ">", "<"]:
            if op in cond:
                chosen = op
                parts = cond.split(op)
                left = parts[0].strip()
                right = parts[1].strip()
                val = right.strip()
                if val.startswith('"') and val.endswith('"'):
                    val = val[1:-1]
                elif val.lower() in ["true", "false"]:
                    val = True if val.lower() == "true" else False
                else:
                    try:
                        if "." in val:
                            val = float(val)
                        else:
                            val = int(val)
                    except Exception:
                        pass

                def predicate(record, l=left, o=chosen, v=val):
                    rv = record.get(l)
                    if o == "==":
                        return rv == v
                    elif o == "!=":
                        return rv != v
                    elif o == ">=":
                        try:
                            return rv >= v
                        except Exception:
                            return False
                    elif o == "<=":
                        try:
                            return rv <= v
                        except Exception:
                            return False
                    elif o == ">":
                        try:
                            return rv > v
                        except Exception:
                            return False
                    elif o == "<":
                        try:
                            return rv < v
                        except Exception:
                            return False
                    return False

                return predicate
        if " contains " in cond:
            left, right = cond.split(" contains ", 1)
            left = left.strip()
            right = right.strip().strip('"')
            def predicate(record, l=left, r=right):
                rv = record.get(l)
                if rv is None:
                    return False
                return str(rv).find(r) != -1
            return predicate
        raise ValueError(f"Unparsable condition: {cond}")


class OpsMaestroEnvWithFeedback(OpsMaestroEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        text = obs.lower()
        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint: Optional[str] = None

        if "invalid action format" in text or "expected \\boxed" in text:
            error_type = "FormatError"
            error_detail["issue"] = "missing_or_bad_boxed_format"
            hint = "Use \\boxed{tool|param=value;...} with a valid tool."
        elif "unsupported tool" in text:
            error_type = "UnsupportedAction"
            error_detail["tool"] = self._extract_tool_from_obs(obs)
            hint = "Use only available tools listed in the instructions."
        elif "protocol violation" in text or "execution error" in text or "no dataset loaded" in text:
            error_type = "ProtocolViolation"
            if "no dataset loaded" in text:
                error_detail["violation"] = "missing_load"
                hint = "Start with \\boxed{load_dataset|source=...} before filtering or formatting."
            else:
                error_detail["violation"] = "tool_usage_error"
                hint = "Check parameters and prerequisites for the tool."
        elif "timeout" in text:
            error_type = "Timeout"
            error_detail["outcome"] = "max_turns_reached"
            hint = "Submit earlier after meeting constraints; plan your tool sequence efficiently."
        elif "submission failed" in text:
            error_type = "WrongDecision"
            missing = self._extract_missing_from_obs(obs)
            error_detail["missing_requirements"] = missing
            hint = self._compose_hint(missing)
        elif "submission success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["steps_taken"] = getattr(self, "steps_taken", None)
            diagnostic["required_steps"] = self.task.get("required_steps")
            diagnostic["required_format"] = self.task.get("format")
            diagnostic["base_source"] = self.task.get("base_source")
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Begin by loading the required dataset using load_dataset.",
            "turn": 0,
            "steps_taken": 0,
            "required_steps": self.task.get("required_steps"),
            "required_format": self.task.get("format"),
            "base_source": self.task.get("base_source"),
        }
        return obs, info

    def _extract_tool_from_obs(self, obs: str) -> Optional[str]:
        m = re.search(r"Unsupported tool: ([^\.]+)", obs)
        return m.group(1) if m else None

    def _extract_missing_from_obs(self, obs: str) -> List[str]:
        missing = []
        for key in ["filter", "select", "sort", "metric", "join", "format", "artifact", "dataset", "insufficient"]:
            if key in obs.lower():
                missing.append(key)
        return missing

    def _compose_hint(self, missing: List[str]) -> Optional[str]:
        if not missing:
            return "Ensure all constraints are met and submit the correct artifact name."
        if "dataset" in missing:
            return "Load the specified dataset with load_dataset before other operations."
        if "filter" in missing:
            return "Apply each required filter exactly as listed using filter_rows|condition=..."
        if "select" in missing:
            return "Use select_fields with the exact field list required."
        if "sort" in missing:
            return "Use sort_by with the required field and order (asc/desc)."
        if "metric" in missing:
            return "Compute the required metric with compute_metric|metric=...;field=..."
        if "join" in missing:
            return "Perform join_dataset with the specified source and key."
        if "format" in missing:
            return "Render with format_output to the required format (CSV or JSON)."
        if "artifact" in missing:
            return "Save the artifact using save_artifact with the required name."
        if "insufficient" in missing:
            return "Perform additional valid tool calls to meet the step requirement before submitting."
        return "Check unmet constraints and fulfill them step by step."