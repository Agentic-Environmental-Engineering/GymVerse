from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class PipelineConductorEnv(Env):
    def __init__(self, complexity: int = 1, max_turns: Optional[int] = 100, **_):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.max_turns = max_turns if max_turns is not None else 100
        self.min_required_steps = self.complexity
        self.max_required_steps = self.complexity * 2
        self._init_database()
        self.reset()

    def _init_database(self):
        size_a = 8 + self.complexity
        size_b = 8 + self.complexity
        categories = ["alpha", "beta", "gamma", "delta"]
        regions = ["north", "south", "east", "west"]
        tags = ["red", "blue", "green", "yellow", "purple"]
        self.datasets = {}

        data_a = []
        for i in range(1, size_a + 1):
            data_a.append({
                "id": i,
                "category": random.choice(categories),
                "value": random.randint(5, 20),
                "tag": random.choice(tags),
            })
        self.datasets["A"] = data_a

        data_b = []
        for i in range(1, size_b + 1):
            data_b.append({
                "id": i,
                "weight": random.randint(1, 5),
                "region": random.choice(regions),
            })
        self.datasets["B"] = data_b

        if self.complexity >= 5:
            size_c = 6 + self.complexity
            data_c = []
            for i in range(1, size_c + 1):
                data_c.append({
                    "tag": random.choice(tags),
                    "bonus": random.randint(0, 3),
                })
            self.datasets["C"] = data_c

        self.tools = {}
        self.tools["load_data"] = {
            "description": "Load a dataset into working memory.",
            "parameters": [{"name": "source", "type": "string", "choices": list(self.datasets.keys())}],
            "returns": "Sets current_data to records of the dataset."
        }
        self.tools["inspect"] = {
            "description": "Return a summary of the current working data (no state change).",
            "parameters": [],
            "returns": "String summary of current_data."
        }
        self.tools["filter_rows"] = {
            "description": "Filter rows by a condition.",
            "parameters": [
                {"name": "field", "type": "string"},
                {"name": "op", "type": "string", "choices": ["eq", "gt", "lt", "contains"]},
                {"name": "value", "type": "string|number"}
            ],
            "returns": "Filters current_data in place."
        }
        if self.complexity >= 2:
            self.tools["join"] = {
                "description": "Inner join current_data with another dataset on a key.",
                "parameters": [
                    {"name": "source", "type": "string", "choices": list(self.datasets.keys())},
                    {"name": "key", "type": "string", "default": "id"}
                ],
                "returns": "Merges fields from both datasets."
            }
            self.tools["multiply_fields"] = {
                "description": "Create/overwrite target field as a*b for each record.",
                "parameters": [
                    {"name": "target", "type": "string"},
                    {"name": "a", "type": "string"},
                    {"name": "b", "type": "string"}
                ],
                "returns": "Adds numeric field 'target' to each record."
            }
        if self.complexity >= 3:
            self.tools["group_aggregate"] = {
                "description": "Group by a field and aggregate another numeric field.",
                "parameters": [
                    {"name": "group_by", "type": "string"},
                    {"name": "field", "type": "string"},
                    {"name": "agg", "type": "string", "choices": ["sum", "mean", "max", "min"]}
                ],
                "returns": "Sets current_data to group-level results with 'group' and 'agg_value'."
            }
        if self.complexity >= 4:
            self.tools["sort_by"] = {
                "description": "Sort current_data by a field.",
                "parameters": [
                    {"name": "field", "type": "string"},
                    {"name": "order", "type": "string", "choices": ["asc", "desc"], "default": "asc"}
                ],
                "returns": "Reorders current_data."
            }
            self.tools["limit"] = {
                "description": "Keep only first n records.",
                "parameters": [{"name": "n", "type": "int"}],
                "returns": "Truncates current_data."
            }
        if self.complexity >= 6:
            self.tools["scale_field"] = {
                "description": "Multiply a numeric field by a factor.",
                "parameters": [{"name": "field", "type": "string"}, {"name": "factor", "type": "number"}],
                "returns": "Scales field in place."
            }
        self.tools["compute_stat"] = {
            "description": "Compute a statistic over a numeric field.",
            "parameters": [
                {"name": "field", "type": "string"},
                {"name": "stat", "type": "string", "choices": ["sum", "mean", "max", "min"]}
            ],
            "returns": "Computes scalar and stores as last_result."
        }
        self.tools["submit_answer"] = {
            "description": "Submit the final scalar answer.",
            "parameters": [{"name": "answer", "type": "number"}],
            "returns": "Ends the episode."
        }

        self.turn_count = 0
        self.steps_taken = 0
        self.current_data: Optional[List[Dict[str, Any]]] = None
        self.execution_state: Dict[str, Any] = {"last_result": None}
        self.task: Dict[str, Any] = {}
        self.solution: Optional[float] = None

    def _generate_task_requiring_n_steps(self, n: int) -> Dict[str, Any]:
        essential_ops = []
        if n - 1 >= 5 and ("join" in self.tools) and ("multiply_fields" in self.tools):
            target_category = random.choice(["alpha", "beta", "gamma", "delta"])
            essential_ops = [
                {"tool": "load_data", "args": {"source": "A"}},
                {"tool": "join", "args": {"source": "B", "key": "id"}},
                {"tool": "multiply_fields", "args": {"target": "score", "a": "value", "b": "weight"}},
                {"tool": "filter_rows", "args": {"field": "category", "op": "eq", "value": target_category}},
                {"tool": "compute_stat", "args": {"field": "score", "stat": "sum"}},
            ]
            description = f"Compute the sum of score=value*weight for records where category='{target_category}' after joining A with B."
        elif n - 1 >= 4 and ("join" in self.tools):
            chosen_region = random.choice(["north", "south", "east", "west"])
            essential_ops = [
                {"tool": "load_data", "args": {"source": "A"}},
                {"tool": "join", "args": {"source": "B", "key": "id"}},
                {"tool": "filter_rows", "args": {"field": "region", "op": "eq", "value": chosen_region}},
                {"tool": "compute_stat", "args": {"field": "value", "stat": "mean"}},
            ]
            description = f"Compute the mean of value for records in region='{chosen_region}' after joining A with B."
        elif n - 1 >= 4 and ("group_aggregate" in self.tools):
            essential_ops = [
                {"tool": "load_data", "args": {"source": "A"}},
                {"tool": "filter_rows", "args": {"field": "value", "op": "gt", "value": random.randint(8, 14)}},
                {"tool": "group_aggregate", "args": {"group_by": "category", "field": "value", "agg": "sum"}},
                {"tool": "compute_stat", "args": {"field": "agg_value", "stat": "max"}},
            ]
            description = "Compute the maximum of total 'value' per category, after filtering A by a value threshold."
        else:
            chosen_category = random.choice(["alpha", "beta", "gamma", "delta"])
            essential_ops = [
                {"tool": "load_data", "args": {"source": "A"}},
                {"tool": "filter_rows", "args": {"field": "category", "op": "eq", "value": chosen_category}},
                {"tool": "compute_stat", "args": {"field": "value", "stat": random.choice(["sum", "mean", "max", "min"])}},
            ]
            description = f"Compute a statistic over 'value' for A where category='{chosen_category}'."

        essential_count = len(essential_ops)
        extras_needed = max(0, (n - 1) - essential_count)
        extra_ops = [{"tool": "inspect", "args": {}} for _ in range(extras_needed)]
        ops = essential_ops + extra_ops

        solution = self._simulate_pipeline(essential_ops)
        return {"description": description, "required_steps": n, "operations": ops, "solution": solution}

    def _simulate_pipeline(self, ops: List[Dict[str, Any]]) -> float:
        data = None
        last_result = None
        for step in ops:
            tool = step["tool"]
            args = step["args"]
            if tool == "load_data":
                src = args.get("source")
                data = [dict(r) for r in self.datasets.get(src, [])]
            elif tool == "join":
                if data is None:
                    continue
                src = args.get("source")
                key = args.get("key", "id")
                other = self.datasets.get(src, [])
                index = {}
                for r in other:
                    index.setdefault(r.get(key), r)
                joined = []
                for r in data:
                    k = r.get(key)
                    if k in index:
                        merged = dict(r)
                        for kk, vv in index[k].items():
                            if kk not in merged:
                                merged[kk] = vv
                            else:
                                if kk != key:
                                    merged[f"{src}.{kk}"] = vv
                        joined.append(merged)
                data = joined
            elif tool == "filter_rows":
                if data is None:
                    continue
                field, op, value = args.get("field"), args.get("op"), args.get("value")
                filtered = []
                for r in data:
                    rv = r.get(field)
                    keep = False
                    if op == "eq":
                        keep = rv == value
                    elif op == "gt":
                        try:
                            keep = float(rv) > float(value)
                        except Exception:
                            keep = False
                    elif op == "lt":
                        try:
                            keep = float(rv) < float(value)
                        except Exception:
                            keep = False
                    elif op == "contains":
                        if isinstance(rv, str) and isinstance(value, str):
                            keep = value in rv
                        else:
                            keep = False
                    if keep:
                        filtered.append(r)
                data = filtered
            elif tool == "multiply_fields":
                if data is None:
                    continue
                target, a, b = args.get("target"), args.get("a"), args.get("b")
                for r in data:
                    av = r.get(a, 0)
                    bv = r.get(b, 0)
                    try:
                        r[target] = float(av) * float(bv)
                    except Exception:
                        r[target] = 0.0
            elif tool == "group_aggregate":
                if data is None:
                    continue
                group_by, field, agg = args.get("group_by"), args.get("field"), args.get("agg")
                groups: Dict[Any, List[float]] = {}
                for r in data:
                    g = r.get(group_by)
                    v = r.get(field)
                    try:
                        v = float(v)
                    except Exception:
                        v = None
                    if v is not None:
                        groups.setdefault(g, []).append(v)
                results = []
                for g, vals in groups.items():
                    if not vals:
                        continue
                    if agg == "sum":
                        agg_val = sum(vals)
                    elif agg == "mean":
                        agg_val = sum(vals) / len(vals)
                    elif agg == "max":
                        agg_val = max(vals)
                    elif agg == "min":
                        agg_val = min(vals)
                    else:
                        agg_val = sum(vals)
                    results.append({"group": g, "agg_value": agg_val})
                data = results
            elif tool == "compute_stat":
                if data is None:
                    last_result = 0.0
                    continue
                field, stat = args.get("field"), args.get("stat")
                values = []
                for r in data:
                    v = r.get(field)
                    try:
                        v = float(v)
                        values.append(v)
                    except Exception:
                        pass
                if not values:
                    last_result = 0.0
                else:
                    if stat == "sum":
                        last_result = float(sum(values))
                    elif stat == "mean":
                        last_result = float(sum(values) / len(values))
                    elif stat == "max":
                        last_result = float(max(values))
                    elif stat == "min":
                        last_result = float(min(values))
                    else:
                        last_result = float(sum(values))
        return float(last_result if last_result is not None else 0.0)

    def _get_instructions(self) -> str:
        lines = []
        lines.append("You are the Pipeline Conductor. Use tools to load, transform, and analyze datasets to compute the requested metric, then submit it.")
        lines.append("Respond with exactly one action per turn using \\boxed{...}.")
        lines.append("Action format: \\boxed{tool_name param1=value param2=\"string\" ...}")
        lines.append("Available tools:")
        for name, meta in self.tools.items():
            params = ", ".join([p["name"] for p in meta.get("parameters", [])])
            lines.append(f"- {name}({params}): {meta['description']}")
        lines.append("End the episode by calling submit_answer(answer=NUMBER).")
        return "\n".join(lines)

    def get_task_suffix(self) -> str:
        state_desc = []
        state_desc.append(f"Turn: {self.turn_count}, Steps taken: {self.steps_taken}, Required steps range: [{self.min_required_steps}, {self.max_required_steps}]")
        state_desc.append(f"Current data loaded: {'Yes' if self.current_data is not None else 'No'}")
        if self.execution_state.get("last_result") is not None:
            state_desc.append(f"Last computed result: {self.execution_state.get('last_result')}")
        state_desc.append(f"Task: {self.task.get('description', '')}")
        state_desc.append("Input a single boxed action now. Example: \\boxed{load_data source=A}")
        return "\n".join(state_desc)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        required_steps = random.randint(self.min_required_steps, self.max_required_steps)
        self.task = self._generate_task_requiring_n_steps(required_steps)
        self.solution = float(self.task["solution"])
        self.turn_count = 0
        self.steps_taken = 0
        self.current_data = None
        self.execution_state = {"last_result": None}
        obs = self._get_instructions()
        info = {"suffix": self.get_task_suffix()}
        return obs, info

    def _parse_action(self, action: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        m = re.search(r"\\boxed\{(.*)\}", action, re.DOTALL)
        if not m:
            return None
        inner = m.group(1).strip()
        if not inner:
            return None
        token_pattern = r'(?:"[^"]*"|\'[^\']*\'|[^ \t]+)'
        tokens = re.findall(token_pattern, inner)
        if not tokens:
            return None
        tool_name = tokens[0]
        args: Dict[str, Any] = {}
        for tok in tokens[1:]:
            if "=" not in tok:
                continue
            key, val = tok.split("=", 1)
            key = key.strip()
            val = val.strip()
            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                v = val[1:-1]
            else:
                try:
                    if "." in val:
                        v = float(val)
                    else:
                        v = int(val)
                except Exception:
                    v = val
            args[key] = v
        return tool_name, args

    def sample_random_action(self) -> str:
        tool = random.choice(list(self.tools.keys()))
        if tool == "load_data":
            src = random.choice(list(self.datasets.keys()))
            return f"\\boxed{{load_data source={src}}}"
        if tool == "inspect":
            return "\\boxed{inspect}"
        if tool == "filter_rows":
            field = random.choice(["category", "value", "tag"])
            if field == "category":
                return f"\\boxed{{filter_rows field=category op=eq value=\"alpha\"}}"
            if field == "value":
                return f"\\boxed{{filter_rows field=value op=gt value=10}}"
            return f"\\boxed{{filter_rows field=tag op=contains value=\"red\"}}"
        if tool == "join":
            src = random.choice(list(self.datasets.keys()))
            return f"\\boxed{{join source={src} key=id}}"
        if tool == "multiply_fields":
            return "\\boxed{multiply_fields target=score a=value b=weight}"
        if tool == "group_aggregate":
            return "\\boxed{group_aggregate group_by=category field=value agg=sum}"
        if tool == "sort_by":
            return "\\boxed{sort_by field=value order=desc}"
        if tool == "limit":
            return "\\boxed{limit n=5}"
        if tool == "scale_field":
            return "\\boxed{scale_field field=value factor=2}"
        if tool == "compute_stat":
            return "\\boxed{compute_stat field=value stat=sum}"
        if tool == "submit_answer":
            guess = round(self.solution or 0.0, 2)
            return f"\\boxed{{submit_answer answer={guess}}}"
        return "\\boxed{inspect}"

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        info = {"suffix": self.get_task_suffix()}
        parsed = self._parse_action(action)
        if parsed is None:
            obs = "Invalid action format. Use \\boxed{tool_name param=value ...}."
            return obs, LanguageGameReward.format_error_reward, True, False, info

        tool_name, args = parsed
        if tool_name not in self.tools:
            obs = f"Unsupported action: unknown tool '{tool_name}'."
            return obs, 0.0, True, False, info  # Fixed: was -0.5

        terminated = False
        truncated = False
        reward = 0.0
        try:
            result_text = self._execute_tool(tool_name, args)
            obs = f"Executed {tool_name}. Result: {result_text}"
            if tool_name != "inspect":
                self.steps_taken += 1
            if tool_name == "submit_answer":
                correct = self._check_answer(args.get("answer"))
                if correct:
                    obs = f"Success: correct final answer {args.get('answer')}."
                    reward = 1.0
                else:
                    obs = f"Incorrect final answer {args.get('answer')}. Expected a different value."
                    reward = 0.0
                terminated = True
            else:
                terminated = False
        except ValueError as e:
            obs = f"Protocol violation: {str(e)}"
            reward = 0.0  # Fixed: was -0.1, failures should be 0.0
            terminated = True  # Protocol violations should end episode
        except Exception as e:
            obs = f"Execution error: {str(e)}"
            reward = 0.0  # Fixed: was -0.15, failures should be 0.0
            terminated = True  # Execution errors should end episode

        if not terminated and self.turn_count >= self.max_turns:
            obs = "Timeout: turn limit reached."
            terminated = True
            truncated = True

        info["suffix"] = self.get_task_suffix()
        return obs, reward, terminated, truncated, info

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        if tool_name == "load_data":
            source = args.get("source")
            if source not in self.datasets:
                raise ValueError(f"dataset '{source}' not found")
            self.current_data = [dict(r) for r in self.datasets[source]]
            return f"Loaded {len(self.current_data)} records from {source}"

        if tool_name == "inspect":
            if self.current_data is None:
                return "No data loaded."
            fields = set()
            for r in self.current_data[:5]:
                for f in r.keys():
                    fields.add(f)
            return f"{len(self.current_data)} records. Fields: {sorted(list(fields))}"

        if tool_name == "filter_rows":
            if self.current_data is None:
                raise ValueError("no data loaded. Call load_data first.")
            field, op, value = args.get("field"), args.get("op"), args.get("value")
            filtered = []
            for r in self.current_data:
                rv = r.get(field)
                keep = False
                if op == "eq":
                    keep = rv == value
                elif op == "gt":
                    try:
                        keep = float(rv) > float(value)
                    except Exception:
                        keep = False
                elif op == "lt":
                    try:
                        keep = float(rv) < float(value)
                    except Exception:
                        keep = False
                elif op == "contains":
                    if isinstance(rv, str) and isinstance(value, str):
                        keep = value in rv
                    else:
                        keep = False
                else:
                    raise ValueError(f"unsupported op '{op}'")
                if keep:
                    filtered.append(r)
            self.current_data = filtered
            return f"Filtered to {len(self.current_data)} records"

        if tool_name == "join":
            if self.current_data is None:
                raise ValueError("no data loaded. Call load_data first.")
            source = args.get("source")
            key = args.get("key", "id")
            if source not in self.datasets:
                raise ValueError(f"dataset '{source}' not found")
            other = self.datasets[source]
            index = {}
            for r in other:
                index.setdefault(r.get(key), r)
            joined = []
            for r in self.current_data:
                k = r.get(key)
                if k in index:
                    merged = dict(r)
                    other_r = index[k]
                    for kk, vv in other_r.items():
                        if kk not in merged:
                            merged[kk] = vv
                        else:
                            if kk != key:
                                merged[f"{source}.{kk}"] = vv
                    joined.append(merged)
            self.current_data = joined
            return f"Joined with {source} on {key}, resulting in {len(self.current_data)} records"

        if tool_name == "multiply_fields":
            if self.current_data is None:
                raise ValueError("no data loaded. Call load_data first.")
            target, a, b = args.get("target"), args.get("a"), args.get("b")
            for r in self.current_data:
                av = r.get(a, 0)
                bv = r.get(b, 0)
                try:
                    r[target] = float(av) * float(bv)
                except Exception:
                    r[target] = 0.0
            return f"Computed {target} as {a}*{b} for {len(self.current_data)} records"

        if tool_name == "group_aggregate":
            if self.current_data is None:
                raise ValueError("no data loaded. Call load_data first.")
            group_by, field, agg = args.get("group_by"), args.get("field"), args.get("agg")
            groups: Dict[Any, List[float]] = {}
            for r in self.current_data:
                g = r.get(group_by)
                v = r.get(field)
                try:
                    v = float(v)
                except Exception:
                    v = None
                if v is not None:
                    groups.setdefault(g, []).append(v)
            results = []
            for g, vals in groups.items():
                if not vals:
                    continue
                if agg == "sum":
                    agg_val = sum(vals)
                elif agg == "mean":
                    agg_val = sum(vals) / len(vals)
                elif agg == "max":
                    agg_val = max(vals)
                elif agg == "min":
                    agg_val = min(vals)
                else:
                    raise ValueError(f"unsupported agg '{agg}'")
                results.append({"group": g, "agg_value": agg_val})
            self.current_data = results
            return f"Aggregated {field} by {group_by} using {agg}, produced {len(results)} groups"

        if tool_name == "sort_by":
            if self.current_data is None:
                raise ValueError("no data loaded. Call load_data first.")
            field = args.get("field")
            order = args.get("order", "asc")
            try:
                self.current_data.sort(key=lambda r: r.get(field), reverse=(order == "desc"))
            except Exception:
                pass
            return f"Sorted by {field} in {order} order"

        if tool_name == "limit":
            if self.current_data is None:
                raise ValueError("no data loaded. Call load_data first.")
            n = int(args.get("n", 0))
            if n < 0:
                n = 0
            self.current_data = self.current_data[:n]
            return f"Limited to first {n} records"

        if tool_name == "scale_field":
            if self.current_data is None:
                raise ValueError("no data loaded. Call load_data first.")
            field = args.get("field")
            factor = float(args.get("factor", 1.0))
            for r in self.current_data:
                try:
                    r[field] = float(r.get(field, 0)) * factor
                except Exception:
                    pass
            return f"Scaled field {field} by factor {factor}"

        if tool_name == "compute_stat":
            if self.current_data is None:
                raise ValueError("no data loaded. Call load_data first.")
            field, stat = args.get("field"), args.get("stat")
            values = []
            for r in self.current_data:
                v = r.get(field)
                try:
                    v = float(v)
                    values.append(v)
                except Exception:
                    pass
            if not values:
                res = 0.0
            else:
                if stat == "sum":
                    res = float(sum(values))
                elif stat == "mean":
                    res = float(sum(values) / len(values))
                elif stat == "max":
                    res = float(max(values))
                elif stat == "min":
                    res = float(min(values))
                else:
                    raise ValueError(f"unsupported stat '{stat}'")
            self.execution_state["last_result"] = res
            return f"Computed {stat}({field}) = {res}"

        if tool_name == "submit_answer":
            ans = args.get("answer")
            if ans is None:
                raise ValueError("missing 'answer' parameter")
            return f"Submitted answer {ans}"
        raise ValueError(f"unknown tool '{tool_name}'")

    def _check_answer(self, ans: Any) -> bool:
        try:
            val = float(ans)
            sol = float(self.solution if self.solution is not None else 0.0)
            return abs(val - sol) <= 1e-6
        except Exception:
            return False


class PipelineConductorEnvWithFeedback(PipelineConductorEnv):
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
            hint = "Use \\boxed{tool_name param=value} with one tool per turn."
        elif "unsupported action" in text or "unknown tool" in text:
            error_type = "UnsupportedAction"
            error_detail["tool"] = "unrecognized"
            hint = "Choose one of the documented tools in the instructions."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "no data loaded" in text:
                error_detail["violation"] = "data_not_loaded"
                hint = "Start with \\boxed{load_data source=A} or another dataset."
            elif "unsupported op" in text:
                error_detail["violation"] = "bad_operator"
                hint = "Use one of: eq, gt, lt, contains."
            elif "unsupported stat" in text or "unsupported agg" in text:
                error_detail["violation"] = "bad_stat_or_agg"
                hint = "Choose from: sum, mean, max, min."
            else:
                error_detail["violation"] = "other"
                hint = "Check tool parameters against the instruction list."
        elif "execution error" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "runtime_error"
            hint = "Verify field names exist and parameter types are correct."
        elif "incorrect final answer" in text:
            error_type = "WrongDecision"
            try:
                expected = float(self.solution if self.solution is not None else 0.0)
            except Exception:
                expected = None
            error_detail["expected"] = expected
            error_detail["got"] = None
            m = re.search(r"incorrect final answer ([^\.]+)", text)
            if m:
                error_detail["got"] = m.group(1)
            hint = "Compute the required statistic using the tools (e.g., load, join, transform, compute_stat) and resubmit."
        elif "timeout" in text:
            error_type = "Timeout"
            error_detail["outcome"] = "turn_limit_reached"
            hint = "Plan fewer exploratory calls; prioritize essential operations then submit."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "steps_taken": getattr(self, "steps_taken", None),
                "last_result": getattr(self, "execution_state", {}).get("last_result"),
                "data_loaded": self.current_data is not None,
            }
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint
        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        start_hint = "Begin with \\boxed{load_data source=A}, then follow the task specification to transform and compute."
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": start_hint if self.feedback_level >= 2 else None,
            "turn": 0,
            "state": {
                "steps_taken": 0,
                "last_result": None,
                "data_loaded": False,
            },
        }
        return obs, info