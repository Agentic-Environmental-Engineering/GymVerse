from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List, Union

class ArtisanToolroomEnv(Env):
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
            "open": {
                "description": "Load a source dataset into a named frame.",
                "params": ["source", "as"],
                "returns": "Frame name with data"
            },
            "select": {
                "description": "Keep only specified columns in a frame.",
                "params": ["frame", "columns"],
                "returns": "Modified frame"
            },
            "filter": {
                "description": "Filter rows by simple condition. Supported: ==, >, <, in [a|b|c].",
                "params": ["frame", "where"],
                "returns": "Filtered frame"
            },
            "derive": {
                "description": "Create a new numeric column from an expression like colA*colB or colA+colB.",
                "params": ["frame", "into", "formula"],
                "returns": "Modified frame"
            },
            "aggregate": {
                "description": "Group by and aggregate a numeric column. op in {sum, avg, count}.",
                "params": ["frame", "by", "op", "target", "into"],
                "returns": "Aggregated frame"
            },
            "join": {
                "description": "Inner join two frames on matching key names.",
                "params": ["left", "right", "on", "as"],
                "returns": "New joined frame"
            },
            "sort": {
                "description": "Sort a frame by a column. order in {asc, desc}.",
                "params": ["frame", "by", "order"],
                "returns": "Modified frame"
            },
            "head": {
                "description": "Keep first k rows.",
                "params": ["frame", "k"],
                "returns": "Modified frame"
            },
            "unique": {
                "description": "Get unique values from a column; stores last_unique for inspection.",
                "params": ["frame", "column"],
                "returns": "Unique values (observation)"
            },
            "preview": {
                "description": "Show first n rows of a frame.",
                "params": ["frame", "n"],
                "returns": "Preview in observation"
            },
            "final": {
                "description": "Submit final answer.",
                "params": ["answer"],
                "returns": "Terminates if answer format acceptable"
            }
        }
        # Simulated data generation parameters scale with complexity
        base_customers = 6 + self.complexity * 2
        base_items = 6 + self.complexity * 2
        base_sales = 20 + self.complexity * 10

        random_regions = ["North", "South", "East", "West", "Central", "Coastal"]
        random_segments = ["Enterprise", "SMB", "Consumer", "Public"]
        random_brands = ["Acme", "Zenith", "Nimbus", "Orion", "Helix"]
        random_categories = ["Gadgets", "Tools", "Supplies", "Accessories"]

        # Customers
        self.customers = []
        for cid in range(1, base_customers + 1):
            self.customers.append({
                "customer_id": cid,
                "name": f"Customer{cid}",
                "segment": random.choice(random_segments),
                "region": random.choice(random_regions)
            })

        # Products
        self.products = []
        for pid in range(1, base_items + 1):
            self.products.append({
                "item_id": pid,
                "brand": random.choice(random_brands),
                "category": random.choice(random_categories)
            })

        # Sales
        self.sales = []
        for sid in range(1, base_sales + 1):
            cust = random.choice(self.customers)
            item = random.choice(self.products)
            qty = random.randint(1, 9)
            price = random.randint(5, 100)
            self.sales.append({
                "sale_id": sid,
                "customer_id": cust["customer_id"],
                "item_id": item["item_id"],
                "qty": qty,
                "price": price,
                "region": cust["region"]
            })

        # Sources registry
        self.sources = {
            "sales.csv": {"type": "table", "data": self.sales},
            "customers.json": {"type": "table", "data": self.customers},
            "products.csv": {"type": "table", "data": self.products}
        }

    def _generate_task_requiring_n_steps(self, required_steps: int) -> Dict[str, Any]:
        # Select a task pattern based on required_steps and complexity
        # Define several task templates with increasing complexity
        task_types = []
        if required_steps <= 2:
            task_types = ["count_rows", "unique_regions"]
        elif required_steps <= 4:
            task_types = ["unique_regions", "top_item_by_qty_in_region"]
        else:
            # longer pipelines
            task_types = ["top_region_by_revenue", "top_segment_by_revenue", "top_brand_by_revenue", "top_item_by_qty_in_region"]

        ttype = random.choice(task_types)

        if ttype == "count_rows":
            source = "sales.csv"
            expected = len(self.sources[source]["data"])
            desc = f"Determine how many rows are in {source}. Return the integer count."
            return {"type": ttype, "expected": expected, "description": desc}

        if ttype == "unique_regions":
            # compute unique regions in sales
            regs = sorted(list({r["region"] for r in self.sales}))
            expected = regs
            desc = "List all unique regions present in sales.csv as a list of strings."
            return {"type": ttype, "expected": expected, "description": desc}

        if ttype == "top_item_by_qty_in_region":
            region = random.choice(list({r["region"] for r in self.sales}))
            # compute totals by item within region
            tally = {}
            for row in self.sales:
                if row["region"] == region:
                    tally[row["item_id"]] = tally.get(row["item_id"], 0) + row["qty"]
            if not tally:
                # Edge case; pick again with generic total
                ttype = "count_rows"
                source = "sales.csv"
                expected = len(self.sources[source]["data"])
                desc = f"Determine how many rows are in {source}. Return the integer count."
                return {"type": ttype, "expected": expected, "description": desc}
            item_id = max(tally.items(), key=lambda x: x[1])[0]
            expected = item_id
            desc = f"In sales.csv, restricted to region='{region}', find the item_id with highest total qty."
            return {"type": ttype, "expected": expected, "description": desc, "region": region}

        if ttype == "top_region_by_revenue":
            # compute revenue=qty*price grouped by region
            totals = {}
            for row in self.sales:
                rev = row["qty"] * row["price"]
                totals[row["region"]] = totals.get(row["region"], 0) + rev
            if not totals:
                expected = None
            else:
                expected = max(totals.items(), key=lambda x: x[1])[0]
            desc = "Compute total revenue (qty*price) by region from sales.csv and return the region with highest revenue."
            return {"type": ttype, "expected": expected, "description": desc}

        if ttype == "top_segment_by_revenue":
            # join sales with customers on customer_id; group by segment sum revenue
            cust_map = {c["customer_id"]: c for c in self.customers}
            totals = {}
            for row in self.sales:
                rev = row["qty"] * row["price"]
                seg = cust_map.get(row["customer_id"], {}).get("segment", None)
                if seg is not None:
                    totals[seg] = totals.get(seg, 0) + rev
            expected = max(totals.items(), key=lambda x: x[1])[0] if totals else None
            desc = "Join sales.csv with customers.json on customer_id. Compute revenue (qty*price) by segment and return the top segment."
            return {"type": ttype, "expected": expected, "description": desc}

        if ttype == "top_brand_by_revenue":
            # join sales with products on item_id; group by brand sum revenue
            prod_map = {p["item_id"]: p for p in self.products}
            totals = {}
            for row in self.sales:
                rev = row["qty"] * row["price"]
                brand = prod_map.get(row["item_id"], {}).get("brand", None)
                if brand is not None:
                    totals[brand] = totals.get(brand, 0) + rev
            expected = max(totals.items(), key=lambda x: x[1])[0] if totals else None
            desc = "Join sales.csv with products.csv on item_id. Compute revenue (qty*price) by brand and return the top brand."
            return {"type": ttype, "expected": expected, "description": desc}

        # fallback
        source = "sales.csv"
        expected = len(self.sources[source]["data"])
        desc = f"Determine how many rows are in {source}. Return the integer count."
        return {"type": "count_rows", "expected": expected, "description": desc}

    def _get_instructions(self) -> str:
        tools_desc = []
        for name, meta in self.tools.items():
            params = ", ".join(meta["params"])
            tools_desc.append(f"- {name}({params}): {meta['description']}")
        tool_text = "\n".join(tools_desc)

        datasets = []
        for s, meta in self.sources.items():
            datasets.append(f"- {s}: columns={list(meta['data'][0].keys()) if meta['data'] else []}")
        ds_text = "\n".join(datasets)

        req = f"You must perform between {self.required_steps_min} and {self.required_steps_max} tool calls BEFORE final."
        return (
            "You are in a tool-use workspace. Execute tools to solve the analytics task.\n"
            f"Task: {self.task['description']}\n"
            f"{req}\n"
            "Available datasets:\n"
            f"{ds_text}\n"
            "Available tools and signatures:\n"
            f"{tool_text}\n"
            "Protocol:\n"
            "- Each step, respond with exactly one action in the format: \\boxed{tool: key1=value1, key2=value2}\n"
            "- Strings may be unquoted if no commas; lists use [a|b|c].\n"
            "- For open, pick a unique frame name via 'as'. For transformations, refer to frame names.\n"
            "- Finish with final: answer=... to submit the result."
        )

    def get_task_suffix(self) -> str:
        frames_list = ", ".join(sorted(self.frames.keys())) if self.frames else "(none)"
        step_info = f"Steps taken: {self.steps_taken}. You must take between {self.required_steps_min} and {self.required_steps_max} tool calls before final."
        return f"Frames: {frames_list}. {step_info} Respond with one action: \\boxed{{tool: key=value, ...}}"

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        required_steps = random.randint(self.min_required_steps, self.max_required_steps)
        self.task = self._generate_task_requiring_n_steps(required_steps)
        self.turn_count = 0
        self.steps_taken = 0
        self.required_steps_min = required_steps
        self.required_steps_max = max(required_steps, min(self.max_required_steps, required_steps + random.randint(0, self.complexity)))
        self.frames: Dict[str, List[Dict[str, Any]]] = {}
        self.last_unique: Optional[List[Any]] = None
        self.terminated = False
        self.truncated = False
        self.submission = None
        obs = self._get_instructions()
        return obs, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not isinstance(action, str):
            return None
        m = re.search(r"\\boxed\{(.*)\}\s*$", action.strip())
        if not m:
            return None
        inner = m.group(1).strip()
        if ":" not in inner:
            return None
        tool, params_str = inner.split(":", 1)
        tool = tool.strip()
        params = {}
        # split by commas but allow brackets grouping
        parts = []
        buf = ""
        bracket = 0
        for ch in params_str:
            if ch == "[":
                bracket += 1
                buf += ch
            elif ch == "]":
                bracket = max(0, bracket - 1)
                buf += ch
            elif ch == "," and bracket == 0:
                parts.append(buf.strip())
                buf = ""
            else:
                buf += ch
        if buf.strip():
            parts.append(buf.strip())
        for p in parts:
            if not p:
                continue
            if "=" not in p:
                continue
            k, v = p.split("=", 1)
            k = k.strip()
            v = v.strip()
            if v.startswith("[") and v.endswith("]"):
                inner_list = v[1:-1].strip()
                if inner_list == "":
                    params[k] = []
                else:
                    params[k] = [x.strip() for x in inner_list.split("|")]
            else:
                v = v.strip("'").strip('"')
                params[k] = v
        return {"tool": tool, "params": params}

    def sample_random_action(self) -> str:
        options = [
            r"\boxed{open: source=sales.csv, as=sales}",
            r"\boxed{preview: frame=sales, n=3}",
            r"\boxed{unique: frame=sales, column=region}",
            r"\boxed{final: answer=42}"
        ]
        return random.choice(options)

    def _execute_tool(self, tool: str, params: Dict[str, Any]) -> str:
        def ensure_frame(name: str):
            if name not in self.frames:
                raise ValueError(f"Frame not found: {name}")

        if tool == "open":
            source = params.get("source")
            alias = params.get("as")
            if not source or not alias:
                raise ValueError("open requires source and as")
            if alias in self.frames:
                raise ValueError(f"Frame name already exists: {alias}")
            if source not in self.sources:
                raise ValueError(f"Unknown source: {source}")
            data = [dict(r) for r in self.sources[source]["data"]]
            self.frames[alias] = data
            return f"Opened {source} into frame '{alias}' with {len(data)} rows and columns {list(data[0].keys()) if data else []}."

        if tool == "select":
            frame = params.get("frame")
            cols = params.get("columns")
            if not frame or not cols:
                raise ValueError("select requires frame and columns")
            ensure_frame(frame)
            if isinstance(cols, list):
                columns = cols
            else:
                columns = [c.strip() for c in re.split(r"[|;]", cols)]
            new_data = []
            for row in self.frames[frame]:
                new_row = {c: row[c] for c in columns if c in row}
                new_data.append(new_row)
            self.frames[frame] = new_data
            return f"Selected columns {columns} in frame '{frame}'. Now columns={columns}."

        if tool == "filter":
            frame = params.get("frame")
            where = params.get("where")
            if not frame or not where:
                raise ValueError("filter requires frame and where")
            ensure_frame(frame)
            cond = where.strip()
            def parse_value(s: str) -> Any:
                s = s.strip().strip("'").strip('"')
                if re.fullmatch(r"-?\d+(\.\d+)?", s):
                    if "." in s:
                        return float(s)
                    else:
                        return int(s)
                return s

            rows = self.frames[frame]
            out = []
            if " in [" in cond and cond.endswith("]"):
                m = re.match(r"(\w+)\s+in\s+\[(.*)\]$", cond)
                if not m:
                    raise ValueError("Unsupported where syntax")
                col = m.group(1)
                values = [parse_value(x) for x in m.group(2).split("|") if x.strip() != ""]
                for r in rows:
                    if r.get(col) in values:
                        out.append(r)
            elif "==" in cond:
                col, val = cond.split("==", 1)
                col = col.strip()
                val = parse_value(val)
                for r in rows:
                    if r.get(col) == val:
                        out.append(r)
            elif ">" in cond:
                col, val = cond.split(">", 1)
                col = col.strip()
                val = parse_value(val)
                for r in rows:
                    try:
                        if float(r.get(col, float("-inf"))) > float(val):
                            out.append(r)
                    except:
                        pass
            elif "<" in cond:
                col, val = cond.split("<", 1)
                col = col.strip()
                val = parse_value(val)
                for r in rows:
                    try:
                        if float(r.get(col, float("inf"))) < float(val):
                            out.append(r)
                    except:
                        pass
            else:
                raise ValueError("Unsupported where operator")
            self.frames[frame] = out
            return f"Filtered frame '{frame}' to {len(out)} rows by condition [{cond}]."

        if tool == "derive":
            frame = params.get("frame")
            into = params.get("into")
            formula = params.get("formula")
            if not frame or not into or not formula:
                raise ValueError("derive requires frame, into, formula")
            ensure_frame(frame)
            expr = formula.strip()
            op = None
            if "*" in expr:
                left, right = expr.split("*", 1)
                op = "*"
            elif "+" in expr:
                left, right = expr.split("+", 1)
                op = "+"
            else:
                raise ValueError("Only '*' and '+' supported")
            left = left.strip()
            right = right.strip()
            for r in self.frames[frame]:
                a = r.get(left)
                b = r.get(right)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    r[into] = a * b if op == "*" else a + b
                else:
                    raise ValueError("Non-numeric operands for derive")
            return f"Derived column '{into}' as {left}{op}{right} in frame '{frame}'."

        if tool == "aggregate":
            frame = params.get("frame")
            by = params.get("by")
            op = params.get("op")
            target = params.get("target")
            into = params.get("into")
            if not (frame and by and op and target and into):
                raise ValueError("aggregate requires frame, by, op, target, into")
            ensure_frame(frame)
            rows = self.frames[frame]
            groups = {}
            for r in rows:
                key = r.get(by)
                if key not in groups:
                    groups[key] = []
                groups[key].append(r.get(target, 0))
            out = []
            for k, vals in groups.items():
                nums = [v for v in vals if isinstance(v, (int, float))]
                if op == "sum":
                    agg = sum(nums)
                elif op == "avg":
                    agg = sum(nums) / len(nums) if nums else 0
                elif op == "count":
                    agg = len(vals)
                else:
                    raise ValueError("Unsupported aggregate op")
                out.append({by: k, into: agg})
            self.frames[frame] = out
            return f"Aggregated frame '{frame}' by {by} with {op} on {target} -> {into}. Rows={len(out)}."

        if tool == "join":
            left = params.get("left")
            right = params.get("right")
            on = params.get("on")
            alias = params.get("as")
            if not (left and right and on and alias):
                raise ValueError("join requires left, right, on, as")
            ensure_frame(left)
            ensure_frame(right)
            if alias in self.frames:
                raise ValueError(f"Frame name already exists: {alias}")
            right_map = {}
            for r in self.frames[right]:
                rk = r.get(on)
                if rk not in right_map:
                    right_map[rk] = []
                right_map[rk].append(r)
            joined = []
            for l in self.frames[left]:
                lk = l.get(on)
                if lk in right_map:
                    for rr in right_map[lk]:
                        merged = dict(l)
                        # avoid key collisions by preferring left on duplicates
                        for k, v in rr.items():
                            if k not in merged:
                                merged[k] = v
                        joined.append(merged)
            self.frames[alias] = joined
            return f"Joined '{left}' with '{right}' on '{on}' into '{alias}'. Rows={len(joined)}."

        if tool == "sort":
            frame = params.get("frame")
            by = params.get("by")
            order = params.get("order", "asc")
            if not (frame and by):
                raise ValueError("sort requires frame and by")
            ensure_frame(frame)
            rev = True if str(order).lower() == "desc" else False
            self.frames[frame].sort(key=lambda r: r.get(by, 0), reverse=rev)
            return f"Sorted frame '{frame}' by {by} order={order}."

        if tool == "head":
            frame = params.get("frame")
            k = params.get("k")
            if not frame or k is None:
                raise ValueError("head requires frame and k")
            ensure_frame(frame)
            try:
                n = int(k)
            except:
                raise ValueError("k must be integer")
            self.frames[frame] = self.frames[frame][:max(0, n)]
            return f"Kept first {n} rows of frame '{frame}'. Now rows={len(self.frames[frame])}."

        if tool == "unique":
            frame = params.get("frame")
            col = params.get("column")
            if not frame or not col:
                raise ValueError("unique requires frame and column")
            ensure_frame(frame)
            values = []
            seen = set()
            for r in self.frames[frame]:
                v = r.get(col)
                if v not in seen:
                    seen.add(v)
                    values.append(v)
            self.last_unique = values
            return f"Unique values from '{frame}.{col}': {values}"

        if tool == "preview":
            frame = params.get("frame")
            n = params.get("n", 5)
            if not frame:
                raise ValueError("preview requires frame")
            ensure_frame(frame)
            try:
                n = int(n)
            except:
                n = 5
            sample = self.frames[frame][:max(0, n)]
            return f"Preview of '{frame}' first {n} rows: {sample}"

        if tool == "final":
            ans = params.get("answer", "")
            self.submission = ans
            return f"Submitted final answer: {ans}"

        raise ValueError(f"Unsupported tool: {tool}")

    def _check_completion(self) -> Tuple[bool, float, str]:
        if self.submission is None:
            return False, 0.0, ""
        # Enforce step bounds before allowing success
        if self.steps_taken < self.required_steps_min:
            return True, 0.0, "Final received but insufficient steps taken."
        if self.steps_taken > self.required_steps_max:
            return True, 0.0, "Final received but exceeded maximum allowed steps."

        expected = self.task["expected"]
        got = self.submission

        def normalize(v: Any) -> Any:
            if isinstance(v, str):
                return v.strip()
            if isinstance(v, list):
                # parse submission lists
                return v
            return v

        # Try to parse submission for list-answers
        parsed_submission: Union[str, int, float, List[str]] = got
        if isinstance(got, str):
            s = got.strip()
            if s.startswith("[") and s.endswith("]"):
                inner = s[1:-1].strip()
                if inner == "":
                    parsed_submission = []
                else:
                    parsed_submission = [x.strip().strip("'").strip('"') for x in inner.split("|")]
            elif re.fullmatch(r"-?\d+", s):
                parsed_submission = int(s)
            elif re.fullmatch(r"-?\d+\.\d+", s):
                parsed_submission = float(s)
            else:
                parsed_submission = s

        ok = False
        if isinstance(expected, list):
            # compare as sets of strings
            if isinstance(parsed_submission, list):
                ok = set([str(x) for x in parsed_submission]) == set([str(x) for x in expected])
            else:
                ok = False
        else:
            ok = str(parsed_submission) == str(expected)

        if ok:
            return True, LanguageGameReward.success_reward, "Success: final answer is correct."
        else:
            return True, 0.0, f"Final answer incorrect. Expected={expected}, Got={parsed_submission}"

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self.terminated or self.truncated:
            return "Episode already finished.", 0.0, True, self.truncated, {"suffix": self.get_task_suffix()}
        self.turn_count += 1

        parsed = self._parse_action(action)
        if not parsed:
            obs = "Invalid action format. Use \\boxed{tool: key=value, key2=value2}."
            return obs, LanguageGameReward.format_error_reward, False, False, {"suffix": self.get_task_suffix()}

        tool = parsed["tool"]
        params = parsed["params"]

        if tool not in self.tools:
            obs = f"UnsupportedAction: Unknown tool '{tool}'."
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        reward = 0.0
        terminated = False
        truncated = False
        info = {"suffix": self.get_task_suffix()}

        try:
            result_text = self._execute_tool(tool, params)
            # Count only non-final, non-preview as a 'tool call' toward steps
            if tool != "final":
                self.steps_taken += 1
            obs = f"Tool {tool} executed.\n{result_text}\n{self.get_task_suffix()}"
            if tool == "final":
                terminated, reward, msg = self._check_completion()
                self.terminated = terminated
                obs = f"{result_text}\n{msg}"
            # Check max turns
            if self.turn_count >= self.max_turns and not self.terminated:
                terminated = True
                truncated = True
                self.truncated = True
                obs = "Timeout: Reached maximum turns."
            return obs, reward, terminated, truncated, info

        except Exception as e:
            obs = f"ProtocolViolation: {str(e)}"
            # do not count step on violation
            if self.turn_count >= self.max_turns and not self.terminated:
                terminated = True
                truncated = True
                self.truncated = True
                obs = "Timeout: Reached maximum turns."
            return obs, 0.0, False, truncated, info


class ArtisanToolroomEnvWithFeedback(ArtisanToolroomEnv):
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
            error_detail["issue"] = "missing_or_bad_boxed_format"
            hint = "Use exactly one action like: \\boxed{open: source=sales.csv, as=sales}"

        elif "unsupportedaction" in text and "unknown tool" in text:
            error_type = "UnsupportedAction"
            m = re.search(r"unknown tool '([^']+)'", obs, flags=re.IGNORECASE)
            if m:
                error_detail["tool"] = m.group(1)
            hint = "Use only the documented tools listed in the instructions."

        elif "protocolviolation" in text:
            error_type = "ProtocolViolation"
            error_detail["message"] = obs
            if "frame not found" in text:
                hint = "Open a source first with open: source=..., as=... and use that frame name."
            elif "requires" in text:
                hint = "Check the required parameters for this tool and include them."
            else:
                hint = "Verify tool prerequisites and parameter names."

        elif "timeout" in text:
            error_type = "Timeout"
            error_detail["turn"] = getattr(self, "turn_count", None)
            hint = "Submit final before running out of turns."

        elif "final received but insufficient steps" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "insufficient_steps_before_final"
            hint = "Perform more tool calls (load, filter, join, aggregate, etc.) before final."

        elif "final received but exceeded maximum" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "too_many_steps_before_final"
            hint = "Reduce unnecessary tool calls and finalize earlier."

        elif "final answer incorrect" in text:
            error_type = "WrongDecision"
            exp = re.search(r"expected=(.*?), got=", obs, flags=re.IGNORECASE)
            got = re.search(r"got=(.*)$", obs, flags=re.IGNORECASE)
            if exp:
                error_detail["expected_hint"] = exp.group(1).strip()
            if got:
                error_detail["got"] = got.group(1).strip()
            hint = "Re-check your transformations. Ensure you derived revenue if needed, grouped correctly, and sorted before taking the head."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "steps_taken": getattr(self, "steps_taken", None),
                "required_min": getattr(self, "required_steps_min", None),
                "required_max": getattr(self, "required_steps_max", None),
                "frames": list(getattr(self, "frames", {}).keys())
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
            "hint": "Start by opening a dataset: \\boxed{open: source=sales.csv, as=sales}",
            "turn": 0,
            "state": {
                "steps_taken": 0,
                "required_min": getattr(self, "required_steps_min", None),
                "required_max": getattr(self, "required_steps_max", None),
                "frames": []
            }
        }
        return obs, info