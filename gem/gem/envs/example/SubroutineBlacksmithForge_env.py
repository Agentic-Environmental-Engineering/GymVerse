from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional

class SubroutineBlacksmithForgeEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        max_turns: Optional[int] = 100,
        drift_probability: float = 0.35,
        macro_persist: bool = True,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.max_turns = max_turns if max_turns is not None else 100
        self.min_required_steps = self.complexity
        self.max_required_steps = self.complexity * 2

        self.drift_probability = float(drift_probability)
        self.macro_persist = bool(macro_persist)

        self._init_database()

        self.library_macros: Dict[str, Dict[str, Any]] = {}
        self.macro_call_counts: Dict[str, int] = {}
        self.macro_success_counts: Dict[str, int] = {}
        self.episode_id = 0

        self.reset()

    def _init_database(self):
        self.datasets = {
            "tickets_alpha": {
                "description": "Support tickets. Fields: id, customer_tier, status, priority, created_day, minutes_open, tags (list).",
                "records": [
                    {"id": "A-001", "customer_tier": "gold", "status": "open", "priority": 3, "created_day": 1, "minutes_open": 180, "tags": ["billing"]},
                    {"id": "A-002", "customer_tier": "silver", "status": "closed", "priority": 1, "created_day": 2, "minutes_open": 20, "tags": ["howto"]},
                    {"id": "A-003", "customer_tier": "gold", "status": "open", "priority": 2, "created_day": 3, "minutes_open": 60, "tags": ["bug", "mobile"]},
                    {"id": "A-004", "customer_tier": "bronze", "status": "open", "priority": 2, "created_day": 3, "minutes_open": 600, "tags": ["billing", "refund"]},
                    {"id": "A-005", "customer_tier": "gold", "status": "closed", "priority": 3, "created_day": 4, "minutes_open": 45, "tags": ["bug"]},
                    {"id": "A-006", "customer_tier": "silver", "status": "open", "priority": 1, "created_day": 5, "minutes_open": 15, "tags": ["howto", "mobile"]},
                    {"id": "A-007", "customer_tier": "bronze", "status": "closed", "priority": 2, "created_day": 6, "minutes_open": 90, "tags": ["bug"]},
                    {"id": "A-008", "customer_tier": "gold", "status": "open", "priority": 3, "created_day": 7, "minutes_open": 240, "tags": ["refund"]},
                ],
            },
            "usage_beta": {
                "description": "Daily app usage. Fields: user_id, plan, day, sessions, minutes, region.",
                "records": [
                    {"user_id": "U10", "plan": "pro", "day": 1, "sessions": 3, "minutes": 30, "region": "NA"},
                    {"user_id": "U11", "plan": "free", "day": 1, "sessions": 1, "minutes": 5, "region": "EU"},
                    {"user_id": "U12", "plan": "pro", "day": 2, "sessions": 2, "minutes": 25, "region": "NA"},
                    {"user_id": "U10", "plan": "pro", "day": 2, "sessions": 4, "minutes": 42, "region": "NA"},
                    {"user_id": "U13", "plan": "team", "day": 2, "sessions": 5, "minutes": 55, "region": "APAC"},
                    {"user_id": "U14", "plan": "free", "day": 3, "sessions": 2, "minutes": 8, "region": "EU"},
                    {"user_id": "U15", "plan": "team", "day": 3, "sessions": 3, "minutes": 35, "region": "NA"},
                    {"user_id": "U13", "plan": "team", "day": 4, "sessions": 4, "minutes": 48, "region": "APAC"},
                ],
            },
            "orders_gamma": {
                "description": "E-commerce orders. Fields: order_id, sku, category, amount, currency, day, country, expedited (bool).",
                "records": [
                    {"order_id": "O900", "sku": "SKU-1", "category": "accessories", "amount": 25.0, "currency": "USD", "day": 1, "country": "US", "expedited": False},
                    {"order_id": "O901", "sku": "SKU-2", "category": "hardware", "amount": 199.0, "currency": "USD", "day": 1, "country": "US", "expedited": True},
                    {"order_id": "O902", "sku": "SKU-3", "category": "software", "amount": 49.0, "currency": "EUR", "day": 2, "country": "DE", "expedited": False},
                    {"order_id": "O903", "sku": "SKU-2", "category": "hardware", "amount": 199.0, "currency": "USD", "day": 2, "country": "CA", "expedited": False},
                    {"order_id": "O904", "sku": "SKU-4", "category": "accessories", "amount": 15.0, "currency": "USD", "day": 3, "country": "US", "expedited": False},
                    {"order_id": "O905", "sku": "SKU-5", "category": "hardware", "amount": 499.0, "currency": "USD", "day": 4, "country": "US", "expedited": True},
                    {"order_id": "O906", "sku": "SKU-3", "category": "software", "amount": 49.0, "currency": "USD", "day": 4, "country": "US", "expedited": False},
                    {"order_id": "O907", "sku": "SKU-1", "category": "accessories", "amount": 30.0, "currency": "EUR", "day": 5, "country": "FR", "expedited": False},
                ],
            },
        }

        self.clients = {
            "HelioSupport": {
                "style": "plain",
                "default_dataset": "tickets_alpha",
                "needs": ["triage_summary", "open_count", "priority_filter", "tag_filter"],
            },
            "KestrelGrowth": {
                "style": "json",
                "default_dataset": "usage_beta",
                "needs": ["region_breakdown", "top_users", "plan_filter", "sum_minutes"],
            },
            "OrchidOps": {
                "style": "table",
                "default_dataset": "orders_gamma",
                "needs": ["revenue_by_category", "expedited_only", "currency_convert", "day_range"],
            },
        }

        self.primitive_tool_schemas = {
            "load_dataset": {
                "description": "Load a dataset into the workspace.",
                "parameters": [{"name": "dataset_id", "type": "string"}],
                "returns": "list[dict]",
                "examples": [r'\boxed{call tool=load_dataset dataset_id=tickets_alpha}'],
            },
            "select_fields": {
                "description": "Project records to a subset of fields (keeps only these keys).",
                "parameters": [{"name": "fields", "type": "csv_strings"}],
                "returns": "list[dict]",
                "examples": [r'\boxed{call tool=select_fields fields=id,status,priority}'],
            },
            "filter_records": {
                "description": "Filter current records by a simple condition: field op value. Ops: ==, !=, >, >=, <, <=, contains (for list or string).",
                "parameters": [{"name": "field", "type": "string"}, {"name": "op", "type": "string"}, {"name": "value", "type": "string"}],
                "returns": "list[dict]",
                "examples": [r'\boxed{call tool=filter_records field=status op=== value=open}'],
            },
            "group_by": {
                "description": "Group current records by a field, storing a dict: key -> list[records].",
                "parameters": [{"name": "field", "type": "string"}],
                "returns": "dict[str, list[dict]]",
                "examples": [r'\boxed{call tool=group_by field=category}'],
            },
            "aggregate": {
                "description": "Aggregate grouped data or flat records. mode=count|sum|avg. For sum/avg provide field.",
                "parameters": [{"name": "mode", "type": "string"}, {"name": "field", "type": "string_optional"}],
                "returns": "dict or number",
                "examples": [r'\boxed{call tool=aggregate mode=count}', r'\boxed{call tool=aggregate mode=sum field=amount}'],
            },
            "sort_records": {
                "description": "Sort current flat records by field. order=asc|desc.",
                "parameters": [{"name": "field", "type": "string"}, {"name": "order", "type": "string"}],
                "returns": "list[dict]",
                "examples": [r'\boxed{call tool=sort_records field=minutes order=desc}'],
            },
            "format_output": {
                "description": "Format current result into a string. style=plain|json|table.",
                "parameters": [{"name": "style", "type": "string"}],
                "returns": "string",
                "examples": [r'\boxed{call tool=format_output style=json}'],
            },
            "deliver": {
                "description": "Deliver the final formatted report to the client. Must be called last.",
                "parameters": [{"name": "client_id", "type": "string"}, {"name": "message", "type": "string"}],
                "returns": "receipt",
                "examples": [r'\boxed{call tool=deliver client_id=HelioSupport message=$last_output}'],
            },
            "inspect_tools": {
                "description": "List currently available tools and their parameter names (after drift).",
                "parameters": [],
                "returns": "dict",
                "examples": [r'\boxed{call tool=inspect_tools}'],
            },
            "test_macro": {
                "description": "Dry-run a macro without delivering; updates workspace and returns outcome.",
                "parameters": [{"name": "macro", "type": "string"}],
                "returns": "result",
                "examples": [r'\boxed{call tool=test_macro macro=triage_spell}'],
            },
            "reset_workspace": {
                "description": "Clear workspace data/results (does not erase macros).",
                "parameters": [],
                "returns": "ok",
                "examples": [r'\boxed{call tool=reset_workspace}'],
            },
        }

        self.drift_catalog = [
            {
                "kind": "rename_tool",
                "from": "filter_records",
                "to": "where",
                "param_map": {"field": "col", "op": "op", "value": "val"},
            },
            {
                "kind": "rename_tool",
                "from": "group_by",
                "to": "bucket_by",
                "param_map": {"field": "key"},
            },
            {
                "kind": "rename_param",
                "tool": "load_dataset",
                "from": "dataset_id",
                "to": "source",
            },
            {
                "kind": "rename_param",
                "tool": "format_output",
                "from": "style",
                "to": "fmt",
            },
            {
                "kind": "rename_tool",
                "from": "sort_records",
                "to": "order_by",
                "param_map": {"field": "col", "order": "direction"},
            },
        ]

    def _get_instructions(self) -> str:
        return (
            "You are the Subroutine Blacksmith. Solve client workflow requests by calling tools or forging macros.\n"
            "Actions must be in \\boxed{...}.\n\n"
            "Action types:\n"
            "  1) Call a tool: \\boxed{call tool=TOOL_NAME key=value ...}\n"
            "  2) Forge a macro (persists across episodes): \\boxed{forge name=MACRO steps=\"call tool=...; call tool=...\"}\n"
            "  3) Cast a macro: \\boxed{cast name=MACRO}\n\n"
            "Rules:\n"
            "- Tools execute and update a workspace (current records, grouped data, last_output).\n"
            "- Interface drift may rename tools/parameters. Use inspect_tools to see the current interface.\n"
            "- You must finish by calling deliver with the correct client_id and a message.\n"
        )

    def get_task_suffix(self) -> str:
        tools_list = ", ".join(sorted(self.tools.keys()))
        macro_list = ", ".join(sorted(self.library_macros.keys())) if self.library_macros else "(none)"
        ws = self._workspace_brief()

        return (
            f"\n=== TASK ===\n{self.task['request']}\n"
            f"Required pipeline stages (must be satisfied): {', '.join(self.task['required_stages'])}\n"
            f"Client style preference: {self.task['style']}\n"
            f"Dataset hint: {self.task['dataset_id']}\n"
            f"Interface drift active: {self.drift_active}\n"
            f"Available tools: {tools_list}\n"
            f"Known macros: {macro_list}\n"
            f"Workspace: {ws}\n\n"
            "Respond with one action in this exact format:\n"
            "\\boxed{call tool=TOOL_NAME key=value ...}\n"
            "or \\boxed{forge name=MACRO steps=\"call tool=...; call tool=...\"}\n"
            "or \\boxed{cast name=MACRO}\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self.episode_id += 1
        self.turn_count = 0
        self.steps_taken = 0

        self.workspace: Dict[str, Any] = {
            "dataset_id": None,
            "records": None,
            "grouped": None,
            "aggregate": None,
            "last_output": None,
            "delivered": False,
            "delivery_receipt": None,
            "history": [],
        }

        if not self.macro_persist:
            self.library_macros = {}
            self.macro_call_counts = {}
            self.macro_success_counts = {}

        self.drift_active = random.random() < self.drift_probability
        self._apply_drift()

        required_steps = random.randint(self.min_required_steps, self.max_required_steps)
        self.task = self._generate_task_requiring_n_steps(required_steps)

        obs = self._get_instructions()
        info = {"suffix": self.get_task_suffix()}
        return obs, info

    def _apply_drift(self):
        self.tools = {}
        self.tool_aliases = {}
        self.param_aliases = {}

        for k, v in self.primitive_tool_schemas.items():
            self.tools[k] = dict(v)

        self.drift_events = []
        if self.drift_active:
            max_events = 1 + (self.complexity // 3)
            events = random.sample(self.drift_catalog, k=min(max_events, len(self.drift_catalog)))
            for ev in events:
                self.drift_events.append(ev)

            for ev in self.drift_events:
                if ev["kind"] == "rename_tool":
                    frm = ev["from"]
                    to = ev["to"]
                    if frm in self.tools and to not in self.tools:
                        self.tools[to] = self.tools.pop(frm)
                        self.tool_aliases[frm] = to
                        self.param_aliases[to] = dict(ev.get("param_map", {}))
                elif ev["kind"] == "rename_param":
                    tool = ev["tool"]
                    frm = ev["from"]
                    to = ev["to"]
                    target = tool
                    if tool in self.tool_aliases:
                        target = self.tool_aliases[tool]
                    if target in self.tools:
                        self.param_aliases.setdefault(target, {})
                        self.param_aliases[target][frm] = to

    def _generate_task_requiring_n_steps(self, required_steps: int) -> Dict[str, Any]:
        client_id = random.choice(list(self.clients.keys()))
        client = self.clients[client_id]
        dataset_id = client["default_dataset"]
        style = client["style"]

        stage_pool = []
        if dataset_id == "tickets_alpha":
            stage_pool = ["load", "filter", "filter", "aggregate", "format", "deliver"]
        elif dataset_id == "usage_beta":
            stage_pool = ["load", "filter", "group", "aggregate", "sort", "format", "deliver"]
        else:
            stage_pool = ["load", "filter", "group", "aggregate", "format", "deliver"]

        min_stages = min(len(stage_pool), max(3, required_steps))
        chosen = stage_pool[:]
        while len(chosen) < min_stages:
            chosen.append(random.choice(stage_pool))

        if required_steps <= len(chosen):
            chosen = chosen[:required_steps]
        else:
            while len(chosen) < required_steps:
                chosen.append(random.choice(stage_pool))

        required_stages = []
        for s in chosen:
            if s not in required_stages:
                required_stages.append(s)

        spec = self._make_stage_spec(client_id, dataset_id, style, required_stages)
        request = self._render_request_text(client_id, dataset_id, style, spec)

        return {
            "client_id": client_id,
            "dataset_id": dataset_id,
            "style": style,
            "required_steps": required_steps,
            "required_stages": required_stages,
            "spec": spec,
            "request": request,
        }

    def _make_stage_spec(self, client_id: str, dataset_id: str, style: str, required_stages: Any) -> Dict[str, Any]:
        if dataset_id == "tickets_alpha":
            tiers = ["gold", "silver", "bronze"]
            tier = random.choice(tiers)
            only_open = random.random() < 0.8
            min_priority = random.choice([1, 2, 3])
            tag = random.choice(["billing", "bug", "refund", "mobile", "howto"])
            return {
                "filter": [
                    ("customer_tier", "==", tier) if random.random() < 0.7 else None,
                    ("status", "==", "open") if only_open else None,
                    ("priority", ">=", str(min_priority)) if random.random() < 0.7 else None,
                    ("tags", "contains", tag) if random.random() < 0.5 else None,
                ],
                "group": None,
                "aggregate": random.choice([("count", None), ("avg", "minutes_open")]),
                "sort": None,
                "format_style": style,
            }
        if dataset_id == "usage_beta":
            plan = random.choice(["free", "pro", "team"])
            region = random.choice(["NA", "EU", "APAC"])
            return {
                "filter": [
                    ("plan", "==", plan) if random.random() < 0.7 else None,
                    ("region", "==", region) if random.random() < 0.6 else None,
                    ("sessions", ">=", str(random.choice([2, 3, 4]))) if random.random() < 0.6 else None,
                ],
                "group": random.choice(["user_id", "region", "plan"]),
                "aggregate": random.choice([("sum", "minutes"), ("count", None), ("avg", "sessions")]),
                "sort": ("minutes", "desc") if random.random() < 0.7 else ("sessions", "desc"),
                "format_style": style,
            }
        expedited = random.random() < 0.6
        category = random.choice(["hardware", "software", "accessories"])
        country = random.choice(["US", "CA", "DE", "FR"])
        return {
            "filter": [
                ("category", "==", category) if random.random() < 0.7 else None,
                ("country", "==", country) if random.random() < 0.5 else None,
                ("expedited", "==", "True") if expedited else None,
            ],
            "group": "category" if random.random() < 0.7 else "country",
            "aggregate": ("sum", "amount"),
            "sort": None,
            "format_style": style,
        }

    def _render_request_text(self, client_id: str, dataset_id: str, style: str, spec: Dict[str, Any]) -> str:
        parts = [f"Client {client_id} requests a report from dataset '{dataset_id}'."]
        filters = [f for f in spec.get("filter", []) if f]
        if filters:
            parts.append("Apply filters: " + ", ".join([f"{a} {b} {c}" for (a, b, c) in filters]) + ".")
        if spec.get("group"):
            parts.append(f"Group by '{spec['group']}'.")
        if spec.get("aggregate"):
            mode, field = spec["aggregate"]
            if field:
                parts.append(f"Aggregate: {mode} of '{field}'.")
            else:
                parts.append(f"Aggregate: {mode}.")
        if spec.get("sort"):
            field, order = spec["sort"]
            parts.append(f"Sort by '{field}' {order}.")
        parts.append(f"Format as {style} and then deliver to the client.")
        parts.append("Note: tools may have drifted; inspect_tools can reveal the current interface.")
        return " ".join(parts)

    def _workspace_brief(self) -> str:
        ds = self.workspace.get("dataset_id")
        recs = self.workspace.get("records")
        grouped = self.workspace.get("grouped")
        agg = self.workspace.get("aggregate")
        out = self.workspace.get("last_output")
        delivered = self.workspace.get("delivered")
        nrecs = None if recs is None else len(recs)
        ngroups = None if grouped is None else len(grouped)
        return f"dataset={ds}, records={nrecs}, groups={ngroups}, aggregate={'set' if agg is not None else 'none'}, last_output={'set' if out else 'none'}, delivered={delivered}"

    def _parse_action(self, action: str):
        if action is None:
            return None
        m = re.search(r"\\boxed\{(.+?)\}\s*$", action.strip(), flags=re.DOTALL)
        if not m:
            return None
        inner = m.group(1).strip()

        if inner.startswith("call "):
            rest = inner[len("call ") :].strip()
            tokens = self._tokenize_kv(rest)
            if "tool" not in tokens:
                return {"type": "call", "tool": None, "args": tokens}
            tool = tokens.pop("tool")
            return {"type": "call", "tool": tool, "args": tokens}

        if inner.startswith("forge "):
            rest = inner[len("forge ") :].strip()
            tokens = self._tokenize_kv(rest, allow_quoted=True)
            name = tokens.get("name")
            steps = tokens.get("steps")
            return {"type": "forge", "name": name, "steps": steps}

        if inner.startswith("cast "):
            rest = inner[len("cast ") :].strip()
            tokens = self._tokenize_kv(rest)
            name = tokens.get("name")
            return {"type": "cast", "name": name}

        return {"type": "unknown", "raw": inner}

    def _tokenize_kv(self, s: str, allow_quoted: bool = True) -> Dict[str, str]:
        out: Dict[str, str] = {}
        i = 0
        n = len(s)
        while i < n:
            while i < n and s[i].isspace():
                i += 1
            if i >= n:
                break
            j = i
            while j < n and s[j] not in "= ":
                j += 1
            key = s[i:j].strip()
            while j < n and s[j].isspace():
                j += 1
            if j >= n or s[j] != "=":
                k = j
                while k < n and not s[k].isspace():
                    k += 1
                i = k
                continue
            j += 1
            while j < n and s[j].isspace():
                j += 1
            if allow_quoted and j < n and s[j] == '"':
                j += 1
                k = j
                while k < n and s[k] != '"':
                    k += 1
                val = s[j:k]
                j = k + 1 if k < n else k
            else:
                k = j
                while k < n and not s[k].isspace():
                    k += 1
                val = s[j:k]
                j = k
            if key:
                out[key] = val
            i = j
        return out

    def sample_random_action(self) -> str:
        choices = [
            r"\boxed{call tool=inspect_tools}",
            r"\boxed{call tool=load_dataset dataset_id=tickets_alpha}",
            r"\boxed{call tool=reset_workspace}",
            r"\boxed{forge name=quick_spell steps=\"call tool=inspect_tools; call tool=reset_workspace\"}",
        ]
        if self.library_macros:
            any_macro = random.choice(list(self.library_macros.keys()))
            choices.append(rf"\boxed{{cast name={any_macro}}}")
        return random.choice(choices)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        info: Dict[str, Any] = {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            obs = "TIMEOUT: max_turns reached."
            return obs, -1.0, True, True, info

        parsed = self._parse_action(action)
        if parsed is None:
            obs = "INVALID ACTION FORMAT: Use \\boxed{...} with a supported action (call/forge/cast)."
            return obs, float(LanguageGameReward.format_error_reward), True, False, info

        if parsed["type"] == "unknown":
            obs = f"UNSUPPORTED ACTION: '{parsed.get('raw','')}'. Use call, forge, or cast."
            return obs, -0.5, True, False, info

        if parsed["type"] == "forge":
            name = parsed.get("name")
            steps = parsed.get("steps")
            if not name or not re.match(r"^[A-Za-z_][A-Za-z0-9_]{0,32}$", name):
                return "PROTOCOL VIOLATION: forge requires name=[A-Za-z_][A-Za-z0-9_]*.", -0.5, True, False, info
            if not steps or len(steps.strip()) == 0:
                return "PROTOCOL VIOLATION: forge requires steps=\"...\" containing semicolon-separated call actions.", -0.5, True, False, info
            calls = [c.strip() for c in steps.split(";") if c.strip()]
            if not calls:
                return "PROTOCOL VIOLATION: macro has no steps.", -0.5, True, False, info
            if len(calls) > max(3, 4 + self.complexity):
                return "PROTOCOL VIOLATION: macro too long for this forge (reduce steps).", -0.5, True, False, info
            normalized_steps = []
            for c in calls:
                fake = r"\boxed{" + c + "}"
                p = self._parse_action(fake)
                if not p or p["type"] != "call":
                    return "PROTOCOL VIOLATION: macro steps must each be 'call tool=...'.", -0.5, True, False, info
                if not p.get("tool"):
                    return "PROTOCOL VIOLATION: each macro step must include tool=...", -0.5, True, False, info
                normalized_steps.append({"tool": p["tool"], "args": p["args"]})
            self.library_macros[name] = {"steps": normalized_steps, "created_episode": self.episode_id}
            self.macro_call_counts.setdefault(name, 0)
            self.macro_success_counts.setdefault(name, 0)
            obs = f"OK: Forged macro '{name}' with {len(normalized_steps)} steps. It will persist across episodes."
            return obs, 0.2, False, False, info

        if parsed["type"] == "cast":
            name = parsed.get("name")
            if not name or name not in self.library_macros:
                return f"UNSUPPORTED ACTION: unknown macro '{name}'.", -0.5, True, False, info
            self.macro_call_counts[name] = self.macro_call_counts.get(name, 0) + 1
            obs_lines = [f"OK: Casting macro '{name}' ({len(self.library_macros[name]['steps'])} steps)."]
            macro_ok = True
            for idx, st in enumerate(self.library_macros[name]["steps"], start=1):
                tool = st["tool"]
                args = dict(st["args"])
                step_obs, ok, proto, unsup = self._execute_call(tool, args, count_step=True)
                obs_lines.append(f"  step {idx}: {step_obs}")
                if not ok:
                    macro_ok = False
                    break
            if macro_ok:
                self.macro_success_counts[name] = self.macro_success_counts.get(name, 0) + 1
            obs = "\n".join(obs_lines)
            done_obs, done_reward, done_term = self._maybe_score_and_finish()
            if done_term:
                return obs + "\n" + done_obs, done_reward, True, False, info
            return obs, 0.0 if macro_ok else -0.2, False, False, info

        if parsed["type"] == "call":
            tool = parsed.get("tool")
            args = parsed.get("args", {})
            if not tool:
                return "PROTOCOL VIOLATION: call requires tool=TOOL_NAME.", -0.5, True, False, info
            step_obs, ok, proto, unsup = self._execute_call(tool, args, count_step=True)
            if unsup:
                return step_obs, -0.5, True, False, info
            if proto:
                return step_obs, -0.5, True, False, info
            if not ok:
                return step_obs, -0.2, False, False, info

            done_obs, done_reward, done_term = self._maybe_score_and_finish()
            if done_term:
                return step_obs + "\n" + done_obs, done_reward, True, False, info
            return step_obs, 0.0, False, False, info

        return "UNSUPPORTED ACTION.", -0.5, True, False, info

    def _maybe_score_and_finish(self) -> Tuple[str, float, bool]:
        if not self.workspace.get("delivered"):
            return "OK: Continue.", 0.0, False

        receipt = self.workspace.get("delivery_receipt", {})
        if not receipt.get("ok"):
            return "FAILURE: deliver was called but receipt indicates failure.", -1.0, True

        stage_cov = self._stage_coverage()
        missing = [s for s in self.task["required_stages"] if s not in stage_cov]
        if missing:
            return f"WRONG DECISION: delivered but missing required stages: {', '.join(missing)}.", -1.0, True

        msg = receipt.get("message", "")
        if not isinstance(msg, str) or len(msg.strip()) == 0:
            return "WRONG DECISION: delivered empty message.", -1.0, True

        style = self.task["style"]
        if style == "json":
            if not (msg.strip().startswith("{") and msg.strip().endswith("}")):
                return "WRONG DECISION: expected json-like message (must start with '{' and end with '}').", -1.0, True
        if style == "table":
            if "|" not in msg:
                return "WRONG DECISION: expected table-like message (must contain '|').", -1.0, True

        reward = 1.0
        if self.steps_taken <= max(1, self.task["required_steps"]):
            reward += 0.2
        if self.library_macros:
            reusable = 0
            for name in self.library_macros:
                if self.macro_call_counts.get(name, 0) >= 1:
                    reusable += 1
            reward += min(0.3, 0.1 * reusable)

        return "SUCCESS: client workflow delivered with required stages satisfied.", reward, True

    def _stage_coverage(self):
        cov = set()
        for e in self.workspace.get("history", []):
            if e.get("stage"):
                cov.add(e["stage"])
        if self.workspace.get("delivered"):
            cov.add("deliver")
        return cov

    def _coerce_value(self, val: str):
        if val is None:
            return None
        if val.lower() == "true":
            return True
        if val.lower() == "false":
            return False
        if re.fullmatch(r"-?\d+", val):
            return int(val)
        if re.fullmatch(r"-?\d+\.\d+", val):
            return float(val)
        return val

    def _apply_param_aliases(self, tool: str, args: Dict[str, str]) -> Dict[str, str]:
        amap = self.param_aliases.get(tool, {})
        out = {}
        for k, v in args.items():
            nk = amap.get(k, k)
            out[nk] = v
        return out

    def _execute_call(self, tool: str, args: Dict[str, str], count_step: bool) -> Tuple[str, bool, bool, bool]:
        if tool in self.tool_aliases:
            return (
                f"PROTOCOL VIOLATION: tool '{tool}' drifted to '{self.tool_aliases[tool]}'. Use inspect_tools and update your call.",
                False,
                True,
                False,
            )
        if tool not in self.tools:
            return f"UNSUPPORTED ACTION: unknown tool '{tool}'. Try inspect_tools.", False, False, True

        args = dict(args)
        args = self._apply_param_aliases(tool, args)

        if count_step:
            self.steps_taken += 1

        try:
            if tool == "inspect_tools":
                schema_view = {}
                for tname, meta in self.tools.items():
                    params = [p["name"] for p in meta.get("parameters", [])]
                    aliases = self.param_aliases.get(tname, {})
                    schema_view[tname] = {"params": params, "param_aliases": aliases, "description": meta.get("description", "")}
                self.workspace["history"].append({"tool": tool, "stage": "inspect"})
                return f"OK: Tools schema: {schema_view}", True, False, False

            if tool == "reset_workspace":
                keep_hist = self.workspace.get("history", [])
                self.workspace = {
                    "dataset_id": None,
                    "records": None,
                    "grouped": None,
                    "aggregate": None,
                    "last_output": None,
                    "delivered": False,
                    "delivery_receipt": None,
                    "history": keep_hist + [{"tool": tool, "stage": "reset"}],
                }
                return "OK: Workspace reset (macros preserved).", True, False, False

            if tool in ("load_dataset",):
                key = "dataset_id"
                if self.drift_active:
                    if self.param_aliases.get(tool, {}).get("dataset_id") == "source":
                        key = "source"
                ds = args.get(key)
                if ds not in self.datasets:
                    return f"WRONG DECISION: dataset '{ds}' not found.", False, False, False
                self.workspace["dataset_id"] = ds
                self.workspace["records"] = [dict(r) for r in self.datasets[ds]["records"]]
                self.workspace["grouped"] = None
                self.workspace["aggregate"] = None
                self.workspace["last_output"] = None
                self.workspace["history"].append({"tool": tool, "stage": "load", "dataset_id": ds})
                return f"OK: Loaded dataset '{ds}' with {len(self.workspace['records'])} records.", True, False, False

            if tool in ("select_fields",):
                if self.workspace.get("records") is None:
                    return "PROTOCOL VIOLATION: no records loaded. Call load_dataset first.", False, False, False
                fields = args.get("fields", "")
                flds = [x.strip() for x in fields.split(",") if x.strip()]
                if not flds:
                    return "PROTOCOL VIOLATION: select_fields requires fields=a,b,c.", False, False, False
                newrecs = []
                for r in self.workspace["records"]:
                    nr = {}
                    for f in flds:
                        if f in r:
                            nr[f] = r[f]
                    newrecs.append(nr)
                self.workspace["records"] = newrecs
                self.workspace["grouped"] = None
                self.workspace["aggregate"] = None
                self.workspace["last_output"] = None
                self.workspace["history"].append({"tool": tool, "stage": "select", "fields": flds})
                return f"OK: Selected fields {flds}. Records now have keys subset.", True, False, False

            if tool in ("filter_records", "where"):
                if self.workspace.get("records") is None:
                    return "PROTOCOL VIOLATION: no records loaded. Call load_dataset first.", False, False, False

                field_key = "field"
                value_key = "value"
                if tool == "where":
                    field_key = "col"
                    value_key = "val"

                field = args.get(field_key)
                op = args.get("op")
                value = args.get(value_key)
                if field is None or op is None or value is None:
                    return f"PROTOCOL VIOLATION: {tool} requires {field_key}=... op=... {value_key}=...", False, False, False

                v = self._coerce_value(value)
                filtered = []
                for r in self.workspace["records"]:
                    rv = r.get(field, None)
                    ok = False
                    if op == "==":
                        ok = rv == v
                    elif op == "!=":
                        ok = rv != v
                    elif op == ">":
                        ok = isinstance(rv, (int, float)) and isinstance(v, (int, float)) and (rv > v)
                    elif op == ">=":
                        ok = isinstance(rv, (int, float)) and isinstance(v, (int, float)) and (rv >= v)
                    elif op == "<":
                        ok = isinstance(rv, (int, float)) and isinstance(v, (int, float)) and (rv < v)
                    elif op == "<=":
                        ok = isinstance(rv, (int, float)) and isinstance(v, (int, float)) and (rv <= v)
                    elif op == "contains":
                        if isinstance(rv, list):
                            ok = v in rv
                        elif isinstance(rv, str) and isinstance(v, str):
                            ok = v in rv
                        else:
                            ok = False
                    else:
                        return f"PROTOCOL VIOLATION: unsupported op '{op}'.", False, False, False
                    if ok:
                        filtered.append(r)
                self.workspace["records"] = filtered
                self.workspace["grouped"] = None
                self.workspace["aggregate"] = None
                self.workspace["last_output"] = None
                self.workspace["history"].append({"tool": tool, "stage": "filter", "field": field, "op": op, "value": v})
                return f"OK: Filtered records on {field} {op} {value}. Remaining {len(filtered)} records.", True, False, False

            if tool in ("group_by", "bucket_by"):
                if self.workspace.get("records") is None:
                    return "PROTOCOL VIOLATION: no records loaded. Call load_dataset first.", False, False, False
                field_key = "field" if tool == "group_by" else "key"
                field = args.get(field_key)
                if not field:
                    return f"PROTOCOL VIOLATION: {tool} requires {field_key}=...", False, False, False
                groups: Dict[str, Any] = {}
                for r in self.workspace["records"]:
                    k = r.get(field, None)
                    ks = str(k)
                    groups.setdefault(ks, []).append(r)
                self.workspace["grouped"] = groups
                self.workspace["aggregate"] = None
                self.workspace["last_output"] = None
                self.workspace["history"].append({"tool": tool, "stage": "group", "field": field})
                return f"OK: Grouped by '{field}' into {len(groups)} groups.", True, False, False

            if tool in ("aggregate",):
                mode = args.get("mode")
                field = args.get("field")
                if not mode:
                    return "PROTOCOL VIOLATION: aggregate requires mode=count|sum|avg.", False, False, False

                if self.workspace.get("grouped") is not None:
                    groups = self.workspace["grouped"]
                    result = {}
                    for gk, items in groups.items():
                        if mode == "count":
                            result[gk] = len(items)
                        elif mode in ("sum", "avg"):
                            if not field:
                                return "PROTOCOL VIOLATION: aggregate sum/avg requires field=...", False, False, False
                            vals = [it.get(field, None) for it in items]
                            vals = [x for x in vals if isinstance(x, (int, float))]
                            s = sum(vals) if vals else 0.0
                            result[gk] = s if mode == "sum" else (s / len(vals) if vals else 0.0)
                        else:
                            return f"PROTOCOL VIOLATION: unknown mode '{mode}'.", False, False, False
                    self.workspace["aggregate"] = result
                else:
                    if self.workspace.get("records") is None:
                        return "PROTOCOL VIOLATION: nothing to aggregate. Load a dataset first.", False, False, False
                    recs = self.workspace["records"]
                    if mode == "count":
                        self.workspace["aggregate"] = len(recs)
                    elif mode in ("sum", "avg"):
                        if not field:
                            return "PROTOCOL VIOLATION: aggregate sum/avg requires field=...", False, False, False
                        vals = [r.get(field, None) for r in recs]
                        vals = [x for x in vals if isinstance(x, (int, float))]
                        s = sum(vals) if vals else 0.0
                        self.workspace["aggregate"] = s if mode == "sum" else (s / len(vals) if vals else 0.0)
                    else:
                        return f"PROTOCOL VIOLATION: unknown mode '{mode}'.", False, False, False

                self.workspace["last_output"] = None
                self.workspace["history"].append({"tool": tool, "stage": "aggregate", "mode": mode, "field": field})
                return f"OK: Aggregated with mode={mode}" + (f", field={field}" if field else "") + ".", True, False, False

            if tool in ("sort_records", "order_by"):
                if self.workspace.get("records") is None:
                    return "PROTOCOL VIOLATION: no records loaded. Call load_dataset first.", False, False, False
                col_key = "field" if tool == "sort_records" else "col"
                ord_key = "order" if tool == "sort_records" else "direction"
                field = args.get(col_key)
                order = args.get(ord_key, "asc")
                if not field:
                    return f"PROTOCOL VIOLATION: {tool} requires {col_key}=...", False, False, False
                if order not in ("asc", "desc"):
                    return "PROTOCOL VIOLATION: order/direction must be asc|desc.", False, False, False
                rev = order == "desc"
                self.workspace["records"] = sorted(
                    self.workspace["records"],
                    key=lambda r: (r.get(field) is None, r.get(field)),
                    reverse=rev,
                )
                self.workspace["grouped"] = None
                self.workspace["aggregate"] = None
                self.workspace["last_output"] = None
                self.workspace["history"].append({"tool": tool, "stage": "sort", "field": field, "order": order})
                return f"OK: Sorted records by '{field}' {order}.", True, False, False

            if tool in ("format_output",):
                fmt_key = "style"
                if self.drift_active:
                    if self.param_aliases.get(tool, {}).get("style") == "fmt":
                        fmt_key = "fmt"
                style = args.get(fmt_key)
                if style not in ("plain", "json", "table"):
                    return f"PROTOCOL VIOLATION: format_output requires {fmt_key}=plain|json|table.", False, False, False

                rendered = None
                if self.workspace.get("aggregate") is not None:
                    rendered = self._render_value(self.workspace["aggregate"], style)
                elif self.workspace.get("grouped") is not None:
                    summary = {k: len(v) for k, v in self.workspace["grouped"].items()}
                    rendered = self._render_value(summary, style)
                elif self.workspace.get("records") is not None:
                    rendered = self._render_value(self.workspace["records"][: min(8, len(self.workspace["records"]))], style)
                else:
                    return "PROTOCOL VIOLATION: nothing to format. Load/compute something first.", False, False, False

                self.workspace["last_output"] = rendered
                self.workspace["history"].append({"tool": tool, "stage": "format", "style": style})
                return f"OK: Formatted output as {style}. last_output is set.", True, False, False

            if tool in ("deliver",):
                client_id = args.get("client_id")
                message = args.get("message")
                if not client_id:
                    return "PROTOCOL VIOLATION: deliver requires client_id=...", False, False, False
                if client_id != self.task["client_id"]:
                    self.workspace["delivered"] = True
                    self.workspace["delivery_receipt"] = {"ok": False, "reason": "wrong_client", "client_id": client_id, "message": message}
                    self.workspace["history"].append({"tool": tool, "stage": "deliver", "ok": False})
                    return f"WRONG DECISION: delivered to wrong client '{client_id}'.", False, False, True
                if message == "$last_output":
                    message = self.workspace.get("last_output")
                if message is None:
                    self.workspace["delivered"] = True
                    self.workspace["delivery_receipt"] = {"ok": False, "reason": "missing_message", "client_id": client_id, "message": ""}
                    self.workspace["history"].append({"tool": tool, "stage": "deliver", "ok": False})
                    return "PROTOCOL VIOLATION: deliver message is missing (use message=$last_output after format_output).", False, False, False

                self.workspace["delivered"] = True
                self.workspace["delivery_receipt"] = {"ok": True, "client_id": client_id, "message": str(message)}
                self.workspace["history"].append({"tool": tool, "stage": "deliver", "ok": True})
                return "OK: Delivered report. Receipt ok=true.", True, False, False

            if tool in ("test_macro",):
                macro = args.get("macro")
                if not macro or macro not in self.library_macros:
                    return f"UNSUPPORTED ACTION: unknown macro '{macro}'.", False, False, True
                original_delivered = self.workspace.get("delivered")
                original_receipt = self.workspace.get("delivery_receipt")
                self.workspace["delivered"] = False
                self.workspace["delivery_receipt"] = None
                ok_all = True
                results = []
                for st in self.library_macros[macro]["steps"]:
                    step_obs, ok, proto, unsup = self._execute_call(st["tool"], dict(st["args"]), count_step=True)
                    results.append(step_obs)
                    if not ok:
                        ok_all = False
                        break
                self.workspace["delivered"] = original_delivered
                self.workspace["delivery_receipt"] = original_receipt
                self.workspace["history"].append({"tool": tool, "stage": "test", "macro": macro, "ok": ok_all})
                return "OK: test_macro result=" + ("success" if ok_all else "failure") + ". " + " | ".join(results[-3:]), True, False, False

            return f"UNSUPPORTED ACTION: tool '{tool}' exists but is not executable.", False, False, True

        except Exception as e:
            return f"EXECUTION ERROR: {type(e).__name__}: {e}", False, False, False

    def _render_value(self, val: Any, style: str) -> str:
        if style == "plain":
            return str(val)
        if style == "json":
            return self._to_json_like(val)
        return self._to_table_like(val)

    def _to_json_like(self, val: Any) -> str:
        if isinstance(val, dict):
            items = []
            for k in sorted(val.keys(), key=lambda x: str(x)):
                items.append(f"\"{str(k)}\": {self._to_json_like(val[k])}")
            return "{" + ", ".join(items) + "}"
        if isinstance(val, list):
            return "[" + ", ".join(self._to_json_like(x) for x in val) + "]"
        if isinstance(val, str):
            return "\"" + val.replace('"', '\\"') + "\""
        if isinstance(val, bool):
            return "true" if val else "false"
        if val is None:
            return "null"
        return str(val)

    def _to_table_like(self, val: Any) -> str:
        if isinstance(val, dict):
            rows = [("key", "value")]
            for k in sorted(val.keys(), key=lambda x: str(x)):
                rows.append((str(k), str(val[k])))
            return self._render_rows(rows)
        if isinstance(val, list) and val and isinstance(val[0], dict):
            keys = sorted({kk for r in val for kk in r.keys()}, key=lambda x: str(x))
            rows = [tuple(keys)]
            for r in val[: min(8, len(val))]:
                rows.append(tuple(str(r.get(k, "")) for k in keys))
            return self._render_rows(rows)
        return self._render_rows([("value",), (str(val),)])

    def _render_rows(self, rows: Any) -> str:
        str_rows = [tuple(str(x) for x in row) for row in rows]
        widths = []
        max_cols = max(len(r) for r in str_rows)
        for c in range(max_cols):
            widths.append(max(len(r[c]) if c < len(r) else 0 for r in str_rows))
        out_lines = []
        for i, r in enumerate(str_rows):
            cells = []
            for c in range(max_cols):
                cell = r[c] if c < len(r) else ""
                cells.append(cell.ljust(widths[c]))
            line = "| " + " | ".join(cells) + " |"
            out_lines.append(line)
            if i == 0:
                sep = "|-" + "-|-".join("-" * w for w in widths) + "-|"
                out_lines.append(sep)
        return "\n".join(out_lines)


class SubroutineBlacksmithForgeEnvWithFeedback(SubroutineBlacksmithForgeEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = int(feedback_level)
        super().__init__(**kwargs)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        diag = {"error_type": "OK"}
        if self.feedback_level >= 1:
            diag["error_detail"] = {"outcome": "episode_start"}
            diag["turn"] = 0
            diag["state"] = {
                "drift_active": getattr(self, "drift_active", None),
                "known_macros": sorted(list(getattr(self, "library_macros", {}).keys())),
                "required_stages": getattr(self, "task", {}).get("required_stages", None),
            }
        if self.feedback_level >= 2:
            diag["hint"] = "Consider starting with \\boxed{call tool=inspect_tools} if drift is active, then load the hinted dataset."
        info["diagnostic"] = diag
        return obs, info

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        text = (obs or "").lower()
        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint = None

        if "timeout" in text:
            error_type = "Timeout"
            error_detail["issue"] = "max_turns"
            hint = "Use macros to compress repeated multi-step pipelines and deliver earlier."
        elif "invalid action format" in text:
            error_type = "FormatError"
            error_detail["issue"] = "missing_boxed"
            hint = "Wrap exactly one action in \\boxed{...}, e.g. \\boxed{call tool=inspect_tools}."
        elif "unsupported action" in text and "unknown tool" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_tool"
            hint = "Call inspect_tools to see the current tool names (drift may have renamed them)."
        elif "unsupported action" in text and "unknown macro" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_macro"
            hint = "Forge the macro first, or cast an existing macro name exactly as listed."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "drifted to" in text:
                error_detail["violation"] = "used_old_tool_name"
                hint = "Use the new tool name shown in the message, or inspect_tools and update your macro/call."
            elif "no records loaded" in text or "call load_dataset first" in text:
                error_detail["violation"] = "missing_prerequisite_load"
                hint = "Load the dataset before filtering/grouping/sorting/selecting."
            elif "nothing to format" in text:
                error_detail["violation"] = "format_without_data"
                hint = "Compute an aggregate or keep records available, then call format_output before deliver."
            elif "deliver requires" in text:
                error_detail["violation"] = "deliver_args"
                hint = "Use deliver with client_id=... and message=$last_output after format_output."
            else:
                error_detail["violation"] = "generic_protocol"
                hint = "Follow the tool's required parameters exactly (inspect_tools reveals current parameter names)."
        elif "wrong decision" in text:
            error_type = "WrongDecision"
            if "wrong client" in text:
                error_detail["issue"] = "wrong_client"
                hint = "Deliver to the exact client_id in the task."
            elif "missing required stages" in text:
                error_detail["issue"] = "missing_stages"
                hint = "Ensure you performed the required pipeline stages (load/filter/group/aggregate/format) before deliver."
            elif "expected json-like" in text or "expected table-like" in text:
                error_detail["issue"] = "style_mismatch"
                hint = "Use format_output with the client’s preferred style, then deliver message=$last_output."
            else:
                error_detail["issue"] = "generic_wrong"
                hint = "Compare your workspace state with the requested filters/group/aggregate and adjust steps."
        elif "execution error" in text:
            error_type = "ProtocolViolation"
            error_detail["issue"] = "runtime_exception"
            hint = "Check parameter names and value types; inspect_tools helps under drift."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "workspace": getattr(self, "workspace", None) and {
                    "dataset_id": self.workspace.get("dataset_id"),
                    "records_loaded": self.workspace.get("records") is not None,
                    "grouped": self.workspace.get("grouped") is not None,
                    "aggregate": self.workspace.get("aggregate") is not None,
                    "last_output": self.workspace.get("last_output") is not None,
                    "delivered": self.workspace.get("delivered"),
                },
                "drift_active": getattr(self, "drift_active", None),
                "available_tools": sorted(list(getattr(self, "tools", {}).keys())),
                "known_macros": sorted(list(getattr(self, "library_macros", {}).keys())),
                "steps_taken": getattr(self, "steps_taken", None),
            }
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info