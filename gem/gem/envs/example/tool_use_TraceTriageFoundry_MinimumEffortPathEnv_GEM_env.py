from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class TraceTriageFoundryEnv(Env):
    def __init__(self, complexity: int = 1, max_turns: Optional[int] = 100, **_):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.max_turns = max_turns if max_turns is not None else 100

        self.min_required_steps = self.complexity
        self.max_required_steps = self.complexity * 2

        self._init_database()
        self.reset()

    def _init_database(self) -> None:
        base_tools = [
            "catalog_tools",
            "list_sources",
            "load_source",
            "select_active",
            "grep",
            "extract_errors",
            "timeline",
            "count_by_tool",
            "diff_sources",
            "summarize_state",
            "submit_root_cause",
        ]

        optional_tools = [
            "filter_by_severity",
            "top_k_messages",
            "unique_request_ids",
            "inspect_tool_schema",
        ]

        num_optional = max(0, min(len(optional_tools), self.complexity - 1))
        tool_names = base_tools + optional_tools[:num_optional]

        self.tools: Dict[str, Dict[str, Any]] = {}
        for name in tool_names:
            self.tools[name] = {"name": name}

        self.tool_schemas: Dict[str, Dict[str, Any]] = {
            "catalog_tools": {
                "description": "List available tools and brief descriptions.",
                "parameters": [],
                "returns": "text",
                "example": r"\boxed{catalog_tools()}",
            },
            "inspect_tool_schema": {
                "description": "Show detailed schema for a tool.",
                "parameters": [{"name": "name", "type": "string"}],
                "returns": "json",
                "example": r"\boxed{inspect_tool_schema(name='grep')}",
            },
            "list_sources": {
                "description": "List log sources available in the incident workspace.",
                "parameters": [],
                "returns": "list[string]",
                "example": r"\boxed{list_sources()}",
            },
            "load_source": {
                "description": "Load a source into memory for analysis.",
                "parameters": [{"name": "source", "type": "string"}],
                "returns": "summary",
                "example": r"\boxed{load_source(source='svc_payments')}",
            },
            "select_active": {
                "description": "Select which loaded source is active for subsequent tools.",
                "parameters": [{"name": "source", "type": "string"}],
                "returns": "ok",
                "example": r"\boxed{select_active(source='svc_payments')}",
            },
            "grep": {
                "description": "Search active source for a substring; returns matching lines with indices.",
                "parameters": [{"name": "pattern", "type": "string"}, {"name": "limit", "type": "int"}],
                "returns": "matches",
                "example": r"\boxed{grep(pattern='ERROR', limit=10)}",
            },
            "extract_errors": {
                "description": "Parse active source and extract structured error events.",
                "parameters": [{"name": "limit", "type": "int"}],
                "returns": "events",
                "example": r"\boxed{extract_errors(limit=50)}",
            },
            "filter_by_severity": {
                "description": "Filter extracted events by severity (INFO/WARN/ERROR).",
                "parameters": [{"name": "severity", "type": "string"}],
                "returns": "events",
                "example": r"\boxed{filter_by_severity(severity='ERROR')}",
            },
            "timeline": {
                "description": "Show time-ordered view of extracted events (active source).",
                "parameters": [{"name": "limit", "type": "int"}],
                "returns": "timeline",
                "example": r"\boxed{timeline(limit=20)}",
            },
            "count_by_tool": {
                "description": "Count extracted events by tool/function name.",
                "parameters": [],
                "returns": "dict",
                "example": r"\boxed{count_by_tool()}",
            },
            "top_k_messages": {
                "description": "Show the most frequent error messages from extracted events.",
                "parameters": [{"name": "k", "type": "int"}],
                "returns": "list",
                "example": r"\boxed{top_k_messages(k=3)}",
            },
            "unique_request_ids": {
                "description": "List unique request IDs seen in extracted events.",
                "parameters": [{"name": "limit", "type": "int"}],
                "returns": "list[string]",
                "example": r"\boxed{unique_request_ids(limit=20)}",
            },
            "diff_sources": {
                "description": "Compare two loaded sources for shared request IDs in extracted events.",
                "parameters": [{"name": "source_a", "type": "string"}, {"name": "source_b", "type": "string"}],
                "returns": "summary",
                "example": r"\boxed{diff_sources(source_a='svc_api', source_b='svc_payments')}",
            },
            "summarize_state": {
                "description": "Get a compact summary of current workspace state.",
                "parameters": [],
                "returns": "summary",
                "example": r"\boxed{summarize_state()}",
            },
            "submit_root_cause": {
                "description": "Terminate and submit your answer: the single tool name that caused the incident.",
                "parameters": [{"name": "tool", "type": "string"}],
                "returns": "final",
                "example": r"\boxed{submit_root_cause(tool='token_sign')}",
            },
        }

        self.tool_descriptions: Dict[str, str] = {
            "catalog_tools": "Show the tool catalog you can call.",
            "inspect_tool_schema": "Get the schema for one tool.",
            "list_sources": "List available log sources.",
            "load_source": "Load a log source into memory.",
            "select_active": "Choose which loaded source is active.",
            "grep": "Search active source for a substring.",
            "extract_errors": "Extract structured error events from active source.",
            "filter_by_severity": "Filter extracted events by severity.",
            "timeline": "Show extracted events in time order.",
            "count_by_tool": "Count extracted events by tool name.",
            "top_k_messages": "Show most frequent error messages.",
            "unique_request_ids": "List request IDs from extracted events.",
            "diff_sources": "Compare extracted request IDs across two sources.",
            "summarize_state": "Summarize current workspace state.",
            "submit_root_cause": "Submit final root-cause tool name and end the episode.",
        }

        tool_pool = [
            "token_sign",
            "token_verify",
            "rate_limit_check",
            "db_connect",
            "db_query",
            "cache_get",
            "cache_set",
            "json_parse",
            "schema_validate",
            "retry_policy",
            "circuit_breaker",
            "http_client_send",
            "payment_capture",
            "refund_issue",
            "idempotency_guard",
            "metrics_emit",
            "trace_propagate",
        ]
        pool_size = min(len(tool_pool), 6 + self.complexity * 2)
        self.root_cause_pool = tool_pool[:pool_size]

        message_templates = [
            "bad signature",
            "missing field",
            "schema mismatch",
            "upstream timeout",
            "rate limited",
            "connection refused",
            "unexpected null",
            "invalid json",
            "duplicate request",
            "permission denied",
        ]
        message_size = min(len(message_templates), 5 + self.complexity)
        self.message_pool = message_templates[:message_size]

        self.severities = ["INFO", "WARN", "ERROR"]

    def _generate_task_requiring_n_steps(self, required_steps: int) -> Dict[str, Any]:
        num_sources = max(2, min(6, 2 + (required_steps // 2)))
        source_names = [f"svc_{name}" for name in ["api", "payments", "auth", "ledger", "checkout", "webhook"]]
        random.shuffle(source_names)
        sources = source_names[:num_sources]

        root_cause_tool = random.choice(self.root_cause_pool)
        incident_id = f"INC-{random.randint(1000, 9999)}"
        window = f"{random.randint(10,23):02d}:{random.randint(0,59):02d}Z-{random.randint(10,23):02d}:{random.randint(0,59):02d}Z"

        shared_req_ids = [f"req_{random.randint(10000,99999)}" for _ in range(max(3, min(10, required_steps + 2)))]
        root_req_ids = random.sample(shared_req_ids, k=max(2, min(len(shared_req_ids), 2 + required_steps // 3)))

        noise_tools = [t for t in self.root_cause_pool if t != root_cause_tool]
        if not noise_tools:
            noise_tools = ["noop_tool"]

        def make_line(ts_i: int, source: str, severity: str, tool: str, req: str, msg: str) -> str:
            return f"{ts_i:04d} {source} {severity} tool={tool} req={req} msg='{msg}'"

        logs: Dict[str, Dict[str, Any]] = {}
        for s in sources:
            lines = []
            base_len = 25 + required_steps * 8
            length = max(20, min(140, base_len + random.randint(-5, 10)))
            for i in range(length):
                req = random.choice(shared_req_ids)
                if req in root_req_ids and random.random() < 0.55:
                    tool = root_cause_tool
                    severity = "ERROR" if random.random() < 0.85 else "WARN"
                    msg = random.choice(self.message_pool)
                else:
                    tool = random.choice(noise_tools)
                    r = random.random()
                    if r < 0.08 + 0.01 * self.complexity:
                        severity = "ERROR"
                        msg = random.choice(self.message_pool)
                    elif r < 0.22:
                        severity = "WARN"
                        msg = random.choice(self.message_pool)
                    else:
                        severity = "INFO"
                        msg = "ok"
                lines.append(make_line(i, s, severity, tool, req, msg))
            logs[s] = {"lines": lines}

        tool_hint = "Root cause is a single tool name; infer it by extracting and counting ERROR events across sources."
        if required_steps <= 2:
            tool_hint = "Start by loading a source, extracting errors, then count by tool."

        return {
            "incident_id": incident_id,
            "time_window": window,
            "required_steps": required_steps,
            "sources": sources,
            "logs": logs,
            "root_cause_tool": root_cause_tool,
            "root_req_ids": set(root_req_ids),
            "hint": tool_hint,
        }

    def _get_instructions(self) -> str:
        return (
            "You are on-call. Diagnose a production incident by using the available tools to inspect log sources. "
            "Your goal is to identify the single root-cause tool/function name responsible for the incident, "
            "then submit it with submit_root_cause.\n"
            "Actions must be in the form \\boxed{tool_name(arg='value', ...)}.\n"
            "Use list_sources -> load_source -> select_active, then use analysis tools like grep/extract_errors/count_by_tool.\n"
            "Ordinary tool calls give 0 reward. Submitting the correct tool ends the episode with +1; wrong submission ends with -1.\n"
            "Invalid action format ends the episode with a format penalty."
        )

    def get_task_suffix(self) -> str:
        if not hasattr(self, "task") or self.task is None:
            return "State: (uninitialized). Use \\boxed{list_sources()} to begin."
        required = self.task.get("required_steps", "?")
        remaining_turns = max(0, self.max_turns - getattr(self, "turn_count", 0))
        loaded = sorted(list(getattr(self, "loaded_sources", set())))
        active = getattr(self, "active_source", None)
        extracted = getattr(self, "extracted_events", {}).get(active, [])
        extracted_n = len(extracted) if active in getattr(self, "extracted_events", {}) else 0
        return (
            f"Incident={self.task['incident_id']} Window={self.task['time_window']} RequiredToolCalls≈{required}. "
            f"TurnsLeft={remaining_turns}. LoadedSources={loaded}. ActiveSource={active}. "
            f"ActiveExtractedEvents={extracted_n}. "
            "Respond with exactly one action in \\boxed{...} format."
        )

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not isinstance(action, str):
            return None
        m = re.search(r"\\boxed\{(.+)\}\s*$", action.strip())
        if not m:
            return None
        inner = m.group(1).strip()

        call = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)\s*$", inner)
        if not call:
            return None
        tool_name = call.group(1)
        args_str = call.group(2).strip()

        args: Dict[str, Any] = {}
        if args_str == "":
            return {"tool": tool_name, "args": args}

        parts = []
        buf = ""
        in_single = False
        in_double = False
        for ch in args_str:
            if ch == "'" and not in_double:
                in_single = not in_single
                buf += ch
            elif ch == '"' and not in_single:
                in_double = not in_double
                buf += ch
            elif ch == "," and not in_single and not in_double:
                parts.append(buf.strip())
                buf = ""
            else:
                buf += ch
        if buf.strip():
            parts.append(buf.strip())

        for p in parts:
            kv = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)\s*$", p)
            if not kv:
                return None
            k = kv.group(1)
            v = kv.group(2).strip()
            if (len(v) >= 2) and ((v[0] == v[-1] == "'") or (v[0] == v[-1] == '"')):
                args[k] = v[1:-1]
            elif re.fullmatch(r"-?\d+", v):
                args[k] = int(v)
            else:
                return None

        return {"tool": tool_name, "args": args}

    def sample_random_action(self) -> str:
        examples = [
            r"\boxed{list_sources()}",
            r"\boxed{load_source(source='svc_api')}",
            r"\boxed{select_active(source='svc_api')}",
            r"\boxed{extract_errors(limit=50)}",
            r"\boxed{count_by_tool()}",
            r"\boxed{submit_root_cause(tool='token_sign')}",
        ]
        return random.choice(examples)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        required_steps = random.randint(self.min_required_steps, self.max_required_steps)
        self.task = self._generate_task_requiring_n_steps(required_steps)

        self.turn_count = 0
        self.steps_taken = 0

        self.loaded_sources: set = set()
        self.active_source: Optional[str] = None
        self.extracted_events: Dict[str, list] = {}
        self.last_result: Optional[Any] = None

        info = {"suffix": self.get_task_suffix()}
        return self._get_instructions(), info

    def _require_active_loaded(self) -> Optional[str]:
        if not self.loaded_sources:
            return "protocol violation: no sources loaded. call load_source first."
        if self.active_source is None:
            return "protocol violation: no active source selected. call select_active after loading."
        if self.active_source not in self.loaded_sources:
            return "protocol violation: active source not loaded. load it and select_active again."
        return None

    def _get_active_lines(self) -> list:
        return self.task["logs"][self.active_source]["lines"]

    def _parse_line_to_event(self, line: str) -> Optional[Dict[str, Any]]:
        m = re.match(
            r"^(?P<ts>\d{4})\s+(?P<src>\S+)\s+(?P<sev>INFO|WARN|ERROR)\s+tool=(?P<tool>\S+)\s+req=(?P<req>\S+)\s+msg='(?P<msg>.*)'\s*$",
            line,
        )
        if not m:
            return None
        return {
            "ts": int(m.group("ts")),
            "source": m.group("src"),
            "severity": m.group("sev"),
            "tool": m.group("tool"),
            "req": m.group("req"),
            "msg": m.group("msg"),
        }

    def _execute_tool(self, tool: str, args: Dict[str, Any]) -> str:
        if tool == "catalog_tools":
            names = sorted([k for k in self.tools.keys()])
            lines = ["tool catalog:"]
            for n in names:
                desc = self.tool_descriptions.get(n, "")
                lines.append(f"- {n}: {desc}")
            return "\n".join(lines)

        if tool == "inspect_tool_schema":
            name = args.get("name")
            if not isinstance(name, str) or name == "":
                return "execution error: name must be a non-empty string."
            if name not in self.tool_schemas:
                return f"execution error: unknown tool schema for '{name}'."
            schema = self.tool_schemas[name]
            return f"schema {name}: params={schema.get('parameters', [])} returns={schema.get('returns')} example={schema.get('example')}"

        if tool == "list_sources":
            return "sources: " + ", ".join(self.task["sources"])

        if tool == "load_source":
            source = args.get("source")
            if not isinstance(source, str) or source == "":
                return "execution error: source must be a non-empty string."
            if source not in self.task["logs"]:
                return f"execution error: source '{source}' not found."
            self.loaded_sources.add(source)
            self.active_source = self.active_source or source
            return f"loaded source '{source}'. loaded_count={len(self.loaded_sources)} active='{self.active_source}'."

        if tool == "select_active":
            source = args.get("source")
            if not isinstance(source, str) or source == "":
                return "execution error: source must be a non-empty string."
            if source not in self.loaded_sources:
                return f"protocol violation: cannot select '{source}' because it is not loaded."
            self.active_source = source
            return f"active source set to '{source}'."

        if tool == "grep":
            pre = self._require_active_loaded()
            if pre:
                return pre
            pattern = args.get("pattern")
            limit = args.get("limit", 10)
            if not isinstance(pattern, str) or pattern == "":
                return "execution error: pattern must be a non-empty string."
            if not isinstance(limit, int) or limit <= 0:
                return "execution error: limit must be a positive int."
            matches = []
            for i, line in enumerate(self._get_active_lines()):
                if pattern in line:
                    matches.append((i, line))
                    if len(matches) >= limit:
                        break
            if not matches:
                return f"grep result: 0 matches for '{pattern}' in active='{self.active_source}'."
            out = [f"grep result: {len(matches)} matches (showing up to {limit}) in active='{self.active_source}':"]
            for idx, line in matches:
                out.append(f"[{idx}] {line}")
            return "\n".join(out)

        if tool == "extract_errors":
            pre = self._require_active_loaded()
            if pre:
                return pre
            limit = args.get("limit", 50)
            if not isinstance(limit, int) or limit <= 0:
                return "execution error: limit must be a positive int."
            events = []
            for line in self._get_active_lines():
                ev = self._parse_line_to_event(line)
                if ev and ev["severity"] in ("WARN", "ERROR"):
                    events.append(ev)
                    if len(events) >= limit:
                        break
            self.extracted_events[self.active_source] = events
            n_err = sum(1 for e in events if e["severity"] == "ERROR")
            n_warn = sum(1 for e in events if e["severity"] == "WARN")
            return f"extracted events from active='{self.active_source}': total={len(events)} warn={n_warn} error={n_err}."

        if tool == "filter_by_severity":
            pre = self._require_active_loaded()
            if pre:
                return pre
            severity = args.get("severity")
            if not isinstance(severity, str) or severity not in self.severities:
                return "execution error: severity must be one of INFO/WARN/ERROR."
            if self.active_source not in self.extracted_events:
                return "protocol violation: no extracted events for active source. call extract_errors first."
            filtered = [e for e in self.extracted_events[self.active_source] if e["severity"] == severity]
            self.extracted_events[self.active_source] = filtered
            return f"filtered extracted events in active='{self.active_source}' to severity='{severity}': total={len(filtered)}."

        if tool == "timeline":
            pre = self._require_active_loaded()
            if pre:
                return pre
            limit = args.get("limit", 20)
            if not isinstance(limit, int) or limit <= 0:
                return "execution error: limit must be a positive int."
            if self.active_source not in self.extracted_events:
                return "protocol violation: no extracted events for active source. call extract_errors first."
            events = sorted(self.extracted_events[self.active_source], key=lambda e: e["ts"])
            show = events[:limit]
            if not show:
                return f"timeline: 0 events in active='{self.active_source}'."
            out = [f"timeline: showing {len(show)} events in active='{self.active_source}':"]
            for e in show:
                out.append(f"{e['ts']:04d} {e['severity']} tool={e['tool']} req={e['req']} msg='{e['msg']}'")
            return "\n".join(out)

        if tool == "count_by_tool":
            pre = self._require_active_loaded()
            if pre:
                return pre
            if self.active_source not in self.extracted_events:
                return "protocol violation: no extracted events for active source. call extract_errors first."
            counts: Dict[str, int] = {}
            for e in self.extracted_events[self.active_source]:
                if e["severity"] == "ERROR":
                    counts[e["tool"]] = counts.get(e["tool"], 0) + 1
            if not counts:
                return f"count_by_tool: 0 ERROR events in active='{self.active_source}'."
            items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            summary = ", ".join([f"{k}={v}" for k, v in items[: min(8, len(items))]])
            return f"count_by_tool (ERROR only) in active='{self.active_source}': {summary}"

        if tool == "top_k_messages":
            pre = self._require_active_loaded()
            if pre:
                return pre
            k = args.get("k", 3)
            if not isinstance(k, int) or k <= 0:
                return "execution error: k must be a positive int."
            if self.active_source not in self.extracted_events:
                return "protocol violation: no extracted events for active source. call extract_errors first."
            counts: Dict[str, int] = {}
            for e in self.extracted_events[self.active_source]:
                if e["severity"] == "ERROR":
                    counts[e["msg"]] = counts.get(e["msg"], 0) + 1
            if not counts:
                return f"top_k_messages: 0 ERROR events in active='{self.active_source}'."
            items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
            out = [f"top_k_messages: k={k} in active='{self.active_source}':"]
            for msg, c in items:
                out.append(f"- '{msg}': {c}")
            return "\n".join(out)

        if tool == "unique_request_ids":
            pre = self._require_active_loaded()
            if pre:
                return pre
            limit = args.get("limit", 20)
            if not isinstance(limit, int) or limit <= 0:
                return "execution error: limit must be a positive int."
            if self.active_source not in self.extracted_events:
                return "protocol violation: no extracted events for active source. call extract_errors first."
            reqs = []
            seen = set()
            for e in self.extracted_events[self.active_source]:
                if e["severity"] == "ERROR" and e["req"] not in seen:
                    seen.add(e["req"])
                    reqs.append(e["req"])
                    if len(reqs) >= limit:
                        break
            if not reqs:
                return f"unique_request_ids: 0 ERROR request IDs in active='{self.active_source}'."
            return f"unique_request_ids (ERROR only) in active='{self.active_source}': " + ", ".join(reqs)

        if tool == "diff_sources":
            a = args.get("source_a")
            b = args.get("source_b")
            if not isinstance(a, str) or not isinstance(b, str) or a == "" or b == "":
                return "execution error: source_a and source_b must be non-empty strings."
            if a not in self.loaded_sources or b not in self.loaded_sources:
                return "protocol violation: diff_sources requires both sources to be loaded."
            if a not in self.extracted_events or b not in self.extracted_events:
                return "protocol violation: diff_sources requires extract_errors to be run on both sources."
            ra = {e["req"] for e in self.extracted_events[a] if e["severity"] == "ERROR"}
            rb = {e["req"] for e in self.extracted_events[b] if e["severity"] == "ERROR"}
            shared = sorted(list(ra.intersection(rb)))
            return f"diff_sources: shared_error_req_ids={len(shared)} example={shared[:5]}"

        if tool == "summarize_state":
            loaded = sorted(list(self.loaded_sources))
            active = self.active_source
            extracted_sources = sorted(list(self.extracted_events.keys()))
            active_n = len(self.extracted_events.get(active, [])) if active else 0
            return (
                f"state summary: incident={self.task['incident_id']} loaded={loaded} active={active} "
                f"extracted_sources={extracted_sources} active_extracted={active_n} steps_taken={self.steps_taken}."
            )

        if tool == "submit_root_cause":
            cand = args.get("tool")
            if not isinstance(cand, str) or cand == "":
                return "execution error: tool must be a non-empty string."
            if cand == self.task["root_cause_tool"]:
                return f"success: correct root cause '{cand}'."
            return f"wrong decision: submitted '{cand}', but that is not the root cause."

        return "unsupported action: unknown tool."

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1

        if self.turn_count >= self.max_turns:
            obs = "timeout: reached max turns."
            info = {"suffix": self.get_task_suffix()}
            return obs, 0.0, True, True, info

        parsed = self._parse_action(action)
        if not parsed:
            obs = "invalid action format: use \\boxed{tool_name(arg='value', ...)}."
            info = {"suffix": self.get_task_suffix()}
            return obs, float(LanguageGameReward.format_error_reward), True, False, info

        tool = parsed["tool"]
        args = parsed["args"]

        if tool not in self.tools:
            obs = f"unsupported action: unknown tool '{tool}'."
            info = {"suffix": self.get_task_suffix()}
            return obs, 0.0, True, False, info  # Fixed: was -1.0, failures should be 0.0

        self.steps_taken += 1
        obs = self._execute_tool(tool, args)
        self.last_result = obs

        if tool == "submit_root_cause":
            if obs.startswith("success:"):
                reward = 1.0
            else:
                reward = 0.0  # Fixed: was -1.0, failures should be 0.0
            info = {"suffix": self.get_task_suffix()}
            return obs, reward, True, False, info

        info = {"suffix": self.get_task_suffix()}
        return obs, 0.0, False, False, info


class TraceTriageFoundryEnvWithFeedback(TraceTriageFoundryEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        diagnostic = {"error_type": "OK"}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = {"outcome": "episode_start"}
            diagnostic["turn"] = 0
            diagnostic["steps_taken"] = getattr(self, "steps_taken", None)
            diagnostic["loaded_sources"] = sorted(list(getattr(self, "loaded_sources", set())))
            diagnostic["active_source"] = getattr(self, "active_source", None)
        if self.feedback_level >= 2:
            diagnostic["hint"] = "Start with \\boxed{list_sources()} then \\boxed{load_source(...)} and \\boxed{extract_errors(limit=...)}."
        info["diagnostic"] = diagnostic
        return obs, info

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        text = (obs or "").lower()
        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint: Optional[str] = None

        if "timeout:" in text:
            error_type = "Timeout"
            error_detail["issue"] = "max_turns_reached"
            hint = "Use fewer exploratory calls: load -> extract_errors -> count_by_tool -> submit_root_cause."

        elif "invalid action format" in text and "boxed" in text:
            error_type = "FormatError"
            error_detail["issue"] = "missing_or_malformed_boxed"
            hint = r"Wrap exactly one call like \boxed{list_sources()} or \boxed{load_source(source='svc_api')}."

        elif text.startswith("unsupported action:"):
            error_type = "UnsupportedAction"
            m = re.search(r"unknown tool '([^']+)'", obs)
            if m:
                error_detail["tool"] = m.group(1)
            hint = r"Call \boxed{catalog_tools()} to see valid tool names."

        elif "protocol violation:" in text:
            error_type = "ProtocolViolation"
            if "no sources loaded" in text:
                error_detail["violation"] = "missing_load_source"
                hint = r"Call \boxed{list_sources()} then \boxed{load_source(source='...')}."
            elif "no active source selected" in text:
                error_detail["violation"] = "missing_select_active"
                hint = r"After loading, call \boxed{select_active(source='...')}."
            elif "call extract_errors first" in text:
                error_detail["violation"] = "missing_extract_errors"
                hint = r"Run \boxed{extract_errors(limit=50)} before count/timeline/message tools."
            elif "requires both sources to be loaded" in text:
                error_detail["violation"] = "diff_requires_loaded"
                hint = r"Load both sources with \boxed{load_source(...)} before \boxed{diff_sources(...)}."
            elif "requires extract_errors to be run on both sources" in text:
                error_detail["violation"] = "diff_requires_extracted"
                hint = r"Run \boxed{extract_errors(limit=...)} while each source is active, then diff."
            else:
                error_detail["violation"] = "protocol_violation"

        elif "execution error:" in text:
            error_type = "ProtocolViolation"
            error_detail["issue"] = "bad_parameters_or_preconditions"
            hint = r"Check the tool schema with \boxed{inspect_tool_schema(name='...')} and pass required args."

        elif text.startswith("wrong decision:"):
            error_type = "WrongDecision"
            m = re.search(r"submitted '([^']+)'", obs)
            if m:
                error_detail["got"] = m.group(1)
            hint = "Extract ERROR events and use count_by_tool; the root cause tends to dominate ERROR counts."

        elif text.startswith("success:"):
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["steps_taken"] = getattr(self, "steps_taken", None)
            diagnostic["loaded_sources"] = sorted(list(getattr(self, "loaded_sources", set())))
            diagnostic["active_source"] = getattr(self, "active_source", None)
            diagnostic["required_steps"] = getattr(self, "task", {}).get("required_steps") if getattr(self, "task", None) else None
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info