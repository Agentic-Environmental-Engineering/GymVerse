from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class TelemetrySmithEnv(Env):
    def __init__(self, complexity: int = 1, max_turns: Optional[int] = 100, **_):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.max_turns = max_turns if max_turns is not None else 100
        self.min_required_steps = self.complexity
        self.max_required_steps = self.complexity * 2
        self._init_database()
        self.reset()

    def _init_database(self):
        base_names = [
            "lark", "ember", "quartz", "nebula", "aurora", "zephyr",
            "solstice", "caldera", "onyx", "cirrus", "atlas", "hydra",
            "spire", "comet", "saffron", "topaz", "meridian", "falcon",
            "sage", "violet"
        ]
        n_services = min(len(base_names), 3 + self.complexity + random.randint(0, 2))
        T = 8 + self.complexity * 2 + random.randint(0, 4)
        self.time_horizon = T
        self.services: Dict[str, Dict[str, Any]] = {}
        for name in random.sample(base_names, n_services):
            total_devices = random.randint(40, 140)
            base_ms = random.randint(70, 280)
            variability = random.uniform(0.15, 0.35)
            err_base = random.uniform(0.01, 0.16)
            ts = []
            for t in range(T):
                reports = random.randint(max(1, int(total_devices * 0.4)), total_devices)
                requests = random.randint(reports * 2, reports * 4)
                # errors influenced by err_base, with noise
                er = max(0.0, min(0.5, err_base + random.uniform(-0.02, 0.02)))
                errors = max(0, int(requests * er))
                latencies: List[int] = []
                for _ in range(requests):
                    val = int(random.gauss(base_ms, base_ms * variability))
                    # occasional spikes
                    if random.random() < 0.03:
                        val += random.randint(100, 800)
                    val = max(1, min(val, 3000))
                    latencies.append(val)
                ts.append({"t": t, "reports": reports, "requests": requests, "errors": errors, "latencies": latencies})
            self.services[name] = {
                "total_devices": total_devices,
                "base_ms": base_ms,
                "err_base": err_base,
                "series": ts
            }

        self.tools: Dict[str, Dict[str, Any]] = {
            "list_services": {
                "description": "List available services and their device counts",
                "parameters": [],
                "returns": "List of service names"
            },
            "select_service": {
                "description": "Select a target service for metric computations",
                "parameters": [{"name": "name", "type": "string"}],
                "returns": "Sets current selection"
            },
            "time_window": {
                "description": "Set inclusive time window indices [start, end]",
                "parameters": [{"name": "start", "type": "int"}, {"name": "end", "type": "int"}],
                "returns": "Sets current time window"
            },
            "describe": {
                "description": "Summary of current selection and window",
                "parameters": [],
                "returns": "Text summary"
            },
            "percentile": {
                "description": "Compute latency percentile p for current selection & window",
                "parameters": [{"name": "p", "type": "int"}],
                "returns": "Number (ms)"
            },
            "error_rate": {
                "description": "Compute error rate in current selection & window",
                "parameters": [],
                "returns": "Number (0..1)"
            },
            "coverage": {
                "description": "Compute average reporting coverage over window",
                "parameters": [],
                "returns": "Number (0..1)"
            },
            "verify_sla": {
                "description": "Verify SLA feasibility for current selection & window using task thresholds",
                "parameters": [],
                "returns": "Feasible/NotFeasible"
            },
            "show_targets": {
                "description": "Display SLA thresholds defined by the task",
                "parameters": [],
                "returns": "Thresholds"
            },
            "submit": {
                "description": "Submit final decision yes/no for the task",
                "parameters": [{"name": "decision", "type": "string"}],
                "returns": "Terminal"
            }
        }

        self.current_service: Optional[str] = None
        self.window: Tuple[int, int] = (0, self.time_horizon - 1)
        self.turn_count = 0
        self.steps_taken = 0
        self.required_steps = self.min_required_steps
        self.task: Dict[str, Any] = {}

    def _get_instructions(self) -> str:
        lines = []
        lines.append("You are operating a telemetry lab. Use tools to inspect and verify SLA for a target service and time window.")
        lines.append("Goal: Decide whether the service satisfies the SLA in the specified window with thresholds (latency p95 <= L_max, error_rate <= E_max, coverage >= C_min).")
        lines.append("Action format: Use \\boxed{tool arg=value arg2=value}. Submit final decision using \\boxed{submit decision=yes} or \\boxed{submit decision=no}.")
        lines.append("Tools available: " + ", ".join(sorted(self.tools.keys())))
        lines.append("You must execute at least {} valid tool calls before submitting.".format(self.required_steps))
        lines.append("Malformed or unsupported actions terminate the episode with negative reward.")
        return "\n".join(lines)

    def get_task_suffix(self) -> str:
        svc = self.task.get("target_service")
        w = self.task.get("window")
        L = self.task.get("L_max")
        E = self.task.get("E_max")
        C = self.task.get("C_min")
        state = []
        state.append("Task: Evaluate service '{}' in window [{}..{}] with p95<= {} ms, error_rate<= {:.3f}, coverage>= {:.3f}."
                     .format(svc, w[0], w[1], L, E, C))
        state.append("Selected service: {}".format(self.current_service if self.current_service else "none"))
        state.append("Selected window: [{}..{}]".format(self.window[0], self.window[1]))
        state.append("Steps taken: {} / required: {}".format(self.steps_taken, self.required_steps))
        state.append("Turn: {} / max: {}".format(self.turn_count, self.max_turns))
        state.append("Format: \\boxed{tool arg=value}. Example: \\boxed{select_service name=" + str(svc) + "}")
        return "\n".join(state)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self.turn_count = 0
        self.steps_taken = 0
        self.required_steps = random.randint(self.min_required_steps, self.max_required_steps)
        target_service = random.choice(list(self.services.keys()))
        window_len = random.randint(max(3, int(self.time_horizon * 0.2)), max(3, int(self.time_horizon * 0.5)))
        start = random.randint(0, self.time_horizon - window_len)
        end = start + window_len - 1
        L_max = random.randint(120, 400)
        E_max = round(random.uniform(0.04, 0.18), 3)
        C_min = round(random.uniform(0.50, 0.90), 3)

        self.task = {
            "target_service": target_service,
            "window": (start, end),
            "L_max": L_max,
            "E_max": E_max,
            "C_min": C_min,
            "p": 95
        }
        self.current_service = None
        self.window = (0, self.time_horizon - 1)

        correct = self._compute_feasibility(target_service, (start, end), L_max, E_max, C_min, 95)
        self.task["correct"] = "yes" if correct else "no"
        obs = self._get_instructions()
        info = {"suffix": self.get_task_suffix()}
        return obs, info

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        if self.turn_count > self.max_turns:
            obs = "Timeout: Episode reached maximum turns."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}
        parsed = self._parse_action(action)
        if not parsed:
            obs = "Invalid action format. Use \\boxed{tool arg=value}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
        tool, args = parsed
        if tool not in self.tools:
            obs = "UnsupportedAction: Unknown tool '{}'.".format(tool)
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}  # Fixed: was -0.5
        try:
            if tool == "submit":
                decision = str(args.get("decision", "")).strip().lower()
                if decision not in ("yes", "no"):
                    raise ValueError("ProtocolViolation: 'decision' must be yes or no.")
                if self.steps_taken < self.required_steps:
                    obs = "ProtocolViolation: Not enough tool calls before submit. Required: {}, Taken: {}."
                    obs = obs.format(self.required_steps, self.steps_taken)
                    return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}  # Fixed: was -0.5
                expected = self.task["correct"]
                if decision == expected:
                    obs = "Final decision accepted: {}. Success.".format(decision)
                    return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
                else:
                    obs = "Final decision incorrect: {}. Expected {}.".format(decision, expected)
                    return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}  # Fixed: was -1.0

            result = self._execute_tool(tool, args)
            self.steps_taken += 1
            obs = "Tool '{}' executed.\nResult: {}".format(tool, result)
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
        except Exception as e:
            msg = str(e)
            obs = "ProtocolViolation: {}".format(msg)
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}  # Fixed: was -0.5

    def _compute_feasibility(self, service: str, window: Tuple[int, int], L_max: int, E_max: float, C_min: float, p: int) -> bool:
        svc = self.services[service]
        start, end = window
        latencies: List[int] = []
        requests = 0
        errors = 0
        reports = 0
        total_devices = svc["total_devices"]
        for t in range(start, end + 1):
            seg = svc["series"][t]
            latencies.extend(seg["latencies"])
            requests += seg["requests"]
            errors += seg["errors"]
            reports += seg["reports"]
        if len(latencies) == 0 or requests == 0:
            return False
        latency_p = self._percentile(latencies, p)
        err_rate = errors / requests
        coverage_avg = reports / ((end - start + 1) * total_devices)
        return (latency_p <= L_max) and (err_rate <= E_max) and (coverage_avg >= C_min)

    def _execute_tool(self, tool: str, args: Dict[str, Any]) -> str:
        if tool == "list_services":
            names = sorted(self.services.keys())
            details = ["{}(devices={})".format(n, self.services[n]["total_devices"]) for n in names]
            return "Available: " + ", ".join(details)
        if tool == "select_service":
            name = str(args.get("name", "")).strip()
            if name not in self.services:
                raise ValueError("select_service requires a valid 'name'.")
            self.current_service = name
            return "Selected service '{}'.".format(name)
        if tool == "time_window":
            start = args.get("start", None)
            end = args.get("end", None)
            if start is None or end is None:
                raise ValueError("time_window requires 'start' and 'end'.")
            try:
                start = int(start)
                end = int(end)
            except:
                raise ValueError("time_window 'start' and 'end' must be integers.")
            if not (0 <= start <= end < self.time_horizon):
                raise ValueError("time_window bounds must be within [0, {}] and start<=end.".format(self.time_horizon - 1))
            self.window = (start, end)
            return "Window set to [{}..{}].".format(start, end)
        if tool == "describe":
            if self.current_service is None:
                raise ValueError("select_service must be called first.")
            svc = self.services[self.current_service]
            start, end = self.window
            total_devices = svc["total_devices"]
            requests = sum(seg["requests"] for seg in svc["series"][start:end+1])
            errors = sum(seg["errors"] for seg in svc["series"][start:end+1])
            reports = sum(seg["reports"] for seg in svc["series"][start:end+1])
            coverage_avg = reports / ((end - start + 1) * total_devices)
            return "Service '{}', window [{}..{}], requests={}, errors={}, coverage_avg={:.3f}.".format(
                self.current_service, start, end, requests, errors, coverage_avg
            )
        if tool == "percentile":
            if self.current_service is None:
                raise ValueError("select_service must be called first.")
            p = args.get("p", None)
            if p is None:
                raise ValueError("percentile requires 'p'.")
            try:
                p = int(p)
            except:
                raise ValueError("percentile 'p' must be integer.")
            if not (1 <= p <= 99):
                raise ValueError("percentile 'p' must be between 1 and 99.")
            svc = self.services[self.current_service]
            start, end = self.window
            latencies: List[int] = []
            for t in range(start, end + 1):
                latencies.extend(svc["series"][t]["latencies"])
            if not latencies:
                raise ValueError("No latency data in selected window.")
            val = self._percentile(latencies, p)
            return "Latency p{} = {} ms.".format(p, val)
        if tool == "error_rate":
            if self.current_service is None:
                raise ValueError("select_service must be called first.")
            svc = self.services[self.current_service]
            start, end = self.window
            requests = sum(seg["requests"] for seg in svc["series"][start:end+1])
            errors = sum(seg["errors"] for seg in svc["series"][start:end+1])
            if requests == 0:
                raise ValueError("No requests in selected window.")
            rate = errors / requests
            return "Error rate = {:.4f}.".format(rate)
        if tool == "coverage":
            if self.current_service is None:
                raise ValueError("select_service must be called first.")
            svc = self.services[self.current_service]
            start, end = self.window
            reports = sum(seg["reports"] for seg in svc["series"][start:end+1])
            total_devices = svc["total_devices"]
            coverage_avg = reports / ((end - start + 1) * total_devices)
            return "Coverage avg = {:.4f}.".format(coverage_avg)
        if tool == "verify_sla":
            if self.current_service is None:
                raise ValueError("select_service and time_window must be called before verify_sla.")
            L_max = self.task["L_max"]
            E_max = self.task["E_max"]
            C_min = self.task["C_min"]
            p = self.task["p"]
            res = self._compute_feasibility(self.current_service, self.window, L_max, E_max, C_min, p)
            svc = self.services[self.current_service]
            start, end = self.window
            latencies: List[int] = []
            requests = 0
            errors = 0
            reports = 0
            total_devices = svc["total_devices"]
            for t in range(start, end + 1):
                seg = svc["series"][t]
                latencies.extend(seg["latencies"])
                requests += seg["requests"]
                errors += seg["errors"]
                reports += seg["reports"]
            pval = self._percentile(latencies, p) if latencies else None
            rate = errors / requests if requests > 0 else None
            cov = reports / ((end - start + 1) * total_devices)
            label = "Feasible" if res else "NotFeasible"
            return "Verification: {} (p{}={} ms, error_rate={:.4f}, coverage={:.4f}).".format(label, p, pval, rate, cov)
        if tool == "show_targets":
            return "SLA thresholds: p95<= {} ms, error_rate<= {:.3f}, coverage>= {:.3f}.".format(
                self.task["L_max"], self.task["E_max"], self.task["C_min"]
            )
        raise ValueError("Unsupported tool path.")

    def _percentile(self, data: List[int], p: int) -> int:
        if not data:
            return 0
        data_sorted = sorted(data)
        n = len(data_sorted)
        idx = int((p / 100.0) * n)
        idx = max(0, min(n - 1, idx - 1))
        return data_sorted[idx]

    def _parse_action(self, action: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        match = re.search(r"\\boxed\{(.*)\}", action.strip())
        if not match:
            return None
        content = match.group(1).strip()
        if not content:
            return None
        tokens = content.split()
        tool = tokens[0]
        args: Dict[str, Any] = {}
        for tok in tokens[1:]:
            if "=" in tok:
                k, v = tok.split("=", 1)
                v = v.strip()
                v = v.strip('"').strip("'")
                args[k.strip()] = v
            else:
                args[tok.strip()] = True
        return tool, args

    def sample_random_action(self) -> str:
        if random.random() < 0.5:
            return "\\boxed{list_services}"
        else:
            svc = random.choice(list(self.services.keys()))
            return "\\boxed{select_service name=" + svc + "}"


class TelemetrySmithEnvWithFeedback(TelemetrySmithEnv):
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
            hint = "Wrap your command in \\boxed{...} and include key=value pairs."
        elif "unsupportedaction" in text or "unknown tool" in text:
            error_type = "UnsupportedAction"
            error_detail["tool"] = "unknown"
            hint = "Use one of the listed tools: {}".format(", ".join(sorted(self.tools.keys())))
        elif "protocolviolation" in text:
            error_type = "ProtocolViolation"
            if "not enough tool calls" in text:
                error_detail["violation"] = "early_submit"
                hint = "Run more tools (e.g., select_service, time_window, percentile, error_rate, coverage) before submitting."
            elif "select_service" in text and "must be called first" in text:
                error_detail["violation"] = "missing_selection"
                hint = "Call \\boxed{select_service name=<target>} first."
            elif "time_window" in text and "requires" in text:
                error_detail["violation"] = "bad_window_args"
                hint = "Provide integers: \\boxed{time_window start=0 end=5} within bounds."
            else:
                error_detail["violation"] = "prerequisite_missing"
                hint = "Check prerequisites: service selection and time window as needed."
        elif "final decision incorrect" in text or ("expected" in text and "final decision" in text):
            error_type = "WrongDecision"
            error_detail["expected"] = self.task.get("correct")
            error_detail["got"] = "yes" if "incorrect: yes" in text else ("no" if "incorrect: no" in text else None)
            hint = "Compute p95 latency, error rate, and coverage for the specified window before submitting."
        elif "timeout" in text:
            error_type = "Timeout"
            error_detail["outcome"] = "max_turns_reached"
            hint = "Plan tool calls to reach the required steps earlier; start with list_services and select_service."
        elif "success" in text or "final decision accepted" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["selected_service"] = getattr(self, "current_service", None)
            diagnostic["selected_window"] = getattr(self, "window", None)
            diagnostic["required_steps"] = getattr(self, "required_steps", None)
            diagnostic["steps_taken"] = getattr(self, "steps_taken", None)
            diagnostic["task_service"] = self.task.get("target_service")
            diagnostic["task_window"] = self.task.get("window")
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint
        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start with \\boxed{list_services}, then \\boxed{select_service name=<target>} and \\boxed{time_window start=<s> end=<e>}.",
            "turn": 0,
            "selected_service": self.current_service,
            "selected_window": self.window,
            "required_steps": self.required_steps,
            "steps_taken": self.steps_taken,
        }
        return obs, info