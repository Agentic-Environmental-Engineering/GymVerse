from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional

class ToolchestMaestroEnv(Env):
    def __init__(self, complexity: int = 1, max_turns: Optional[int] = 100, **_):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.max_turns = max_turns if max_turns is not None else 100
        self.min_required_steps = self.complexity
        self.max_required_steps = self.complexity * 2
        self._init_database()
        self.reset()

    def _init_database(self):
        self.all_categories = [
            "missing", "range", "duplicates", "schema", "dtype", "units", "id_unique", "derived"
        ]
        self.schemas = {
            "sensor_v1": {
                "required_columns": ["id", "temp", "pressure", "status"],
                "ranges": {"temp": (-50, 150), "pressure": (0, 500)},
                "units": {"temp": "C", "pressure": "kPa"},
                "dtype": {"id": "int", "temp": "float", "pressure": "float", "status": "str"},
                "derived": ["health_score"]
            }
        }
        self.tools = {
            "help": {"description": "List tools and usage", "parameters": [], "returns": "str"},
            "show_task": {"description": "Show current task description", "parameters": [], "returns": "str"},
            "list_tools": {"description": "List available tools", "parameters": [], "returns": "list"},
            "list_datasets": {"description": "List dataset names in scope", "parameters": [], "returns": "list"},
            "load_dataset": {"description": "Load a dataset by name", "parameters": [{"name": "name", "type": "str"}], "returns": "status"},
            "describe_dataset": {"description": "Describe currently loaded dataset", "parameters": [], "returns": "summary"},
            "list_checks": {"description": "List all check categories", "parameters": [], "returns": "list"},
            "check_missing": {"description": "Check presence of missing-value faults", "parameters": [], "returns": "bool"},
            "check_range": {"description": "Check out-of-range value faults", "parameters": [], "returns": "bool"},
            "check_duplicates": {"description": "Check duplicate rows/keys faults", "parameters": [], "returns": "bool"},
            "check_schema": {"description": "Check column/schema mismatch faults", "parameters": [], "returns": "bool"},
            "check_dtype": {"description": "Check wrong data-type faults", "parameters": [], "returns": "bool"},
            "check_units": {"description": "Check wrong unit faults", "parameters": [], "returns": "bool"},
            "check_id_unique": {"description": "Check id uniqueness faults", "parameters": [], "returns": "bool"},
            "check_derived": {"description": "Check required derived column faults", "parameters": [], "returns": "bool"},
            "note_fault": {"description": "Record a discovered fault for planning", "parameters": [{"name": "category", "type": "str"}], "returns": "status"},
            "submit": {"description": "Submit final verdict", "parameters": [{"name": "answer", "type": "str"}, {"name": "interventions", "type": "int"}], "returns": "terminal"}
        }
        self.all_possible_datasets = {}
        for i in range(1, 21):
            name = f"DS{i:02d}"
            size = random.randint(100, 5000)
            cols = ["id", "temp", "pressure", "status"]
            self.all_possible_datasets[name] = {"size": size, "columns": cols}
        self.execution_state = {
            "current_dataset": None,
            "notes": {},
        }

    def _generate_task_requiring_n_steps(self, required_steps: int) -> Dict[str, Any]:
        max_cat = len(self.all_categories)
        min_d = max(1, (required_steps + max_cat - 1) // max_cat)
        max_d = min(5, required_steps)
        num_datasets = random.randint(min_d, max_d) if min_d <= max_d else min_d
        chosen = random.sample(list(self.all_possible_datasets.keys()), num_datasets)
        faults = {d: set() for d in chosen}
        remaining = required_steps
        cap = num_datasets * max_cat
        remaining = min(remaining, cap)
        dataset_cycle = list(faults.keys())
        idx = 0
        while remaining > 0:
            d = dataset_cycle[idx % num_datasets]
            available = [c for c in self.all_categories if c not in faults[d]]
            if not available:
                idx += 1
                continue
            c = random.choice(available)
            faults[d].add(c)
            remaining -= 1
            idx += 1
        total_min_interventions = sum(len(v) for v in faults.values())
        schema = "sensor_v1"
        return {
            "schema": schema,
            "datasets": chosen,
            "faults": faults,
            "total_min_interventions": total_min_interventions,
            "required_steps": total_min_interventions
        }

    def _get_instructions(self) -> str:
        schema = self.task["schema"]
        ds = ", ".join(self.task["datasets"])
        tools_list = ", ".join(sorted(self.tools.keys()))
        checks = ", ".join([c for c in self.all_categories])
        return (
            "You are auditing multiple datasets with specialized tools. Your goal is to determine whether the global "
            "property holds: all datasets comply with the schema and quality rules (no faults). If any faults exist, "
            "report the minimal number of interventions required to make all datasets compliant. Each distinct fault "
            "category in a dataset requires exactly one intervention to fix.\n"
            f"Schema: {schema}. Datasets in scope: {ds}.\n"
            f"Available tools: {tools_list}.\n"
            f"Check categories: {checks}.\n"
            "A typical workflow: list_datasets() -> load_dataset(name) -> run check_* tools -> optionally note_fault(category) -> submit(answer=SAT|UNSAT, interventions=K).\n"
            "Important protocol: You must load a dataset before running check_* or note_fault. Each check reveals whether that fault category is present in the currently loaded dataset.\n"
            "Terminal submission format: \\boxed{submit(answer=SAT|UNSAT, interventions=INTEGER)}.\n"
            "All actions must be in a single \\boxed{...} block."
        )

    def get_task_suffix(self) -> str:
        current = self.execution_state.get("current_dataset")
        current_str = current if current else "None"
        turn_str = f"Turn {self.turn_count}/{self.max_turns}. Steps taken: {self.steps_taken}."
        return (
            f"\nState: current_dataset={current_str}. {turn_str}\n"
            "Respond with a single action using \\boxed{tool_name(arg=value, ...)}. "
            "When ready, submit using \\boxed{submit(answer=SAT|UNSAT, interventions=K)}."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        required_steps = random.randint(self.min_required_steps, self.max_required_steps)
        self.task = self._generate_task_requiring_n_steps(required_steps)
        self.turn_count = 0
        self.steps_taken = 0
        self.terminated = False
        self.truncated = False
        self.execution_state = {"current_dataset": None, "notes": {}}
        obs = self._get_instructions()
        return obs, {"suffix": self.get_task_suffix()}

    def _check_loaded(self) -> bool:
        return self.execution_state.get("current_dataset") is not None

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        if name == "help":
            entries = []
            for k, v in sorted(self.tools.items()):
                params = ", ".join([p["name"] for p in v["parameters"]])
                entries.append(f"- {k}({params}): {v['description']}")
            return "Tools:\n" + "\n".join(entries)

        if name == "show_task":
            schema = self.task["schema"]
            ds = ", ".join(self.task["datasets"])
            return f"Task: Audit datasets [{ds}] against schema {schema}. Submit minimal interventions needed."

        if name == "list_tools":
            return "Available tools: " + ", ".join(sorted(self.tools.keys()))

        if name == "list_datasets":
            return "Datasets: " + ", ".join(self.task["datasets"])

        if name == "load_dataset":
            dname = args.get("name")
            if not dname or dname not in self.task["datasets"]:
                return "Protocol violation: dataset not found or name missing. Use list_datasets() first."
            self.execution_state["current_dataset"] = dname
            if dname not in self.execution_state["notes"]:
                self.execution_state["notes"][dname] = set()
            return f"Loaded dataset: {dname}"

        if name == "describe_dataset":
            if not self._check_loaded():
                return "Protocol violation: no dataset loaded. Use load_dataset(name=...)."
            d = self.execution_state["current_dataset"]
            meta = self.all_possible_datasets.get(d, {})
            return f"Dataset {d} | size={meta.get('size','?')} | columns={','.join(meta.get('columns',[]))}"

        if name == "list_checks":
            return "Checks: " + ", ".join(self.all_categories)

        if name.startswith("check_"):
            if not self._check_loaded():
                return "Protocol violation: no dataset loaded. Load a dataset before checks."
            d = self.execution_state["current_dataset"]
            cat_map = {
                "check_missing": "missing",
                "check_range": "range",
                "check_duplicates": "duplicates",
                "check_schema": "schema",
                "check_dtype": "dtype",
                "check_units": "units",
                "check_id_unique": "id_unique",
                "check_derived": "derived",
            }
            if name not in cat_map:
                return "Unsupported check."
            cat = cat_map[name]
            present = cat in self.task["faults"][d]
            return f"Check {cat}: present={str(present)}"

        if name == "note_fault":
            if not self._check_loaded():
                return "Protocol violation: no dataset loaded. Load a dataset before noting faults."
            cat = args.get("category")
            if cat not in self.all_categories:
                return "Protocol violation: unknown category for note_fault."
            d = self.execution_state["current_dataset"]
            if d not in self.execution_state["notes"]:
                self.execution_state["notes"][d] = set()
            self.execution_state["notes"][d].add(cat)
            return f"Noted fault for {d}: {cat}"

        return "UNSUPPORTED ACTION"

    def _parse_action(self, action: str):
        if not isinstance(action, str):
            return None
        m = re.search(r"\\boxed\{(.*)\}\s*$", action.strip(), flags=re.DOTALL)
        if not m:
            return None
        content = m.group(1).strip()
        m2 = re.match(r"^\s*([a-zA-Z_][\w]*)\s*\((.*)\)\s*$", content, flags=re.DOTALL)
        if not m2:
            return None
        tool = m2.group(1)
        arg_str = m2.group(2).strip()
        args = {}
        if arg_str == "":
            return tool, args
        parts = []
        buf = ""
        in_quotes = False
        escape = False
        for ch in arg_str:
            if escape:
                buf += ch
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_quotes = not in_quotes
                buf += ch
                continue
            if ch == "," and not in_quotes:
                parts.append(buf.strip())
                buf = ""
            else:
                buf += ch
        if buf.strip():
            parts.append(buf.strip())
        for p in parts:
            if "=" not in p:
                continue
            k, v = p.split("=", 1)
            k = k.strip()
            v = v.strip()
            if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
                v_parsed = v[1:-1]
            else:
                vl = v.lower()
                if vl == "true":
                    v_parsed = True
                elif vl == "false":
                    v_parsed = False
                else:
                    if re.fullmatch(r"-?\d+", v):
                        v_parsed = int(v)
                    else:
                        v_parsed = v
            args[k] = v_parsed
        return tool, args

    def sample_random_action(self) -> str:
        if self.execution_state.get("current_dataset") is None:
            choice = random.choice([
                r'\boxed{list_datasets()}',
                r'\boxed{list_tools()}',
                r'\boxed{show_task()}'
            ])
            return choice
        else:
            checks = ["check_missing", "check_range", "check_duplicates", "check_schema",
                      "check_dtype", "check_units", "check_id_unique", "check_derived"]
            if random.random() < 0.7:
                return f'\\boxed{{{random.choice(checks)}()}}'
            else:
                cat = random.choice(self.all_categories)
                return f'\\boxed{{note_fault(category="{cat}")}}'

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self.terminated or self.truncated:
            return "Episode already ended.", 0.0, True, self.truncated, {"suffix": self.get_task_suffix()}

        self.turn_count += 1

        parsed = self._parse_action(action)
        if not parsed:
            obs = "FORMAT ERROR: Invalid action format. Use \\boxed{tool_name(arg=value,...)}."
            reward = LanguageGameReward.format_error_reward
            self.terminated = True
            return obs, reward, True, False, {"suffix": self.get_task_suffix()}

        tool, args = parsed

        if tool == "submit":
            ans = str(args.get("answer")).upper() if "answer" in args else None
            k = args.get("interventions", None)
            if ans not in ("SAT", "UNSAT") or not isinstance(k, int):
                obs = "FORMAT ERROR: submit requires answer=SAT|UNSAT and interventions=INTEGER."
                reward = LanguageGameReward.format_error_reward
                self.terminated = True
                return obs, reward, True, False, {"suffix": self.get_task_suffix()}

            true_k = self.task["total_min_interventions"]
            true_ans = "SAT" if true_k == 0 else "UNSAT"
            correct = (ans == true_ans) and (k == true_k if ans == "UNSAT" else k == 0)
            if correct:
                obs = f"SUCCESS: Final submission accepted and correct. Minimal interventions={true_k}."
                reward = 1.0
            else:
                obs = f"FINAL SUBMISSION INCORRECT: Expected ({true_ans}, {true_k}) but got ({ans}, {k})."
                reward = 0.0
            self.terminated = True
            return obs, reward, True, False, {"suffix": self.get_task_suffix()}

        if tool not in self.tools:
            obs = f"UNSUPPORTED ACTION: Unknown tool '{tool}'."
            reward = 0.0  # Fixed: was -0.5, failures should be 0.0
            self.terminated = True
            return obs, reward, True, False, {"suffix": self.get_task_suffix()}

        result = self._execute_tool(tool, args)
        if result.startswith("Protocol violation"):
            obs = result
            reward = 0.0
            terminated = False
            truncated = False
        elif result == "UNSUPPORTED ACTION":
            obs = "UNSUPPORTED ACTION: This check/tool is not available."
            reward = 0.0  # Fixed: was -0.5, failures should be 0.0
            self.terminated = True
            return obs, reward, True, False, {"suffix": self.get_task_suffix()}
        else:
            obs = f"OK: {result}"
            reward = 0.0
            terminated = False
            truncated = False
            self.steps_taken += 1

        if not self.terminated and not terminated and self.turn_count >= self.max_turns:
            obs = "TIMEOUT: Maximum turns reached without final submission."
            reward = 0.0
            self.truncated = True
            return obs, reward, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}


class ToolchestMaestroEnvWithFeedback(ToolchestMaestroEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        text = obs.lower()
        error_type = "OK"
        error_detail = {}
        hint = None

        if "format error" in text:
            error_type = "FormatError"
            error_detail["issue"] = "invalid_boxed_format_or_submit_args"
            hint = 'Use \\boxed{tool(args)}. For final: \\boxed{submit(answer=SAT|UNSAT, interventions=INTEGER)}.'

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["tool"] = "unknown"
            hint = "Call list_tools() to see available tools."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "no dataset loaded" in text:
                error_detail["violation"] = "missing_load_before_check"
                hint = "Call list_datasets() then load_dataset(name=...) before running check_* or note_fault."
            elif "dataset not found" in text:
                error_detail["violation"] = "invalid_dataset_name"
                hint = "Use list_datasets() to get a valid dataset name."
            else:
                error_detail["violation"] = "general_protocol"
                hint = "Ensure prerequisites are met (e.g., dataset loaded)."

        elif "timeout" in text:
            error_type = "Timeout"
            error_detail["outcome"] = "max_turns_reached"
            hint = "Plan to submit earlier. Use list_checks() and targeted checks before submit."

        elif "final submission incorrect" in text:
            error_type = "WrongDecision"
            true_k = getattr(self, "task", {}).get("total_min_interventions", None)
            error_detail["expected_interventions"] = true_k
            if "expected (" in text:
                m = re.search(r"expected \((sat|unsat),\s*(-?\d+)\).*got \((sat|unsat),\s*(-?\d+)\)", text)
                if m:
                    error_detail["expected"] = {"answer": m.group(1).upper(), "interventions": int(m.group(2))}
                    error_detail["got"] = {"answer": m.group(3).upper(), "interventions": int(m.group(4))}
            hint = "Systematically run check_* on each loaded dataset to count distinct fault categories; sum them and submit."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "current_dataset": getattr(self, "execution_state", {}).get("current_dataset", None),
                "datasets": getattr(self, "task", {}).get("datasets", []),
                "minimal_interventions": getattr(self, "task", {}).get("total_min_interventions", None),
                "steps_taken": getattr(self, "steps_taken", None),
                "max_turns": getattr(self, "max_turns", None),
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
            "hint": "Start with list_datasets(), then load_dataset(name=...), run relevant check_* tools, and submit with submit(answer=..., interventions=...).",
            "turn": 0,
            "state": {
                "current_dataset": None,
                "datasets": self.task["datasets"],
                "minimal_interventions": self.task["total_min_interventions"],
                "steps_taken": 0,
                "max_turns": self.max_turns,
            }
        }
        return obs, info