from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class AlgorithmDAGEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 100,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 100

        # Evolvable parameters
        self.complexity_params = {
            # Number of nodes in DAG: larger → more computation paths to resolve → harder
            "num_nodes": (6, 28),
            # Number of input/source nodes: more inputs → more initial values and branches → harder
            "num_inputs": (2, 7),
            # Maximum number of parents per non-input node: higher fan-in → more dependencies → harder
            "max_fan_in": (2, 4),
            # Operation variety: more different operations → more cognitive load → harder
            "op_variety": (4, 9),
            # Input value magnitude range (absolute): larger numbers → harder arithmetic, potential overflow control
            "value_range": (9, 50),
            # REVERSED: maximum number of peek/neighbors queries allowed; fewer queries → less information → harder
            "max_peeks": (30, 10),
        }

        # Variance settings
        self.param_variance = {
            "num_nodes": 2,
            "num_inputs": 1,
            "max_fan_in": 0,
            "op_variety": 1,
            "value_range": 5,
            "max_peeks": 3,
        }

        # Placeholder attributes
        self.num_nodes: int = 0
        self.num_inputs: int = 0
        self.max_fan_in: int = 0
        self.op_variety: int = 0
        self.value_range: int = 0
        self.max_peeks: int = 0

        # Domain state
        self.turn_count: int = 0
        self.peeks_used: int = 0
        self.sink_id: int = 0
        self.parents: Dict[int, list] = {}
        self.children: Dict[int, list] = {}
        self.op_types: Dict[int, Optional[str]] = {}
        self.op_params: Dict[int, Optional[int]] = {}
        self.initial_inputs: Dict[int, int] = {}
        self.true_values: Dict[int, int] = {}
        self.collected: set = set()
        self.collected_values: Dict[int, int] = {}

        self.all_ops = [
            "add", "max", "min", "avg", "xor", "mul", "abs", "neg", "mod", "sub", "thresh"
        ]

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for param_name, (min_val, max_val) in self.complexity_params.items():
            base = min_val + (max_val - min_val) * normalized
            actual_value = base
            if self.enable_param_randomization:
                var = self.param_variance.get(param_name, 0)
                if var > 0:
                    actual_value = base + random.uniform(-var, var)
            # Clamp and handle reversed ranges
            low = min(min_val, max_val)
            high = max(min_val, max_val)
            actual_value = max(low, min(high, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        return (
            "You are tracing a hidden algorithm represented as a computation DAG.\n"
            "Each node outputs an integer by applying its operation to parent node values.\n"
            "Inputs are fixed integers at source nodes; internal nodes depend on their parents.\n"
            "Your goal is to compute the final sink node's output and submit it.\n"
            "\n"
            "Available actions (use \\boxed{...}):\n"
            "- help\n"
            "- status\n"
            "- peek i          (reveal node i's definition; i is an integer id)\n"
            "- neighbors i     (list parents and children of node i)\n"
            "- collect i       (compute and store node i's value if parents are collected)\n"
            "- answer v        (submit final integer v as the sink's value; ends episode)\n"
            "\n"
            f"Example: {self.sample_random_action()}\n"
            "Malformed or unsupported actions end the episode with a penalty.\n"
        )

    def get_task_suffix(self) -> str:
        turns_left = self.max_turns - self.turn_count
        peeks_left = max(0, self.max_peeks - self.peeks_used)
        collected_list = sorted(list(self.collected))
        show = collected_list[:10]
        more = "" if len(collected_list) <= 10 else f" and {len(collected_list)-10} more"
        return (
            f"DAG summary: nodes={self.num_nodes}, inputs={self.num_inputs}, sink={self.sink_id}.\n"
            f"Progress: collected={len(self.collected)} ({show}{more}), peeks_left={peeks_left}, turns_left={turns_left}.\n"
            "Enter your action as \\boxed{command}, e.g., \\boxed{collect 3} or \\boxed{answer 42}."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()
        self.turn_count = 0
        self.peeks_used = 0

        # Generate DAG
        n = self.num_nodes
        k_inputs = min(self.num_inputs, n - 1)
        self.sink_id = n

        self.parents = {i: [] for i in range(1, n + 1)}
        self.children = {i: [] for i in range(1, n + 1)}
        self.op_types = {i: None for i in range(1, n + 1)}
        self.op_params = {i: None for i in range(1, n + 1)}
        self.initial_inputs = {}
        self.true_values = {}
        self.collected = set()
        self.collected_values = {}

        # Determine operation set
        ops = random.sample(self.all_ops, min(self.op_variety, len(self.all_ops)))

        # Assign inputs: nodes 1..k_inputs
        for i in range(1, k_inputs + 1):
            val = random.randint(-self.value_range, self.value_range)
            self.initial_inputs[i] = val
            self.op_types[i] = "input"
            self.op_params[i] = None

        # Assign parents and operations for internal nodes
        for i in range(k_inputs + 1, n + 1):
            avail = list(range(1, i))
            fan_in = random.randint(1, self.max_fan_in)
            fan_in = min(fan_in, len(avail))
            ps = random.sample(avail, fan_in)
            self.parents[i] = ps
            for p in ps:
                self.children[p].append(i)

            # Choose op consistent with arity
            arity = fan_in
            candidates = []
            for op in ops:
                if op in ["abs", "neg", "mod"] and arity == 1:
                    candidates.append(op)
                elif op in ["sub"] and arity >= 2:
                    candidates.append(op)
                elif op in ["thresh"] and arity >= 1:
                    candidates.append(op)
                elif op in ["mul"] and arity >= 2:
                    candidates.append(op)
                elif op in ["add", "max", "min", "avg", "xor"] and arity >= 1:
                    candidates.append(op)
            if not candidates:
                # Fallback to add
                candidates = ["add"]
            chosen = random.choice(candidates)
            self.op_types[i] = chosen
            if chosen in ["mod", "thresh"]:
                self.op_params[i] = random.randint(3, max(3, self.value_range))
            else:
                self.op_params[i] = None

        # Compute true values topologically
        def clamp(x: int) -> int:
            return max(-10**9, min(10**9, x))

        for i in range(1, n + 1):
            if self.op_types[i] == "input":
                self.true_values[i] = self.initial_inputs[i]
            else:
                ps = self.parents[i]
                vals = [self.true_values[p] for p in ps]
                op = self.op_types[i]
                param = self.op_params[i]
                if op == "add":
                    self.true_values[i] = clamp(sum(vals))
                elif op == "max":
                    self.true_values[i] = max(vals) if vals else 0
                elif op == "min":
                    self.true_values[i] = min(vals) if vals else 0
                elif op == "avg":
                    self.true_values[i] = int(sum(vals) // max(1, len(vals)))
                elif op == "xor":
                    v = 0
                    for x in vals:
                        v ^= x
                    self.true_values[i] = v
                elif op == "mul":
                    v = 1
                    for x in vals:
                        v *= x
                        if abs(v) > 10**9:
                            v = clamp(v)
                            break
                    self.true_values[i] = clamp(v)
                elif op == "abs":
                    self.true_values[i] = abs(vals[0]) if vals else 0
                elif op == "neg":
                    self.true_values[i] = clamp(-vals[0]) if vals else 0
                elif op == "mod":
                    c = param if param is not None else 3
                    base = abs(vals[0]) if vals else 0
                    self.true_values[i] = base % c
                elif op == "sub":
                    a = vals[0] if len(vals) >= 1 else 0
                    b = vals[1] if len(vals) >= 2 else 0
                    self.true_values[i] = clamp(a - b)
                elif op == "thresh":
                    t = param if param is not None else 0
                    s = sum(vals)
                    self.true_values[i] = 1 if s >= t else 0
                else:
                    self.true_values[i] = 0

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if parsed is None:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        cmd, arg = parsed

        if cmd == "help":
            obs = self._get_instructions().strip()
            reward = 0.0
        elif cmd == "status":
            peeks_left = max(0, self.max_peeks - self.peeks_used)
            obs = (
                f"Status: collected={len(self.collected)} nodes, sink={self.sink_id}, peeks_left={peeks_left}. "
                f"Collected ids: {sorted(list(self.collected))[:10]}"
            )
            reward = 0.0
        elif cmd in ("peek", "neighbors"):
            if not isinstance(arg, int) or arg < 1 or arg > self.num_nodes:
                obs = f"Unsupported action: node id {arg} is out of range 1..{self.num_nodes}."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            self.peeks_used += 1
            if self.peeks_used > self.max_peeks:
                obs = "Protocol violation: exceeded peek/neighbors query budget."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if cmd == "peek":
                t = self.op_types[arg]
                if t == "input":
                    v = self.initial_inputs[arg]
                    obs = f"Peek result: node {arg} is INPUT with value {v}."
                else:
                    par = self.parents[arg]
                    if self.op_params[arg] is not None:
                        obs = f"Peek result: node {arg} op={t}({self.op_params[arg]}) arity={len(par)} parents={par}."
                    else:
                        obs = f"Peek result: node {arg} op={t} arity={len(par)} parents={par}."
                reward = 0.0
            else:
                obs = (
                    f"Neighbors: node {arg} parents={self.parents[arg]} children={self.children[arg]}."
                )
                reward = 0.0
        elif cmd == "collect":
            if not isinstance(arg, int) or arg < 1 or arg > self.num_nodes:
                obs = f"Unsupported action: node id {arg} is out of range 1..{self.num_nodes}."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            i = arg
            if i in self.collected:
                obs = f"Collect: node {i} is already collected with value {self.collected_values[i]}."
                reward = 0.0
            else:
                t = self.op_types[i]
                if t == "input":
                    val = self.true_values[i]
                    self.collected.add(i)
                    self.collected_values[i] = val
                    obs = f"Collect success: node {i} value={val}."
                    reward = 0.0
                else:
                    missing = [p for p in self.parents[i] if p not in self.collected]
                    if missing:
                        obs = f"Cannot collect node {i}: missing parents {missing}."
                        reward = 0.0
                    else:
                        val = self.true_values[i]
                        self.collected.add(i)
                        self.collected_values[i] = val
                        obs = f"Collect success: node {i} value={val}."
                        reward = 0.0
        elif cmd == "answer":
            true_val = self.true_values[self.sink_id]
            if arg == true_val:
                obs = "Success! Correct final value submitted."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Failed! Incorrect final value {arg}."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
        else:
            obs = "Unsupported action. Available: help, status, peek i, neighbors i, collect i, answer v."
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            truncated = True
            terminated = True
            obs = f"Reached max turns ({self.max_turns})."
            return obs, 0.0, terminated, truncated, {"suffix": self.get_task_suffix()}

        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Any]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        m = pattern.findall(action)
        if not m:
            return None
        content = m[-1].strip()
        low = content.lower().strip()

        if low == "help":
            return ("help", None)
        if low == "status":
            return ("status", None)

        m_peek = re.match(r'^\s*peek\s+(\d+)\s*$', low)
        if m_peek:
            return ("peek", int(m_peek.group(1)))

        m_nei = re.match(r'^\s*neighbors\s+(\d+)\s*$', low)
        if m_nei:
            return ("neighbors", int(m_nei.group(1)))

        m_col = re.match(r'^\s*collect\s+(\d+)\s*$', low)
        if m_col:
            return ("collect", int(m_col.group(1)))

        m_ans = re.match(r'^\s*answer\s+(-?\d+)\s*$', low)
        if m_ans:
            return ("answer", int(m_ans.group(1)))

        return None

    def sample_random_action(self) -> str:
        if self.num_nodes >= 2:
            act = f"peek {random.randint(1, self.num_nodes)}"
        else:
            act = "status"
        return f"\\boxed{{{act}}}"


class AlgorithmDAGEnvWithFeedback(AlgorithmDAGEnv):
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
            hint = "Wrap your command using \\boxed{...}, e.g., \\boxed{collect 3}."
        elif "unsupported action:" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "invalid_node_id"
            hint = f"Use a valid node id between 1 and {self.num_nodes}. Check neighbors or status first."
        elif "unsupported action. available" in text:
            error_type = "UnsupportedAction"
            error_detail["allowed"] = ["help", "status", "peek i", "neighbors i", "collect i", "answer v"]
            hint = "Use one of the allowed commands exactly."
        elif "protocol violation: exceeded peek" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "peek_budget_exceeded"
            error_detail["peeks_used"] = getattr(self, "peeks_used", None)
            error_detail["max_peeks"] = getattr(self, "max_peeks", None)
            hint = "Limit your peek/neighbors queries; rely on collect attempts and status to discover structure."
        elif "failed! incorrect final value" in text:
            error_type = "WrongDecision"
            error_detail["got"] = self._extract_number(text)
            error_detail["expected_known"] = False
            hint = "Ensure you collected all required parent nodes before answering; verify arithmetic."
        elif "cannot collect node" in text and "missing parents" in text:
            error_type = "OK"
            missing = self._extract_missing(text)
            error_detail["missing_parents"] = missing
            if self.feedback_level >= 2:
                hint = f"Collect missing parents {missing} first. Use \\boxed{{collect p}} for each parent."
        elif "success! correct final value" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["max_turns"] = getattr(self, "max_turns", None)
            hint = "Plan fewer queries and collect nodes efficiently; answer promptly when ready."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "collected_count": len(getattr(self, "collected", [])),
                "peeks_left": max(0, getattr(self, "max_peeks", 0) - getattr(self, "peeks_used", 0)),
                "sink_id": getattr(self, "sink_id", None),
                "num_nodes": getattr(self, "num_nodes", None),
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
            "hint": "Start by collecting input nodes (try small ids) and then progress to their children.",
            "turn": 0,
            "state": {
                "collected_count": 0,
                "peeks_left": getattr(self, "max_peeks", 0),
                "sink_id": getattr(self, "sink_id", None),
                "num_nodes": getattr(self, "num_nodes", None),
            },
        }
        return obs, info

    def _extract_number(self, text: str) -> Optional[int]:
        m = re.search(r'(-?\d+)', text)
        return int(m.group(1)) if m else None

    def _extract_missing(self, text: str) -> list:
        m = re.search(r'missing parents \[([^\]]*)\]', text)
        if not m:
            return []
        part = m.group(1).strip()
        if not part:
            return []
        nums = re.findall(r'(\d+)', part)
        return [int(x) for x in nums]