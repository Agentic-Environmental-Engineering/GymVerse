from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class CodeTraceEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 30,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 30

        self.complexity_params = {
            # program_length: number of top-level statements; longer program increases tracing difficulty
            "program_length": (4, 16),
            # num_variables: number of mutable variables; more variables increases state space
            "num_variables": (2, 6),
            # loop_count: number of top-level loops; more loops increases iterative reasoning complexity
            "loop_count": (0, 6),
            # max_loop_depth: maximum nesting depth of loops; deeper nesting increases complexity
            "max_loop_depth": (0, 3),
            # const_range: absolute bound of constants used; larger magnitudes produce larger values and harder mental arithmetic
            "const_range": (5, 25),
            # ops_variety: number of operation types allowed among {add, sub, mul}; more variety increases reasoning complexity
            "ops_variety": (1, 3),
            # REVERSED: max_reveal_actions: fewer reveals allowed makes it harder due to partial observability pressure
            "max_reveal_actions": (10, 3),
        }

        self.param_variance = {
            "program_length": 1,
            "num_variables": 1,
            "loop_count": 1,
            "max_loop_depth": 0,
            "const_range": 3,
            "ops_variety": 0,
            "max_reveal_actions": 1,
        }

        self.program_length: int = 0
        self.num_variables: int = 0
        self.loop_count: int = 0
        self.max_loop_depth: int = 0
        self.const_range: int = 0
        self.ops_variety: int = 0
        self.max_reveal_actions: int = 0

        self.turn_count: int = 0
        self.program: List[Any] = []
        self.variables: Dict[str, int] = {}
        self.var_names: List[str] = []
        self.target_var: str = ""
        self.target_value: int = 0
        self.reveals_used: int = 0
        self.memo: Dict[str, str] = {}
        self.ops_available: List[str] = []

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for param_name, (min_val, max_val) in self.complexity_params.items():
            center_value = min_val + (max_val - min_val) * normalized
            actual_value = center_value
            if self.enable_param_randomization:
                variance = self.param_variance.get(param_name, 0)
                if variance > 0:
                    actual_value = center_value + random.uniform(-variance, variance)
            # clamp considering reversed ranges
            if min_val > max_val:
                low, high = max_val, min_val
            else:
                low, high = min_val, max_val
            actual_value = max(low, min(high, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        lines = []
        lines.append("You are tracing a hidden pseudo-code program.")
        lines.append("Goal: compute the final integer value of a hidden target variable after executing the program.")
        lines.append("Program DSL semantics:")
        lines.append("- set x = C            : assign constant C to variable x")
        lines.append("- set x = y + C        : assign x to y plus constant C")
        lines.append("- add x C              : x = x + C")
        lines.append("- sub x C              : x = x - C")
        lines.append("- mul x C              : x = x * C   (C is a small positive integer)")
        lines.append("- loop N { ... }       : repeat the body exactly N times (loops can be nested)")
        lines.append("- // comment           : no-op")
        lines.append("Actions:")
        lines.append("- meta                 : show summary metadata (target unknown)")
        lines.append("- list_vars            : list variable names")
        lines.append("- show <path>          : reveal a statement at path (e.g., 'show 2', 'show 3.1', 'show 1.2.1')")
        lines.append("- run                  : execute full program and reveal all final variable values")
        lines.append("- memo key=value       : store a note in local memory")
        lines.append("- submit <int>         : submit final answer for the target variable (terminates)")
        lines.append("- help                 : show the DSL and action guide")
        lines.append("Format all actions as \\boxed{...}.")
        example = self.sample_random_action()
        lines.append(f"Example: {example}")
        return "\n".join(lines)

    def get_task_suffix(self) -> str:
        lines = []
        lines.append(f"Turns: {self.turn_count}/{self.max_turns}")
        lines.append(f"Reveals used: {self.reveals_used}/{self.max_reveal_actions}")
        lines.append(f"Known variables: {', '.join(self.var_names)}")
        lines.append("Target variable is hidden; use meta/show/run to explore.")
        lines.append("Enter your action in \\boxed{...} format.")
        return "\n".join(lines)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()
        self.turn_count = 0
        self.reveals_used = 0
        self.memo = {}

        base_names = ["x", "y", "z", "u", "v", "w", "p", "q", "r"]
        self.var_names = base_names[: self.num_variables]
        self.variables = {name: 0 for name in self.var_names}

        ops_pool = ["add", "sub", "mul"]
        random.shuffle(ops_pool)
        self.ops_available = ops_pool[: self.ops_variety]

        self.program = self._generate_program()
        self.target_var = random.choice(self.var_names)
        self.target_value = self._execute_program()

        obs = self._get_instructions()
        info = {"suffix": self.get_task_suffix()}
        return obs, info

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if not parsed:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}"
            return (
                obs,
                LanguageGameReward.format_error_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        cmd, payload = parsed

        if cmd == "meta":
            obs = self._meta_observation()
            reward = 0.0

        elif cmd == "list_vars":
            obs = f"Variables: {', '.join(self.var_names)}"
            reward = 0.0

        elif cmd == "show":
            if self.reveals_used >= self.max_reveal_actions:
                obs = (
                    f"Protocol violation: reveal budget exceeded "
                    f"(max_reveal_actions={self.max_reveal_actions})."
                )
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            path = payload
            node = self._resolve_path(path)
            if node is None:
                obs = "Protocol violation: invalid path."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            desc = self._describe_node(node)
            self.reveals_used += 1
            obs = f"Reveal {'.'.join(str(i) for i in path)}: {desc}"
            reward = 0.0

        elif cmd == "run":
            finals = self._execute_program(finals=True)
            kv = ", ".join(f"{k}={v}" for k, v in finals.items())
            obs = f"Program executed. Final values: {kv}. Submit with \\boxed{{submit <int>}}."
            reward = 0.0

        elif cmd == "memo":
            key, value = payload
            self.memo[key] = value
            obs = f"Memo stored: {key}={value}"
            reward = 0.0

        elif cmd == "submit":
            val = payload
            correct = (val == self.target_value)
            if correct:
                obs = f"Success! Correct value for target: {val}"
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Failed! Submitted {val}, expected {self.target_value}."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        elif cmd == "help":
            obs = self._get_instructions()
            reward = 0.0

        else:
            obs = f"Unsupported action: {cmd}"
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Tuple[str, Any]]:
        if not action:
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        if not content:
            return None

        if content.lower() == "meta":
            return ("meta", None)
        if content.lower() == "run":
            return ("run", None)
        if content.lower() == "list_vars":
            return ("list_vars", None)
        if content.lower() == "help":
            return ("help", None)

        m_show = re.match(r'^\s*show\s+([0-9]+(?:\.[0-9]+)*)\s*$', content, re.IGNORECASE)
        if m_show:
            path_str = m_show.group(1)
            path = [int(x) for x in path_str.split(".")]
            return ("show", path)

        m_submit = re.match(r'^\s*submit\s+(-?\d+)\s*$', content, re.IGNORECASE)
        if m_submit:
            try:
                val = int(m_submit.group(1))
            except ValueError:
                return None
            return ("submit", val)

        m_memo = re.match(r'^\s*memo\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)\s*$', content, re.IGNORECASE)
        if m_memo:
            key = m_memo.group(1)
            value = m_memo.group(2)
            return ("memo", (key, value))

        return ("unsupported", content)

    def sample_random_action(self) -> str:
        choices = [
            "\\boxed{meta}",
            "\\boxed{list_vars}",
            "\\boxed{show 1}",
            "\\boxed{run}",
            "\\boxed{memo note=trace first loop}",
            "\\boxed{submit 0}",
            "\\boxed{help}",
        ]
        return random.choice(choices)

    def _generate_program(self) -> List[Any]:
        def rnd_const():
            return random.randint(-self.const_range, self.const_range)

        def rnd_small_pos():
            return random.randint(2, min(5, max(2, self.const_range)))

        def make_assign():
            var = random.choice(self.var_names)
            form = random.choice(["const", "var_plus_const", "copy_var"])
            if form == "const":
                return {"type": "assign", "var": var, "expr": {"kind": "const", "value": rnd_const()}}
            elif form == "copy_var":
                src = random.choice(self.var_names)
                return {"type": "assign", "var": var, "expr": {"kind": "var", "name": src}}
            else:
                src = random.choice(self.var_names)
                c = rnd_const()
                return {"type": "assign", "var": var, "expr": {"kind": "var_plus_const", "name": src, "const": c}}

            # unreachable

        def make_update():
            var = random.choice(self.var_names)
            op = random.choice(self.ops_available)
            if op in ["add", "sub"]:
                c = rnd_const()
            else:
                c = rnd_small_pos()
            return {"type": "update", "op": op, "var": var, "const": c}

        def make_noop():
            text = random.choice([
                "TODO optimize",
                "temporary logging",
                "refactor later",
                "placeholder",
                "no-op",
            ])
            return {"type": "noop", "text": f"// {text}"}

        def make_loop(depth: int) -> Dict[str, Any]:
            count = random.randint(2, min(6, max(2, self.const_range)))
            body_len = random.randint(1, 4)
            body = []
            for _ in range(body_len):
                choice = random.random()
                if self.max_loop_depth > depth and choice < 0.25 and self.loop_count > 0:
                    body.append(make_loop(depth + 1))
                elif choice < 0.70:
                    body.append(make_update())
                elif choice < 0.90:
                    body.append(make_assign())
                else:
                    body.append(make_noop())
            return {"type": "loop", "count": count, "body": body}

        prog: List[Any] = []
        loops_to_place = min(self.loop_count, max(0, self.program_length - 1))
        for i in range(self.program_length):
            p = random.random()
            if loops_to_place > 0 and p < 0.35:
                prog.append(make_loop(1))
                loops_to_place -= 1
            elif p < 0.70:
                prog.append(make_assign())
            elif p < 0.90:
                prog.append(make_update())
            else:
                prog.append(make_noop())
        return prog

    def _resolve_path(self, path: List[int]) -> Optional[Dict[str, Any]]:
        node_list = self.program
        node = None
        for idx in path:
            if idx < 1 or idx > len(node_list):
                return None
            node = node_list[idx - 1]
            if idx == path[-1]:
                return node
            if node.get("type") != "loop":
                return None
            node_list = node.get("body", [])
        return node

    def _describe_node(self, node: Dict[str, Any]) -> str:
        t = node.get("type")
        if t == "assign":
            var = node["var"]
            expr = node["expr"]
            if expr["kind"] == "const":
                return f"set {var} = {expr['value']}"
            elif expr["kind"] == "var":
                return f"set {var} = {expr['name']}"
            else:
                return f"set {var} = {expr['name']} + {expr['const']}"
        elif t == "update":
            op = node["op"]
            var = node["var"]
            const = node["const"]
            if op == "add":
                return f"add {var} {const}"
            elif op == "sub":
                return f"sub {var} {const}"
            else:
                return f"mul {var} {const}"
        elif t == "loop":
            return f"loop {node['count']} {{ ... {len(node.get('body', []))} statements ... }}"
        elif t == "noop":
            return node.get("text", "// comment")
        else:
            return "unknown"

    def _meta_observation(self) -> str:
        def count_loops(nodes: List[Any]) -> int:
            c = 0
            for n in nodes:
                if n.get("type") == "loop":
                    c += 1 + count_loops(n.get("body", []))
            return c

        total_loops = count_loops(self.program)
        obs = (
            f"Metadata: top-level statements={len(self.program)}, "
            f"total_loops={total_loops}, max_loop_depth={self.max_loop_depth}, "
            f"ops_available={','.join(self.ops_available)}. Target variable remains hidden."
        )
        return obs

    def _execute_program(self, finals: bool = False) -> Any:
        env = {k: 0 for k in self.var_names}

        def eval_expr(expr: Dict[str, Any]) -> int:
            k = expr["kind"]
            if k == "const":
                return int(expr["value"])
            elif k == "var":
                return int(env.get(expr["name"], 0))
            elif k == "var_plus_const":
                return int(env.get(expr["name"], 0)) + int(expr["const"])
            else:
                return 0

        def exec_node(node: Dict[str, Any]):
            t = node.get("type")
            if t == "assign":
                env[node["var"]] = eval_expr(node["expr"])
            elif t == "update":
                v = node["var"]
                c = int(node["const"])
                if node["op"] == "add":
                    env[v] = env.get(v, 0) + c
                elif node["op"] == "sub":
                    env[v] = env.get(v, 0) - c
                elif node["op"] == "mul":
                    env[v] = env.get(v, 0) * c
            elif t == "loop":
                count = int(node.get("count", 0))
                body = node.get("body", [])
                for _ in range(max(0, count)):
                    for b in body:
                        exec_node(b)
            elif t == "noop":
                pass

        for n in self.program:
            exec_node(n)

        if finals:
            return dict(env)
        return int(env.get(self.target_var, 0))


class CodeTraceEnvWithFeedback(CodeTraceEnv):
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
            hint = "Wrap your command like \\boxed{run} or \\boxed{show 2.1}."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "reveal budget exceeded" in text:
                error_detail["violation"] = "reveal_budget_exceeded"
                error_detail["max_reveals"] = getattr(self, "max_reveal_actions", None)
                error_detail["reveals_used"] = getattr(self, "reveals_used", None)
                hint = "Stop revealing; use \\boxed{run} to compute final values, then \\boxed{submit <int>}."
            elif "invalid path" in text:
                error_detail["violation"] = "invalid_path"
                hint = "Use a valid path like 1 or 3.2. Paths are 1-indexed; show 2.1 reveals inside loop at top-level 2."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            m = re.search(r"unsupported action: ([^\n]+)", obs, re.IGNORECASE)
            error_detail["action"] = m.group(1).strip() if m else None
            hint = "Use one of: meta, list_vars, show <path>, run, memo key=value, submit <int>, help."

        elif "failed! submitted" in text:
            error_type = "WrongDecision"
            m = re.search(r"failed! submitted\s+(-?\d+),\s+expected\s+(-?\d+)", obs, re.IGNORECASE)
            got = int(m.group(1)) if m else None
            expected = int(m.group(2)) if m else getattr(self, "target_value", None)
            error_detail["got"] = got
            error_detail["expected"] = expected
            hint = "Execute the program via \\boxed{run} to see final values before submitting."

        elif truncated:
            error_type = "Timeout"
            error_detail["max_turns"] = getattr(self, "max_turns", None)
            error_detail["turn"] = getattr(self, "turn_count", None)
            hint = "Act sooner: use \\boxed{run} then \\boxed{submit <int>}."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["reveal_used"] = getattr(self, "reveals_used", None)
            diagnostic["max_reveals"] = getattr(self, "max_reveal_actions", None)
            diagnostic["variables"] = getattr(self, "var_names", None)
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start with \\boxed{meta} or \\boxed{run}, then plan your submission.",
            "turn": 0,
            "reveal_used": 0,
            "max_reveals": getattr(self, "max_reveal_actions", None),
        }
        return obs, info