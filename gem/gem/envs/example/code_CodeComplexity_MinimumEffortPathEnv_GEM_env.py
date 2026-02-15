from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class CodeComplexityEnv(Env):
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
            # Total nodes in AST: larger program → more exploration and reasoning
            "num_nodes": (6, 40),
            # Maximum nesting depth: deeper nesting → more complex composition of exponents
            "max_depth": (2, 6),
            # Maximum number of IF nodes to permit: more branching → harder worst-case reasoning
            "num_if_nodes": (0, 10),
            # Maximum number of SEQ nodes to permit: more composition → more paths to consider
            "num_seq_nodes": (1, 8),
            # Maximum allowed loop exponent p in n^p: higher variability increases reasoning burden
            "max_loop_power": (1, 5),
            # Maximum branches per IF: more branches → more paths for worst-case selection
            "max_branches_per_if": (2, 3),
            # REVERSED: number of free HELP hints; fewer hints → harder
            "help_quota": (2, 0),
        }

        # Parameter variance
        self.param_variance = {
            "num_nodes": 3,               # large range → ±3 nodes variation
            "max_depth": 1,               # medium range → ±1
            "num_if_nodes": 1,            # medium range → ±1
            "num_seq_nodes": 1,           # medium range → ±1
            "max_loop_power": 1,          # medium range → ±1
            "max_branches_per_if": 0,     # small range → fix
            "help_quota": 0,              # small range reversed → fix by level
        }

        # Placeholder attributes set in _apply_complexity_params
        self.num_nodes: int = 0
        self.max_depth: int = 0
        self.num_if_nodes: int = 0
        self.num_seq_nodes: int = 0
        self.max_loop_power: int = 0
        self.max_branches_per_if: int = 0
        self.help_quota: int = 0

        # State
        self.turn_count: int = 0
        self.nodes: Dict[int, Dict[str, Any]] = {}
        self.root_id: int = 0
        self.true_exponent: int = 0
        self.visited: set = set()
        self.stack: list = []
        self.notes: Dict[int, int] = {}
        self.remaining_help: int = 0

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for param_name, (min_val, max_val) in self.complexity_params.items():
            center_value = min_val + (max_val - min_val) * normalized
            variance = self.param_variance.get(param_name, 0)
            actual_value = center_value
            if self.enable_param_randomization and variance > 0:
                actual_value = center_value + random.uniform(-variance, variance)
            # clamp support reversed ranges
            low = min(min_val, max_val)
            high = max(min_val, max_val)
            actual_value = max(low, min(high, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        return (
            "You are analyzing a hidden pseudo-program to determine its worst-case polynomial time exponent k with respect to n.\n"
            "Program structure uses nodes:\n"
            "- WORK: constant-cost leaf (exponent 0)\n"
            "- LOOP p: iterates n^p times; total exponent adds p plus its body's exponent\n"
            "- SEQ: sequential composition; sum dominated by max exponent among children\n"
            "- IF: conditional; worst-case exponent is max among branches\n"
            "Available actions:\n"
            "- ROOT: return the root node id\n"
            "- INSPECT i: show type and parameters of node i and mark it visited\n"
            "- CHILDREN i: list the children ids of node i\n"
            "- STACK_PUSH i: push node id i onto your internal stack\n"
            "- STACK_POP: pop one id from your internal stack\n"
            "- NOTE i k: record a note that node i has exponent k (integer)\n"
            "- READ_NOTE i: read the noted exponent for node i\n"
            "- STATUS: show progress summary\n"
            "- HELP: get a guidance hint (limited quota)\n"
            "- ANSWER k: submit your final integer exponent k and end the episode\n"
            "Rules:\n"
            "- Use \\boxed{...} format for actions.\n"
            "- Invalid actions or indices terminate the episode with a penalty.\n"
            "- Only the final ANSWER returns non-zero reward.\n"
            f"Example action: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        visited_list = sorted(list(self.visited))
        stack_preview = list(self.stack)[-3:]
        notes_summary = ", ".join([f"{k}:{v}" for k, v in sorted(self.notes.items())]) if self.notes else "none"
        return (
            f"Turn {self.turn_count} | visited={visited_list} | stack_tail={stack_preview} | notes={notes_summary} | help_left={self.remaining_help}\n"
            "Enter your action using \\boxed{ACTION [args]}."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()

        self.turn_count = 0
        self.nodes = {}
        self.root_id = 0
        self.true_exponent = 0
        self.visited = set()
        self.stack = []
        self.notes = {}
        self.remaining_help = self.help_quota

        self._build_program()
        self.true_exponent = self._compute_exponent(self.root_id)

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if not parsed:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        name = parsed["name"]
        reward = 0.0

        # Protocol checks and execution
        if name == "ROOT":
            obs = f"Root node id: {self.root_id}"
            return obs, reward, False, False, {"suffix": self.get_task_suffix()}

        elif name == "INSPECT":
            nid = parsed.get("id", None)
            if nid is None or nid not in self.nodes:
                obs = f"Protocol violation: node {nid} does not exist."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            node = self.nodes[nid]
            self.visited.add(nid)
            if node["type"] == "LOOP":
                obs = (
                    f"Node {nid}: TYPE=LOOP, power={node['power']}, "
                    f"children={node['children']}"
                )
            else:
                obs = f"Node {nid}: TYPE={node['type']}, children={node['children']}"
            return obs, reward, False, False, {"suffix": self.get_task_suffix()}

        elif name == "CHILDREN":
            nid = parsed.get("id", None)
            if nid is None or nid not in self.nodes:
                obs = f"Protocol violation: node {nid} does not exist."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            ch = self.nodes[nid]["children"]
            obs = f"Children of {nid}: {ch if ch else 'none'}"
            return obs, reward, False, False, {"suffix": self.get_task_suffix()}

        elif name == "STACK_PUSH":
            nid = parsed.get("id", None)
            if nid is None or nid not in self.nodes:
                obs = f"Protocol violation: node {nid} does not exist."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            self.stack.append(nid)
            obs = f"Pushed {nid}. Stack size now {len(self.stack)}."
            return obs, reward, False, False, {"suffix": self.get_task_suffix()}

        elif name == "STACK_POP":
            if len(self.stack) == 0:
                obs = "Stack is empty. Nothing popped."
            else:
                val = self.stack.pop()
                obs = f"Popped {val}. Stack size now {len(self.stack)}."
            return obs, reward, False, False, {"suffix": self.get_task_suffix()}

        elif name == "NOTE":
            nid = parsed.get("id", None)
            val = parsed.get("value", None)
            if nid is None or nid not in self.nodes or not isinstance(val, int):
                obs = "Protocol violation: NOTE requires valid node id and integer k."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            self.notes[nid] = val
            obs = f"Noted node {nid} exponent={val}."
            return obs, reward, False, False, {"suffix": self.get_task_suffix()}

        elif name == "READ_NOTE":
            nid = parsed.get("id", None)
            if nid is None or nid not in self.nodes:
                obs = f"Protocol violation: node {nid} does not exist."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if nid in self.notes:
                obs = f"Note for node {nid}: {self.notes[nid]}"
            else:
                obs = f"No note recorded for node {nid}."
            return obs, reward, False, False, {"suffix": self.get_task_suffix()}

        elif name == "STATUS":
            obs = self.get_task_suffix()
            return obs, reward, False, False, {"suffix": self.get_task_suffix()}

        elif name == "HELP":
            if self.remaining_help > 0:
                self.remaining_help -= 1
                obs = (
                    "Hint: Use ROOT to get the starting id. Explore with INSPECT and CHILDREN. "
                    "For LOOP p, exponent adds p plus child's exponent. "
                    "For SEQ and IF, take the max of child/branch exponents for worst-case."
                )
            else:
                obs = "No help remaining."
            return obs, reward, False, False, {"suffix": self.get_task_suffix()}

        elif name == "ANSWER":
            val = parsed.get("value", None)
            if not isinstance(val, int):
                obs = "Protocol violation: ANSWER requires an integer k."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if val == self.true_exponent:
                obs = f"Success! Correct exponent k={val}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Failed! Your k={val} but the correct k was different."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = f"Unsupported action: {name}."
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        # Timeout check
        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        if not content:
            return None
        tokens = re.split(r'\s+', content)
        if not tokens:
            return None
        name = tokens[0].strip().upper()
        parsed: Dict[str, Any] = {"name": name}
        try:
            if name in ("ROOT", "STACK_POP", "STATUS", "HELP"):
                pass
            elif name in ("INSPECT", "CHILDREN", "STACK_PUSH", "READ_NOTE"):
                if len(tokens) < 2:
                    parsed["id"] = None
                else:
                    parsed["id"] = int(tokens[1])
            elif name == "NOTE":
                if len(tokens) < 3:
                    parsed["id"] = None
                    parsed["value"] = None
                else:
                    parsed["id"] = int(tokens[1])
                    parsed["value"] = int(tokens[2])
            elif name == "ANSWER":
                if len(tokens) < 2:
                    parsed["value"] = None
                else:
                    parsed["value"] = int(tokens[1])
            else:
                parsed["name"] = name
            return parsed
        except ValueError:
            return {"name": name, "id": None, "value": None}

    def sample_random_action(self) -> str:
        choices = [
            "\\boxed{ROOT}",
            "\\boxed{STATUS}",
            "\\boxed{HELP}",
            "\\boxed{STACK_POP}",
        ]
        return random.choice(choices)

    # Internal helpers
    def _add_node(self, typ: str, power: Optional[int] = None) -> int:
        nid = len(self.nodes) + 1
        self.nodes[nid] = {"type": typ, "children": [], "power": power}
        return nid

    def _build_program(self):
        # Create root node respecting budget
        # If tiny program budget, make root WORK
        if self.num_nodes <= 2 or self.max_depth <= 1:
            self.root_id = self._add_node("WORK")
            return

        # Root as SEQ for richer top-level composition (if budget allows)
        self.root_id = self._add_node("SEQ")
        # Build children under constraints
        # We will fill until num_nodes reached using recursive generation
        def make_subtree(depth: int) -> int:
            # If budget or depth exhausted, leaf
            if len(self.nodes) >= self.num_nodes or depth >= self.max_depth:
                return self._add_node("WORK")

            # Determine possible types considering budget
            # Ensure at least one child fits in budget for composite nodes
            budget_left = self.num_nodes - len(self.nodes)
            possible = []

            # LOOP needs 2 nodes minimum (loop + child)
            if budget_left >= 2:
                possible.append("LOOP")
            # IF needs 1 + branches; check minimal 1 + 2
            if budget_left >= 3:
                possible.append("IF")
            # SEQ needs 1 + children; minimal 1 + 2
            if budget_left >= 3:
                possible.append("SEQ")
            # WORK always possible
            possible.append("WORK")

            # Limit IF/SEQ usage by caps
            # bias selection towards LOOP to ensure loops are present
            weights = []
            for t in possible:
                if t == "LOOP":
                    weights.append(0.4)
                elif t == "IF":
                    # reduce if cap reached
                    used_if = sum(1 for n in self.nodes.values() if n["type"] == "IF")
                    weights.append(0.3 if used_if < self.num_if_nodes else 0.05)
                elif t == "SEQ":
                    used_seq = sum(1 for n in self.nodes.values() if n["type"] == "SEQ")
                    weights.append(0.3 if used_seq < self.num_seq_nodes else 0.05)
                else:
                    weights.append(0.2)
            # Normalize weights
            s = sum(weights)
            if s <= 0:
                chosen = "WORK"
            else:
                r = random.random() * s
                cum = 0.0
                chosen = possible[-1]
                for t, w in zip(possible, weights):
                    cum += w
                    if r <= cum:
                        chosen = t
                        break

            if chosen == "WORK":
                return self._add_node("WORK")
            elif chosen == "LOOP":
                power = random.randint(1, max(1, self.max_loop_power))
                nid = self._add_node("LOOP", power=power)
                child_id = make_subtree(depth + 1)
                self.nodes[nid]["children"].append(child_id)
                return nid
            elif chosen == "IF":
                nid = self._add_node("IF")
                max_b = max(2, self.max_branches_per_if)
                # determine branches within budget
                budget_left = self.num_nodes - len(self.nodes)
                # max children we can add is min(max_b, budget_left) but we need nodes for branches
                branches = min(max_b, max(2, min(max_b, budget_left)))
                # ensure at least 2 branches
                branches = max(2, branches)
                for _ in range(branches):
                    if len(self.nodes) >= self.num_nodes:
                        break
                    cid = make_subtree(depth + 1)
                    self.nodes[nid]["children"].append(cid)
                return nid
            elif chosen == "SEQ":
                nid = self._add_node("SEQ")
                # choose 2-3 children within budget
                budget_left = self.num_nodes - len(self.nodes)
                children_count = min(3, max(2, budget_left))
                for _ in range(children_count):
                    if len(self.nodes) >= self.num_nodes:
                        break
                    cid = make_subtree(depth + 1)
                    self.nodes[nid]["children"].append(cid)
                return nid
            else:
                return self._add_node("WORK")

        # Add some children under root
        # Ensure at least two children under root if budget allows
        child_target = 2 if self.num_nodes - len(self.nodes) >= 2 else 1
        for _ in range(child_target):
            cid = make_subtree(1)
            self.nodes[self.root_id]["children"].append(cid)

        # If root ends up with no children due to budget constraints, convert to WORK
        if len(self.nodes[self.root_id]["children"]) == 0:
            self.nodes[self.root_id]["type"] = "WORK"
            self.nodes[self.root_id]["power"] = None

    def _compute_exponent(self, nid: int) -> int:
        node = self.nodes[nid]
        typ = node["type"]
        if typ == "WORK":
            return 0
        elif typ == "LOOP":
            child = node["children"][0] if node["children"] else None
            body_exp = self._compute_exponent(child) if child is not None else 0
            return node["power"] + body_exp
        elif typ == "SEQ":
            if not node["children"]:
                return 0
            return max(self._compute_exponent(c) for c in node["children"])
        elif typ == "IF":
            if not node["children"]:
                return 0
            return max(self._compute_exponent(c) for c in node["children"])
        else:
            return 0


class CodeComplexityEnvWithFeedback(CodeComplexityEnv):
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
            error_detail["issue"] = "missing_or_wrong_boxed_format"
            hint = "Wrap your action like \\boxed{ACTION args}. Example: \\boxed{ROOT}."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "does not exist" in text:
                error_detail["violation"] = "nonexistent_node"
                hint = "Start with \\boxed{ROOT}, then \\boxed{INSPECT id} and \\boxed{CHILDREN id} to discover valid ids."
            elif "note requires" in text:
                error_detail["violation"] = "note_arguments"
                hint = "Use NOTE with an existing node id and integer k, e.g., \\boxed{NOTE 3 2}."
            elif "answer requires" in text:
                error_detail["violation"] = "answer_not_integer"
                hint = "Submit integer k, e.g., \\boxed{ANSWER 3}."
            else:
                error_detail["violation"] = "unspecified"
                hint = "Follow the action formats in the instructions."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["action"] = "unknown_command"
            hint = "Supported actions: ROOT, INSPECT i, CHILDREN i, STACK_PUSH i, STACK_POP, NOTE i k, READ_NOTE i, STATUS, HELP, ANSWER k."
        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Plan exploration steps, compute k, and use \\boxed{ANSWER k} before hitting the turn limit."
        elif "failed!" in text:
            error_type = "WrongDecision"
            error_detail["outcome"] = "incorrect_answer"
            hint = "Revisit LOOP nodes: k adds loop power plus body exponent; for SEQ/IF take the max of child exponents."
        elif "success!" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None
        else:
            error_type = "OK"
            error_detail["outcome"] = "ongoing"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["visited_count"] = len(getattr(self, "visited", []))
            diagnostic["stack_size"] = len(getattr(self, "stack", []))
            diagnostic["notes_count"] = len(getattr(self, "notes", {}))
            diagnostic["help_left"] = getattr(self, "remaining_help", 0)
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Begin with \\boxed{ROOT}, then \\boxed{INSPECT id} and \\boxed{CHILDREN id} to map the structure.",
            "turn": 0,
            "visited_count": 0,
            "stack_size": 0,
            "notes_count": 0,
            "help_left": getattr(self, "remaining_help", 0),
        }
        return obs, info