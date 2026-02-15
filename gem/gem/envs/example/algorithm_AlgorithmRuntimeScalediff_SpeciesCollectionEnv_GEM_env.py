from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class AlgorithmRuntimeScalediffEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = False,
        max_turns: Optional[int] = 100,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 100

        # Evolvable parameters with explanations:
        self.complexity_params = {
            "num_leaves": (5, 53),            # Target leaves; more leaves increase exploration/calculation difficulty
            "max_depth": (2, 6),              # Tree depth; deeper trees increase dependency complexity
            "p_parallel": (20, 75),           # Percent PAR internal nodes; more PAR increases max-based reasoning difficulty
            "max_branching": (2, 5),          # Max children per internal node; larger branching increases fan-out complexity
            "cost_max": (10, 215),            # Max leaf cost; larger range increases numeric magnitude without changing rules
            "start_reveals": (2, 0),          # REVERSED: automatic initial reveals; fewer freebies = harder
        }
        # Variances tuned per range size
        self.param_variance = {
            "num_leaves": 0,
            "max_depth": 0,
            "p_parallel": 0,
            "max_branching": 0,
            "cost_max": 0,
            "start_reveals": 0,
        }

        # Placeholders
        self.num_leaves: int = 0
        self.max_depth: int = 0
        self.p_parallel: int = 0
        self.max_branching: int = 0
        self.cost_max: int = 0
        self.start_reveals: int = 0

        # State
        self.turn_count: int = 0
        self.nodes: Dict[int, Dict[str, Any]] = {}
        self.root_id: int = 0
        self.root_runtime: int = 0
        self.known_types: Dict[int, str] = {}
        self.known_costs: Dict[int, int] = {}
        self.marked_leaves: set = set()
        self.visited_nodes: set = set()
        self.last_answer_attempt: Optional[int] = None

        self.reset()

    def _apply_complexity_params(self):
        table = {

            1: (5, 2, 20, 2, 10, 2),

            2: (9, 2, 30, 2, 40, 2),

            3: (14, 3, 40, 2, 80, 1),

            4: (20, 3, 50, 3, 120, 1),

            5: (27, 4, 55, 3, 150, 0),

            6: (34, 4, 60, 3, 180, 0),

            7: (41, 5, 65, 4, 200, 0),

            8: (48, 5, 68, 4, 200, 0),

            9: (50, 6, 70, 4, 200, 0),

            10: (53, 6, 75, 5, 215, 0),

        }

        level = int(self.complexity)
        params = table.get(level, table[max(table.keys())])
        (num_leaves, max_depth, p_parallel, max_branching, cost_max, start_reveals) = params

    def _get_instructions(self) -> str:
        return (
            "Algorithm Runtime Composition Game.\n"
            "A hidden algorithm is represented as a rooted tree: internal nodes are SEQ (sum of children runtimes) "
            "or PAR (max of children runtimes), and leaves are atomic tasks with integer costs.\n"
            "Goal: compute the root runtime exactly and submit it.\n"
            "Actions:\n"
            "- help: show instructions\n"
            "- task: show brief task reminder\n"
            "- inspect <id>: reveal node type (SEQ/PAR) or leaf cost\n"
            "- children <id>: list children of an internal node\n"
            "- mark <id>: mark a node; marking a leaf collects its cost\n"
            "- compute <id>: compute the subtree runtime if all descendant leaves are marked; otherwise reports progress\n"
            "- count: show progress counts\n"
            "- answer <integer>: submit your final runtime for the root\n"
            "Rules:\n"
            "- Use \\boxed{...} format for all actions\n"
            "- Refer only to valid node IDs shown by observations\n"
            "- Invalid or unsupported commands terminate with penalty\n"
            f"Example action: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        total_leaves = sum(1 for n in self.nodes.values() if n["type"] == "LEAF")
        marked = len(self.marked_leaves)
        visited = len(self.visited_nodes)
        return (
            f"Turn {self.turn_count}/{self.max_turns}. "
            f"Marked leaves: {marked}/{total_leaves}. Visited nodes: {visited}. "
            f"Root node ID: {self.root_id}. "
            "Enter your action in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        self.turn_count = 0
        self.nodes = {}
        self.root_id = 0
        self.known_types = {}
        self.known_costs = {}
        self.marked_leaves = set()
        self.visited_nodes = set()
        self.last_answer_attempt = None

        self._generate_tree()
        self.root_runtime = self._compute_runtime(self.root_id)

        # initial reveals
        self._initial_reveals()

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if not parsed:
            obs = "Invalid action format. Use \\boxed{...}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        cmd = parsed.get("cmd")
        reward = 0.0
        obs = ""

        if cmd in {"help", "task"}:
            obs = self._get_instructions()

        elif cmd == "inspect":
            node_id = parsed.get("id")
            if node_id is None or node_id not in self.nodes:
                obs = f"Protocol violation: node id {node_id} does not exist."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            node = self.nodes[node_id]
            self.visited_nodes.add(node_id)
            if node["type"] == "LEAF":
                cost = node["cost"]
                self.known_costs[node_id] = cost
                obs = f"Node {node_id} is LEAF with cost {cost}."
            else:
                typ = node["type"]
                self.known_types[node_id] = typ
                obs = f"Node {node_id} is {typ}."

        elif cmd == "children":
            node_id = parsed.get("id")
            if node_id is None or node_id not in self.nodes:
                obs = f"Protocol violation: node id {node_id} does not exist."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            node = self.nodes[node_id]
            self.visited_nodes.add(node_id)
            if node["type"] == "LEAF":
                obs = "Protocol violation: leaf has no children."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                children = node["children"]
                obs = f"Children of {node_id}: {', '.join(map(str, children))}."

        elif cmd == "mark":
            node_id = parsed.get("id")
            if node_id is None or node_id not in self.nodes:
                obs = f"Protocol violation: node id {node_id} does not exist."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            self.visited_nodes.add(node_id)
            node = self.nodes[node_id]
            if node["type"] == "LEAF":
                self.marked_leaves.add(node_id)
                obs = f"Marked leaf {node_id}."
            else:
                obs = f"Marked node {node_id}."

        elif cmd == "compute":
            node_id = parsed.get("id")
            if node_id is None or node_id not in self.nodes:
                obs = f"Protocol violation: node id {node_id} does not exist."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            self.visited_nodes.add(node_id)
            leaves = self._descendant_leaves(node_id)
            total = len(leaves)
            collected = sum(1 for l in leaves if l in self.marked_leaves)
            if collected == total:
                val = self._compute_runtime(node_id)
                obs = f"Subtree runtime of {node_id} is {val}."
            else:
                obs = f"Subtree incomplete: {collected}/{total} leaves marked."

        elif cmd == "count":
            total_leaves = sum(1 for n in self.nodes.values() if n["type"] == "LEAF")
            marked = len(self.marked_leaves)
            visited = len(self.visited_nodes)
            obs = f"Progress: marked_leaves={marked}/{total_leaves}, visited_nodes={visited}."

        elif cmd == "answer":
            val = parsed.get("value")
            if val is None or not isinstance(val, int):
                obs = "Unsupported action: answer must be an integer."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            self.last_answer_attempt = val
            if val == self.root_runtime:
                obs = f"Success! Correct runtime is {self.root_runtime}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = "Incorrect answer."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = f"Unsupported action: {cmd}."
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        m = pattern.findall(action)
        if not m:
            return None
        content = m[-1].strip()
        parts = content.split()
        if not parts:
            return None
        cmd = parts[0].lower()
        if cmd in {"help", "task", "count"}:
            return {"cmd": cmd}
        if cmd in {"inspect", "children", "mark", "compute"}:
            if len(parts) < 2:
                return None
            try:
                node_id = int(parts[1])
            except ValueError:
                return None
            return {"cmd": cmd, "id": node_id}
        if cmd == "answer":
            if len(parts) < 2:
                return {"cmd": cmd, "value": None}
            try:
                val = int(parts[1])
            except ValueError:
                val = None
            return {"cmd": cmd, "value": val}
        return {"cmd": cmd}

    def sample_random_action(self) -> str:
        return "\\boxed{inspect 0}"

    def _generate_tree(self):
        self.nodes = {}
        next_id = 0
        self.root_id = next_id
        next_id += 1
        root_type = self._rand_internal_type()
        self.nodes[self.root_id] = {"type": root_type, "children": [], "cost": None, "depth": 0}

        queue = [self.root_id]
        leaf_count = 0

        def create_child(parent_id, depth):
            nonlocal next_id, leaf_count
            child_id = next_id
            next_id += 1
            if depth < self.max_depth and random.random() < self._internal_prob(leaf_count):
                t = self._rand_internal_type()
                self.nodes[child_id] = {"type": t, "children": [], "cost": None, "depth": depth}
                queue.append(child_id)
            else:
                c = random.randint(1, self.cost_max)
                self.nodes[child_id] = {"type": "LEAF", "children": [], "cost": c, "depth": depth}
                leaf_count += 1
            self.nodes[parent_id]["children"].append(child_id)

        while queue and leaf_count < self.num_leaves:
            parent = queue.pop(0)
            parent_depth = self.nodes[parent]["depth"]
            k = random.randint(2, self.max_branching)
            for _ in range(k):
                create_child(parent, parent_depth + 1)

        # If not enough leaves, expand convertible leaves (depth < max_depth)
        convertible = [nid for nid, n in self.nodes.items() if n["type"] == "LEAF" and n["depth"] < self.max_depth]
        tries = 0
        while leaf_count < self.num_leaves and convertible and tries < 1000:
            tries += 1
            nid = random.choice(convertible)
            # convert to internal
            self.nodes[nid]["type"] = self._rand_internal_type()
            self.nodes[nid]["cost"] = None
            queue.append(nid)
            convertible.remove(nid)
            # expand this node
            parent = nid
            parent_depth = self.nodes[parent]["depth"]
            k = random.randint(2, self.max_branching)
            for _ in range(k):
                new_id = next_id
                next_id += 1
                depth = parent_depth + 1
                if depth < self.max_depth and random.random() < self._internal_prob(leaf_count):
                    t = self._rand_internal_type()
                    self.nodes[new_id] = {"type": t, "children": [], "cost": None, "depth": depth}
                    queue.append(new_id)
                else:
                    c = random.randint(1, self.cost_max)
                    self.nodes[new_id] = {"type": "LEAF", "children": [], "cost": c, "depth": depth}
                    leaf_count += 1
                self.nodes[parent]["children"].append(new_id)

        # Guarantee no leafless internal nodes
        for nid, node in list(self.nodes.items()):
            if node["type"] != "LEAF" and not node["children"]:
                # add at least two leaves
                for _ in range(2):
                    cid = next_id
                    next_id += 1
                    depth = node["depth"] + 1
                    c = random.randint(1, self.cost_max)
                    self.nodes[cid] = {"type": "LEAF", "children": [], "cost": c, "depth": depth}
                    node["children"].append(cid)
                    leaf_count += 1

    def _rand_internal_type(self) -> str:
        return "PAR" if random.randint(1, 100) <= self.p_parallel else "SEQ"

    def _internal_prob(self, current_leaves: int) -> float:
        target = max(1, self.num_leaves)
        deficit = max(0, target - current_leaves)
        base = 0.2 + 0.6 * (deficit / target)
        return max(0.05, min(0.85, base))

    def _compute_runtime(self, nid: int) -> int:
        node = self.nodes[nid]
        if node["type"] == "LEAF":
            return node["cost"]
        child_vals = [self._compute_runtime(c) for c in node["children"]]
        if node["type"] == "SEQ":
            return sum(child_vals)
        else:
            return max(child_vals)

    def _descendant_leaves(self, nid: int) -> list:
        leaves = []
        stack = [nid]
        while stack:
            cur = stack.pop()
            n = self.nodes[cur]
            if n["type"] == "LEAF":
                leaves.append(cur)
            else:
                stack.extend(n["children"])
        return leaves

    def _initial_reveals(self):
        # reveal root type
        root_type = self.nodes[self.root_id]["type"]
        self.known_types[self.root_id] = root_type
        self.visited_nodes.add(self.root_id)
        # optionally reveal one child or leaf cost
        reveals = self.start_reveals
        if reveals > 0:
            children = self.nodes[self.root_id]["children"]
            if children:
                cid = random.choice(children)
                cnode = self.nodes[cid]
                self.visited_nodes.add(cid)
                if cnode["type"] == "LEAF":
                    self.known_costs[cid] = cnode["cost"]
                else:
                    self.known_types[cid] = cnode["type"]


class AlgorithmRuntimeScalediffEnvWithFeedback(AlgorithmRuntimeScalediffEnv):
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
            hint = "Wrap the command in \\boxed{...}, e.g., \\boxed{inspect 0}."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "does not exist" in text:
                error_detail["violation"] = "invalid_node_id"
                hint = "Request children to discover valid IDs, then inspect/mark those."
            elif "leaf has no children" in text:
                error_detail["violation"] = "children_of_leaf"
                hint = "Use inspect or mark on leaves; use children only on SEQ/PAR nodes."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["action"] = "unknown_command"
            hint = "Valid actions: help, task, inspect <id>, children <id>, mark <id>, compute <id>, count, answer <int>."

        elif "incorrect answer" in text:
            error_type = "WrongDecision"
            error_detail["got"] = self.last_answer_attempt
            error_detail["expected"] = self.root_runtime
            hint = "Compute subtrees using compute <id> after marking all descendant leaves, then answer with the root runtime."

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Plan actions: enumerate children from the root, mark leaves systematically, compute bottom-up, then answer."

        elif "subtree incomplete" in text:
            error_type = "OK"
            # Extract k/m if present
            m = re.search(r"subtree incomplete:\s*(\d+)\s*/\s*(\d+)", text)
            if m:
                error_detail["marked"] = int(m.group(1))
                error_detail["total"] = int(m.group(2))
            hint = "Mark all leaves under the target node; use children to navigate and inspect to confirm leaf status."

        elif "success!" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "root_id": self.root_id,
                "marked_leaves": len(self.marked_leaves),
                "total_leaves": sum(1 for n in self.nodes.values() if n["type"] == "LEAF"),
                "visited_nodes": len(self.visited_nodes),
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
            "hint": "Start by \\boxed{children 0} to list root children, then \\boxed{inspect <id>} and \\boxed{mark <leaf_id>}.",
            "turn": 0,
            "state": {
                "root_id": self.root_id,
                "marked_leaves": len(self.marked_leaves),
                "total_leaves": sum(1 for n in self.nodes.values() if n["type"] == "LEAF"),
                "visited_nodes": len(self.visited_nodes),
            },
        }
        return obs, info
