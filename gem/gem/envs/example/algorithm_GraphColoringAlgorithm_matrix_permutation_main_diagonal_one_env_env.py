from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class GraphColoringAlgorithmEnv(Env):
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

        self.complexity_params = {
            # Number of nodes: more nodes increase search space and constraints
            'num_nodes': (6, 30),
            # Max degree cap: higher cap increases adjacency constraints (harder)
            'max_degree_cap': (2, 8),
            # REVERSED: allowed colors: fewer colors increase difficulty
            'allowed_colors': (5, 3),
            # REVERSED: precolored count: fewer helpful precolors increase difficulty
            'precolored_count': (2, 0),
            # REVERSED: recolor budget: fewer allowed repairs increases difficulty
            'recolor_budget': (50, 5),
        }

        self.param_variance = {
            'num_nodes': 3,
            'max_degree_cap': 1,
            'allowed_colors': 0,
            'precolored_count': 0,
            'recolor_budget': 5,
        }

        self.num_nodes: int = 0
        self.max_degree_cap: int = 0
        self.allowed_colors: int = 0
        self.precolored_count: int = 0
        self.recolor_budget: int = 0

        self.turn_count: int = 0
        self.adj: Dict[int, List[int]] = {}
        self.colors: Dict[int, Optional[int]] = {}
        self.remaining_recolor: int = 0

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
            low = min(min_val, max_val)
            high = max(min_val, max_val)
            actual_value = max(low, min(high, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _generate_graph(self):
        self.adj = {i: [] for i in range(self.num_nodes)}
        degrees = [0] * self.num_nodes
        max_deg_allowed = min(self.max_degree_cap, self.allowed_colors - 1)
        if max_deg_allowed < 1:
            max_deg_allowed = 1
        max_possible_edges = int(self.num_nodes * max_deg_allowed / 2)
        target_edges = max(1, int(0.6 * max_possible_edges))
        attempts = 0
        edges_added = 0
        added = set()
        while edges_added < target_edges and attempts < target_edges * 10:
            i = random.randint(0, self.num_nodes - 1)
            j = random.randint(0, self.num_nodes - 1)
            if i == j:
                attempts += 1
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in added:
                attempts += 1
                continue
            if degrees[a] >= max_deg_allowed or degrees[b] >= max_deg_allowed:
                attempts += 1
                continue
            added.add((a, b))
            self.adj[a].append(b)
            self.adj[b].append(a)
            degrees[a] += 1
            degrees[b] += 1
            edges_added += 1
            attempts += 1

    def _precolor(self):
        self.colors = {i: None for i in range(self.num_nodes)}
        nodes = list(range(self.num_nodes))
        random.shuffle(nodes)
        count = 0
        for i in nodes:
            if count >= self.precolored_count:
                break
            available = set(range(1, self.allowed_colors + 1))
            for nb in self.adj[i]:
                c = self.colors.get(nb)
                if c is not None and c in available:
                    available.discard(c)
            if available:
                chosen = random.choice(list(available))
                self.colors[i] = chosen
                count += 1

    def _count_conflicts(self) -> int:
        conflicts = 0
        seen = set()
        for u in range(self.num_nodes):
            cu = self.colors[u]
            if cu is None:
                continue
            for v in self.adj[u]:
                if v < u:
                    continue
                cv = self.colors[v]
                if cv is None:
                    continue
                if cu == cv and (u, v) not in seen:
                    conflicts += 1
                    seen.add((u, v))
        return conflicts

    def _is_proper_complete(self) -> bool:
        if any(self.colors[i] is None for i in range(self.num_nodes)):
            return False
        return self._count_conflicts() == 0

    def _get_instructions(self) -> str:
        return (
            "Algorithmic Graph Coloring Challenge.\n"
            "Goal: produce a proper coloring of the given graph using colors in [1..C].\n"
            "A proper coloring assigns a color to every node such that no adjacent nodes share the same color.\n"
            "You may use these actions:\n"
            "- COLOR i c: assign color c to node i (if uncolored)\n"
            "- RECOLOR i c: change color of node i (requires recolor budget)\n"
            "- UNCOLOR i: remove color from node i (requires recolor budget)\n"
            "- FINISH: submit your coloring for evaluation\n"
            "Formatting: wrap your action in \\boxed{...}. Example:\n"
            f"{self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        lines = []
        lines.append(f"Nodes: {self.num_nodes}, Allowed colors C: {self.allowed_colors}")
        lines.append(f"Recolor budget remaining: {self.remaining_recolor}")
        lines.append("Adjacency:")
        for i in range(self.num_nodes):
            nbrs = ",".join(str(x) for x in sorted(self.adj[i]))
            lines.append(f"  {i}: [{nbrs}]")
        colored = []
        for i in range(self.num_nodes):
            c = self.colors[i]
            colored.append(f"{i}:{c if c is not None else '-'}")
        lines.append("Current coloring: " + " ".join(colored))
        lines.append("Use \\boxed{ACTION args} with one of: COLOR i c, RECOLOR i c, UNCOLOR i, FINISH")
        return "\n".join(lines)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()
        if self.allowed_colors <= self.max_degree_cap:
            self.max_degree_cap = max(1, self.allowed_colors - 1)
        self._generate_graph()
        self._precolor()
        self.turn_count = 0
        self.remaining_recolor = self.recolor_budget
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)
        terminated = False
        truncated = False

        if parsed is None:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}"
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        act_type = parsed.get("type")
        obs = ""
        reward = 0.0

        if act_type == "COLOR":
            i = parsed.get("node")
            c = parsed.get("color")
            if i is None or c is None:
                obs = "Protocol violation: missing_arguments for COLOR"
            elif not (0 <= i < self.num_nodes):
                obs = f"Protocol violation: invalid_node {i}"
            elif not (1 <= c <= self.allowed_colors):
                obs = f"Protocol violation: invalid_color {c}"
            elif self.colors[i] is not None:
                obs = f"Protocol violation: node_already_colored {i}"
            else:
                self.colors[i] = c
                conflicts = self._count_conflicts()
                remaining = sum(1 for k in range(self.num_nodes) if self.colors[k] is None)
                obs = f"Colored node {i} with color {c}. Conflicts: {conflicts}. Uncolored left: {remaining}. Recolor left: {self.remaining_recolor}"

        elif act_type == "RECOLOR":
            i = parsed.get("node")
            c = parsed.get("color")
            if i is None or c is None:
                obs = "Protocol violation: missing_arguments for RECOLOR"
            elif not (0 <= i < self.num_nodes):
                obs = f"Protocol violation: invalid_node {i}"
            elif not (1 <= c <= self.allowed_colors):
                obs = f"Protocol violation: invalid_color {c}"
            elif self.colors[i] is None:
                obs = f"Protocol violation: node_not_colored {i}"
            elif self.remaining_recolor <= 0:
                obs = "Protocol violation: recolor_budget_exceeded"
            else:
                self.colors[i] = c
                self.remaining_recolor -= 1
                conflicts = self._count_conflicts()
                remaining = sum(1 for k in range(self.num_nodes) if self.colors[k] is None)
                obs = f"Recolored node {i} to color {c}. Conflicts: {conflicts}. Uncolored left: {remaining}. Recolor left: {self.remaining_recolor}"

        elif act_type == "UNCOLOR":
            i = parsed.get("node")
            if i is None:
                obs = "Protocol violation: missing_arguments for UNCOLOR"
            elif not (0 <= i < self.num_nodes):
                obs = f"Protocol violation: invalid_node {i}"
            elif self.colors[i] is None:
                obs = f"Protocol violation: node_not_colored {i}"
            elif self.remaining_recolor <= 0:
                obs = "Protocol violation: recolor_budget_exceeded"
            else:
                self.colors[i] = None
                self.remaining_recolor -= 1
                conflicts = self._count_conflicts()
                remaining = sum(1 for k in range(self.num_nodes) if self.colors[k] is None)
                obs = f"Uncolored node {i}. Conflicts: {conflicts}. Uncolored left: {remaining}. Recolor left: {self.remaining_recolor}"

        elif act_type == "FINISH":
            if self._is_proper_complete():
                obs = "Success! Proper complete coloring achieved."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                conflicts = self._count_conflicts()
                remaining = sum(1 for k in range(self.num_nodes) if self.colors[k] is None)
                obs = f"Failed! Coloring invalid. Conflicts: {conflicts}. Uncolored left: {remaining}."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = f"Unsupported action: {act_type}. Use COLOR i c, RECOLOR i c, UNCOLOR i, FINISH."

        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns}). Episode ended."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action:
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        tokens = content.split()
        if not tokens:
            return None
        cmd = tokens[0].upper()
        try:
            if cmd == "COLOR" and len(tokens) == 3:
                i = int(tokens[1])
                c = int(tokens[2])
                return {"type": "COLOR", "node": i, "color": c}
            elif cmd == "RECOLOR" and len(tokens) == 3:
                i = int(tokens[1])
                c = int(tokens[2])
                return {"type": "RECOLOR", "node": i, "color": c}
            elif cmd == "UNCOLOR" and len(tokens) == 2:
                i = int(tokens[1])
                return {"type": "UNCOLOR", "node": i}
            elif cmd == "FINISH" and len(tokens) == 1:
                return {"type": "FINISH"}
            else:
                return {"type": cmd}
        except ValueError:
            return None

    def sample_random_action(self) -> str:
        uncolored = [i for i in range(self.num_nodes) if self.colors.get(i) is None]
        if uncolored:
            i = random.choice(uncolored)
            c = random.randint(1, self.allowed_colors)
            return f"\\boxed{{COLOR {i} {c}}}"
        else:
            return "\\boxed{FINISH}"


class GraphColoringAlgorithmEnvWithFeedback(GraphColoringAlgorithmEnv):
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
            hint = f"Wrap actions like {self.sample_random_action()}"

        elif "protocol violation:" in text:
            error_type = "ProtocolViolation"
            detail_msg = obs.split("Protocol violation:")[-1].strip()
            error_detail["violation"] = detail_msg
            if "invalid_node" in detail_msg:
                hint = "Use a valid node id in [0..N-1] shown in the suffix."
            elif "invalid_color" in detail_msg:
                hint = "Choose a color in the range [1..C] given in the suffix."
            elif "node_already_colored" in detail_msg:
                hint = "Use RECOLOR i c or UNCOLOR i to change a colored node."
            elif "node_not_colored" in detail_msg:
                hint = "Color the node first or choose a different node."
            elif "recolor_budget_exceeded" in detail_msg:
                hint = "Avoid further RECOLOR/UNCOLOR; resolve conflicts with careful COLOR choices or FINISH."
            elif "missing_arguments" in detail_msg:
                hint = "Provide all required arguments: COLOR i c, RECOLOR i c, UNCOLOR i."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["allowed"] = ["COLOR i c", "RECOLOR i c", "UNCOLOR i", "FINISH"]
            hint = "Use one of the supported actions exactly as listed."

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["max_turns"] = self.max_turns
            hint = "Act decisively and issue FINISH once you have a proper coloring."

        elif "failed!" in text:
            error_type = "WrongDecision"
            conflicts = self._count_conflicts()
            remaining = sum(1 for k in range(self.num_nodes) if self.colors[k] is None)
            error_detail["expected"] = "All nodes colored with no adjacent nodes sharing a color."
            error_detail["conflicts"] = conflicts
            error_detail["uncolored"] = remaining
            hint = "Resolve conflicts by RECOLOR/UNCOLOR and fill all uncolored nodes before FINISH."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "num_nodes": self.num_nodes,
                "allowed_colors": self.allowed_colors,
                "remaining_recolor": self.remaining_recolor,
                "conflicts": self._count_conflicts(),
                "uncolored": sum(1 for k in range(self.num_nodes) if self.colors[k] is None),
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
            "hint": "Start by coloring low-degree nodes with colors unused by their neighbors.",
            "turn": 0,
            "state": {
                "num_nodes": self.num_nodes,
                "allowed_colors": self.allowed_colors,
                "remaining_recolor": self.remaining_recolor,
                "conflicts": self._count_conflicts(),
                "uncolored": sum(1 for k in range(self.num_nodes) if self.colors[k] is None),
            },
        }
        return obs, info