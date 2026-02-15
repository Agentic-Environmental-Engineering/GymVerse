from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class AquaConfluxEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 20,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 20

        # Evolvable parameters
        self.complexity_params = {
            # Number of internal layers between source and sinks: deeper = harder
            'num_layers': (3, 6),
            # Max branching factor at a split: more branches = harder
            'branching_max': (2, 4),
            # Approximate pressure for merges: larger → smaller next layer size more often → more merges = harder
            'merge_bias': (0, 3),
            # Number of sinks at final layer: more sinks = more options and paths = harder
            'num_sinks': (2, 5),
            # Number of edges with leaks: more leaks = harder arithmetic and reasoning
            'num_leaks': (0, 8),
            # Maximum leak magnitude (units) on an edge: higher → harder
            'leak_max': (0, 50),
            # Base inflow at source: larger values yield larger arithmetic = slightly harder
            'base_flow': (200, 1000),
        }
        self.param_variance = {
            'num_layers': 0,
            'branching_max': 0,
            'merge_bias': 0,
            'num_sinks': 1,
            'num_leaks': 1,
            'leak_max': 5,
            'base_flow': 50,
        }

        # Placeholder attributes set by _apply_complexity_params
        self.num_layers: int = 0
        self.branching_max: int = 0
        self.merge_bias: int = 0
        self.num_sinks: int = 0
        self.num_leaks: int = 0
        self.leak_max: int = 0
        self.base_flow: int = 0

        # State
        self.turn_count: int = 0
        self.layers: List[List[str]] = []
        self.edges: List[Dict[str, Any]] = []
        self.adj_by_src: Dict[str, List[int]] = {}
        self.in_by_dst: Dict[str, List[int]] = {}
        self.node_inflow: Dict[str, int] = {}
        self.edge_delivered: List[int] = []
        self.sink_flows: Dict[str, int] = {}
        self.target_sink: Optional[str] = None
        self.instance_id: int = 0

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

    def _build_network(self):
        self.layers = []
        self.edges = []
        self.adj_by_src = {}
        self.in_by_dst = {}
        self.node_inflow = {}
        self.edge_delivered = []
        self.sink_flows = {}

        self.layers.append(['SRC'])
        prev_count = 1
        internal_layer_count = max(1, self.num_layers - 2)

        # Build internal layers
        for i in range(1, 1 + internal_layer_count):
            layer_nodes = []
            # size tends to be around prev_count * k, but with merge bias it can be smaller
            min_next = max(1, prev_count - self.merge_bias)
            max_next = max(prev_count, min(prev_count * self.branching_max, prev_count * (self.branching_max + 1)))
            next_count = random.randint(min_next, max(prev_count, min(max_next, prev_count * self.branching_max)))
            # Cap to avoid explosion
            next_count = max(1, min(next_count, 10))
            for j in range(next_count):
                layer_nodes.append(f"N{i}_{j+1}")
            self.layers.append(layer_nodes)
            prev_count = next_count

        # Build sink layer
        sinks = [f"T{k+1}" for k in range(self.num_sinks)]
        self.layers.append(sinks)

        # Create edges from each layer to next
        for li in range(len(self.layers) - 1):
            src_layer = self.layers[li]
            dst_layer = self.layers[li + 1]
            for src in src_layer:
                k_out = random.randint(1, min(self.branching_max, len(dst_layer)))
                dests = random.sample(dst_layer, k_out)
                # Temporary edges without percent assigned; we will assign percents per source
                for dst in dests:
                    self.edges.append({'src': src, 'dst': dst, 'percent': 0, 'leak': 0})

        # Assign split percentages per source
        self.adj_by_src = {}
        self.in_by_dst = {}
        for idx, e in enumerate(self.edges):
            src = e['src']
            dst = e['dst']
            self.adj_by_src.setdefault(src, []).append(idx)
            self.in_by_dst.setdefault(dst, []).append(idx)

        for src, idxs in self.adj_by_src.items():
            k = len(idxs)
            if k == 1:
                self.edges[idxs[0]]['percent'] = 100
            else:
                cuts = sorted(random.sample(range(1, 100), k - 1))
                parts = []
                last = 0
                for c in cuts:
                    parts.append(c - last)
                    last = c
                parts.append(100 - last)
                random.shuffle(parts)
                for i_part, ei in enumerate(idxs):
                    self.edges[ei]['percent'] = parts[i_part]

        # Assign leaks to a subset of edges
        candidates = list(range(len(self.edges)))
        random.shuffle(candidates)
        leaks_to_place = min(self.num_leaks, len(candidates))
        for i in range(leaks_to_place):
            ei = candidates[i]
            leak = 0
            if self.leak_max > 0:
                leak = random.randint(0, self.leak_max)
                # Snap to nearest multiple of 5 for neatness
                leak = int(round(leak / 5.0)) * 5
            self.edges[ei]['leak'] = max(0, leak)

    def _compute_flows(self):
        self.node_inflow = {node: 0 for layer in self.layers for node in layer}
        self.node_inflow['SRC'] = self.base_flow
        self.edge_delivered = [0 for _ in self.edges]

        # Process layers from 0 to last-1
        for li in range(len(self.layers) - 1):
            for node in self.layers[li]:
                inflow = self.node_inflow.get(node, 0)
                out_idxs = self.adj_by_src.get(node, [])
                if not out_idxs:
                    continue
                # Split inflow by integer rounding with remainder distribution
                percents = [self.edges[i]['percent'] for i in out_idxs]
                raws = [inflow * p / 100.0 for p in percents]
                floors = [int(raw // 1) for raw in raws]
                remainder = inflow - sum(floors)
                fracs = [(raw - int(raw // 1), i) for i, raw in enumerate(raws)]
                fracs.sort(reverse=True)
                for r in range(remainder):
                    floors[fracs[r % len(fracs)][1]] += 1
                # Apply leaks and propagate
                for k, ei in enumerate(out_idxs):
                    sent = floors[k]
                    leak = self.edges[ei]['leak']
                    delivered = max(0, sent - leak)
                    self.edge_delivered[ei] = delivered
                    dst = self.edges[ei]['dst']
                    self.node_inflow[dst] = self.node_inflow.get(dst, 0) + delivered

        self.sink_flows = {sink: self.node_inflow.get(sink, 0) for sink in self.layers[-1]}

    def _pick_target_sink(self):
        sinks = self.layers[-1]
        self.target_sink = random.choice(sinks) if sinks else None

    def _network_summary_text(self) -> str:
        lines = []
        lines.append(f"- Base flow at source: {self.base_flow} units")
        lines.append("- Nodes by layer:")
        for i, layer in enumerate(self.layers):
            kind = "Source" if i == 0 else ("Sinks" if i == len(self.layers) - 1 else f"Layer {i}")
            lines.append(f"  {kind}: {', '.join(layer)}")
        lines.append("- Edges (percent split and leak per edge):")
        for src, idxs in self.adj_by_src.items():
            parts = []
            for ei in idxs:
                e = self.edges[ei]
                parts.append(f"{src} -> {e['dst']} [{e['percent']}% - leak {e['leak']}]")
            lines.append("  " + "; ".join(parts))
        lines.append(f"- Target sink: {self.target_sink}")
        return "\n".join(lines)

    def _count_paths(self, target: str) -> int:
        # Dynamic programming count of paths from SRC to each node
        order = []
        for layer in self.layers:
            order.extend(layer)
        count = {node: 0 for node in order}
        count['SRC'] = 1
        for li in range(len(self.layers) - 1):
            for node in self.layers[li]:
                for ei in self.adj_by_src.get(node, []):
                    dst = self.edges[ei]['dst']
                    count[dst] = count.get(dst, 0) + count.get(node, 0)
        return count.get(target, 0)

    def _sample_path_to(self, target: str) -> List[str]:
        # Greedy backtrack via predecessor with largest delivered flow
        path = [target]
        current = target
        for li in range(len(self.layers) - 1, 0, -1):
            preds = self.in_by_dst.get(current, [])
            if not preds:
                break
            best = None
            best_val = -1
            best_src = None
            for ei in preds:
                delivered = self.edge_delivered[ei]
                src = self.edges[ei]['src']
                if delivered > best_val:
                    best_val = delivered
                    best = ei
                    best_src = src
            if best_src is None:
                break
            path.append(best_src)
            current = best_src
            if current == 'SRC':
                break
        path.reverse()
        return path

    def _get_instructions(self) -> str:
        return (
            "Aqua Conflux Puzzle:\n"
            "Water enters at source (SRC) with the given base flow (units). At each junction, outgoing edges split the\n"
            "incoming flow according to their specified percentages (summing to 100% for that junction). Each edge may\n"
            "have a leak that subtracts a fixed number of units from the flow on that edge. Merges simply sum incoming\n"
            "flows. Flows cannot go below zero after a leak (they clamp to 0). Your task is to compute the final flow\n"
            f"reaching the designated target sink as an integer number of units.\n\n"
            "Available actions (use \\boxed{...}):\n"
            "- list_nodes                        → list all nodes by type\n"
            "- show node=NODEID                  → show details of a specific node\n"
            "- trace sink=SINKID                 → report path count and a sample path to the sink\n"
            "- help                              → reprint the commands\n"
            "- answer value=INTEGER              → submit your computed final flow for the target sink\n\n"
            "Formatting: Wrap your command in \\boxed{...}. Examples:\n"
            f"{self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        return (
            "Current network:\n"
            f"{self._network_summary_text()}\n\n"
            "Enter your action in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()
        self.turn_count = 0
        self.instance_id += 1
        # Generate solvable instance (ensure some flow reaches sinks)
        attempts = 0
        while True:
            attempts += 1
            self._build_network()
            self._compute_flows()
            self._pick_target_sink()
            if self.target_sink is None:
                continue
            # Ensure at least one sink has positive flow to avoid degenerate puzzle
            if max(self.sink_flows.values()) > 0:
                break
            if attempts > 10:
                break
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if not parsed:
            obs = "INVALID ACTION FORMAT: Use \\boxed{...} with a supported action."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        act = parsed.get('action', '').lower()

        if act == 'help':
            obs = self._get_instructions()
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if act == 'list_nodes':
            lines = []
            types = {}
            # Determine types
            indeg = {node: len(self.in_by_dst.get(node, [])) for layer in self.layers for node in layer}
            outdeg = {node: len(self.adj_by_src.get(node, [])) for layer in self.layers for node in layer}
            for layer in self.layers:
                for node in layer:
                    if node == 'SRC':
                        typ = 'source'
                    elif node in self.layers[-1]:
                        typ = 'sink'
                    elif outdeg.get(node, 0) > 1:
                        typ = 'split'
                    elif indeg.get(node, 0) > 1 and outdeg.get(node, 0) == 1:
                        typ = 'merge'
                    else:
                        typ = 'pass'
                    types[node] = typ
            lines.append("NODES:")
            for node, typ in types.items():
                lines.append(f"- {node}: {typ}")
            obs = "\n".join(lines)
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if act == 'show':
            node = parsed.get('node', None)
            if not node:
                obs = "PROTOCOL VIOLATION: 'show' requires parameter node=NODEID."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            all_nodes = {n for layer in self.layers for n in layer}
            all_nodes.add('SRC')
            if node not in all_nodes:
                obs = f"PROTOCOL VIOLATION: Unknown node '{node}'."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            indeg = len(self.in_by_dst.get(node, []))
            out_idxs = self.adj_by_src.get(node, [])
            outdeg = len(out_idxs)
            if node == 'SRC':
                header = "SRC (source)"
            elif node in self.layers[-1]:
                header = f"{node} (sink)"
            elif outdeg > 1:
                header = f"{node} (split junction)"
            elif indeg > 1 and outdeg == 1:
                header = f"{node} (merge)"
            else:
                header = f"{node} (pass-through)"
            lines = [header]
            if outdeg > 0:
                lines.append("Outgoing edges:")
                for ei in out_idxs:
                    e = self.edges[ei]
                    lines.append(f"- {node} -> {e['dst']} [{e['percent']}% - leak {e['leak']}]")
            else:
                lines.append("No outgoing edges.")
            obs = "\n".join(lines)
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if act == 'trace':
            sink = parsed.get('sink', None)
            if not sink:
                obs = "PROTOCOL VIOLATION: 'trace' requires parameter sink=SINKID."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            if sink not in self.layers[-1]:
                obs = f"PROTOCOL VIOLATION: Unknown sink '{sink}'."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            count = self._count_paths(sink)
            sample_path = self._sample_path_to(sink)
            path_str = " -> ".join(sample_path) if sample_path else "(none)"
            obs = f"TRACE: paths_to_{sink}={count}; sample_path: {path_str}"
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if act == 'answer':
            val_str = parsed.get('value', None)
            if val_str is None:
                obs = "PROTOCOL VIOLATION: 'answer' requires parameter value=INTEGER."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            try:
                guess = int(val_str.strip())
            except Exception:
                obs = "PROTOCOL VIOLATION: 'value' must be an integer."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            correct = self.sink_flows.get(self.target_sink, 0)
            if guess == correct:
                obs = f"Success! Correct flow to {self.target_sink} is {correct} units."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Incorrect. Your answer {guess} is not equal to the correct flow {correct}."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        # Unsupported action
        obs = f"UNSUPPORTED ACTION: '{act}' is not recognized."
        return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if action is None:
            return None
        m = re.search(r"\\boxed\{(.+?)\}\s*$", action.strip(), flags=re.DOTALL)
        if not m:
            return None
        inner = m.group(1).strip()
        if not inner:
            return None
        parts = inner.split()
        tokens: Dict[str, Any] = {}
        tokens['action'] = parts[0]
        for part in parts[1:]:
            if '=' in part:
                key, value = part.split('=', 1)
                tokens[key] = value
        return tokens

    def sample_random_action(self) -> str:
        choices = []
        # Use existing nodes to produce plausible examples
        node_choices = [n for layer in self.layers for n in layer]
        sink_choices = list(self.layers[-1]) if self.layers else []
        if node_choices:
            choices.append(rf"\boxed{{show node={random.choice(node_choices)}}}")
        if sink_choices:
            choices.append(rf"\boxed{{trace sink={random.choice(sink_choices)}}}")
        choices.append(r"\boxed{list_nodes}")
        # Provide an example answer with a random integer near expected
        target = self.target_sink or "T1"
        correct = self.sink_flows.get(target, 0)
        guess = max(0, correct + random.randint(-20, 20))
        choices.append(rf"\boxed{{answer value={guess}}}")
        return random.choice(choices)


class AquaConfluxEnvWithFeedback(AquaConfluxEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        text = obs.lower()
        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint: Optional[str] = None

        if "invalid action format" in text:
            error_type = "FormatError"
            error_detail["issue"] = "missing_boxed_format"
            hint = "Wrap your command inside \\boxed{...}, e.g., \\boxed{list_nodes}"

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["action"] = action
            hint = "Use supported actions: list_nodes, show node=..., trace sink=..., help, answer value=INTEGER"

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "requires parameter" in text:
                error_detail["violation"] = "missing_parameter"
                hint = "Provide required parameters, e.g., \\boxed{show node=N2_1} or \\boxed{trace sink=T1}"
            elif "unknown node" in text or "unknown sink" in text:
                error_detail["violation"] = "unknown_identifier"
                hint = "List nodes with \\boxed{list_nodes} and check sink IDs in the network summary"
            elif "'value' must be an integer" in text:
                error_detail["violation"] = "non_integer_answer"
                hint = "Submit integers only, like \\boxed{answer value=120}"
            else:
                error_detail["violation"] = "generic_protocol_error"
                hint = "Review commands with \\boxed{help}"

        elif "incorrect" in text:
            error_type = "WrongDecision"
            # Extract expected and got
            m_correct = re.search(r"correct flow (\d+)", obs, flags=re.IGNORECASE)
            m_got = re.search(r"your answer\s+(-?\d+)", obs, flags=re.IGNORECASE)
            if m_correct:
                error_detail["expected"] = int(m_correct.group(1))
            if m_got:
                error_detail["got"] = int(m_got.group(1))
            error_detail["target_sink"] = self.target_sink
            hint = "Double-check split percentages and subtract leaks after splitting; merges add incoming delivered flows"

        elif "reached max turns" in text or "timeout" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Plan your exploration: list nodes, inspect relevant junctions, trace the target sink, then answer"

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["target_sink"] = getattr(self, "target_sink", None)
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start", "target_sink": self.target_sink},
            "hint": "Start with \\boxed{list_nodes} and \\boxed{show node=...}, then \\boxed{trace sink=...} before answering",
            "turn": 0,
        }
        return obs, info