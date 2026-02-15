from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class GraphCycleStreamingEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = None,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, complexity))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns  # can be None; we compute a turn_limit per episode

        # Evolvable parameters with ranges and difficulty rationale:
        # - num_nodes: larger graph increases state size and operations required → harder
        # - edge_multiplier_pct: extra edges beyond a spanning forest, proportional to n; more edges → denser → harder
        # - buffer_size (REVERSED): smaller buffer forces stricter streaming constraints → harder
        # - max_reveal_k (REVERSED): smaller reveal batch size → more steps and less lookahead → harder
        self.complexity_params = {
            'num_nodes': (6, 60),              # more nodes = harder
            'edge_multiplier_pct': (0, 200),   # more edges beyond n-1 = harder
            'buffer_size': (12, 3),            # REVERSED: less buffer = harder
            'max_reveal_k': (6, 1),            # REVERSED: smaller batch reveal = harder
        }

        # Variance settings: integers with medium-to-large ranges get small ± variance
        self.param_variance = {
            'num_nodes': 3,            # ~5-10% variance
            'edge_multiplier_pct': 10, # ~5-10% variance
            'buffer_size': 1,          # small discrete range
            'max_reveal_k': 1,         # small discrete range
        }

        # Placeholder attributes
        self.num_nodes: int = 0
        self.edge_multiplier_pct: int = 0
        self.buffer_size: int = 0
        self.max_reveal_k: int = 0

        # Episode state
        self.turn_count: int = 0
        self.turn_limit: int = 0
        self.edges_stream: List[Tuple[int, int]] = []
        self.stream_pos: int = 0
        self.buffer: List[Tuple[int, int]] = []
        self.parent: List[int] = []
        self.rank: List[int] = []
        self.processed_count: int = 0
        self.cycle_detected_so_far: bool = False
        self.ground_truth_is_forest: bool = False
        self.total_edges: int = 0

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for name, (min_val, max_val) in self.complexity_params.items():
            center_value = min_val + (max_val - min_val) * normalized
            if self.enable_param_randomization:
                var = self.param_variance.get(name, 0)
                if var > 0:
                    v = random.uniform(-var, var)
                    val = center_value + v
                else:
                    val = center_value
            else:
                val = center_value
            lo, hi = (min_val, max_val)
            if lo > hi:
                lo, hi = hi, lo
            val = max(lo, min(hi, val))
            setattr(self, name, int(round(val)))

    def _uf_init(self, n: int):
        self.parent = list(range(n + 1))
        self.rank = [0] * (n + 1)

    def _uf_find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def _uf_union(self, a: int, b: int) -> bool:
        ra, rb = self._uf_find(a), self._uf_find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[rb] < self.rank[ra]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1
        return True

    def _gen_graph(self):
        n = self.num_nodes
        max_possible = n * (n - 1) // 2
        extra = int((self.edge_multiplier_pct * n) // 100)
        target_m = n - 1 + extra
        target_m = max(0, min(max_possible, target_m))
        all_pairs = []
        for u in range(1, n + 1):
            for v in range(u + 1, n + 1):
                all_pairs.append((u, v))
        random.shuffle(all_pairs)
        chosen = all_pairs[:target_m]
        self.edges_stream = chosen
        self.total_edges = len(chosen)
        self.stream_pos = 0

        # Ground truth check for forest via a fresh DSU
        parent = list(range(n + 1))
        rank = [0] * (n + 1)

        def ffind(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def funion(a: int, b: int) -> bool:
            ra, rb = ffind(a), ffind(b)
            if ra == rb:
                return False
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[rb] < rank[ra]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1
            return True

        cycle = False
        for (u, v) in chosen:
            if not funion(u, v):
                cycle = True
                break
        self.ground_truth_is_forest = not cycle

    def _get_instructions(self) -> str:
        return (
            "You are playing GraphCycleStreaming.\n"
            "Goal: Decide whether the entire undirected graph (defined by a hidden edge stream) is a forest (acyclic).\n"
            "You can reveal edges into a bounded buffer, process edges using union-find semantics, reorder the buffer, query counts, and then finalize with a yes/no.\n"
            "Actions (use \\boxed{...}):\n"
            "- reveal k: Reveal up to k edges from the stream into the buffer (k ≤ max_reveal_k and buffer has capacity).\n"
            "- process u v: Process edge (u,v) if it exists in the buffer. If it connects nodes already connected, a cycle is detected.\n"
            "- shuffle asc|desc: Reorder the buffered edges lexicographically.\n"
            "- query: Get aggregates (components, buffer size, remaining edges).\n"
            "- finalize yes|no: Submit your final answer: yes if the graph is a forest; no otherwise.\n"
            "Rules:\n"
            "- Vertices are 1..N. Edges are undirected and listed with u<v.\n"
            "- Invalid formats or protocol violations end the episode with a penalty.\n"
            f"Example action: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        remaining = self.total_edges - self.stream_pos
        comps = len({self._uf_find(i) for i in range(1, self.num_nodes + 1)})
        buf_str = ", ".join([f"{u}-{v}" for (u, v) in self.buffer]) if self.buffer else "(empty)"
        return (
            f"State:\n"
            f"- nodes: {self.num_nodes}\n"
            f"- processed_edges: {self.processed_count}/{self.total_edges}\n"
            f"- remaining_stream: {remaining}\n"
            f"- buffer_capacity: {len(self.buffer)}/{self.buffer_size}\n"
            f"- buffer: {buf_str}\n"
            f"- components_now: {comps}\n"
            f"- cycle_detected_so_far: {'yes' if self.cycle_detected_so_far else 'no'}\n"
            f"Enter your action using \\boxed{{...}} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()

        self.turn_count = 0
        self.buffer = []
        self.processed_count = 0
        self.cycle_detected_so_far = False

        # Build the graph instance
        self._gen_graph()
        self._uf_init(self.num_nodes)

        # Determine the per-episode turn limit
        if self.max_turns is not None:
            self.turn_limit = int(self.max_turns)
        else:
            # Allow multiple actions per edge plus reordering/query overhead
            self.turn_limit = 3 * max(1, self.total_edges) + 20

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if parsed is None:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        kind = parsed.get("type")

        # Handle actions
        if kind == "reveal":
            k = parsed.get("k", 0)
            if k < 1 or k > self.max_reveal_k:
                obs = f"Protocol violation: reveal k must be 1..{self.max_reveal_k}."
                return obs, -0.25, True, False, {"suffix": self.get_task_suffix()}
            capacity = self.buffer_size - len(self.buffer)
            if capacity <= 0:
                obs = "Protocol violation: buffer is full; process edges before revealing more."
                return obs, -0.25, True, False, {"suffix": self.get_task_suffix()}
            available = self.total_edges - self.stream_pos
            to_add = min(k, capacity, available)
            if to_add <= 0:
                obs = "Reveal: no edges available to reveal."
                # Non-terminal no-op allowed
                if self.turn_count >= self.turn_limit:
                    return "Reached max turns (timeout).", 0.0, True, True, {"suffix": self.get_task_suffix()}
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            new_edges = self.edges_stream[self.stream_pos:self.stream_pos + to_add]
            self.stream_pos += to_add
            self.buffer.extend(new_edges)
            obs = f"Revealed {to_add} edge(s). Buffer now holds {len(self.buffer)}."
            if self.turn_count >= self.turn_limit:
                return "Reached max turns (timeout).", 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        elif kind == "process":
            u, v = parsed.get("u"), parsed.get("v")
            if not isinstance(u, int) or not isinstance(v, int):
                obs = "Protocol violation: process requires integer endpoints."
                return obs, -0.25, True, False, {"suffix": self.get_task_suffix()}
            if not (1 <= u <= self.num_nodes and 1 <= v <= self.num_nodes and u != v):
                obs = "Protocol violation: endpoints must be distinct in [1..N]."
                return obs, -0.25, True, False, {"suffix": self.get_task_suffix()}
            a, b = (u, v) if u < v else (v, u)
            if (a, b) not in self.buffer:
                obs = "Protocol violation: edge not in buffer."
                return obs, -0.25, True, False, {"suffix": self.get_task_suffix()}
            # Remove from buffer, apply union-find
            self.buffer.remove((a, b))
            merged = self._uf_union(a, b)
            if not merged:
                self.cycle_detected_so_far = True
            self.processed_count += 1
            status = "cycle" if not merged else "merged"
            obs = f"Processed edge {a}-{b}: {status}."
            if self.turn_count >= self.turn_limit:
                return "Reached max turns (timeout).", 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        elif kind == "shuffle":
            order = parsed.get("order")
            if order not in ("asc", "desc"):
                obs = "Unsupported action: shuffle requires 'asc' or 'desc'."
                return obs, -0.25, True, False, {"suffix": self.get_task_suffix()}
            reverse = (order == "desc")
            self.buffer.sort(key=lambda e: (e[0], e[1]), reverse=reverse)
            obs = f"Buffer shuffled {order}."
            if self.turn_count >= self.turn_limit:
                return "Reached max turns (timeout).", 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        elif kind == "query":
            comps = len({self._uf_find(i) for i in range(1, self.num_nodes + 1)})
            remaining = self.total_edges - self.stream_pos
            obs = (
                f"Query: components={comps}, buffer={len(self.buffer)}, remaining_stream={remaining}, "
                f"cycle_detected_so_far={'yes' if self.cycle_detected_so_far else 'no'}."
            )
            if self.turn_count >= self.turn_limit:
                return "Reached max turns (timeout).", 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        elif kind == "finalize":
            ans = parsed.get("ans")
            if ans not in ("yes", "no"):
                obs = "Unsupported action: finalize requires 'yes' or 'no'."
                return obs, -0.25, True, False, {"suffix": self.get_task_suffix()}
            guess_is_forest = (ans == "yes")
            correct = (guess_is_forest == self.ground_truth_is_forest)
            if correct:
                detail = "a forest" if self.ground_truth_is_forest else "not a forest"
                obs = f"Success! Correct final answer: the graph is {detail}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                detail = "a forest" if self.ground_truth_is_forest else "not a forest"
                obs = f"Incorrect final answer. The graph is {detail}."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = f"Unsupported action type: {kind}."
            return obs, -0.25, True, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action:
            return None
        m = re.findall(r'\\boxed\{(.+?)\}', action, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            return None
        content = m[-1].strip()
        content = re.sub(r'\s+', ' ', content).strip().lower()
        # Grammar
        # reveal k
        mr = re.match(r'^reveal\s+(\d+)$', content)
        if mr:
            return {"type": "reveal", "k": int(mr.group(1))}
        # process u v
        mp = re.match(r'^process\s+(\d+)\s+(\d+)$', content)
        if mp:
            return {"type": "process", "u": int(mp.group(1)), "v": int(mp.group(2))}
        # shuffle asc|desc
        ms = re.match(r'^shuffle\s+(asc|desc)$', content)
        if ms:
            return {"type": "shuffle", "order": ms.group(1)}
        # query
        if content == "query":
            return {"type": "query"}
        # finalize yes|no
        mf = re.match(r'^finalize\s+(yes|no)$', content)
        if mf:
            return {"type": "finalize", "ans": mf.group(1)}
        return {"type": "unsupported", "raw": content}

    def sample_random_action(self) -> str:
        choices = []
        remaining = self.total_edges - self.stream_pos
        capacity = self.buffer_size - len(self.buffer)
        if remaining > 0 and capacity > 0:
            k = random.randint(1, max(1, min(self.max_reveal_k, capacity, remaining)))
            choices.append(f"reveal {k}")
        if self.buffer:
            u, v = random.choice(self.buffer)
            choices.append(f"process {u} {v}")
            choices.append(random.choice(["shuffle asc", "shuffle desc"]))
        choices.append("query")
        if random.random() < 0.3:
            choices.append(random.choice(["finalize yes", "finalize no"]))
        return f"\\boxed{{{random.choice(choices)}}}"


class GraphCycleStreamingEnvWithFeedback(GraphCycleStreamingEnv):
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
            if self.feedback_level >= 2:
                hint = "Wrap your command like \\boxed{reveal 3} or \\boxed{process 2 5}."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "reveal k must" in text:
                error_detail["violation"] = "reveal_limit_exceeded_or_invalid"
                if self.feedback_level >= 2:
                    hint = "Choose k within 1..max_reveal_k and ensure buffer has free capacity."
            elif "buffer is full" in text:
                error_detail["violation"] = "buffer_full"
                if self.feedback_level >= 2:
                    hint = "Process some buffered edges before revealing more."
            elif "edge not in buffer" in text:
                error_detail["violation"] = "process_edge_not_in_buffer"
                if self.feedback_level >= 2:
                    hint = "Only process edges currently listed in the buffer."
            elif "endpoints must be distinct" in text:
                error_detail["violation"] = "invalid_endpoints"
                if self.feedback_level >= 2:
                    hint = "Use distinct endpoints within [1..N]."
            else:
                error_detail["violation"] = "other_protocol"
                if self.feedback_level >= 2:
                    hint = "Follow the command grammar and buffer/stream constraints."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_or_bad_arguments"
            if self.feedback_level >= 2:
                hint = "Supported actions: reveal k | process u v | shuffle asc|desc | query | finalize yes|no."
        elif "reached max turns" in text or "timeout" in text:
            error_type = "Timeout"
            error_detail["limit"] = getattr(self, "turn_limit", None)
            if self.feedback_level >= 2:
                hint = "Reveal in small batches and process edges consistently; avoid unnecessary queries/shuffles."
        elif "incorrect final answer" in text:
            error_type = "WrongDecision"
            error_detail["got"] = "wrong_finalization"
            if self.feedback_level >= 2:
                # Contextual hint depending on progress
                if not self.cycle_detected_so_far:
                    hint = "To assert 'forest=yes' safely, process all edges or detect no cycles through full stream."
                else:
                    hint = "If any processed edge connects already-connected nodes, the graph is not a forest."
        elif "success" in text and "correct final answer" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "nodes": getattr(self, "num_nodes", None),
                "processed": getattr(self, "processed_count", None),
                "buffer_len": len(getattr(self, "buffer", [])),
                "remaining": max(0, getattr(self, "total_edges", 0) - getattr(self, "stream_pos", 0)),
                "cycle_detected_so_far": getattr(self, "cycle_detected_so_far", None),
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
            "hint": "Start by using reveal k to bring a few edges into the buffer, then process them.",
            "turn": 0,
            "state": {
                "nodes": getattr(self, "num_nodes", None),
                "processed": getattr(self, "processed_count", None),
                "buffer_len": len(getattr(self, "buffer", [])),
                "remaining": max(0, getattr(self, "total_edges", 0) - getattr(self, "stream_pos", 0)),
                "cycle_detected_so_far": getattr(self, "cycle_detected_so_far", None),
            },
        }
        return obs, info