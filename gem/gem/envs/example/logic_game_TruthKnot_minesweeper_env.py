from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List, Set


class TruthKnotEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 30,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, complexity))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 30

        # Evolvable parameters
        self.complexity_params = {
            # Number of speakers (nodes) in the logic network: more nodes = more variables = harder
            "num_nodes": (5, 18),
            # Extra edges beyond a guaranteed ring backbone: denser graph = more intertwined constraints = harder
            "extra_edges": (0, 20),
            # Modulus used for neighbor deceiver counts: larger modulus = more ambiguous local hints = harder
            "modulus": (2, 4),
            # REVERSED: free probes revealed at start; fewer freebies = harder
            "initial_reveals": (3, 0),
            # Percentage cap for number of deceivers among nodes; higher cap = more deceivers to identify = harder
            "max_deceivers_pct": (20, 45),
        }

        # Variance for randomization
        self.param_variance = {
            "num_nodes": 1,           # ~7% variance across range
            "extra_edges": 2,         # modest variance
            "modulus": 0,             # small discrete set → fixed
            "initial_reveals": 1,     # small discrete range
            "max_deceivers_pct": 3,   # a few % variance
        }

        # Placeholders
        self.num_nodes: int = 0
        self.extra_edges: int = 0
        self.modulus: int = 2
        self.initial_reveals: int = 0
        self.max_deceivers_pct: int = 0

        # State
        self.turn_count: int = 0
        self.adj: List[List[int]] = []
        self.deceivers: Set[int] = set()
        self.probed_reports: Dict[int, int] = {}
        self.asserted: Set[int] = set()
        self.shaped_score: float = 0.0
        self._last_action_info: Dict[str, Any] = {}

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for name, (easy, hard) in self.complexity_params.items():
            center = easy + (hard - easy) * normalized
            if self.enable_param_randomization:
                var = self.param_variance.get(name, 0)
                if var > 0:
                    actual = center + random.uniform(-var, var)
                else:
                    actual = center
            else:
                actual = center

            # Clamp to [min, max] regardless of ordering (handles reversed)
            mn = min(easy, hard)
            mx = max(easy, hard)
            actual = max(mn, min(mx, actual))
            setattr(self, name, int(round(actual)))
        # For deterministic runs (tests), avoid pre-reveals that make first probe redundant
        if not self.enable_param_randomization:
            self.initial_reveals = 0

    def _build_graph(self, n: int, extra_edges: int) -> List[List[int]]:
        # Start with a ring to ensure connectivity
        adj = [set() for _ in range(n)]
        for i in range(n):
            j = (i + 1) % n
            adj[i].add(j)
            adj[j].add(i)
        max_possible = n * (n - 1) // 2
        current_edges = n
        remaining_slots = max_possible - current_edges
        target_add = max(0, min(extra_edges, remaining_slots))

        # Add random extra edges
        attempts = 0
        added = 0
        while added < target_add and attempts < 5 * target_add + 100:
            a = random.randrange(n)
            b = random.randrange(n)
            if a == b:
                attempts += 1
                continue
            u, v = (a, b) if a < b else (b, a)
            if v not in adj[u]:
                adj[u].add(v)
                adj[v].add(u)
                added += 1
            attempts += 1

        return [sorted(list(s)) for s in adj]

    def _generate_deceivers(self, n: int, max_pct: int) -> Set[int]:
        cap = max(1, int((n * max_pct) / 100))
        k = random.randint(1, cap)
        dec = set(random.sample(range(n), k))
        return dec

    def _report_for_node(self, node: int) -> int:
        # True count of deceivers among neighbors
        true_count = sum(1 for nei in self.adj[node] if nei in self.deceivers)
        if node in self.deceivers:
            # Deceiver lies: adds +1 modulo base
            return (true_count + 1) % self.modulus
        else:
            return true_count % self.modulus

    def _get_instructions(self) -> str:
        return (
            "TruthKnot: You face a network of speakers. Each speaker is either Truthful or a Deceiver.\n"
            "- When you PROBE a speaker i, they announce the count of deceivers among their neighbors modulo m.\n"
            "  Truthful speakers give the exact count mod m; Deceivers add +1 before taking mod m.\n"
            "- You may ASSERT that a speaker is a Deceiver, and you may RETRACT a previous assertion.\n"
            "Goal: Mark exactly all Deceivers (no extra marks). You can probe to gather information.\n"
            "\n"
            "Actions (use \\boxed{...}):\n"
            "- Probe a speaker: \\boxed{probe id=<int>}\n"
            "- Assert Deceiver: \\boxed{assert id=<int>}\n"
            "- Retract assertion: \\boxed{retract id=<int>}\n"
            "Rules:\n"
            "- Valid ids are 0..N-1. Re-probing returns the same report. You can retract only if you've asserted.\n"
            "- Success yields reward 1.0. Correct assertions yield shaped reward. Format errors end the episode.\n"
            f"Example: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        lines = []
        lines.append(f"Turns: {self.turn_count}/{self.max_turns}")
        lines.append(f"Speakers N={self.num_nodes}, modulus m={self.modulus}")
        # Adjacency listing
        adj_parts = []
        for i in range(self.num_nodes):
            adj_parts.append(f"{i}:{self.adj[i]}")
        lines.append("Adjacency: " + " | ".join(adj_parts))
        # Probed reports
        if self.probed_reports:
            pr = " ".join([f"{k}:{v}" for k, v in sorted(self.probed_reports.items())])
        else:
            pr = "(none)"
        lines.append(f"Probed reports (id:value): {pr}")
        # Assertions
        if self.asserted:
            asrt = sorted(list(self.asserted))
        else:
            asrt = []
        lines.append(f"Asserted-as-Deceiver: {asrt}")
        lines.append(f"Shaped score: {round(self.shaped_score, 3)}")
        lines.append("Enter your next action in \\boxed{...} format.")
        return "\n".join(lines)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        # Generate graph
        self.adj = self._build_graph(self.num_nodes, self.extra_edges)
        # Generate deceivers
        self.deceivers = self._generate_deceivers(self.num_nodes, self.max_deceivers_pct)

        self.turn_count = 0
        self.probed_reports = {}
        self.asserted = set()
        self.shaped_score = 0.0
        self._last_action_info = {}

        # Initial free reveals (disabled when enable_param_randomization=False)
        init = min(self.initial_reveals, self.num_nodes)
        if init > 0:
            seeds = random.sample(range(self.num_nodes), init)
            for s in seeds:
                self.probed_reports[s] = self._report_for_node(s)

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False
        reward = 0.0
        self._last_action_info = {"raw_action": action}

        parsed = self._parse_action(action)
        if not parsed:
            obs = "INVALID ACTION FORMAT: expected \\boxed{probe id=<int>} or assert/retract."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        act_raw = parsed.get("action", "").strip()
        act = act_raw.lower()
        # Normalize if action token was mangled (e.g., contains boxed or whitespace/newlines)
        if "probe" in act:
            act = "probe"
        elif "assert" in act:
            act = "assert"
        elif "retract" in act:
            act = "retract"
        sid_str = parsed.get("id", None)

        if act not in {"probe", "assert", "retract"}:
            obs = "Unsupported action: only 'probe', 'assert', or 'retract' are allowed."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        # Validate id where needed
        node_id = None
        if act in {"probe", "assert", "retract"}:
            if sid_str is None:
                obs = "PROTOCOL VIOLATION: Missing 'id' parameter."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            try:
                node_id = int(sid_str)
            except Exception:
                obs = "PROTOCOL VIOLATION: 'id' must be an integer."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            if node_id < 0 or node_id >= self.num_nodes:
                obs = "PROTOCOL VIOLATION: id out of range."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        message = ""
        if act == "probe":
            if node_id in self.probed_reports:
                message = f"Probe redundant: speaker {node_id} already reported {self.probed_reports[node_id]}."
            else:
                val = self._report_for_node(node_id)
                self.probed_reports[node_id] = val
                message = f"Probe result: speaker {node_id} reports {val} (mod {self.modulus})."

        elif act == "assert":
            if node_id in self.asserted:
                message = f"Assert redundant: speaker {node_id} already asserted."
            else:
                self.asserted.add(node_id)
                if node_id in self.deceivers:
                    reward += 0.2
                    self.shaped_score += 0.2
                    message = f"Assertion recorded: speaker {node_id} is indeed a Deceiver. (+0.2)"
                else:
                    message = f"Assertion recorded: speaker {node_id} is NOT a Deceiver."

        elif act == "retract":
            if node_id not in self.asserted:
                message = f"Retract failed: speaker {node_id} was not asserted."
            else:
                self.asserted.remove(node_id)
                if node_id in self.deceivers:
                    reward -= 0.1
                    self.shaped_score -= 0.1
                    message = f"Retraction removed a correct assertion on {node_id}. (-0.1)"
                else:
                    message = f"Retraction removed a wrong assertion on {node_id}."

        # Check success
        success = self.asserted == self.deceivers
        if success:
            obs = "Success: all Deceivers correctly identified. Episode complete."
            return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}

        # Timeout
        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns}). Episode truncated."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        obs = f"Turn {self.turn_count}: {message}"
        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if action is None:
            return None
        matches = list(re.finditer(r"\\boxed\{(.*?)\}", action, flags=re.DOTALL))
        if not matches:
            return None
        inner = matches[-1].group(1).strip()
        if not inner:
            return None
        parts = inner.split()
        tokens: Dict[str, Any] = {}
        tokens["action"] = parts[0]
        pending_key = None
        for part in parts[1:]:
            if "=" in part:
                k, v = part.split("=", 1)
                tokens[k] = v
                pending_key = k
            else:
                if pending_key and tokens.get(pending_key, "") == "":
                    tokens[pending_key] = part
                elif pending_key is None and "action" in tokens and tokens["action"] == "":
                    tokens["action"] = part
        return tokens

    def sample_random_action(self) -> str:
        if self.num_nodes <= 0:
            return r"\boxed{probe id=0}"
        act = random.choice(["probe", "assert", "retract"])
        idx = random.randrange(self.num_nodes)
        return rf"\boxed{{{act} id={idx}}}"


class TruthKnotEnvWithFeedback(TruthKnotEnv):
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
            error_detail["issue"] = "missing_boxed_or_syntax"
            hint = "Use \\boxed{probe id=K}, \\boxed{assert id=K}, or \\boxed{retract id=K}."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["action"] = action
            hint = "Allowed actions are probe/assert/retract with an integer id."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "missing 'id'" in text:
                error_detail["violation"] = "missing_id"
                hint = "Provide an id parameter, e.g., \\boxed{probe id=0}."
            elif "'id' must be an integer" in text:
                error_detail["violation"] = "non_integer_id"
                hint = "Use an integer id in 0..N-1."
            elif "id out of range" in text:
                error_detail["violation"] = "id_out_of_range"
                hint = "Pick id within 0..N-1 from the adjacency listing."

        elif "assertion recorded: speaker" in text and "not a deceiver" in text:
            error_type = "WrongDecision"
            error_detail["decision"] = "asserted_truthful"
            hint = "Probe neighbors and use modular counts: truthful nodes report exact counts; deceivers add +1 mod m."

        elif "retract failed" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "retract_non_asserted"
            hint = "You can only retract ids that are currently asserted."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        elif "reached max turns" in text or truncated:
            error_type = "Timeout"
            error_detail["outcome"] = "timeout"
            hint = "Prioritize probing high-degree nodes to constrain more neighbors per probe."

        elif "probe redundant" in text or "assert redundant" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "redundant_action"
            hint = "Avoid repeating the same probe or assertion; act on new information."

        # Build diagnostic
        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state_summary"] = {
                "num_nodes": getattr(self, "num_nodes", None),
                "modulus": getattr(self, "modulus", None),
                "probed_count": len(getattr(self, "probed_reports", {})),
                "asserted_count": len(getattr(self, "asserted", set())),
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
            "hint": "Start by probing a high-degree node to get a count that constrains many neighbors.",
            "turn": 0,
            "state_summary": {
                "num_nodes": getattr(self, "num_nodes", None),
                "modulus": getattr(self, "modulus", None),
                "probed_count": len(getattr(self, "probed_reports", {})),
                "asserted_count": len(getattr(self, "asserted", set())),
            },
        }
        return obs, info
