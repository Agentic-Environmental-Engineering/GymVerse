from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class BSTAuditEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 50,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 50

        # Evolvable parameters
        self.complexity_params = {
            'num_nodes': (7, 60),              # More nodes = larger search space and more permutations -> harder
            'fault_swaps': (0, 20),            # Number of disjoint pair swaps injected. Higher = more repairs -> harder
            'max_full_dumps': (3, 0),          # REVERSED: fewer full dumps (dump_inorder) allowed -> harder
            'trial_swap_budget': (10, 2),      # REVERSED: fewer working-copy trial swaps -> harder
            'reveal_budget_factor': (1.0, 0.3) # REVERSED: fewer per-node reveals allowed relative to n -> harder
        }

        # Variance for randomization
        self.param_variance = {
            'num_nodes': 4,            # ~7% of range
            'fault_swaps': 2,          # small integer variance
            'max_full_dumps': 0,       # small range → conservative (no randomization)
            'trial_swap_budget': 1,    # small integer variance
            'reveal_budget_factor': 0.05  # slight continuous variance
        }

        # Placeholder attributes
        self.num_nodes: int = 0
        self.fault_swaps: int = 0
        self.max_full_dumps: int = 0
        self.trial_swap_budget: int = 0
        self.reveal_budget_factor: float = 0.0

        # State
        self.turn_count: int = 0
        self.terminated: bool = False

        # Latent world
        self.tree_children: Dict[int, Tuple[Optional[int], Optional[int]]] = {}
        self.latent_keys: Dict[int, int] = {}

        # Representation
        self.working_keys: Dict[int, int] = {}
        self.marked_nodes: List[int] = []
        self.full_dumps_used: int = 0
        self.trial_swaps_used: int = 0
        self.reveals_used: int = 0
        self.reveal_budget: int = 0

        # Ground truth objective
        self.min_swaps_to_fix: int = 0

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
                    lo = min(min_val, max_val)
                    hi = max(min_val, max_val)
                    if param_name == 'reveal_budget_factor':
                        actual_value = max(lo, min(hi, actual_value))
                        setattr(self, param_name, float(actual_value))
                        continue
                    actual_value = max(lo, min(hi, actual_value))
            if param_name == 'reveal_budget_factor':
                setattr(self, param_name, float(actual_value))
            else:
                setattr(self, param_name, int(round(actual_value)))

    def _inorder_ids(self) -> List[int]:
        ids = []
        def dfs(i: int):
            if i is None or i < 1 or i > self.num_nodes:
                return
            left, right = self.tree_children[i]
            dfs(left if left is not None else None)
            ids.append(i)
            dfs(right if right is not None else None)
        dfs(1)
        return ids

    def _inorder_keys(self, use_working: bool = False) -> List[int]:
        ids = self._inorder_ids()
        keys = []
        store = self.working_keys if use_working else self.latent_keys
        for i in ids:
            keys.append(store[i])
        return keys

    def _compute_inversions(self, arr: List[int]) -> int:
        # Count inversions via modified merge sort
        def merge_count(seq):
            n = len(seq)
            if n <= 1:
                return seq, 0
            mid = n // 2
            left, li = merge_count(seq[:mid])
            right, ri = merge_count(seq[mid:])
            i = j = inv = 0
            merged = []
            while i < len(left) and j < len(right):
                if left[i] <= right[j]:
                    merged.append(left[i])
                    i += 1
                else:
                    merged.append(right[j])
                    j += 1
                    inv += len(left) - i
            merged.extend(left[i:])
            merged.extend(right[j:])
            return merged, inv + li + ri
        _, inv = merge_count(arr)
        return inv

    def _get_instructions(self) -> str:
        return (
            "You are auditing a binary search tree (BST) with unique keys.\n"
            "The latent tree shape is fixed; keys may be swapped, potentially breaking the BST invariant.\n"
            "Goal: submit whether the tree currently satisfies the BST invariant (YES), or if not, submit the minimal number of key swaps needed to fix it (NO k).\n"
            "A swap exchanges keys at two nodes; structural links never change. Minimal swaps refer to the latent world, not your working copy.\n"
            "Actions:\n"
            "- dump_structure: show node ids with left/right children\n"
            "- inorder_ids: list node ids in inorder\n"
            "- dump_inorder: full dump of inorder keys (limited by max_full_dumps)\n"
            "- reveal <node_id>: show the key at a node (limited by reveal budget)\n"
            "- mark <node_id>: mark a node as suspected violation (representation-only)\n"
            "- trial_swap <a> <b>: swap keys in your working copy (limited by trial_swap_budget); returns current inversions of working inorder\n"
            "- check: check if your working copy is currently a valid BST and report inversions\n"
            "- submit YES\n"
            "- submit NO <k>\n"
            "Use \\boxed{...} to send actions. For example: "
            f"{self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        status = []
        status.append(f"turn={self.turn_count}/{self.max_turns}")
        status.append(f"n={self.num_nodes}")
        status.append(f"full_dumps_used={self.full_dumps_used}/{self.max_full_dumps}")
        status.append(f"trial_swaps_used={self.trial_swaps_used}/{self.trial_swap_budget}")
        status.append(f"reveals_used={self.reveals_used}/{self.reveal_budget}")
        status.append(f"marks={len(self.marked_nodes)}")
        return (
            "State summary: "
            + ", ".join(status)
            + ". Enter your next action using \\boxed{...}."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        self.turn_count = 0
        self.terminated = False
        self.tree_children = {}
        self.latent_keys = {}
        self.working_keys = {}
        self.marked_nodes = []
        self.full_dumps_used = 0
        self.trial_swaps_used = 0
        self.reveals_used = 0

        # Build complete binary tree shape with ids 1..n
        for i in range(1, self.num_nodes + 1):
            left = 2 * i if 2 * i <= self.num_nodes else None
            right = 2 * i + 1 if 2 * i + 1 <= self.num_nodes else None
            self.tree_children[i] = (left, right)

        inorder_ids = self._inorder_ids()
        # Assign sorted keys to inorder to create a valid BST initially
        for idx, node_id in enumerate(inorder_ids):
            self.latent_keys[node_id] = idx + 1

        # Inject disjoint pair swaps to create a permutation with known minimal swaps
        max_pairs = self.num_nodes // 2
        fault_swaps = min(self.fault_swaps, max_pairs)
        used_positions = set()
        positions = list(range(len(inorder_ids)))
        random.shuffle(positions)
        pairs = []
        for i in range(0, len(positions), 2):
            if len(pairs) >= fault_swaps:
                break
            if i + 1 >= len(positions):
                break
            a = positions[i]
            b = positions[i + 1]
            if a in used_positions or b in used_positions:
                continue
            used_positions.add(a)
            used_positions.add(b)
            pairs.append((a, b))

        for a, b in pairs:
            ida = inorder_ids[a]
            idb = inorder_ids[b]
            self.latent_keys[ida], self.latent_keys[idb] = self.latent_keys[idb], self.latent_keys[ida]

        # Working copy mirrors latent initial state
        self.working_keys = dict(self.latent_keys)

        # Ground truth minimal swaps equals number of disjoint pairs used
        self.min_swaps_to_fix = len(pairs)

        # Reveal budget scales with n
        self.reveal_budget = int(round(self.num_nodes * self.reveal_budget_factor))

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1

        parsed = self._parse_action(action)
        if parsed is None:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        cmd = parsed.get("cmd")
        args = parsed.get("args", [])

        # Handle commands
        if cmd == "help":
            obs = self._get_instructions()
            terminated = False
            truncated = False
            if self.turn_count >= self.max_turns:
                obs = f"Reached max turns ({self.max_turns})"
                return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        elif cmd == "dump_structure":
            lines = []
            for i in range(1, self.num_nodes + 1):
                l, r = self.tree_children[i]
                lines.append(f"{i}: L={l if l is not None else 'None'}, R={r if r is not None else 'None'}")
            obs = "STRUCTURE:\n" + "\n".join(lines)
            reward = 0.0

        elif cmd == "inorder_ids":
            ids = self._inorder_ids()
            obs = "INORDER_IDS: [" + ", ".join(str(x) for x in ids) + "]"
            reward = 0.0

        elif cmd == "dump_inorder":
            if self.full_dumps_used >= self.max_full_dumps:
                obs = "NO_BUDGET: dump_inorder exhausted"
                reward = 0.0
            else:
                self.full_dumps_used += 1
                keys = self._inorder_keys(use_working=False)
                obs = "INORDER_KEYS: [" + ", ".join(str(k) for k in keys) + "]"
                reward = 0.0

        elif cmd == "reveal":
            if len(args) != 1 or not isinstance(args[0], int):
                obs = "ERROR: reveal expects one integer node_id"
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if self.reveals_used >= self.reveal_budget:
                obs = "NO_BUDGET: reveal exhausted"
                reward = 0.0
            else:
                node = args[0]
                if node < 1 or node > self.num_nodes:
                    obs = f"ERROR: node_id {node} out of range"
                    return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
                self.reveals_used += 1
                obs = f"NODE {node} KEY {self.latent_keys[node]}"
                reward = 0.0

        elif cmd == "mark":
            if len(args) != 1 or not isinstance(args[0], int):
                obs = "ERROR: mark expects one integer node_id"
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            node = args[0]
            if node < 1 or node > self.num_nodes:
                obs = f"ERROR: node_id {node} out of range"
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if node not in self.marked_nodes:
                self.marked_nodes.append(node)
            obs = f"MARKED: {node}; marks={len(self.marked_nodes)}"
            reward = 0.0

        elif cmd == "trial_swap":
            if len(args) != 2 or not all(isinstance(x, int) for x in args):
                obs = "ERROR: trial_swap expects two integer node_ids"
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if self.trial_swaps_used >= self.trial_swap_budget:
                obs = "NO_BUDGET: trial_swap exhausted"
                reward = 0.0
            else:
                a, b = args
                if not (1 <= a <= self.num_nodes) or not (1 <= b <= self.num_nodes):
                    obs = "ERROR: node_id out of range for trial_swap"
                    return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
                self.trial_swaps_used += 1
                self.working_keys[a], self.working_keys[b] = self.working_keys[b], self.working_keys[a]
                inv = self._compute_inversions(self._inorder_keys(use_working=True))
                obs = f"SWAPPED {a} {b}; inversions_now={inv}; trial_swaps_left={self.trial_swap_budget - self.trial_swaps_used}"
                reward = 0.0

        elif cmd == "check":
            keys = self._inorder_keys(use_working=True)
            inv = self._compute_inversions(keys)
            ok = "YES" if inv == 0 else "NO"
            obs = f"WORKING_CHECK: valid={ok}; inversions={inv}"
            reward = 0.0

        elif cmd == "submit":
            sub_type = args[0] if len(args) >= 1 else None
            if sub_type is None:
                obs = "ERROR: submit expects YES or NO <k>"
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if isinstance(sub_type, str) and sub_type.upper() == "YES":
                correct = (self.min_swaps_to_fix == 0)
                if correct:
                    obs = "SUBMISSION: YES\nSuccess! BST invariant holds."
                    return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
                else:
                    obs = f"SUBMISSION: YES\nFailed! BST requires {self.min_swaps_to_fix} swaps."
                    return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            elif isinstance(sub_type, str) and sub_type.upper() == "NO":
                if len(args) != 2 or not isinstance(args[1], int):
                    obs = "ERROR: submit NO expects integer k"
                    return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
                k = args[1]
                correct = (self.min_swaps_to_fix == k and self.min_swaps_to_fix > 0)
                if correct:
                    obs = f"SUBMISSION: NO {k}\nSuccess! Minimal swaps = {k}."
                    return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
                else:
                    obs = f"SUBMISSION: NO {k}\nFailed! Minimal swaps = {self.min_swaps_to_fix}."
                    return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = "ERROR: submit expects YES or NO <k>"
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = f"Unsupported action: {cmd}"
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        # Truncation check
        if self.turn_count >= self.max_turns:
            obs_timeout = f"Reached max turns ({self.max_turns})"
            return obs_timeout, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        # Tokenize
        tokens = content.split()
        if not tokens:
            return None
        cmd = tokens[0].lower()
        args: List[Any] = []
        if cmd in ("dump_structure", "inorder_ids", "dump_inorder", "check", "help"):
            return {"cmd": cmd, "args": []}
        if cmd == "reveal":
            if len(tokens) >= 2 and tokens[1].isdigit():
                args.append(int(tokens[1]))
                return {"cmd": "reveal", "args": args}
            else:
                return {"cmd": "reveal", "args": [None]}
        if cmd == "mark":
            if len(tokens) >= 2 and tokens[1].isdigit():
                args.append(int(tokens[1]))
                return {"cmd": "mark", "args": args}
            else:
                return {"cmd": "mark", "args": [None]}
        if cmd == "trial_swap":
            if len(tokens) >= 3 and tokens[1].isdigit() and tokens[2].isdigit():
                args.append(int(tokens[1]))
                args.append(int(tokens[2]))
                return {"cmd": "trial_swap", "args": args}
            else:
                return {"cmd": "trial_swap", "args": [None, None]}
        if cmd == "submit":
            if len(tokens) >= 2:
                sub = tokens[1].upper()
                if sub == "YES":
                    return {"cmd": "submit", "args": [sub]}
                elif sub == "NO":
                    if len(tokens) >= 3 and re.fullmatch(r'-?\d+', tokens[2]):
                        return {"cmd": "submit", "args": [sub, int(tokens[2])]}
                    else:
                        return {"cmd": "submit", "args": [sub, None]}
            return {"cmd": "submit", "args": []}
        return {"cmd": cmd, "args": []}

    def sample_random_action(self) -> str:
        candidates = [
            "\\boxed{dump_structure}",
            "\\boxed{inorder_ids}",
            "\\boxed{dump_inorder}",
            "\\boxed{reveal 3}",
            "\\boxed{trial_swap 2 5}",
            "\\boxed{check}",
            "\\boxed{submit YES}",
            "\\boxed{submit NO 2}",
        ]
        return random.choice(candidates)


class BSTAuditEnvWithFeedback(BSTAuditEnv):
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
            hint = "Wrap your action in \\boxed{...} with a valid command."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["command"] = "unknown"
            hint = "Use documented commands like dump_structure, reveal <id>, trial_swap <a> <b>, check, submit YES/NO k."

        elif "no_budget" in text:
            error_type = "ProtocolViolation"
            if "dump_inorder" in text:
                error_detail["violation"] = "full_dump_exhausted"
                hint = "Use inorder_ids and reveal <node_id> selectively to reconstruct keys without full dump."
            elif "trial_swap" in text:
                error_detail["violation"] = "trial_swap_exhausted"
                hint = "Run check before swapping; prioritize swaps where adjacent inorder keys are inverted."
            elif "reveal exhausted" in text:
                error_detail["violation"] = "reveal_exhausted"
                hint = "Stop revealing; combine structure knowledge with previous reveals to infer the minimal swaps."

        elif "error:" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "bad_arguments"
            hint = "Check argument counts and ranges: reveal <id>, trial_swap <a> <b>, submit YES or submit NO <k>."

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Plan fewer exploratory actions; use dump_inorder early if budget allows, then compute swaps."

        elif "failed!" in text and "submission:" in text:
            error_type = "WrongDecision"
            # Try to extract submitted answer
            submitted = None
            m_yes = re.search(r"submission:\s*yes", text)
            m_no = re.search(r"submission:\s*no\s+(-?\d+)", text)
            if m_yes:
                submitted = "YES"
            elif m_no:
                submitted = f"NO {m_no.group(1)}"
            error_detail["submitted"] = submitted
            error_detail["expected"] = f"YES" if self.min_swaps_to_fix == 0 else f"NO {self.min_swaps_to_fix}"
            hint = "To compute minimal swaps, inspect the inorder keys and count permutation cycles (or inversions as a proxy)."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "n": getattr(self, "num_nodes", None),
                "full_dumps_used": getattr(self, "full_dumps_used", None),
                "max_full_dumps": getattr(self, "max_full_dumps", None),
                "trial_swaps_used": getattr(self, "trial_swaps_used", None),
                "trial_swap_budget": getattr(self, "trial_swap_budget", None),
                "reveals_used": getattr(self, "reveals_used", None),
                "reveal_budget": getattr(self, "reveal_budget", None),
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
            "hint": "Start with dump_structure or inorder_ids; if budget permits, use dump_inorder to see all keys.",
            "turn": 0,
            "state": {
                "n": getattr(self, "num_nodes", None),
                "full_dumps_used": getattr(self, "full_dumps_used", None),
                "max_full_dumps": getattr(self, "max_full_dumps", None),
                "trial_swaps_used": getattr(self, "trial_swaps_used", None),
                "trial_swap_budget": getattr(self, "trial_swap_budget", None),
                "reveals_used": getattr(self, "reveals_used", None),
                "reveal_budget": getattr(self, "reveal_budget", None),
            },
        }
        return obs, info