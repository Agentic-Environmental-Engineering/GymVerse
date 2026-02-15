from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List, Set


class ClueCraftersEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 24,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 24

        # Evolvable parameters with rationale:
        # - num_symbols: size of the symbol set to assign (larger = more complex structure to resolve)
        # - subset_size: K, number of clues to pick (larger K increases combinatorics; we keep ranges reasonable)
        # - num_clues: number of available clues (more options = harder combinatorial selection)
        # - redundancy_level: how many overlapping/less-informative clues exist (higher = harder to find optimal)
        # - cost_spread: max extra cost variation beyond base cost 1 (higher = harder trade-offs)
        self.complexity_params = {
            "num_symbols": (3, 7),           # 3→7 symbols increases hidden structure complexity
            "subset_size": (2, 4),           # pick K clues; more K increases choice space
            "num_clues": (6, 16),            # more clues increases combinatorial difficulty
            "redundancy_level": (1, 6),      # more redundant clues makes optimal selection harder
            "cost_spread": (0, 3),           # max additional cost beyond base cost 1; larger spread = tougher trade-offs
        }

        # Variance settings
        self.param_variance = {
            "num_symbols": 0,         # small discrete range
            "subset_size": 0,         # small discrete range
            "num_clues": 1,           # medium discrete range → ±1
            "redundancy_level": 1,    # medium discrete range → ±1
            "cost_spread": 1,         # small-mid range → ±1
        }

        # Placeholders applied at reset
        self.num_symbols: int = 0
        self.subset_size: int = 0
        self.num_clues: int = 0
        self.redundancy_level: int = 0
        self.cost_spread: int = 0

        # State
        self.turn_count: int = 0
        self.symbols: List[str] = []
        self.positions: List[int] = []
        self.hidden_assignment: Dict[int, str] = {}
        self.clues: List[Dict[str, Any]] = []
        self.optimal_ratio: float = 0.0
        self.optimal_indices: Set[int] = set()
        self.terminated_flag: bool = False

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
                    lo, hi = (min_val, max_val) if min_val <= max_val else (max_val, min_val)
                    actual_value = max(lo, min(hi, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _generate_hidden_assignment(self):
        self.symbols = [chr(ord('A') + i) for i in range(self.num_symbols)]
        self.positions = list(range(self.num_symbols))
        shuffled = self.symbols[:]
        random.shuffle(shuffled)
        self.hidden_assignment = {pos: sym for pos, sym in enumerate(shuffled)}

    def _make_clues(self):
        # Clue types:
        # - PosIsSym: "Position i contains symbol S" (coverage: (i,S))
        # - PosNotSym: "Position i is not S" (coverage: (i,S))
        # - SymNotPos: "Symbol S is not at position i" (coverage: (i,S)) alias
        # - PairOrder: "Symbol S1 is before S2" (coverage: (S1<S2) ordering pair)
        # - PairApart: "S1 is not adjacent to S2" (coverage: adjacency exclusion)
        # Coverage is how many distinct constraint units a clue contributes to:
        #   - Pos/Sym units: num_symbols^2 possibilities (position-symbol matrix)
        #   - Order units: number of ordered symbol pairs
        #   - Adjacency units: unordered adjacent disallowances

        # Build a candidate pool with some redundancy
        clues = []
        base_cost = 1
        spread = self.cost_spread

        def mk_cost():
            return base_cost + (random.randint(0, spread) if spread > 0 else 0)

        N = self.num_symbols
        syms = self.symbols
        # Always include a few highly-informative direct facts
        for i in range(min(self.num_symbols, max(2, self.num_clues // 4))):
            pos = i % N
            sym = self.hidden_assignment[pos]
            clues.append({
                "type": "PosIsSym",
                "text": f"Position {pos} is {sym}.",
                "covers": {("possym", pos, sym)},
                "cost": mk_cost()
            })

        # Add negative facts (redundant/distractors)
        for _ in range(self.redundancy_level + self.num_clues // 3):
            pos = random.randrange(N)
            sym = random.choice(syms)
            if self.hidden_assignment[pos] == sym:
                continue
            clues.append({
                "type": "PosNotSym",
                "text": f"Position {pos} is not {sym}.",
                "covers": {("possym", pos, sym)},
                "cost": mk_cost()
            })

        # Order constraints
        for _ in range(max(1, self.num_clues // 4)):
            s1, s2 = random.sample(syms, 2)
            text = f"{s1} appears before {s2}."
            covers = {("order", min(s1, s2), max(s1, s2), "lt" if s1 < s2 else "lt")}
            clues.append({
                "type": "PairOrder",
                "text": text,
                "covers": covers,
                "cost": mk_cost()
            })

        # Apart constraints
        for _ in range(max(1, self.num_clues // 5)):
            s1, s2 = random.sample(syms, 2)
            text = f"{s1} is not adjacent to {s2}."
            covers = {("apart", min(s1, s2), max(s1, s2))}
            clues.append({
                "type": "PairApart",
                "text": text,
                "covers": covers,
                "cost": mk_cost()
            })

        # Light redundancy by duplicating coverage with different phrasing
        for _ in range(self.redundancy_level):
            pos = random.randrange(N)
            sym = random.choice(syms)
            if self.hidden_assignment[pos] != sym:
                clues.append({
                    "type": "SymNotPos",
                    "text": f"{sym} is not at position {pos}.",
                    "covers": {("possym", pos, sym)},
                    "cost": mk_cost()
                })

        # Ensure we have enough but not too many; trim or pad with safe negatives
        random.shuffle(clues)
        if len(clues) > self.num_clues:
            clues = clues[:self.num_clues]
        while len(clues) < self.num_clues:
            pos = random.randrange(N)
            sym = random.choice(syms)
            if self.hidden_assignment[pos] == sym:
                continue
            clues.append({
                "type": "PosNotSym",
                "text": f"Position {pos} is not {sym}.",
                "covers": {("possym", pos, sym)},
                "cost": mk_cost()
            })

        # Compute coverage size per clue (proxy for informativeness)
        for c in clues:
            c["coverage_size"] = len(c["covers"])

        self.clues = clues

    def _compute_ratio(self, idxs: Set[int]) -> float:
        if not idxs:
            return 0.0
        covers = set()
        total_cost = 0
        for i in idxs:
            c = self.clues[i]
            covers |= c["covers"]
            total_cost += c["cost"]
        if total_cost <= 0:
            return 0.0
        return len(covers) / float(total_cost)

    def _find_optimal_subset(self):
        # Brute-force over all K-size subsets to find max coverage/cost ratio
        # Keep feasibility reasonable by parameter ranges
        K = self.subset_size
        n = len(self.clues)
        best_ratio = -1.0
        best = set()
        # Simple combinational search with random sampling if extremely large (guard)
        limit = 50000
        sampled = 0

        # Generate combinations iteratively
        indices = list(range(n))
        # If combinatorics explode, random sample subsets of size K
        from math import comb
        total = comb(n, K) if 0 <= K <= n else 0
        if total == 0:
            self.optimal_ratio = 0.0
            self.optimal_indices = set()
            return

        def all_combinations_k(lst, k):
            # iterative stack to avoid recursion
            stack = [(0, [])]
            while stack:
                start, curr = stack.pop()
                if len(curr) == k:
                    yield curr
                    continue
                for i in range(start, len(lst)):
                    stack.append((i + 1, curr + [lst[i]]))

        if total <= limit:
            for combi in all_combinations_k(indices, K):
                idxs = set(combi)
                r = self._compute_ratio(idxs)
                if r > best_ratio + 1e-12:
                    best_ratio = r
                    best = idxs
        else:
            # Random sampling fallback
            trials = min(limit, total)
            for _ in range(trials):
                combi = set(random.sample(indices, K))
                r = self._compute_ratio(combi)
                sampled += 1
                if r > best_ratio + 1e-12:
                    best_ratio = r
                    best = combi

        self.optimal_ratio = max(0.0, best_ratio)
        self.optimal_indices = set(best)

    def _get_instructions(self) -> str:
        return (
            "You are playing ClueCrafters.\n"
            "Goal: Select exactly K clues that maximize the information-per-cost ratio.\n"
            "Ratio = (number of distinct constraints covered by your chosen clues) / (sum of their costs).\n"
            "Constraints are units like position-symbol facts, pairwise order, or adjacency exclusions.\n"
            "Rules:\n"
            "- You must select exactly K distinct clue indices from the list.\n"
            "- Use only indices shown; no duplicates allowed.\n"
            "- You can ask to preview the clue list again.\n"
            "\n"
            "Available actions (use \\boxed{...}):\n"
            "- \\boxed{show}: Reprint the available clues and costs.\n"
            "- \\boxed{pick idxs=1,3,5}: Propose a subset of exactly K indices (comma-separated).\n"
            "\n"
            "Feedback:\n"
            "- If your pick is optimal (ties allowed), you succeed.\n"
            "- Valid but non-optimal picks end the episode with neutral reward.\n"
            "- Malformed actions or rule violations end the episode with a small penalty.\n"
            "\n"
            "Example:\n"
            "\\boxed{pick idxs=0,2}\n"
        )

    def get_task_suffix(self) -> str:
        clue_lines = []
        for i, c in enumerate(self.clues):
            clue_lines.append(f"[{i}] cost={c['cost']} :: {c['text']}")
        clue_text = "\n".join(clue_lines)
        return (
            f"Hidden structure size: {self.num_symbols} symbols placed in {self.num_symbols} positions.\n"
            f"K (number of clues to pick): {self.subset_size}\n"
            "Available clues:\n"
            f"{clue_text}\n"
            "Enter your action in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        # Ensure feasibility: subset_size <= num_clues, and minimal values
        self.subset_size = max(1, min(self.subset_size, max(1, self.num_clues - 1)))
        self.num_symbols = max(3, self.num_symbols)
        self.num_clues = max(self.subset_size + 1, self.num_clues)

        self.turn_count = 0
        self.terminated_flag = False
        self._generate_hidden_assignment()
        self._make_clues()
        self._find_optimal_subset()

        # If optimal not found (degenerate), tweak parameters minimally
        if len(self.clues) < self.subset_size:
            # pad with harmless negatives
            while len(self.clues) < self.subset_size:
                pos = random.randrange(self.num_symbols)
                sym = random.choice(self.symbols)
                if self.hidden_assignment[pos] == sym:
                    continue
                self.clues.append({
                    "type": "PosNotSym",
                    "text": f"Position {pos} is not {sym}.",
                    "covers": {("possym", pos, sym)},
                    "cost": 1,
                    "coverage_size": 1
                })
            self._find_optimal_subset()

        obs = self._get_instructions()
        info = {"suffix": self.get_task_suffix()}
        return obs, info

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self.terminated_flag:
            return "Episode already ended.", 0.0, True, False, {"suffix": self.get_task_suffix()}
        self.turn_count += 1

        parsed = self._parse_action(action)
        if not parsed:
            obs = "INVALID ACTION FORMAT: Use \\boxed{show} or \\boxed{pick idxs=a,b,...}."
            self.terminated_flag = True
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        act = parsed.get("action", "")
        if act == "show":
            if self.turn_count >= self.max_turns:
                obs = f"Reached max turns ({self.max_turns})."
                return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}
            obs = "Clue list repeated:\n" + self.get_task_suffix()
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        elif act == "pick":
            idxs_str = parsed.get("idxs", "")
            if not idxs_str:
                obs = "RULE VIOLATION: Missing idxs parameter."
                self.terminated_flag = True
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            # parse indices
            try:
                parts = [p.strip() for p in idxs_str.split(",") if p.strip() != ""]
                chosen = [int(p) for p in parts]
            except Exception:
                obs = "RULE VIOLATION: idxs must be comma-separated integers."
                self.terminated_flag = True
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

            # Validate
            if len(chosen) != self.subset_size:
                obs = f"RULE VIOLATION: You must pick exactly {self.subset_size} indices."
                self.terminated_flag = True
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

            if len(set(chosen)) != len(chosen):
                obs = "RULE VIOLATION: Duplicate indices are not allowed."
                self.terminated_flag = True
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

            if any((i < 0 or i >= len(self.clues)) for i in chosen):
                obs = "RULE VIOLATION: Index out of range."
                self.terminated_flag = True
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

            idxset = set(chosen)
            ratio = self._compute_ratio(idxset)
            # Terminal evaluation
            if abs(ratio - self.optimal_ratio) <= 1e-9:
                covers = set()
                cost = 0
                for i in idxset:
                    covers |= self.clues[i]["covers"]
                    cost += self.clues[i]["cost"]
                obs = (
                    "Success! Your selection achieves the optimal ratio.\n"
                    f"Chosen indices: {sorted(idxset)}\n"
                    f"Coverage units: {len(covers)} | Total cost: {cost} | Ratio: {ratio:.4f}"
                )
                self.terminated_flag = True
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                covers = set()
                cost = 0
                for i in idxset:
                    covers |= self.clues[i]["covers"]
                    cost += self.clues[i]["cost"]
                obs = (
                    "Non-optimal selection. Episode ended.\n"
                    f"Chosen indices: {sorted(idxset)}\n"
                    f"Coverage units: {len(covers)} | Total cost: {cost} | Ratio: {ratio:.4f}\n"
                    f"Optimal ratio was: {self.optimal_ratio:.4f}"
                )
                self.terminated_flag = True
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = "UNSUPPORTED ACTION: Use 'show' or 'pick'."
            self.terminated_flag = True
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
        if not parts:
            return None
        name = parts[0].lower()
        tokens: Dict[str, Any] = {"action": name}
        # Only "show" and "pick" recognized
        if name == "show":
            return tokens
        if name == "pick":
            # Accept forms: "pick idxs=0,2,5"
            for part in parts[1:]:
                if "=" in part:
                    k, v = part.split("=", 1)
                    if k.lower() == "idxs":
                        tokens["idxs"] = v
            return tokens
        return {"action": name}

    def sample_random_action(self) -> str:
        # 50% show, 50% random valid-length pick if possible
        if random.random() < 0.5:
            return r"\boxed{show}"
        if self.clues and self.subset_size > 0 and self.subset_size <= len(self.clues):
            idxs = sorted(random.sample(range(len(self.clues)), self.subset_size))
            s = ",".join(str(i) for i in idxs)
            return rf"\boxed{{pick idxs={s}}}"
        return r"\boxed{show}"


class ClueCraftersEnvWithFeedback(ClueCraftersEnv):
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
            error_detail["issue"] = "missing_or_bad_boxed"
            hint = "Use \\boxed{show} or \\boxed{pick idxs=0,1,...}."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["action"] = "unknown_command"
            hint = "Allowed actions: show, pick."

        elif "rule violation" in text:
            error_type = "ProtocolViolation"
            if "missing idxs" in text:
                error_detail["violation"] = "missing_parameter"
                hint = "Provide indices with idxs=comma_separated_list."
            elif "must pick exactly" in text:
                error_detail["violation"] = "wrong_subset_size"
                hint = "Choose exactly K distinct indices as indicated."
            elif "duplicate indices" in text:
                error_detail["violation"] = "duplicates"
                hint = "List each index once; avoid repeats."
            elif "out of range" in text:
                error_detail["violation"] = "out_of_range"
                hint = "Indices must be within the displayed list bounds."
            else:
                error_detail["violation"] = "general_rule_break"
                hint = "Follow the format and constraints described in the instructions."

        elif "non-optimal selection" in text:
            error_type = "WrongDecision"
            error_detail["outcome"] = "valid_but_not_optimal"
            hint = "Compare costs and aim to cover more distinct constraint units with fewer total cost."

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Decide earlier or use 'show' sparingly."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "num_symbols": getattr(self, "num_symbols", None),
                "subset_size": getattr(self, "subset_size", None),
                "num_clues": len(getattr(self, "clues", [])),
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
            "hint": "Start with \\boxed{show} if you need to re-check clues, then use \\boxed{pick idxs=...}.",
            "turn": 0,
            "state": {
                "num_symbols": getattr(self, "num_symbols", None),
                "subset_size": getattr(self, "subset_size", None),
                "num_clues": len(getattr(self, "clues", [])),
            },
        }
        return obs, info