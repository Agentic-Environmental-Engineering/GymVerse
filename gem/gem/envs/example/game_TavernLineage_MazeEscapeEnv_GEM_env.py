from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class TavernLineageEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 60,
        **_,
    ):
        super().__init__()
        self.complexity = int(max(1, min(10, complexity)))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 60

        # Evolvable parameters
        self.complexity_params = {
            # number of cards in the circle: larger = more states to explore = harder
            'num_cards': (8, 24),
            # number of trap cards: more traps = more dead-ends/misdirection = harder
            'num_traps': (1, 6),
            # number of connectors (non-trap faction cards): more branching reduces certainty but also can help;
            # we tie it to deck size via faction sizes; this parameter controls max faction span per faction (higher = harder search)
            'max_faction_span': (2, 6),
            # visibility budget for peeks: fewer peeks = harder (REVERSED)
            'peek_budget': (10, 4),
            # neighbor reveal size: more neighbors per reveal action makes it easier; scale down (REVERSED)
            'neighbor_reveal_span': (2, 1),
        }

        # Variance settings
        self.param_variance = {
            'num_cards': 1,           # medium discrete
            'num_traps': 1,           # medium discrete
            'max_faction_span': 1,    # medium discrete
            'peek_budget': 1,         # medium discrete, reversed
            'neighbor_reveal_span': 0 # small range (1-2), fix at center per level
        }

        # Placeholder attributes
        self.num_cards: int = 0
        self.num_traps: int = 0
        self.max_faction_span: int = 0
        self.peek_budget: int = 0
        self.neighbor_reveal_span: int = 0

        # Static configuration
        self.cards: List[Dict[str, Any]] = []
        self.start_idx: int = -1
        self.target_idx: int = -1

        # Procedural/bookkeeping
        self.frontier: List[int] = []
        self.visited: set = set()
        self.revealed: set = set()
        self.discovered: bool = False

        # Turn and counters
        self.turn_count: int = 0
        self.remaining_peeks: int = 0
        self.submitted: bool = False
        self.terminated_reason: Optional[str] = None

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for name, (min_val, max_val) in self.complexity_params.items():
            center = min_val + (max_val - min_val) * normalized
            val = center
            if self.enable_param_randomization:
                v = self.param_variance.get(name, 0)
                if v > 0:
                    val = center + random.uniform(-v, v)
            # clamp considering reversed ranges too
            low, high = (min_val, max_val) if min_val <= max_val else (max_val, min_val)
            val = max(low, min(high, val))
            setattr(self, name, int(round(val)))

    def _build_deck(self):
        factions = ["Crowns", "Guild", "Scholars", "Outriders"]
        # Assign factions to non-trap cards; distribute spans
        # Build a circular arrangement
        indices = list(range(self.num_cards))
        random.shuffle(indices)
        # Select trap indices
        trap_indices = set(random.sample(indices, k=min(self.num_traps, self.num_cards - 2)))
        # Ensure at least two non-traps for start/target
        non_trap_pool = [i for i in indices if i not in trap_indices]
        if len(non_trap_pool) < 2:
            # Adjust by removing traps until feasible
            need = 2 - len(non_trap_pool)
            extra = random.sample(list(trap_indices), k=need)
            for x in extra:
                trap_indices.remove(x)
                non_trap_pool.append(x)
        # Initialize cards
        self.cards = []
        for i in range(self.num_cards):
            self.cards.append({
                "type": "trap" if i in trap_indices else "member",
                "faction": None,
                "id": i
            })
        # Assign factions to members
        members = [i for i in range(self.num_cards) if self.cards[i]["type"] == "member"]
        random.shuffle(members)
        # Create faction runs with span up to max_faction_span each, cycling factions
        f_idx = 0
        ptr = 0
        while ptr < len(members):
            span = random.randint(1, max(1, self.max_faction_span))
            take = members[ptr:ptr+span]
            for idx in take:
                self.cards[idx]["faction"] = factions[f_idx % len(factions)]
            f_idx += 1
            ptr += span
        # Choose start and target among members, possibly different factions
        start, target = random.sample([i for i in range(self.num_cards) if self.cards[i]["type"] == "member"], 2)
        self.start_idx = start
        self.target_idx = target
        # Guarantee solvability at generation: with some probability enforce a connecting path via faction-neighbor rules
        # Rule: you can traverse along immediate neighbors in the circle if not trap; you can also jump between any two adjacent same-faction cards if they are adjacent in the circle? We'll define:
        # Allowed moves from card i:
        # - Step to i-1 or i+1 if that neighbor is a member (wrap around).
        # - Additionally, if a neighbor shares faction with current, that connection is strong (still same rule).
        # To ensure solvability, we can carve a simple neighbor path avoiding traps between start and target by flipping trap types along a chosen corridor if needed.
        if not self._has_neighbor_path_member_only(self.start_idx, self.target_idx):
            # Carve a path along shortest circular direction by converting traps on the route to members of appropriate faction
            path = self._shortest_circle_path(self.start_idx, self.target_idx)
            # Convert traps on path to members, assign faction same as predecessor for coherence
            prev_faction = self.cards[self.start_idx]["faction"]
            for idx in path[1:]:
                if self.cards[idx]["type"] == "trap":
                    self.cards[idx]["type"] = "member"
                    self.cards[idx]["faction"] = prev_faction
                else:
                    if self.cards[idx]["faction"] is None:
                        self.cards[idx]["faction"] = prev_faction
                prev_faction = self.cards[idx]["faction"] or prev_faction
        # Final sanity: at least start/target are members
        self.cards[self.start_idx]["type"] = "member"
        if self.cards[self.start_idx]["faction"] is None:
            self.cards[self.start_idx]["faction"] = random.choice(factions)
        self.cards[self.target_idx]["type"] = "member"
        if self.cards[self.target_idx]["faction"] is None:
            self.cards[self.target_idx]["faction"] = random.choice(factions)

    def _neighbors(self, idx: int) -> List[int]:
        left = (idx - 1) % self.num_cards
        right = (idx + 1) % self.num_cards
        span = self.neighbor_reveal_span  # span 1 means only immediate left/right; span 2 would include two steps away
        nb = set()
        for k in range(1, span + 1):
            nb.add((idx - k) % self.num_cards)
            nb.add((idx + k) % self.num_cards)
        return list(sorted(nb))

    def _allowed_moves(self, idx: int) -> List[int]:
        # Allowed to move only onto member cards within neighbor span; traps block moves
        return [j for j in self._neighbors(idx) if self.cards[j]["type"] == "member"]

    def _has_neighbor_path_member_only(self, s: int, t: int) -> bool:
        # BFS strictly via _allowed_moves (member-only moves)
        seen = set([s])
        q = [s]
        while q:
            cur = q.pop(0)
            if cur == t:
                return True
            for nb in self._allowed_moves(cur):
                if nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        return False

    def _shortest_circle_path(self, s: int, t: int) -> List[int]:
        # Return indices along the shorter direction around circle from s to t inclusive
        if s == t:
            return [s]
        n = self.num_cards
        # clockwise
        cw = []
        i = s
        while True:
            cw.append(i)
            if i == t:
                break
            i = (i + 1) % n
        # counterclockwise
        ccw = []
        i = s
        while True:
            ccw.append(i)
            if i == t:
                break
            i = (i - 1) % n
        return cw if len(cw) <= len(ccw) else ccw

    def _get_instructions(self) -> str:
        return (
            "You are in the Tavern Lineage: a circle of hidden cards represents factions and traps.\n"
            "Your goal is to determine if there exists a safe influence path from the start card to the target card.\n"
            "Movement rules for path existence: A path exists if you can traverse from start to target by stepping only onto member cards within the neighbor span (wrap-around circle). Traps block movement.\n"
            "Available actions:\n"
            "- peek index=<i>: Reveal the card type and faction at index i. Consumes 1 peek. Limited by peek_budget.\n"
            "- reveal_neighbors index=<i>: Reveal identities of cards within neighbor span around i. Does not consume peeks.\n"
            "- init_search: Initialize internal search by setting the frontier to [start].\n"
            "- advance: Pop one index from the frontier and enqueue its allowable neighbor moves (member-only). Marks visited.\n"
            "- status: Show your known reveals, visited count, frontier size, remaining peeks.\n"
            "- submit answer=<YES|NO>: Submit final decision about reachability. Exactly one submission ends the game.\n"
            "Notes:\n"
            "- Indices are 0 to num_cards-1 arranged in a circle.\n"
            "- You must use \\boxed{...} to send actions. Example: \\boxed{peek index=3}\n"
            "- The observation will include the start and target indices.\n"
            "Rewards: 1.0 for correct submit, 0.0 for incorrect or timeout, format errors end the game with a penalty.\n"
            f"Example action: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        revealed_list = sorted(list(self.revealed))
        status = (
            f"Turn={self.turn_count} | Start={self.start_idx} Target={self.target_idx} | "
            f"Frontier={list(self.frontier)} | Visited={len(self.visited)} | "
            f"Revealed={revealed_list} | RemainingPeeks={self.remaining_peeks} | "
            f"NeighborSpan={self.neighbor_reveal_span} | DeckSize={self.num_cards}"
        )
        return status + "\nEnter your action in \\boxed{...} format."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        self.turn_count = 0
        self.frontier = []
        self.visited = set()
        self.revealed = set()
        self.discovered = False
        self.submitted = False
        self.terminated_reason = None

        self._build_deck()
        self.remaining_peeks = self.peek_budget

        # Auto-reveal start and target to give anchor points
        self.revealed.add(self.start_idx)
        self.revealed.add(self.target_idx)

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False
        reward = 0.0

        parsed = self._parse_action(action)
        if not parsed:
            obs = "INVALID ACTION FORMAT: Use \\boxed{...} with a supported action."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        name = parsed.get("action", "")
        obs = ""
        protocol_error = False

        if name == "peek":
            idx_str = parsed.get("index")
            if idx_str is None or not idx_str.isdigit():
                obs = "ERROR: Missing or invalid index for peek."
                protocol_error = True
            else:
                idx = int(idx_str)
                if idx < 0 or idx >= self.num_cards:
                    obs = "ERROR: Index out of range."
                    protocol_error = True
                elif self.remaining_peeks <= 0:
                    obs = "ERROR: No peeks remaining."
                    protocol_error = True
                else:
                    self.remaining_peeks -= 1
                    self.revealed.add(idx)
                    c = self.cards[idx]
                    obs = f"PEEK RESULT: index={idx} type={c['type']} faction={c['faction']} remaining_peeks={self.remaining_peeks}"

        elif name == "reveal_neighbors":
            idx_str = parsed.get("index")
            if idx_str is None or not idx_str.isdigit():
                obs = "ERROR: Missing or invalid index for reveal_neighbors."
                protocol_error = True
            else:
                idx = int(idx_str)
                if idx < 0 or idx >= self.num_cards:
                    obs = "ERROR: Index out of range."
                    protocol_error = True
                else:
                    nbs = self._neighbors(idx)
                    reveal_info = []
                    for j in nbs:
                        self.revealed.add(j)
                        cj = self.cards[j]
                        reveal_info.append(f"{j}:{cj['type']}:{cj['faction']}")
                    obs = "REVEAL NEIGHBORS: " + ", ".join(reveal_info)

        elif name == "init_search":
            self.frontier = [self.start_idx]
            self.visited = set()
            obs = f"SEARCH INITIALIZED: frontier={[self.start_idx]} visited=0"

        elif name == "advance":
            if not self.frontier:
                obs = "ERROR: Frontier is empty. Use init_search or reveal to guide exploration."
                protocol_error = True
            else:
                cur = self.frontier.pop(0)
                if cur in self.visited:
                    obs = f"ADVANCE: skipped already visited {cur}; frontier={self.frontier}"
                else:
                    self.visited.add(cur)
                    self.revealed.add(cur)
                    if cur == self.target_idx:
                        self.discovered = True
                    moves = self._allowed_moves(cur)
                    # enqueue unseen moves
                    enqueue = [m for m in moves if m not in self.visited and m not in self.frontier]
                    self.frontier.extend(enqueue)
                    # reveal neighbors passively
                    for m in enqueue:
                        self.revealed.add(m)
                    obs = f"ADVANCE: at={cur} enqueued={enqueue} frontier={self.frontier} visited={len(self.visited)}"

        elif name == "status":
            obs = (
                f"STATUS: start={self.start_idx} target={self.target_idx} "
                f"frontier={self.frontier} visited={len(self.visited)} "
                f"revealed={sorted(list(self.revealed))} remaining_peeks={self.remaining_peeks}"
            )

        elif name == "submit":
            if self.submitted:
                obs = "ERROR: Already submitted."
                protocol_error = True
            else:
                ans = parsed.get("answer", "")
                ans_norm = ans.strip().upper()
                self.submitted = True
                # True label computed from hidden graph reachability
                correct = self._has_neighbor_path_member_only(self.start_idx, self.target_idx)
                if ans_norm not in ["YES", "NO"]:
                    obs = "ERROR: answer must be YES or NO."
                    protocol_error = True
                    self.submitted = False
                else:
                    guess = (ans_norm == "YES")
                    if guess == correct:
                        obs = f"Success! Correct answer: {ans_norm}. Path existence is {correct}."
                        return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
                    else:
                        obs = f"Failed! Wrong answer: {ans_norm}. Path existence is {correct}."
                        return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = "UNSUPPORTED ACTION: Use one of [peek, reveal_neighbors, init_search, advance, status, submit]."
            # Unsupported action ends episode as invalid request
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        if protocol_error:
            # Protocol violations terminate with 0.0 (not format error)
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            obs_timeout = f"TIMEOUT: Reached max turns ({self.max_turns})."
            return obs_timeout, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

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
        tokens["action"] = parts[0]
        for p in parts[1:]:
            if "=" in p:
                k, v = p.split("=", 1)
                tokens[k] = v
        return tokens

    def sample_random_action(self) -> str:
        choices = []
        if self.remaining_peeks > 0:
            idx = random.randint(0, max(0, self.num_cards - 1)) if self.num_cards > 0 else 0
            choices.append(rf"\boxed{{peek index={idx}}}")
        idx2 = random.randint(0, max(0, self.num_cards - 1)) if self.num_cards > 0 else 0
        choices.append(rf"\boxed{{reveal_neighbors index={idx2}}}")
        choices.append(r"\boxed{init_search}")
        choices.append(r"\boxed{advance}")
        choices.append(r"\boxed{status}")
        choices.append(r"\boxed{submit answer=YES}")
        return random.choice(choices)


class TavernLineageEnvWithFeedback(TavernLineageEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        text = obs.lower()
        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint = None

        if "invalid action format" in text:
            error_type = "FormatError"
            error_detail["issue"] = "missing_or_bad_boxed"
            hint = 'Wrap your action like \\boxed{peek index=3}'

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["allowed"] = ["peek", "reveal_neighbors", "init_search", "advance", "status", "submit"]
            hint = "Use a supported function name from the instructions."

        elif "timeout: reached max turns" in text:
            error_type = "Timeout"
            error_detail["max_turns"] = self.max_turns
            hint = "Act earlier: initialize search, advance, and submit before turns run out."

        elif text.startswith("error:"):
            error_type = "ProtocolViolation"
            if "index out of range" in text:
                error_detail["violation"] = "index_range"
                hint = f"Use indices 0..{self.num_cards-1}."
            elif "no peeks remaining" in text:
                error_detail["violation"] = "peek_budget_exceeded"
                hint = "Use reveal_neighbors or advance instead of peek when budget is 0."
            elif "missing or invalid index for peek" in text:
                error_detail["violation"] = "bad_peek_args"
                hint = "Provide a numeric index: \\boxed{peek index=5}"
            elif "missing or invalid index for reveal_neighbors" in text:
                error_detail["violation"] = "bad_reveal_args"
                hint = "Provide a numeric index: \\boxed{reveal_neighbors index=2}"
            elif "frontier is empty" in text:
                error_detail["violation"] = "advance_without_init"
                hint = "Call \\boxed{init_search} before \\boxed{advance}."
            elif "already submitted" in text:
                error_detail["violation"] = "double_submit"
                hint = "You can submit only once. Decide carefully after investigating."
            elif "answer must be yes or no" in text:
                error_detail["violation"] = "bad_submit_answer"
                hint = "Use \\boxed{submit answer=YES} or \\boxed{submit answer=NO}"

        elif "failed! wrong answer" in text:
            error_type = "WrongDecision"
            # we can hint based on whether discovered flag was true or not
            detail = {}
            detail["frontier_size"] = len(getattr(self, "frontier", []))
            detail["visited"] = len(getattr(self, "visited", []))
            error_detail = detail
            hint = "Use init_search and advance to verify reachability before submitting."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            error_detail["turn"] = getattr(self, "turn_count", None)
            error_detail["remaining_peeks"] = getattr(self, "remaining_peeks", None)
            error_detail["neighbor_span"] = getattr(self, "neighbor_reveal_span", None)
            error_detail["deck_size"] = getattr(self, "num_cards", None)
            diagnostic["error_detail"] = error_detail
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {
                "outcome": "episode_start",
                "turn": 0,
                "remaining_peeks": self.remaining_peeks,
                "neighbor_span": self.neighbor_reveal_span,
                "deck_size": self.num_cards
            },
            "hint": "Start by revealing neighbors near the start index, then init_search and advance.",
        }
        return obs, info