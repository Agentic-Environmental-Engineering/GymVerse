from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class RunicPotionGameScalediffEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = False,
        max_turns: Optional[int] = 30,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 30

        # Evolvable parameters
        self.complexity_params = {
            "num_pages": (3, 13),            # strict L1/L10 endpoints
            "page_length": (3, 9),
            "rune_types": (4, 8),
            "rules_active": (1, 4),
        }
        self.param_variance = {
            "num_pages": 0,
            "page_length": 0,
            "rune_types": 0,
            "rules_active": 0,
        }

        # Placeholder attributes populated by _apply_complexity_params
        self.num_pages: int = 0
        self.page_length: int = 0
        self.rune_types: int = 0
        self.rules_active: int = 0

        # State
        self.turn_count: int = 0
        self.pages: Any = None
        self.runes: Any = None
        self.rune_values: Dict[str, int] = {}
        self.target_page_idx: int = 0
        self.focus_start: Optional[int] = None
        self.focus_end: Optional[int] = None
        self.summary_counts: Dict[str, int] = {}
        self.incumbent_total: int = 0
        self.active_rules: Tuple[str, ...] = ()
        self.reference_value: int = 0

        self.reset()

    def _apply_complexity_params(self):
        table = {

            1: (3, 3, 4, 1),

            2: (5, 4, 5, 1),

            3: (6, 5, 5, 2),

            4: (7, 5, 6, 2),

            5: (9, 6, 6, 2),

            6: (10, 7, 7, 3),

            7: (11, 8, 7, 3),

            8: (11, 8, 8, 4),

            9: (12, 9, 8, 4),

            10: (13, 10, 8, 4),

        }

        level = int(self.complexity)
        params = table.get(level, table[max(table.keys())])
        (num_pages, page_length, rune_types, rules_active) = params

    def _generate_instance(self):
        all_runes = ["A", "B", "C", "D", "E", "F", "G", "H"]
        self.runes = all_runes[: self.rune_types]
        self.rune_values = {r: i + 1 for i, r in enumerate(all_runes)}  # fixed mapping A=1..H=8
        self.pages = []
        for _ in range(self.num_pages):
            page = [random.choice(self.runes) for __ in range(self.page_length)]
            self.pages.append(page)
        self.target_page_idx = random.randint(1, self.num_pages)

        rule_pool = [
            "GLOBAL_E_COUNT",        # +count of E across the whole book
            "HAZARD_H_PENALTY",      # -count of H across the whole book
            "PRIME_PAGE_BONUS",      # +3 if target page index is prime
            "ADJACENT_SHARED_BONUS", # +2 per neighbor sharing any rune with target page
            "UNIQUE_TYPES_BONUS",    # +3 if target page has at least 3 unique runes
            "VOWEL_BONUS",           # +3 if vowels (A,E) > consonants on target page
            "EVEN_MULTIPLIER",       # x2 if even-valued sum > odd-valued sum on target page
        ]
        # Sample without replacement; ensure at least 1 rule
        k = max(1, min(len(rule_pool), self.rules_active))
        self.active_rules = tuple(random.sample(rule_pool, k))

        self._update_focus(None, None)
        self.incumbent_total = 0
        self.reference_value = self._compute_reference()

    def _update_focus(self, start: Optional[int], end: Optional[int]):
        self.focus_start = start
        self.focus_end = end
        self.summary_counts = {}
        if start is not None and end is not None:
            s = max(1, min(self.num_pages, int(start)))
            e = max(1, min(self.num_pages, int(end)))
            if s > e:
                s, e = e, s
            for idx in range(s - 1, e):
                for r in self.pages[idx]:
                    self.summary_counts[r] = self.summary_counts.get(r, 0) + 1

    def _is_prime(self, n: int) -> bool:
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        d = 3
        while d * d <= n:
            if n % d == 0:
                return False
            d += 2
        return True

    def _count_global(self, rune: str) -> int:
        c = 0
        for p in self.pages:
            for r in p:
                if r == rune:
                    c += 1
        return c

    def _compute_reference(self) -> int:
        page = self.pages[self.target_page_idx - 1]
        base_sum = sum(self.rune_values[r] for r in page)
        additions = 0
        subtractions = 0
        multiplier = 1

        if "GLOBAL_E_COUNT" in self.active_rules and "E" in self.runes:
            additions += self._count_global("E")

        if "PRIME_PAGE_BONUS" in self.active_rules:
            if self._is_prime(self.target_page_idx):
                additions += 3

        if "UNIQUE_TYPES_BONUS" in self.active_rules:
            if len(set(page)) >= 3:
                additions += 3

        if "VOWEL_BONUS" in self.active_rules:
            vowels = sum(1 for r in page if r in ("A", "E"))
            consonants = len(page) - vowels
            if vowels > consonants:
                additions += 3

        if "ADJACENT_SHARED_BONUS" in self.active_rules:
            target_set = set(page)
            neighbors = []
            if self.target_page_idx - 1 >= 1:
                neighbors.append(self.pages[self.target_page_idx - 2])
            if self.target_page_idx + 1 <= self.num_pages:
                neighbors.append(self.pages[self.target_page_idx])
            for nb in neighbors:
                if target_set.intersection(set(nb)):
                    additions += 2

        if "HAZARD_H_PENALTY" in self.active_rules and "H" in self.runes:
            subtractions += self._count_global("H")

        pre_total = base_sum + additions - subtractions

        if "EVEN_MULTIPLIER" in self.active_rules:
            even_sum = sum(self.rune_values[r] for r in page if self.rune_values[r] % 2 == 0)
            odd_sum = sum(self.rune_values[r] for r in page if self.rune_values[r] % 2 == 1)
            if even_sum > odd_sum:
                multiplier = 2
            else:
                multiplier = 1

        return pre_total * multiplier

    def _get_instructions(self) -> str:
        rule_descs = {
            "GLOBAL_E_COUNT": "Add the total count of rune E across all pages.",
            "HAZARD_H_PENALTY": "Subtract the total count of rune H across all pages.",
            "PRIME_PAGE_BONUS": "Add 3 if the target page index is prime.",
            "ADJACENT_SHARED_BONUS": "Add 2 for each neighboring page that shares any rune with the target page.",
            "UNIQUE_TYPES_BONUS": "Add 3 if the target page has at least 3 unique runes.",
            "VOWEL_BONUS": "Add 3 if vowels (A,E) outnumber consonants on the target page.",
            "EVEN_MULTIPLIER": "Finally, multiply by 2 if even-valued runes sum exceeds odd-valued runes sum on the target page.",
        }
        active_list = "\n".join(f"- {r}: {rule_descs[r]}" for r in self.active_rules)
        mapping = ", ".join(f"{r}={self.rune_values[r]}" for r in self.runes)
        return (
            "Runic Potion Game:\n"
            "Goal: compute and submit the final potency of the target page using the deterministic rules.\n"
            f"- Pages: {self.num_pages}, each with {self.page_length} runes chosen from {self.rune_types} types.\n"
            f"- Target page index: {self.target_page_idx}.\n"
            f"- Rune values: {mapping}.\n"
            "Active modification rules:\n"
            f"{active_list}\n"
            "Actions:\n"
            "- open i: focus on page i\n"
            "- window i-j: focus on pages i..j\n"
            "- peek: show runes in current focus\n"
            "- tally X: count rune X in current focus\n"
            "- compute_base: set total to base sum of target page (requires focus on target)\n"
            "- add n: add n to total\n"
            "- set n: set total to n\n"
            "- check: show current total\n"
            "- rules: list active rules\n"
            "- info: show book summary\n"
            "- submit n: finalize answer\n"
            "Use \\boxed{...} to send actions. Example: "
            f"{self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        focus_desc = "none" if self.focus_start is None else f"[{self.focus_start}-{self.focus_end}]"
        total_desc = f"{self.incumbent_total}"
        remaining = self.max_turns - self.turn_count
        return (
            f"Status: turn={self.turn_count}, remaining={remaining}, focus={focus_desc}, total={total_desc}.\n"
            "Enter your action in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()
        self.turn_count = 0
        self._generate_instance()
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if not parsed:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}"
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        cmd = parsed["cmd"]
        args = parsed.get("args", [])

        reward = 0.0
        text = ""

        if cmd == "open":
            i = args[0] if args else None
            if i is None or not isinstance(i, int):
                text = "Protocol violation: open requires an integer page index."
                return text, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if i < 1 or i > self.num_pages:
                text = f"Protocol violation: page index out of range (1..{self.num_pages})."
                return text, -1.0, True, False, {"suffix": self.get_task_suffix()}
            self._update_focus(i, i)
            page = self.pages[i - 1]
            text = f"Opened page {i}: {' '.join(page)}. Focus set to [{i}-{i}]."

        elif cmd == "window":
            if len(args) != 2 or not all(isinstance(x, int) for x in args):
                text = "Protocol violation: window requires two integer indices i j."
                return text, -1.0, True, False, {"suffix": self.get_task_suffix()}
            i, j = args
            if i < 1 or j < 1 or i > self.num_pages or j > self.num_pages:
                text = f"Protocol violation: window indices out of range (1..{self.num_pages})."
                return text, -1.0, True, False, {"suffix": self.get_task_suffix()}
            s, e = (i, j) if i <= j else (j, i)
            self._update_focus(s, e)
            counts = ", ".join(f"{r}:{self.summary_counts.get(r,0)}" for r in self.runes)
            text = f"Focused window [{s}-{e}]. Summary counts: {counts}."

        elif cmd == "peek":
            if self.focus_start is None:
                text = "Protocol violation: no focus set. Use open i or window i-j first."
                return text, -1.0, True, False, {"suffix": self.get_task_suffix()}
            pages_text = []
            for idx in range(self.focus_start - 1, self.focus_end):
                page = self.pages[idx]
                pages_text.append(f"{idx+1}: {' '.join(page)}")
            text = "Peek:\n" + "\n".join(pages_text)

        elif cmd == "tally":
            r = args[0] if args else None
            if r is None or not isinstance(r, str):
                text = "Protocol violation: tally requires a rune letter."
                return text, -1.0, True, False, {"suffix": self.get_task_suffix()}
            r = r.upper()
            if r not in self.runes:
                text = f"Protocol violation: unknown rune '{r}'. Allowed: {' '.join(self.runes)}."
                return text, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if self.focus_start is None:
                text = "Protocol violation: no focus set. Use open or window first."
                return text, -1.0, True, False, {"suffix": self.get_task_suffix()}
            c = self.summary_counts.get(r, 0)
            text = f"Tally in focus [{self.focus_start}-{self.focus_end}]: {r}={c}."

        elif cmd == "compute_base":
            if self.focus_start is None or not (self.focus_start == self.target_page_idx and self.focus_end == self.target_page_idx):
                text = f"Protocol violation: compute_base requires focus on target page {self.target_page_idx}."
                return text, -1.0, True, False, {"suffix": self.get_task_suffix()}
            page = self.pages[self.target_page_idx - 1]
            base_sum = sum(self.rune_values[r] for r in page)
            self.incumbent_total = base_sum
            text = f"Computed base sum on target page {self.target_page_idx}: total set to {base_sum}."

        elif cmd == "add":
            n = args[0] if args else None
            if n is None or not isinstance(n, int):
                text = "Protocol violation: add requires an integer."
                return text, -1.0, True, False, {"suffix": self.get_task_suffix()}
            self.incumbent_total += n
            text = f"Added {n}. Total is now {self.incumbent_total}."

        elif cmd == "set":
            n = args[0] if args else None
            if n is None or not isinstance(n, int):
                text = "Protocol violation: set requires an integer."
                return text, -1.0, True, False, {"suffix": self.get_task_suffix()}
            self.incumbent_total = n
            text = f"Set total to {self.incumbent_total}."

        elif cmd == "check":
            text = f"Current total: {self.incumbent_total}."

        elif cmd == "rules":
            rule_descs = {
                "GLOBAL_E_COUNT": "Add the total count of rune E across all pages.",
                "HAZARD_H_PENALTY": "Subtract the total count of rune H across all pages.",
                "PRIME_PAGE_BONUS": "Add 3 if the target page index is prime.",
                "ADJACENT_SHARED_BONUS": "Add 2 per neighbor sharing any rune with the target page.",
                "UNIQUE_TYPES_BONUS": "Add 3 if target page has ≥3 unique runes.",
                "VOWEL_BONUS": "Add 3 if vowels (A,E) outnumber consonants on target page.",
                "EVEN_MULTIPLIER": "Multiply by 2 if even sum > odd sum on target page.",
            }
            lines = [f"- {r}: {rule_descs[r]}" for r in self.active_rules]
            text = "Active rules:\n" + "\n".join(lines)

        elif cmd == "info":
            mapping = ", ".join(f"{r}={self.rune_values[r]}" for r in self.runes)
            text = (
                f"Book info: pages={self.num_pages}, page_length={self.page_length}, runes={self.rune_types} ({' '.join(self.runes)}), "
                f"mapping: {mapping}, target={self.target_page_idx}."
            )

        elif cmd == "help":
            text = (
                "Valid actions: open i | window i-j | peek | tally X | compute_base | add n | set n | check | rules | info | submit n"
            )

        elif cmd == "submit":
            n = args[0] if args else None
            if n is None or not isinstance(n, int):
                text = "Protocol violation: submit requires an integer answer."
                return text, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if n == self.reference_value:
                text = f"Success! Correct potency {n}."
                return text, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                text = f"Failed! Wrong potency {n}. Correct was {self.reference_value}."
                return text, -1.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            text = f"Unsupported action '{cmd}'."
            return text, -1.0, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return text, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Any]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r"\\boxed\{(.+?)\}", re.IGNORECASE | re.DOTALL)
        m = list(pattern.finditer(action))
        if not m:
            return None
        content = m[-1].group(1).strip()
        tokens = content.split()
        if not tokens:
            return None
        cmd = tokens[0].lower()
        args = []
        try:
            if cmd == "open":
                if len(tokens) >= 2:
                    args = [int(tokens[1])]
            elif cmd == "window":
                if len(tokens) >= 2:
                    rng = tokens[1]
                    if "-" in rng:
                        parts = rng.split("-")
                        if len(parts) == 2:
                            a = int(parts[0])
                            b = int(parts[1])
                            args = [a, b]
                    elif len(tokens) >= 3:
                        a = int(tokens[1])
                        b = int(tokens[2])
                        args = [a, b]
            elif cmd == "tally":
                if len(tokens) >= 2:
                    args = [tokens[1].upper()]
            elif cmd in ("add", "set", "submit"):
                if len(tokens) >= 2:
                    args = [int(tokens[1])]
            elif cmd in ("peek", "compute_base", "check", "rules", "info", "help"):
                args = []
            else:
                args = []
        except Exception:
            return {"cmd": cmd, "args": []}
        return {"cmd": cmd, "args": args}

    def sample_random_action(self) -> str:
        cmds = []
        if self.num_pages >= 1:
            i = random.randint(1, self.num_pages)
            j = random.randint(1, self.num_pages)
            if i > j:
                i, j = j, i
            cmds.append(f"open {i}")
            cmds.append(f"window {i}-{j}")
        if self.runes:
            r = random.choice(self.runes)
            cmds.append(f"tally {r}")
        cmds.extend(["peek", "compute_base", "check", "rules", "info", "help"])
        choice = random.choice(cmds)
        return f"\\boxed{{{choice}}}"


class RunicPotionGameScalediffEnvWithFeedback(RunicPotionGameScalediffEnv):
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
            hint = "Wrap your command like \\boxed{open 1}."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "open requires an integer" in text:
                error_detail["violation"] = "open_arg_type"
                hint = "Use an integer index: \\boxed{open 2}."
            elif "page index out of range" in text:
                error_detail["violation"] = "open_out_of_range"
                hint = f"Choose 1..{self.num_pages}."
            elif "window requires two integer" in text:
                error_detail["violation"] = "window_args"
                hint = "Provide two integers: \\boxed{window 2-4}."
            elif "window indices out of range" in text:
                error_detail["violation"] = "window_out_of_range"
                hint = f"Use indices within 1..{self.num_pages}."
            elif "no focus set" in text and "peek" in text:
                error_detail["violation"] = "peek_no_focus"
                hint = "Open a page first: \\boxed{open 1}."
            elif "no focus set" in text and "tally" in text:
                error_detail["violation"] = "tally_no_focus"
                hint = "Set a focus window, then tally."
            elif "unknown rune" in text:
                error_detail["violation"] = "tally_unknown_rune"
                hint = f"Use one of: {' '.join(self.runes)}."
            elif "compute_base requires focus on target page" in text:
                error_detail["violation"] = "compute_base_wrong_focus"
                hint = f"Open target page {self.target_page_idx} with \\boxed{{open {self.target_page_idx}}} then \\boxed{{compute_base}}."
            elif "add requires an integer" in text:
                error_detail["violation"] = "add_arg_type"
                hint = "Use integer, e.g., \\boxed{add 3}."
            elif "set requires an integer" in text:
                error_detail["violation"] = "set_arg_type"
                hint = "Use integer, e.g., \\boxed{set 10}."
            elif "submit requires an integer" in text:
                error_detail["violation"] = "submit_arg_type"
                hint = "Finish with \\boxed{submit 42}."
            else:
                error_detail["violation"] = "generic_protocol"
                hint = "Check the help: \\boxed{help}."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["action"] = "unknown_command"
            hint = "Use one of: open, window, peek, tally, compute_base, add, set, check, rules, info, submit, help."

        elif "failed! wrong potency" in text:
            error_type = "WrongDecision"
            try:
                got = int(re.findall(r"wrong potency (\d+)", text)[0])
            except Exception:
                got = None
            error_detail["got"] = got
            error_detail["expected"] = self.reference_value
            hint = "Recompute carefully: base from target page, then apply all listed rules."

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["turn_limit"] = self.max_turns
            hint = "Act more decisively: open target page, compute_base, then apply rule adjustments and submit."

        elif "success! correct potency" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "focus": None if self.focus_start is None else [self.focus_start, self.focus_end],
                "total": self.incumbent_total,
                "target": self.target_page_idx,
                "rules_active": list(self.active_rules),
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
            "hint": f"Start by opening the target page: \\boxed{{open {self.target_page_idx}}}",
            "turn": 0,
            "state": {
                "focus": None,
                "total": self.incumbent_total,
                "target": self.target_page_idx,
                "rules_active": list(self.active_rules),
            },
        }
        return obs, info
