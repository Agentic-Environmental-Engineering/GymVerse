from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List, Set


class DetectiveMysteryEnv(Env):
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

        self.complexity_params = {
            # Number of suspects: more suspects increases search space and difficulty
            "num_suspects": (3, 10),
            # Number of red herring clues: more non-informative clues increases cognitive load
            "num_red_herrings": (0, 3),
            # REVERSED: clue detail level (3=high detail, 1=low detail); lower detail is harder
            "clue_detail_level": (3, 1),
            # REVERSED: allowed number of direct reveal clues (weapon/location); fewer is harder
            "allowed_direct_reveals": (2, 0),
            # Number of informative constraints required before uniqueness; higher is harder
            "required_constraints": (2, 5),
        }

        self.param_variance = {
            "num_suspects": 1,
            "num_red_herrings": 1,
            "clue_detail_level": 0,
            "allowed_direct_reveals": 0,
            "required_constraints": 1,
        }

        self.turn_count: int = 0

        self.suspects: List[str] = []
        self.attributes: Dict[str, Dict[str, str]] = {}
        self.culprit: Optional[str] = None

        self.eliminated: Set[str] = set()

        self.clues: List[Dict[str, Any]] = []
        self.informative_clues: List[Dict[str, Any]] = []
        self.clue_index: int = 0

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

    def _get_instructions(self) -> str:
        actions = [
            "- list_suspects",
            "- view_attributes NAME",
            "- get_clue",
            "- summarize",
            "- eliminate NAME",
            "- accuse NAME (terminal)",
        ]
        example = self.sample_random_action()
        return (
            "You are the lead detective in a deduction mystery.\n"
            "Goal: Identify and accuse the correct culprit among the suspects.\n"
            "Use actions to gather clues, inspect suspects, and eliminate possibilities before accusing.\n"
            "Available actions:\n"
            + "\n".join(actions)
            + "\n"
            "Action format: use \\boxed{...} with a single command per turn.\n"
            f"For example: {example}\n"
        )

    def get_task_suffix(self) -> str:
        remaining = [s for s in self.suspects if s not in self.eliminated]
        seen = self.clue_index
        return (
            f"Turn: {self.turn_count}\n"
            f"Suspects: {', '.join(self.suspects)}\n"
            f"Eliminated: {', '.join(sorted(self.eliminated)) if self.eliminated else '(none)'}\n"
            f"Clues seen: {seen}/{len(self.clues)}\n"
            "Enter your action in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()
        self.turn_count = 0
        self.eliminated = set()
        self.clue_index = 0

        names_pool = ["Alicia", "Bruno", "Carla", "Diego", "Elena", "Felix", "Grace", "Hector", "Iris", "Jonas"]
        random.shuffle(names_pool)
        self.suspects = names_pool[: self.num_suspects]

        weapons = ["Knife", "Rope", "Poison", "Candlestick", "Revolver", "Wrench"]
        locations = ["Study", "Kitchen", "Ballroom", "Library", "Conservatory", "Hall"]
        alibis = ["Strong", "Medium", "Weak"]

        self.attributes = {}
        for s in self.suspects:
            self.attributes[s] = {
                "weapon": random.choice(weapons),
                "location": random.choice(locations),
                "alibi": random.choice(alibis),
            }

        self.culprit = random.choice(self.suspects)
        # Ensure culprit is not trivially impossible (avoid Strong alibi)
        if self.attributes[self.culprit]["alibi"] == "Strong":
            self.attributes[self.culprit]["alibi"] = random.choice(["Medium", "Weak"])

        self._build_clue_sequence()

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if parsed is None:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}} with a single command."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        cmd = parsed["cmd"]
        arg = parsed.get("arg")

        if cmd == "list_suspects":
            obs = (
                f"At turn {self.turn_count}, suspects are: {', '.join(self.suspects)}."
                " Use view_attributes NAME or get_clue to proceed."
            )
            reward = 0.0

        elif cmd == "view_attributes":
            if not arg or arg not in self.suspects:
                obs = f"At turn {self.turn_count}, unsupported suspect for view_attributes."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            att = self.attributes[arg]
            detail = self.clue_detail_level
            if detail >= 3:
                text = f"{arg}'s attributes: weapon={att['weapon']}, location={att['location']}, alibi={att['alibi']}."
            elif detail == 2:
                text = f"{arg}: weapon={att['weapon']}, location={att['location']}."
            else:
                text = f"{arg}: alibi={att['alibi']}."
            obs = f"At turn {self.turn_count}, {text}"
            reward = 0.0

        elif cmd == "get_clue":
            if self.clue_index >= len(self.clues):
                obs = f"At turn {self.turn_count}, no more clues remain. Consider summarize or accuse."
                reward = 0.0
            else:
                clue = self.clues[self.clue_index]
                self.clue_index += 1
                obs = f"At turn {self.turn_count}, Clue #{self.clue_index}: {clue['text']}"
                reward = 0.0

        elif cmd == "summarize":
            possible = self._compute_possible_suspects()
            remaining = [s for s in possible if s not in self.eliminated]
            detail = self.clue_detail_level
            if detail >= 3:
                text = (
                    f"Possible based on seen clues: {', '.join(possible) if possible else '(none)'}; "
                    f"After eliminations: {', '.join(remaining) if remaining else '(none)'}."
                )
            elif detail == 2:
                text = f"Remaining candidates: {', '.join(remaining) if remaining else '(none)'}."
            else:
                text = f"Candidates: {len(remaining)}."
            obs = f"At turn {self.turn_count}, {text}"
            reward = 0.0

        elif cmd == "eliminate":
            if not arg or arg not in self.suspects:
                obs = f"At turn {self.turn_count}, unsupported suspect for eliminate."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            if arg in self.eliminated:
                obs = f"At turn {self.turn_count}, {arg} was already eliminated."
            else:
                self.eliminated.add(arg)
                obs = f"At turn {self.turn_count}, you eliminated {arg} from consideration."
            reward = 0.0

        elif cmd == "accuse":
            if not arg or arg not in self.suspects:
                obs = f"At turn {self.turn_count}, unsupported suspect for accuse."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            if arg == self.culprit:
                obs = f"Success! {arg} is the culprit. Case closed."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Failed! {arg} is innocent. The culprit was {self.culprit}."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = f"At turn {self.turn_count}, unknown command '{cmd}'."
            return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns}). Episode timed out."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, str]]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        m = pattern.search(action)
        if not m:
            return None
        inner = m.group(1).strip()
        if not inner:
            return None
        parts = inner.split()
        cmd = parts[0].strip().lower()
        arg = " ".join(parts[1:]).strip() if len(parts) > 1 else ""
        return {"cmd": cmd, "arg": arg} if cmd else None

    def sample_random_action(self) -> str:
        choices = []
        if self.suspects:
            choices.append(f"\\boxed{{view_attributes {random.choice(self.suspects)}}}")
            choices.append(f"\\boxed{{eliminate {random.choice(self.suspects)}}}")
            choices.append(f"\\boxed{{accuse {random.choice(self.suspects)}}}")
        choices.extend(["\\boxed{list_suspects}", "\\boxed{get_clue}", "\\boxed{summarize}"])
        return random.choice(choices)

    def _build_clue_sequence(self):
        # Build informative clues until uniqueness occurs at required_constraints
        culprit = self.culprit
        c_att = self.attributes[culprit]
        direct_pool = []
        if self.allowed_direct_reveals > 0:
            direct_pool.append(
                {
                    "text": f"The culprit used the {c_att['weapon']}.",
                    "filter": lambda s: self.attributes[s]["weapon"] == c_att["weapon"],
                    "kind": "informative",
                }
            )
        if self.allowed_direct_reveals > 1:
            direct_pool.append(
                {
                    "text": f"The culprit was in the {c_att['location']}.",
                    "filter": lambda s: self.attributes[s]["location"] == c_att["location"],
                    "kind": "informative",
                }
            )
        # Alibi reveal (non-direct but informative)
        direct_pool.append(
            {
                "text": f"The culprit's alibi is {self.attributes[culprit]['alibi']}.",
                "filter": lambda s: self.attributes[s]["alibi"] == self.attributes[culprit]["alibi"],
                "kind": "informative",
            }
        )

        # General eliminations (exclude attributes not matching culprit)
        gen_pool = []
        # eliminate by weapon not equal to culprit
        other_weapons = {self.attributes[s]["weapon"] for s in self.suspects if s != culprit}
        for w in list(other_weapons):
            if w != c_att["weapon"]:
                gen_pool.append(
                    {
                        "text": f"Anyone wielding the {w} is innocent.",
                        "filter": lambda s, w=w: self.attributes[s]["weapon"] != w or s == culprit,
                        "kind": "informative",
                    }
                )
        # eliminate by location not equal to culprit
        other_locs = {self.attributes[s]["location"] for s in self.suspects if s != culprit}
        for loc in list(other_locs):
            if loc != c_att["location"]:
                gen_pool.append(
                    {
                        "text": f"Anyone found in the {loc} is innocent.",
                        "filter": lambda s, loc=loc: self.attributes[s]["location"] != loc or s == culprit,
                        "kind": "informative",
                    }
                )
        # eliminate by strong alibi
        gen_pool.append(
            {
                "text": "Anyone with a Strong alibi is innocent.",
                "filter": lambda s: self.attributes[s]["alibi"] != "Strong" or s == culprit,
                "kind": "informative",
            }
        )
        # negative exclusion (not NAME)
        innocents = [s for s in self.suspects if s != culprit]
        random.shuffle(innocents)
        for name in innocents[: max(1, len(innocents) // 2)]:
            gen_pool.append(
                {
                    "text": f"The culprit is not {name}.",
                    "filter": lambda s, name=name: s != name or s == culprit,
                    "kind": "informative",
                }
            )

        random.shuffle(direct_pool)
        random.shuffle(gen_pool)

        candidates = set(self.suspects)
        selected: List[Dict[str, Any]] = []

        def apply_filters(filters: List[Any]) -> Set[str]:
            out = set(self.suspects)
            for f in filters:
                out = set([s for s in out if f(s)])
            return out

        filters_applied = []

        target_steps = self.required_constraints
        # Select target_steps - 1 clues that do not make unique early
        all_pool = gen_pool + direct_pool
        # Prioritize general clues first to avoid early uniqueness
        all_pool_sorted = gen_pool + direct_pool
        i = 0
        while len(selected) < max(0, target_steps - 1) and i < len(all_pool_sorted):
            candidate_clue = all_pool_sorted[i]
            i += 1
            new_filters = filters_applied + [candidate_clue["filter"]]
            new_candidates = apply_filters(new_filters)
            if len(new_candidates) >= 2:
                selected.append(candidate_clue)
                filters_applied.append(candidate_clue["filter"])
                candidates = new_candidates

        # Final clue to ensure uniqueness
        final_clue = None
        # Try direct reveals first
        for clue in direct_pool:
            new_candidates = apply_filters(filters_applied + [clue["filter"]])
            if len(new_candidates) == 1:
                final_clue = clue
                break
        # If no direct reveal suffices, use any gen clue that yields uniqueness
        if final_clue is None:
            for clue in gen_pool:
                new_candidates = apply_filters(filters_applied + [clue["filter"]])
                if len(new_candidates) == 1:
                    final_clue = clue
                    break
        # Fallback: explicit name (only if needed)
        if final_clue is None:
            final_clue = {
                "text": f"The culprit is {self.culprit}.",
                "filter": lambda s: s == self.culprit,
                "kind": "informative",
            }

        selected.append(final_clue)
        self.informative_clues = selected

        # Build red herrings
        red_herrings = []
        # True statements about innocents, non-filtering
        for _ in range(self.num_red_herrings):
            if not innocents:
                break
            sh = random.choice(innocents)
            att = self.attributes[sh]
            rh_texts = [
                f"{sh}'s alibi is {att['alibi']}.",
                f"{sh} was seen near the {att['location']}.",
                f"{sh} handled the {att['weapon']} earlier.",
            ]
            red_herrings.append(
                {
                    "text": random.choice(rh_texts),
                    "filter": None,
                    "kind": "red_herring",
                }
            )
        # Interleave clues: deterministic order but shuffled mildly
        combined = self.informative_clues + red_herrings
        random.shuffle(combined)
        # Guarantee the final informative clue is placed towards the end
        if final_clue in combined:
            combined.remove(final_clue)
        tail_index = random.randint(max(0, len(combined) - 2), len(combined))
        combined.insert(tail_index, final_clue)

        self.clues = combined

    def _compute_possible_suspects(self) -> List[str]:
        seen_filters = []
        count_seen = 0
        for i in range(self.clue_index):
            c = self.clues[i]
            if c.get("kind") == "informative" and c.get("filter") is not None:
                seen_filters.append(c["filter"])
                count_seen += 1
        possible = set(self.suspects)
        for f in seen_filters:
            possible = set([s for s in possible if f(s)])
        return sorted(list(possible))


class DetectiveMysteryEnvWithFeedback(DetectiveMysteryEnv):
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
            hint = "Use \\boxed{command arg} with one command per turn (e.g., \\boxed{get_clue})."

        elif "unknown command" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_command"
            hint = "Use one of: list_suspects, view_attributes NAME, get_clue, summarize, eliminate NAME, accuse NAME."

        elif "unsupported suspect for view_attributes" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "invalid_view_target"
            hint = "Choose a suspect from the listed names. Try \\boxed{list_suspects} to see them."

        elif "unsupported suspect for eliminate" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "invalid_eliminate_target"
            hint = "Eliminate only from listed suspects. Use summarize to refine candidates first."

        elif "unsupported suspect for accuse" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "invalid_accuse_target"
            hint = "Accuse a listed suspect's exact name."

        elif "failed!" in text:
            error_type = "WrongDecision"
            error_detail["outcome"] = "wrong_accusation"
            error_detail["expected"] = self.culprit
            hint = "Gather more evidence: use \\boxed{get_clue} and \\boxed{summarize} to narrow candidates before accusing."

        elif "reached max turns" in text or "timed out" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Accelerate decision-making: alternate get_clue and summarize, then accuse within the turn limit."

        elif "no more clues remain" in text:
            error_type = "OK"
            error_detail["note"] = "clue_stream_exhausted"
            hint = "Use summarize and accuse based on remaining candidates."

        elif "success!" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            state_info = {
                "suspects_remaining": len([s for s in self.suspects if s not in self.eliminated]),
                "clues_seen": self.clue_index,
                "total_clues": len(self.clues),
                "possible_now": self._compute_possible_suspects(),
            }
            diagnostic["state"] = state_info
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start with \\boxed{list_suspects}, then \\boxed{get_clue}, and use \\boxed{summarize} to narrow candidates.",
            "turn": 0,
            "state": {
                "suspects_remaining": len(self.suspects),
                "clues_seen": 0,
                "total_clues": len(self.clues),
                "possible_now": self.suspects,
            },
        }
        return obs, info