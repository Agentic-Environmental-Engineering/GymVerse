from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class ClueConclaveEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 20,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, complexity))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 20

        # Evolvable parameters
        self.complexity_params = {
            # Number of hypotheses/suspects: more options increases decision space and confusion
            'num_suspects': (3, 10),
            # Number of clues: more clues increases reasoning load and potential contradictions
            'num_clues': (4, 18),
            # REVERSED: fraction of clues that are noisy/misleading; higher noise makes reasoning harder
            'noise_percent': (10, 45),
            # Strength range upper bound: bigger range increases variance and subtlety; higher is harder
            'max_strength': (3, 8),
            # Number of decoy-boosting clusters (clues that jointly favor a wrong suspect); more clusters harder
            'decoy_clusters': (0, 3),
        }

        # Variance settings
        self.param_variance = {
            'num_suspects': 1,
            'num_clues': 2,
            'noise_percent': 3,
            'max_strength': 1,
            'decoy_clusters': 1,
        }

        # Placeholder attributes
        self.num_suspects: int = 0
        self.num_clues: int = 0
        self.noise_percent: int = 0
        self.max_strength: int = 0
        self.decoy_clusters: int = 0

        # State
        self.turn_count: int = 0
        self.active: bool = True
        self.suspects: List[str] = []
        self.true_index: int = -1
        self.clues: List[Dict[str, Any]] = []
        self.evidence_scores: List[int] = []
        self.used_actions: List[str] = []

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for name, (min_v, max_v) in self.complexity_params.items():
            center = min_v + (max_v - min_v) * normalized
            actual = center
            if self.enable_param_randomization:
                var = self.param_variance.get(name, 0)
                if var > 0:
                    actual = center + random.uniform(-var, var)
                    lo, hi = (max_v, min_v) if min_v > max_v else (min_v, max_v)
                    actual = max(lo, min(hi, actual))
            setattr(self, name, int(round(actual)))

    def _generate_case(self):
        # Generate suspects
        self.suspects = [f"Suspect_{chr(65+i)}" for i in range(self.num_suspects)]
        # Choose true culprit
        self.true_index = random.randrange(self.num_suspects)

        # Initialize evidence scores
        self.evidence_scores = [0 for _ in range(self.num_suspects)]
        self.clues = []

        # Decide decoy targets
        decoy_targets = []
        available_indices = [i for i in range(self.num_suspects) if i != self.true_index]
        random.shuffle(available_indices)
        for k in range(min(self.decoy_clusters, len(available_indices))):
            decoy_targets.append(available_indices[k])

        # Build clues
        num_noisy = int(round(self.num_clues * (self.noise_percent / 100.0)))
        noisy_indices = set(random.sample(range(self.num_clues), k=min(num_noisy, self.num_clues)))

        # For each clue, assign target suspect and strength
        for ci in range(self.num_clues):
            is_noisy = ci in noisy_indices
            if is_noisy and decoy_targets:
                target = random.choice(decoy_targets)
            else:
                # Mostly support true, but allow occasional neutral support among non-noisy true-leaning
                target = self.true_index

            strength = random.randint(1, self.max_strength)
            clue_type = "forensic" if random.random() < 0.33 else ("witness" if random.random() < 0.5 else "alibi")

            text = self._render_clue_text(clue_type, target, strength, is_noisy)
            signed_effects = self._clue_effect_vector(target=target, strength=strength, is_noisy=is_noisy)

            self.clues.append({
                "id": ci + 1,
                "type": clue_type,
                "text": text,
                "target": target,
                "strength": strength,
                "is_noisy": is_noisy,
                "effects": signed_effects
            })

        # Aggregate effects to evidence scores
        self.evidence_scores = [0 for _ in range(self.num_suspects)]
        for c in self.clues:
            for s_idx, v in enumerate(c["effects"]):
                self.evidence_scores[s_idx] += v

        # Ensure solvability: enforce unique argmax at true_index
        # If tie or wrong leader, adjust minimally by boosting true's score
        leader = max(range(self.num_suspects), key=lambda i: self.evidence_scores[i])
        leaders = [i for i, sc in enumerate(self.evidence_scores) if sc == self.evidence_scores[leader]]
        if (leader != self.true_index) or (len(leaders) > 1):
            delta = 1 + max(0, max(self.evidence_scores) - self.evidence_scores[self.true_index])
            # Apply a small invisible "meta" boost by appending a balancing virtual clue
            self.evidence_scores[self.true_index] += delta

    def _render_clue_text(self, clue_type: str, target: int, strength: int, is_noisy: bool) -> str:
        sname = self.suspects[target] if self.suspects else f"Suspect_{target}"
        if clue_type == "forensic":
            base = f"Lab trace strength {strength} pointing toward {sname}"
        elif clue_type == "witness":
            base = f"Witness statement credibility {strength} implicating {sname}"
        else:
            base = f"Schedule conflict severity {strength} undermining {sname}'s alibi"

        if is_noisy:
            return base + " (uncorroborated)"
        else:
            return base

    def _clue_effect_vector(self, target: int, strength: int, is_noisy: bool) -> List[int]:
        # Each clue adds positive weight to target and small negative interference to others
        effects = []
        for i in range(self.num_suspects):
            if i == target:
                eff = strength if not is_noisy else max(1, strength - 1)
            else:
                eff = -1 if strength >= 3 else 0
                if is_noisy and random.random() < 0.3:
                    eff = 0  # noisy may avoid penalizing others
            effects.append(eff)
        return effects

    def _get_instructions(self) -> str:
        suspects_list = "\n".join([f"- [{i}] {name}" for i, name in enumerate(self.suspects)])
        clues_list = "\n".join([f"Clue {c['id']}: {c['text']}" for c in self.clues])
        return (
            "Clue Conclave: Deduce the single correct suspect from evidence.\n"
            "Goal: Select exactly one suspect index that best fits the evidence. The case has one true culprit.\n"
            "Rules:\n"
            "- You may only take the DECIDE action exactly once to make your final choice.\n"
            "- Before deciding, you may optionally INSPECT to see current evidence totals.\n"
            "- The episode ends upon a valid DECIDE, an invalid action, or timeout.\n"
            "Actions must be in \\boxed{...} format.\n"
            "Available actions:\n"
            "- \\boxed{INSPECT}: Show current evidence scores per suspect (informational, no reward).\n"
            "- \\boxed{DECIDE index=<k>}: Choose suspect k as the culprit (terminal action).\n"
            "Suspects:\n"
            f"{suspects_list}\n"
            "Clues:\n"
            f"{clues_list}\n"
            "Format strictly requires the \\boxed{...} wrapper. Example:\n"
            f"{self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        return (
            "State:\n"
            f"- Turns used: {self.turn_count}/{self.max_turns}\n"
            "- Awaiting action: use \\boxed{INSPECT} or \\boxed{DECIDE index=<k>} where k is an integer.\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()

        self.turn_count = 0
        self.active = True
        self.used_actions = []

        self._generate_case()
        obs = self._get_instructions()
        return obs, {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if not self.active:
            return "TERMINAL: Episode already ended.", 0.0, True, False, {"suffix": self.get_task_suffix()}

        self.turn_count += 1
        parsed = self._parse_action(action)
        if not parsed:
            self.active = False
            obs = "INVALID ACTION FORMAT: Use \\boxed{INSPECT} or \\boxed{DECIDE index=<k}>."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        act = parsed.get("action", "").upper()

        if act == "INSPECT":
            self.used_actions.append("INSPECT")
            # Informational step
            scores = ", ".join([f"{i}:{sc}" for i, sc in enumerate(self.evidence_scores)])
            if self.turn_count >= self.max_turns:
                self.active = False
                obs = f"INSPECT -> Scores [{scores}]. Reached max turns ({self.max_turns})."
                return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}
            obs = f"INSPECT -> Evidence scores per suspect: [{scores}]."
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        elif act == "DECIDE":
            idx_str = parsed.get("index", None)
            if idx_str is None or not re.fullmatch(r"-?\d+", idx_str):
                self.active = False
                obs = "PROTOCOL VIOLATION: DECIDE requires 'index=<integer>'."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            k = int(idx_str)
            if k < 0 or k >= self.num_suspects:
                self.active = False
                obs = f"PROTOCOL VIOLATION: index out of range. Valid range: 0..{self.num_suspects-1}."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            self.active = False
            self.used_actions.append(f"DECIDE:{k}")
            correct = (k == self.true_index)
            if correct:
                obs = f"SUCCESS: You chose index {k} ({self.suspects[k]}). Correct culprit identified."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = (
                    f"FAILURE: You chose index {k} ({self.suspects[k]}). "
                    f"True culprit was index {self.true_index} ({self.suspects[self.true_index]})."
                )
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            self.active = False
            obs = f"UNSUPPORTED ACTION: '{act}'. Use INSPECT or DECIDE."
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
                k, v = part.split('=', 1)
                tokens[k.strip()] = v.strip()
        return tokens

    def sample_random_action(self) -> str:
        if random.random() < 0.5:
            return r"\boxed{INSPECT}"
        else:
            guess = random.randrange(max(1, self.num_suspects)) if self.num_suspects > 0 else 0
            return rf"\boxed{{DECIDE index={guess}}}"


class ClueConclaveEnvWithFeedback(ClueConclaveEnv):
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
            error_detail["issue"] = "missing_or_bad_boxed_format"
            hint = "Wrap your action exactly like \\boxed{INSPECT} or \\boxed{DECIDE index=2}."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "requires 'index=<integer>'" in text:
                error_detail["violation"] = "missing_index"
                hint = "Provide an integer index with DECIDE, e.g., \\boxed{DECIDE index=0}."
            elif "out of range" in text:
                error_detail["violation"] = "index_out_of_range"
                hint = "Use a valid index within the shown suspect range (see instructions)."
            else:
                error_detail["violation"] = "unknown_protocol"

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["action"] = "unrecognized"
            hint = "Use only INSPECT or DECIDE."

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Act sooner; you can INSPECT once or twice, then DECIDE before the limit."

        elif "failure" in text:
            error_type = "WrongDecision"
            # Try to extract indices
            got_m = re.search(r"you chose index (\-?\d+)", obs, flags=re.IGNORECASE)
            true_m = re.search(r"true culprit was index (\-?\d+)", obs, flags=re.IGNORECASE)
            if got_m:
                error_detail["got"] = int(got_m.group(1))
            if true_m:
                error_detail["expected"] = int(true_m.group(1))
            hint = "Consider using INSPECT to compare evidence totals, then pick the top-scoring suspect."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["suspects"] = list(self.suspects)
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start with \\boxed{INSPECT} to see current evidence totals, then \\boxed{DECIDE index=<k>}.",
            "turn": 0,
            "suspects": list(self.suspects),
        }
        return obs, info