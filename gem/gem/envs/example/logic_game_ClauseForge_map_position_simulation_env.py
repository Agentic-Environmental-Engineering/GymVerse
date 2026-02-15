from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class ClauseForgeEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 20,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 20

        # Evolvable parameters
        self.complexity_params = {
            # number of base propositions p1..pN; larger N increases search space
            "num_props": (4, 12),
            # number of constraints generated; more constraints increase reasoning but can be redundant
            "num_constraints": (4, 18),
            # diversity of clause types influences reasoning difficulty
            # mapped as an integer selector controlling which families are used (1..4)
            "clause_palette": (1, 4),
            # number of decoy variables (aliases and negated aliases) added to text only; more decoys increase parsing/logic trap
            "num_decoys": (0, 4),
            # partial feedback richness: higher gives more granular feedback; lower makes it harder
            # REVERSED: more feedback at easy, less at hard
            "feedback_level_internal": (3, 1),
        }
        self.param_variance = {
            "num_props": 1,
            "num_constraints": 2,
            "clause_palette": 0,
            "num_decoys": 1,
            "feedback_level_internal": 0,
        }

        # Placeholders
        self.num_props: int = 0
        self.num_constraints: int = 0
        self.clause_palette: int = 0
        self.num_decoys: int = 0
        self.feedback_level_internal: int = 0

        # State
        self.turn_count: int = 0
        self.props: List[str] = []
        self.decoy_map: Dict[str, str] = {}
        self.truth: Dict[str, bool] = {}
        self.constraints_text: List[str] = []
        self.constraints_eval = []  # list of callables(state)->bool
        self.active: bool = False

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for name, (min_v, max_v) in self.complexity_params.items():
            center = min_v + (max_v - min_v) * normalized
            v = center
            if self.enable_param_randomization:
                var = self.param_variance.get(name, 0)
                if var > 0:
                    v = center + random.uniform(-var, var)
                # clamp both normal and reversed
                low, high = (max_v, min_v) if min_v > max_v else (min_v, max_v)
                v = max(low, min(high, v))
            setattr(self, name, int(round(v)))

    def _make_literal_text(self, var: str, value: bool) -> str:
        return ("" if value else "NOT ") + var

    def _pick_var(self) -> str:
        return random.choice(self.props)

    def _mk_clause_palette(self) -> List[str]:
        # palette tiers
        base = ["unit", "and", "or"]
        if self.clause_palette >= 2:
            base += ["xor", "if_then"]
        if self.clause_palette >= 3:
            base += ["iff", "atleast_k"]
        if self.clause_palette >= 4:
            base += ["exactly_k", "atmost_k"]
        return base

    def _gen_constraint_from_truth(self) -> Tuple[str, Any]:
        palette = self._mk_clause_palette()
        ctype = random.choice(palette)

        if ctype == "unit":
            v = self._pick_var()
            val = self.truth[v]
            text = f"{v} is {'TRUE' if val else 'FALSE'}."
            def ev(s): return (s[v] is val)
            return text, ev

        if ctype == "and":
            a, b = random.sample(self.props, 2 if len(self.props) >= 2 else 1)
            va = self.truth[a]
            vb = self.truth[b]
            text = f"{a} AND {b} are {'TRUE' if (va and vb) else 'NOT both TRUE'}."
            def ev(s):
                if "NOT both TRUE" in text:
                    return not (s[a] and s[b])
                return s[a] and s[b]
            return text, ev

        if ctype == "or":
            a, b = random.sample(self.props, 2 if len(self.props) >= 2 else 1)
            va = self.truth[a]
            vb = self.truth[b]
            text = f"At least one of {a}, {b} is {'TRUE' if (va or vb) else 'FALSE'}."
            def ev(s):
                want_true = "TRUE" in text
                return ((s[a] or s[b]) if want_true else (not s[a] and not s[b]))
            return text, ev

        if ctype == "xor":
            a, b = random.sample(self.props, 2)
            want = (self.truth[a] != self.truth[b])
            text = f"Exactly one of {a} and {b} is {'TRUE' if want else 'FALSE'}."
            def ev(s):
                if "TRUE" in text:
                    return (s[a] != s[b])
                else:
                    return (s[a] == s[b])
            return text, ev

        if ctype == "if_then":
            a, b = random.sample(self.props, 2)
            # If a then b must be true according to ground truth
            holds = (not self.truth[a]) or self.truth[b]
            text = f"If {a} then {b} is {'TRUE' if holds else 'FALSE'}."
            def ev(s):
                implication = (not s[a]) or s[b]
                return implication if "TRUE" in text else (not implication)
            return text, ev

        if ctype == "iff":
            a, b = random.sample(self.props, 2)
            want = (self.truth[a] == self.truth[b])
            text = f"{a} iff {b} is {'TRUE' if want else 'FALSE'}."
            def ev(s):
                eq = (s[a] == s[b])
                return eq if "TRUE" in text else (not eq)
            return text, ev

        # k-ary constraints
        k = min(len(self.props), max(2, random.randint(2, 4)))
        subset = random.sample(self.props, k)
        true_count = sum(1 for v in subset if self.truth[v])

        if ctype == "atleast_k":
            # choose threshold aligned with truth
            t = random.randint(0, true_count) if true_count > 0 else 0
            text = f"At least {t} of {{{', '.join(subset)}}} are TRUE."
            def ev(s):
                return sum(1 for v in subset if s[v]) >= t
            return text, ev

        if ctype == "atmost_k":
            t = random.randint(true_count, k) if true_count < k else k
            text = f"At most {t} of {{{', '.join(subset)}}} are TRUE."
            def ev(s):
                return sum(1 for v in subset if s[v]) <= t
            return text, ev

        if ctype == "exactly_k":
            t = true_count
            text = f"Exactly {t} of {{{', '.join(subset)}}} are TRUE."
            def ev(s):
                return sum(1 for v in subset if s[v]) == t
            return text, ev

        # Fallback (shouldn't happen)
        v = self._pick_var()
        val = self.truth[v]
        text = f"{v} is {'TRUE' if val else 'FALSE'}."
        def ev(s): return (s[v] is val)
        return text, ev

    def _generate_instance(self):
        self.props = [f"P{i}" for i in range(1, self.num_props + 1)]
        self.truth = {p: bool(random.getrandbits(1)) for p in self.props}

        # Create decoy aliases: Dk refers to either Pi or NOT Pi in the text only
        self.decoy_map = {}
        available = self.props[:]
        for i in range(1, self.num_decoys + 1):
            if not available:
                available = self.props[:]
            p = random.choice(available)
            available.remove(p)
            negate = bool(random.getrandbits(1))
            alias = f"D{i}"
            self.decoy_map[alias] = (p, negate)

        # Build constraints aligned to truth; ensure some diversity
        self.constraints_text = []
        self.constraints_eval = []
        attempts = 0
        while len(self.constraints_text) < self.num_constraints and attempts < self.num_constraints * 5:
            attempts += 1
            text, ev = self._gen_constraint_from_truth()
            # Occasionally substitute proposition names with decoy aliases in the text to increase parsing load
            if self.decoy_map and random.random() < 0.4:
                for alias, (p, neg) in self.decoy_map.items():
                    # Replace p with alias or NOT alias to preserve meaning textually
                    if p in text and random.random() < 0.5:
                        if neg:
                            text = re.sub(rf"\b{p}\b", f"NOT {alias}", text)
                        else:
                            text = re.sub(rf"\b{p}\b", alias, text)
            # Avoid duplicates
            if text not in self.constraints_text:
                self.constraints_text.append(text)
                self.constraints_eval.append(ev)

        # Sanity check: our known truth must satisfy all evals
        assert all(ev(self.truth) for ev in self.constraints_eval), "Generated unsatisfiable constraints"

    def _explain_world(self) -> str:
        lines = []
        lines.append("You are given a set of boolean propositions and textual logical constraints.")
        lines.append("Decoy names may alias propositions or their negations (text only).")
        lines.append("Your task: submit a full assignment for all base propositions P1..Pn.")
        lines.append("Acceptable values: TRUE or FALSE.")
        lines.append("")
        lines.append(f"Propositions: {', '.join(self.props)}")
        if self.decoy_map:
            hints = []
            for alias, (p, neg) in self.decoy_map.items():
                hints.append(f"{alias} refers to {'NOT ' if neg else ''}{p}")
            lines.append("Decoy alias semantics (disclosed to keep solvable): " + "; ".join(hints))
        else:
            lines.append("No decoy aliases in this instance.")
        lines.append("")
        lines.append("Constraints:")
        for i, c in enumerate(self.constraints_text, 1):
            lines.append(f"{i}. {c}")
        return "\n".join(lines)

    def _get_instructions(self) -> str:
        header = (
            "Logic Assignment Challenge:\n"
            "- Provide a truth assignment for all base propositions P1..Pn that satisfies all constraints.\n"
            "- Submit exactly one action per turn; you may refine across turns until success or timeout.\n"
            "- Action format: \\boxed{submit P1=TRUE P2=FALSE ...}\n"
            "- Only base propositions (Pi) are assignable. Decoys (Di) appear only in constraints text.\n"
            "- Feedback will report satisfied constraints and contradictions; success requires all satisfied.\n"
            f"- Example: {self.sample_random_action()}\n"
        )
        # Include the constraint text in the initial observation for prompt-wrapping tests
        return header + "\n" + self._explain_world()

    def get_task_suffix(self) -> str:
        status = f"Turn {self.turn_count}/{self.max_turns}."
        goal = "Enter your full assignment using \\boxed{submit P1=TRUE P2=FALSE ...}"
        return self._explain_world() + "\n\n" + status + "\n" + goal

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()
        self.turn_count = 0
        self.active = True
        self._generate_instance()
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if action is None:
            return None
        # Use the last boxed block if multiple are present
        matches = list(re.finditer(r"\\boxed\{(.*?)\}", action, flags=re.DOTALL))
        if not matches:
            return None
        inner = matches[-1].group(1).strip()
        parts = inner.split()
        if not parts:
            return None
        if parts[0].lower() != "submit":
            return {"action": parts[0].lower()}  # unsupported action detected
        assigns: Dict[str, str] = {}
        for token in parts[1:]:
            if "=" in token:
                k, v = token.split("=", 1)
                assigns[k.strip()] = v.strip().upper().rstrip(",;.")
        return {"action": "submit", "assigns": assigns}

    def _evaluate_assignment(self, assigns: Dict[str, str]) -> Tuple[int, int, List[int], Dict[str, bool], List[str]]:
        parsed_state: Dict[str, bool] = {}
        errors: List[str] = []
        for p in self.props:
            if p not in assigns:
                errors.append(f"Missing {p}")
                continue
            val = assigns[p]
            if val not in ("TRUE", "FALSE"):
                errors.append(f"Invalid value for {p}: {val}")
                continue
            parsed_state[p] = (val == "TRUE")
        if errors:
            return 0, len(self.constraints_eval), list(range(1, len(self.constraints_eval)+1)), parsed_state, errors

        satisfied_idx = []
        for i, ev in enumerate(self.constraints_eval, 1):
            try:
                if ev(parsed_state):
                    satisfied_idx.append(i)
            except Exception:
                pass
        satisfied = len(satisfied_idx)
        total = len(self.constraints_eval)
        violated_idx = [i for i in range(1, total + 1) if i not in satisfied_idx]
        return satisfied, total, violated_idx, parsed_state, errors

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if not self.active:
            return "Episode is not active.", 0.0, True, False, {"suffix": self.get_task_suffix()}
        self.turn_count += 1

        parsed = self._parse_action(action)
        if not parsed:
            obs = "INVALID ACTION FORMAT: Use \\boxed{submit P1=TRUE P2=FALSE ...}"
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        if parsed.get("action") != "submit":
            obs = "UNSUPPORTED ACTION: Only 'submit' is allowed."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        assigns = parsed.get("assigns", {})
        satisfied, total, violated_idx, user_state, errors = self._evaluate_assignment(assigns)

        if errors:
            obs = "PROTOCOL VIOLATION: " + "; ".join(errors)
            # Protocol violations do not terminate; allow further attempts even at max_turns for test coverage
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        success = (satisfied == total)
        if success:
            obs = f"Success! All {total} constraints satisfied."
            self.active = False
            return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}

        # Partial feedback depends on internal level
        details = []
        if self.feedback_level_internal >= 3:
            details.append(f"Satisfied: {satisfied}/{total}")
            if violated_idx:
                sample = violated_idx[:min(3, len(violated_idx))]
                details.append("First unsatisfied constraints: " + ", ".join(str(x) for x in sample))
        elif self.feedback_level_internal == 2:
            details.append(f"Satisfied: {satisfied}/{total}")
        else:
            details.append("Assignment incorrect.")

        obs = "Feedback: " + " | ".join(details)

        if self.turn_count >= self.max_turns:
            self.active = False
            return obs + f" Reached max turns ({self.max_turns}).", 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

    def sample_random_action(self) -> str:
        if not self.props:
            return r"\boxed{submit}"
        assigns = []
        for p in self.props:
            assigns.append(f"{p}={'TRUE' if random.random()<0.5 else 'FALSE'}")
        return r"\boxed{submit " + " ".join(assigns) + "}"


class ClauseForgeEnvWithFeedback(ClauseForgeEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = max(0, min(2, int(feedback_level)))
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        text = obs.lower()

        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint = None

        if "invalid action format" in text:
            error_type = "FormatError"
            error_detail["issue"] = "missing_or_bad_boxed_submit"
            hint = "Use \\boxed{submit P1=TRUE P2=FALSE ...} and assign every Pi."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "only_submit_allowed"
            hint = "Only use the 'submit' action."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            missing = re.findall(r"Missing (P\d+)", obs)
            invalids = re.findall(r"Invalid value for (P\d+): (\w+)", obs)
            error_detail["missing"] = missing
            error_detail["invalids"] = [{"prop": p, "value": v} for p, v in invalids]
            hint = "Assign every listed Pi with TRUE or FALSE exactly."

        elif "reached max turns" in text or truncated:
            error_type = "Timeout"
            error_detail["outcome"] = "max_turns"
            hint = "Propose complete assignments earlier; use feedback about unsatisfied constraints."

        elif "assignment incorrect" in text or "feedback:" in text:
            error_type = "WrongDecision"
            m = re.search(r"Satisfied:\s*(\d+)\s*/\s*(\d+)", obs)
            if m:
                error_detail["satisfied"] = int(m.group(1))
                error_detail["total"] = int(m.group(2))
            idxs = re.search(r"First unsatisfied constraints:\s*([0-9,\s]+)", obs, flags=re.IGNORECASE)
            if idxs:
                nums = [int(x.strip()) for x in idxs.group(1).split(",") if x.strip().isdigit()]
                error_detail["hint_indices"] = nums
            hint = "Focus on unsatisfied constraints; flip variables implicated by those statements and resubmit."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "num_props": getattr(self, "num_props", None),
                "num_constraints": getattr(self, "num_constraints", None),
                "feedback_granularity": getattr(self, "feedback_level_internal", None),
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
            "hint": "Start by submitting a full guess for all Pi. Use feedback to refine.",
            "turn": 0,
            "state": {
                "num_props": getattr(self, "num_props", None),
                "num_constraints": getattr(self, "num_constraints", None),
                "feedback_granularity": getattr(self, "feedback_level_internal", None),
            },
        }
        return obs, info
