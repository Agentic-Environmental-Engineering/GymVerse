from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List, Set


class AxiomSnareEnv(Env):
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

        # Evolvable parameters (logic-native)
        self.complexity_params = {
            # Number of boolean atoms available; more atoms → larger search space → harder
            "num_atoms": (4, 10),
            # Initial hidden core clauses; more base structure to defeat → harder for agent
            "base_clause_count": (2, 6),
            # Maximum literals per clause the agent can add; higher width makes it harder to force contradiction
            "max_clause_width": (3, 5),
            # REVERSED: maximum admissible model Hamming-weight target; smaller weight → adversary has stricter objective → easier for agent
            # We reverse by setting easy>hard and clamping correctly in _apply_complexity_params
            "target_weight": (5, 2),
        }
        self.param_variance = {
            "num_atoms": 1,           # discrete medium range
            "base_clause_count": 1,   # discrete medium range
            "max_clause_width": 0,    # small range; keep fixed to keep rules stable at each level
            "target_weight": 0,       # small span; keep fixed per level to avoid instability
        }

        # placeholders
        self.num_atoms: int = 0
        self.base_clause_count: int = 0
        self.max_clause_width: int = 0
        self.target_weight: int = 0

        # state
        self.turn_count: int = 0
        self.atoms: List[str] = []
        self.base_clauses: List[List[int]] = []
        self.agent_clauses: List[List[int]] = []
        self.derived_units: Dict[int, bool] = {}
        self.last_adversary_model: Optional[Dict[int, bool]] = None
        self.terminated_reason: Optional[str] = None

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for k, (min_v, max_v) in self.complexity_params.items():
            center = min_v + (max_v - min_v) * normalized
            var = self.param_variance.get(k, 0)
            if self.enable_param_randomization and var > 0:
                center = center + random.uniform(-var, var)
            # clamp respecting reversed
            lo, hi = (max_v, min_v) if min_v > max_v else (min_v, max_v)
            center = max(lo, min(hi, center))
            setattr(self, k, int(round(center)))

    def _get_instructions(self) -> str:
        return (
            "You are playing AxiomSnare, a logic-deduction duel.\n"
            "Goal: Add clauses that make the combined knowledge base unsatisfiable before the adversary finds a satisfying assignment.\n"
            "Domain:\n"
            "- Variables are boolean atoms: a1, a2, ... aN. Literals are ai (positive) or ~ai (negated).\n"
            "- Clauses are disjunctions of literals, written as comma-separated tokens. Example: a1,~a3,a4.\n"
            "- The knowledge base (KB) consists of base clauses plus your added clauses.\n"
            "Adversary:\n"
            "- After your move, the adversary deterministically searches for a model (truth assignment) that satisfies the KB and aims to keep total true variables ≤ target_weight.\n"
            "- If such a model exists, the adversary presents it and the episode continues. If not, you win (KB is unsatisfiable).\n"
            "Your actions (use \\boxed{...}):\n"
            "- add_clause literals=a1,~a2,a3    Add a clause with up to max_clause_width literals.\n"
            "- probe                             Ask to see current KB summary and last adversary model.\n"
            "- pass                              Do nothing this turn (lets adversary try immediately).\n"
            "Rules:\n"
            "- Clauses must reference existing atoms and be non-empty.\n"
            "- Max width per clause is limited. Duplicates or tautologies (ai and ~ai together) are allowed but may be unhelpful.\n"
            "- The episode ends if: you render KB UNSAT (success=1.0), the adversary finds a satisfying model at the horizon (timeout gives 0.0), or you format an invalid action (format error reward).\n"
            "Action format examples:\n"
            f"- {r'\\boxed{add_clause literals=a1,~a2}'}\n"
            f"- {r'\\boxed{probe}'}\n"
            f"- {r'\\boxed{pass}'}\n"
        )

    def get_task_suffix(self) -> str:
        base = []
        base.append(f"Turn: {self.turn_count}/{self.max_turns}")
        base.append(f"Atoms: {', '.join(self.atoms)}")
        base.append(f"Target true-count (adversary aims ≤): {self.target_weight}")
        base.append(f"Base clauses: {self._format_clause_list(self.base_clauses)}")
        if self.agent_clauses:
            base.append(f"Your clauses: {self._format_clause_list(self.agent_clauses)}")
        else:
            base.append("Your clauses: []")
        if self.last_adversary_model is not None:
            pretty = self._pretty_model(self.last_adversary_model)
            base.append(f"Last adversary model: {pretty}")
        else:
            base.append("Last adversary model: None")
        base.append("Enter your action in \\boxed{...} format.")
        return "\n".join(base)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()

        self.turn_count = 0
        self.atoms = [f"a{i+1}" for i in range(self.num_atoms)]
        self.base_clauses = self._generate_base_clauses()
        self.agent_clauses = []
        self.derived_units = {}
        self.last_adversary_model = None
        self.terminated_reason = None

        # initial adversary response to base KB
        satisfiable, model = self._adversary_search()
        if satisfiable:
            self.last_adversary_model = model
            obs = "Episode start: Adversary found an initial satisfying model."
        else:
            self.terminated_reason = "unsat"
            obs = "Episode start: Base KB is already UNSAT. You immediately win."
            return obs, {"suffix": self.get_task_suffix()}

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)
        if not parsed:
            obs = "INVALID ACTION FORMAT: Use \\boxed{...} with a supported action."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        act = parsed.get("action", "")
        obs = ""
        terminated = False
        truncated = False

        if act == "add_clause":
            literal_str = parsed.get("literals", "")
            ok, clause_or_msg = self._parse_clause(literal_str)
            if not ok:
                obs = f"PROTOCOL VIOLATION: {clause_or_msg}"
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            clause: List[int] = clause_or_msg
            if len(clause) == 0:
                obs = "PROTOCOL VIOLATION: Clause cannot be empty."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            if len(clause) > self.max_clause_width:
                obs = f"PROTOCOL VIOLATION: Clause exceeds max width {self.max_clause_width}."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            self.agent_clauses.append(self._normalize_clause(clause))
            obs = "Added clause. "
        elif act == "probe":
            obs = "Probe: Showing current KB and last model. "
        elif act == "pass":
            obs = "Passed. "
        else:
            obs = "UNSUPPORTED ACTION: Unknown command."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        satisfiable, model = self._adversary_search()
        if satisfiable:
            self.last_adversary_model = model
            obs += "Adversary produced a satisfying model."
        else:
            obs += "UNSAT achieved. You trapped the adversary."
            return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns}). Adversary still has a model."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        reward = 0.0
        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if action is None:
            return None
        m = re.search(r"\\boxed\{(.+?)\}\s*$", action.strip(), flags=re.DOTALL)
        if not m:
            return None
        inner = m.group(1).strip()
        parts = inner.split()
        if not parts:
            return None
        tokens: Dict[str, Any] = {"action": parts[0]}
        for part in parts[1:]:
            if "=" in part:
                k, v = part.split("=", 1)
                tokens[k.strip()] = v.strip()
        return tokens

    def sample_random_action(self) -> str:
        if random.random() < 0.3:
            return r"\boxed{probe}"
        if random.random() < 0.5:
            return r"\boxed{pass}"
        # generate a small clause
        width = random.randint(1, max(1, self.max_clause_width))
        lits = []
        for _ in range(width):
            var = random.choice(self.atoms) if self.atoms else "a1"
            sign = "~" if random.random() < 0.5 else ""
            lits.append(f"{sign}{var}")
        literal_str = ",".join(lits)
        return rf"\boxed{{add_clause literals={literal_str}}}"

    # ===== Logic utilities =====
    def _generate_base_clauses(self) -> List[List[int]]:
        clauses: List[List[int]] = []
        # Seed with random small clauses that are satisfiable with high probability
        # plus a soft preference encoded via target_weight that adversary uses
        atoms_idx = list(range(1, self.num_atoms + 1))
        for _ in range(self.base_clause_count):
            width = random.randint(1, min(3, self.num_atoms))
            chosen = random.sample(atoms_idx, width)
            lits = []
            for v in chosen:
                s = 1 if random.random() < 0.5 else -1
                lits.append(s * v)
            clauses.append(self._normalize_clause(lits))
        return clauses

    def _normalize_clause(self, clause: List[int]) -> List[int]:
        # remove duplicates, keep sorted by abs, positives before negatives for same var not needed
        s: Dict[int, int] = {}
        for lit in clause:
            var = abs(lit)
            s[lit] = lit
            # If both var and -var present, keep both (tautology allowed)
        uniq = list(s.values())
        uniq.sort(key=lambda x: (abs(x), -1 if x > 0 else 1))
        return uniq

    def _format_clause_list(self, cls: List[List[int]]) -> str:
        return "[" + "; ".join(self._format_clause(c) for c in cls) + "]"

    def _format_clause(self, clause: List[int]) -> str:
        def lit_str(l):
            return ("" if l > 0 else "~") + f"a{abs(l)}"
        return "(" + " v ".join(lit_str(l) for l in clause) + ")"

    def _parse_clause(self, literal_str: str) -> Tuple[bool, Any]:
        if literal_str is None or literal_str.strip() == "":
            return False, "Missing literals= parameter."
        tokens = [t.strip() for t in literal_str.split(",") if t.strip() != ""]
        lits: List[int] = []
        valid_names = set(self.atoms)
        for tok in tokens:
            neg = tok.startswith("~")
            name = tok[1:] if neg else tok
            if name not in valid_names:
                return False, f"Unknown atom '{name}'."
            idx = self.atoms.index(name) + 1
            lits.append(-idx if neg else idx)
        return True, lits

    def _pretty_model(self, model: Dict[int, bool]) -> str:
        items = []
        for i in range(1, self.num_atoms + 1):
            val = model.get(i, False)
            items.append(f"a{i}={'T' if val else 'F'}")
        return "{ " + ", ".join(items) + " }"

    def _kb_clauses(self) -> List[List[int]]:
        return self.base_clauses + self.agent_clauses

    def _adversary_search(self) -> Tuple[bool, Optional[Dict[int, bool]]]:
        # Deterministic search: prioritize assignments with minimal number of True vars,
        # but must be ≤ target_weight to accept. If none with ≤ target_weight, still search
        # for any model; if exists with > target_weight, it counts as satisfiable (adversary succeeds anyway).
        clauses = self._kb_clauses()
        n = self.num_atoms

        # Unit propagation + backtracking (DFS) with branching heuristic: assign False first to keep weight low.
        best_model = None

        def satisfies(cl: List[int], assign: Dict[int, bool]) -> bool:
            for lit in cl:
                var = abs(lit)
                sign = lit > 0
                if var in assign:
                    if assign[var] == sign:
                        return True
                else:
                    # unassigned literal could be true later
                    pass
            return False

        def clause_contradiction(cl: List[int], assign: Dict[int, bool]) -> bool:
            # All literals false under current partial assignment
            all_false = True
            for lit in cl:
                var = abs(lit)
                sign = lit > 0
                if var not in assign:
                    all_false = False
                    break
                if assign[var] == sign:
                    return False
            return all_false

        def unit_propagate(assign: Dict[int, bool]) -> Tuple[bool, Dict[int, bool]]:
            changed = True
            a = dict(assign)
            while changed:
                changed = False
                for cl in clauses:
                    if satisfies(cl, a):
                        continue
                    # check if all but one literal false and remaining unassigned -> unit
                    unassigned = []
                    for lit in cl:
                        var = abs(lit)
                        sign = lit > 0
                        if var not in a:
                            unassigned.append((var, sign))
                        else:
                            if a[var] == sign:
                                unassigned = []
                                break
                    if unassigned and len(unassigned) == 1:
                        var, sign = unassigned[0]
                        # force assignment
                        a[var] = sign
                        changed = True
                        # contradiction check immediate
                        for cl2 in clauses:
                            if clause_contradiction(cl2, a):
                                return False, a
                # also detect immediate contradictions
                for cl in clauses:
                    if clause_contradiction(cl, a):
                        return False, a
            return True, a

        # Try assignments in increasing true-count order up to n
        order_vars = list(range(1, n + 1))

        def dfs(assign: Dict[int, bool]) -> Optional[Dict[int, bool]]:
            ok, a = unit_propagate(assign)
            if not ok:
                return None
            if len(a) == n:
                # full assignment satisfies all clauses (since no contradiction)
                return a
            # choose next var
            for v in order_vars:
                if v not in a:
                    next_var = v
                    break
            # branch: False first, then True
            for val in [False, True]:
                a[next_var] = val
                model = dfs(a)
                del a[next_var]
                if model is not None:
                    return model
            return None

        model = dfs({})
        if model is None:
            return False, None
        # If model found, enforce deterministic tie-breaking by minimal weight preference already induced by False-first branching.
        # Check target weight acceptance (not strictly needed for satisfiability, but stored for display)
        # Adversary succeeds with any satisfying model.
        return True, model


class AxiomSnareEnvWithFeedback(AxiomSnareEnv):
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
            hint = "Wrap your command in \\boxed{...} and use keys like literals= for add_clause."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["allowed"] = ["add_clause", "probe", "pass"]
            hint = "Use one of: add_clause literals=..., probe, or pass."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "unknown atom" in text:
                error_detail["violation"] = "unknown_atom"
                hint = "Use only atoms listed in the state (e.g., a1..aN)."
            elif "clause cannot be empty" in text:
                error_detail["violation"] = "empty_clause"
                hint = "Provide at least one literal like a1 or ~a2."
            elif "exceeds max width" in text:
                error_detail["violation"] = "clause_too_wide"
                hint = "Reduce the number of literals to be within max_clause_width."
            else:
                error_detail["violation"] = "generic_clause_error"
                hint = "Ensure literals are comma-separated, valid names, and within width."

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["outcome"] = "horizon_reached"
            hint = "Aim to add tighter clauses earlier; consider unit clauses or complementary constraints."

        elif "unsat achieved" in text or "already unsat" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        else:
            # Normal step
            if "adversary produced a satisfying model" in text:
                error_type = "OK"
                error_detail["outcome"] = "adversary_model"
                hint = "Try adding narrower clauses (~ai or ai) that conflict with the last model."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "num_atoms": getattr(self, "num_atoms", None),
                "target_weight": getattr(self, "target_weight", None),
                "agent_clause_count": len(getattr(self, "agent_clauses", [])),
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
            "hint": "Start by probing, then add a focused clause targeting variables likely true.",
            "turn": 0,
            "state": {
                "num_atoms": getattr(self, "num_atoms", None),
                "target_weight": getattr(self, "target_weight", None),
                "agent_clause_count": 0,
            },
        }
        return obs, info