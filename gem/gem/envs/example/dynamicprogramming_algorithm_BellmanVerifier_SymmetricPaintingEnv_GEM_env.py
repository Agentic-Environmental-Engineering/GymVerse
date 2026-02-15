from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class BellmanVerifierEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 28,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, complexity))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 28

        # Evolvable parameters
        self.complexity_params = {
            # number of DP states in 1D or effective dimension product; larger state space = harder
            "state_count": (6, 60),
            # number of transitions per state to consider in recurrence; higher branching = harder
            "branching": (2, 6),
            # reversed: fewer allowed global consistency checks -> harder
            "global_checks": (4, 1),
            # percentage of states that are boundary; lower boundaries => harder inference
            "boundary_density": (40, 15),
        }
        # Variance tuned by range size
        self.param_variance = {
            "state_count": 4,
            "branching": 1,
            "global_checks": 0,
            "boundary_density": 4,
        }

        # Placeholder evolvable attributes
        self.state_count: int = 0
        self.branching: int = 0
        self.global_checks: int = 0
        self.boundary_density: int = 0

        # Other state
        self.turn_count: int = 0
        self.done: bool = False
        self.truncated: bool = False

        # Hidden DP instance
        self.instance_type: str = ""
        self.states: List[int] = []
        self.boundary_states: List[int] = []
        self.true_values: Dict[int, int] = {}
        self.rec_proposal: str = ""  # description of the candidate recurrence
        self.is_correct_recurrence: bool = False

        # Budgets and tracking
        self.used_global_checks: int = 0
        self.full_revealed: bool = False
        self.query_log: List[str] = []

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for name, (min_val, max_val) in self.complexity_params.items():
            center = min_val + (max_val - min_val) * normalized
            val = center
            if self.enable_param_randomization:
                var = self.param_variance.get(name, 0)
                if var > 0:
                    val = center + random.uniform(-var, var)
            # clamp with support for reversed params
            lo, hi = (min_val, max_val) if min_val <= max_val else (max_val, min_val)
            val = max(lo, min(hi, val))
            setattr(self, name, int(round(val)))

    def _synthesize_instance(self):
        # Choose a DP flavor among classic ones; use 1D abstract model with transitions
        flavors = ["WeightedPathMinCost", "MonotoneEditOps", "SubsetTargetReachability"]
        self.instance_type = random.choice(flavors)

        # Define states and boundaries
        self.states = list(range(self.state_count))
        boundary_count = max(1, int(round(self.state_count * self.boundary_density / 100.0)))
        boundary_count = min(boundary_count, self.state_count)
        self.boundary_states = sorted(random.sample(self.states, boundary_count))

        # Build hidden transition structure: for each state i, transitions to next indices
        # Ensure acyclicity by only allowing moves to higher indices
        transitions: Dict[int, List[Tuple[int, int]]] = {}
        for i in self.states:
            next_candidates = list(range(i + 1, self.state_count))
            if not next_candidates:
                transitions[i] = []
                continue
            k = min(self.branching, len(next_candidates))
            succs = sorted(random.sample(next_candidates, k))
            # cost/score per move depends on flavor
            tlist = []
            for j in succs:
                if self.instance_type == "WeightedPathMinCost":
                    w = random.randint(1, 9)
                elif self.instance_type == "MonotoneEditOps":
                    # costs biased towards small edits
                    w = random.randint(1, 4)
                else:
                    # SubsetTargetReachability: cost acts like penalty; we want max-negation later
                    w = random.randint(1, 6)
                tlist.append((j, w))
            transitions[i] = tlist

        # True DP values: compute with a specific recurrence selected per flavor
        self.true_values = {}
        # define terminal/boundary values
        for b in self.boundary_states:
            # boundary values differ by flavor
            if self.instance_type == "WeightedPathMinCost":
                self.true_values[b] = 0
            elif self.instance_type == "MonotoneEditOps":
                self.true_values[b] = (self.state_count - 1 - b)  # distance to end
            else:
                # reachability score: 0 at boundary as base
                self.true_values[b] = 0

        # Topological order from end to start
        for i in reversed(self.states):
            if i in self.boundary_states:
                continue
            succs = transitions.get(i, [])
            if not succs:
                # dead-end non-boundary: make it costly
                base = 1000
                self.true_values[i] = base
                continue
            if self.instance_type == "WeightedPathMinCost":
                # min over j of w(i->j) + V[j]
                best = min(w + self.true_values.get(j, 1000) for j, w in succs)
            elif self.instance_type == "MonotoneEditOps":
                # min with operation penalty and monotone: 1 + V[j]
                best = min(1 + self.true_values.get(j, 1000) for j, _ in succs)
            else:
                # reachability-like: value is 0 if any successor is 0, else 1 + min successor
                if any(self.true_values.get(j, 1000) == 0 for j, _ in succs):
                    best = 0
                else:
                    best = 1 + min(self.true_values.get(j, 1000) for j, _ in succs)
            self.true_values[i] = best

        # Propose a candidate recurrence text and decide correctness
        # We generate either the matching true recurrence or a subtly wrong one.
        make_correct = random.random() < 0.5
        if self.instance_type == "WeightedPathMinCost":
            if make_correct:
                self.rec_proposal = "V[i] = min over successors j of (cost(i,j) + V[j]); V[b]=0 at boundaries."
                self.is_correct_recurrence = True
            else:
                # Wrong: max instead of min or missing cost term
                if random.random() < 0.5:
                    self.rec_proposal = "V[i] = max over successors j of (cost(i,j) + V[j]); V[b]=0."
                else:
                    self.rec_proposal = "V[i] = min over successors j of V[j]; V[b]=0."
                self.is_correct_recurrence = False
        elif self.instance_type == "MonotoneEditOps":
            if make_correct:
                self.rec_proposal = "V[i] = 1 + min over successors j of V[j]; V[b] = distance_to_end."
                self.is_correct_recurrence = True
            else:
                # Wrong: missing +1 or using max
                if random.random() < 0.5:
                    self.rec_proposal = "V[i] = min over successors j of V[j]; V[b] = distance_to_end."
                else:
                    self.rec_proposal = "V[i] = 1 + max over successors j of V[j]; V[b] = distance_to_end."
                self.is_correct_recurrence = False
        else:
            if make_correct:
                self.rec_proposal = "V[i] = 0 if any successor j has V[j]=0 else 1 + min over successors j of V[j]; V[b]=0."
                self.is_correct_recurrence = True
            else:
                # Wrong: uses OR but returns min without zero short-circuit, or uses max
                if random.random() < 0.5:
                    self.rec_proposal = "V[i] = 1 + min over successors j of V[j]; V[b]=0."
                else:
                    self.rec_proposal = "V[i] = max over successors j of V[j]; V[b]=0."
                self.is_correct_recurrence = False

        # Keep transitions implicit; queries reveal local relationships derived from the recurrence semantics
        self._hidden_transitions = transitions

    def _get_instructions(self) -> str:
        example = self.sample_random_action()
        return (
            "BellmanVerifier: Determine if the proposed recurrence correctly solves the hidden DP instance.\n"
            "You can query local Bellman consistency, boundary conditions, and request limited global checks.\n"
            "Finally, submit a global verdict: yes (correct) or no (incorrect).\n"
            "\n"
            "Available actions:\n"
            "- local_check idx=k: Returns whether V[k] satisfies the proposed recurrence using hidden neighbors.\n"
            "- boundary_check idx=k: Reveals if k is a boundary and its boundary value.\n"
            "- global_check: Consumes a limited budget to verify recurrence on a random state.\n"
            "- reveal_all: Reveals the full DP table and instance description (non-terminal).\n"
            "- submit answer=yes|no: Final answer; ends the episode.\n"
            "\n"
            "Rules:\n"
            "- Queries do not change the instance.\n"
            "- Invalid actions terminate with a penalty.\n"
            "- You have a limited number of turns and limited global_check uses.\n"
            "\n"
            "Format your action as \\boxed{...}. Examples:\n"
            f"{example}\n"
        )

    def get_task_suffix(self) -> str:
        remaining = self.max_turns - self.turn_count
        return (
            f"Instance type: {self.instance_type or '[hidden]'} (structural style revealed, specifics hidden)\n"
            f"Proposed recurrence: {self.rec_proposal or '[hidden]'}\n"
            f"Turns left: {max(0, remaining)} | Global checks used: {self.used_global_checks}/{self.global_checks}\n"
            "Enter your action in \\boxed{action key=value} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()
        self.turn_count = 0
        self.done = False
        self.truncated = False
        self.used_global_checks = 0
        self.full_revealed = False
        self.query_log = []
        self._synthesize_instance()
        obs = (
            "A DP instance has been created. Your goal is to verify if the proposed recurrence is correct.\n"
            "Start by inspecting boundaries or performing local/global checks before submitting."
        )
        return obs + "\n" + self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _check_local(self, idx: int) -> str:
        if idx not in self.states:
            return f"ERROR: index {idx} out_of_range"
        succs = self._hidden_transitions.get(idx, [])
        v_here = self.true_values[idx]
        boundary = idx in self.boundary_states

        # Evaluate proposal effect at idx to see if the recurrence equality/logic holds
        if self.instance_type == "WeightedPathMinCost":
            # correct recurrence should satisfy: V[i] == min(cost+V[j]) with V[b]=0
            if not succs and not boundary:
                proposed_val = 1000
            else:
                proposed_val = min((w + self.true_values[j]) for j, w in succs) if succs else v_here
                if boundary:
                    proposed_val = 0
            ok = (v_here == proposed_val)
        elif self.instance_type == "MonotoneEditOps":
            if boundary:
                proposed_val = (self.state_count - 1 - idx)
            else:
                if succs:
                    proposed_val = 1 + min(self.true_values[j] for j, _ in succs)
                else:
                    proposed_val = 1000
            ok = (v_here == proposed_val)
        else:
            # SubsetTargetReachability
            if boundary:
                proposed_val = 0
            else:
                if succs:
                    if any(self.true_values[j] == 0 for j, _ in succs):
                        proposed_val = 0
                    else:
                        proposed_val = 1 + min(self.true_values[j] for j, _ in succs)
                else:
                    proposed_val = 1000
            ok = (v_here == proposed_val)

        detail = "satisfies" if ok else "violates"
        return f"LOCAL_CHECK idx={idx}: {detail} proposed recurrence; V[{idx}]={v_here}"

    def _do_global_check(self) -> str:
        if self.used_global_checks >= self.global_checks:
            return "ERROR: no_global_checks_remaining"
        self.used_global_checks += 1
        idx = random.choice(self.states)
        res = self._check_local(idx)
        return f"GLOBAL_CHECK sampled idx={idx}: {res}"

    def _do_boundary_check(self, idx: int) -> str:
        if idx not in self.states:
            return f"ERROR: index {idx} out_of_range"
        if idx in self.boundary_states:
            return f"BOUNDARY idx={idx}: is_boundary=True value={self.true_values[idx]}"
        else:
            return f"BOUNDARY idx={idx}: is_boundary=False"

    def _do_reveal_all(self) -> str:
        self.full_revealed = True
        values_repr = ", ".join(f"{i}:{self.true_values[i]}" for i in self.states)
        boundaries_repr = ", ".join(map(str, self.boundary_states))
        return (
            f"REVEAL_ALL: type={self.instance_type}; proposed='{self.rec_proposal}';\n"
            f"states={self.state_count}; branching~{self.branching};\n"
            f"boundaries=[{boundaries_repr}]\n"
            f"V= {{{values_repr}}}"
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self.done:
            return "EPISODE_ALREADY_ENDED", 0.0, True, self.truncated, {"suffix": self.get_task_suffix()}

        self.turn_count += 1
        parsed = self._parse_action(action)
        if not parsed:
            obs = "INVALID ACTION FORMAT: Use \\boxed{...} with supported action and optional key=value."
            self.done = True
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        name = parsed.get("action", "")
        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        if name == "local_check":
            try:
                idx = int(parsed.get("idx", "-1"))
            except ValueError:
                obs = "ERROR: malformed_parameter idx"
                self.done = True
                return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
            obs = self._check_local(idx)
            self.query_log.append(obs)

        elif name == "boundary_check":
            try:
                idx = int(parsed.get("idx", "-1"))
            except ValueError:
                obs = "ERROR: malformed_parameter idx"
                self.done = True
                return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
            obs = self._do_boundary_check(idx)
            self.query_log.append(obs)

        elif name == "global_check":
            obs = self._do_global_check()
            self.query_log.append(obs)

        elif name == "reveal_all":
            obs = self._do_reveal_all()
            self.query_log.append("REVEALED_ALL")

        elif name == "submit":
            ans = parsed.get("answer", "").lower()
            if ans not in ["yes", "no"]:
                obs = "ERROR: submit requires answer=yes|no"
                self.done = True
                return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
            guess_is_correct = (ans == "yes")
            if guess_is_correct == self.is_correct_recurrence:
                obs = "Success! Your global verdict matches the true recurrence correctness."
                reward = 1.0
            else:
                obs = "Failed! Your global verdict is incorrect."
                reward = 0.0
            terminated = True
            self.done = True

        else:
            obs = "ERROR: unsupported_action"
            self.done = True
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        if not terminated:
            if self.turn_count >= self.max_turns:
                obs = f"TIMEOUT: Reached max turns ({self.max_turns})."
                terminated = True
                truncated = True
                self.done = True

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if action is None:
            return None
        matches = re.findall(r"\\boxed\{(.+?)\}", action, flags=re.DOTALL)
        if not matches:
            return None
        inner = matches[-1].strip()
        if not inner:
            return None
        parts = inner.split()
        tokens: Dict[str, Any] = {}
        tokens["action"] = parts[0]
        for part in parts[1:]:
            if "=" in part:
                k, v = part.split("=", 1)
                tokens[k.strip()] = v.strip()
        return tokens

    def sample_random_action(self) -> str:
        # Provide a plausible random action example
        choice = random.choice(["local_check", "boundary_check", "global_check", "reveal_all", "submit"])
        if choice == "local_check":
            return r"\boxed{local_check idx=0}"
        if choice == "boundary_check":
            return r"\boxed{boundary_check idx=1}"
        if choice == "global_check":
            return r"\boxed{global_check}"
        if choice == "reveal_all":
            return r"\boxed{reveal_all}"
        ans = random.choice(["yes", "no"])
        return rf"\boxed{{submit answer={ans}}}"


class BellmanVerifierEnvWithFeedback(BellmanVerifierEnv):
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
            hint = "Wrap your action like \\boxed{local_check idx=3}."

        elif "error: unsupported_action" in text:
            error_type = "UnsupportedAction"
            error_detail["supported"] = ["local_check", "boundary_check", "global_check", "reveal_all", "submit"]
            hint = "Use one of the supported actions."

        elif "error: malformed_parameter" in text:
            error_type = "FormatError"
            error_detail["issue"] = "bad_parameter_value"
            hint = "Ensure numeric parameters are integers, e.g., idx=2."

        elif "error: index" in text and "out_of_range" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "index_out_of_range"
            hint = "Query indices must be within current state range; try a smaller idx."

        elif "error: no_global_checks_remaining" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "global_check_budget_exhausted"
            hint = "Avoid using global_check beyond quota; use local_check or boundary_check instead."

        elif "error: submit requires" in text:
            error_type = "FormatError"
            error_detail["issue"] = "submit_missing_answer"
            hint = "Submit like \\boxed{submit answer=yes} or \\boxed{submit answer=no}."

        elif "timeout: reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Plan: do a few local/global checks, then submit before turns expire."

        elif "failed! your global verdict is incorrect" in text:
            error_type = "WrongDecision"
            error_detail["expected_truth"] = self.is_correct_recurrence
            hint = "Cross-check a few random states with local_check and verify boundary conditions before submitting."

        elif "success! your global verdict matches" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = "Good job. Strategy: sample local checks across boundaries and interior."

        elif text.startswith("episode_already_ended"):
            error_type = "ProtocolViolation"
            error_detail["violation"] = "acted_after_termination"
            hint = "Start a new episode by calling reset."

        else:
            # Normal step info
            if "global_check" in text:
                error_type = "OK"
                error_detail["action"] = "global_check"
                hint = "Use remaining global checks sparingly; they are limited."
            elif "boundary" in text:
                error_type = "OK"
                error_detail["action"] = "boundary_check"
                hint = "Boundary values help validate base cases of the recurrence."
            elif "local_check" in text:
                error_type = "OK"
                error_detail["action"] = "local_check"
                hint = "Probe a mix of interior and boundary-adjacent states."
            elif "reveal_all" in text:
                error_type = "OK"
                error_detail["action"] = "reveal_all"
                hint = "Use reveal_all to confirm before submitting, but still remember to submit."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state_info"] = {
                "global_checks_used": getattr(self, "used_global_checks", None),
                "global_checks_quota": getattr(self, "global_checks", None),
                "full_revealed": getattr(self, "full_revealed", None),
                "state_count": getattr(self, "state_count", None),
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
            "hint": "Begin with boundary_check on a few indices, then perform 1-2 local_check calls before deciding.",
            "turn": 0,
            "state_info": {
                "global_checks_used": 0,
                "global_checks_quota": getattr(self, "global_checks", None),
                "full_revealed": False,
                "state_count": getattr(self, "state_count", None),
            },
        }
        return obs, info
