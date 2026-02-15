from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class AlgoReliquaryEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 30,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 30

        # Evolvable parameters
        self.complexity_params = {
            # Number of elements in the underlying array: larger = more computation and search space
            "array_length": (6, 40),
            # Value magnitude bound for elements (uniform in [-value_bound, value_bound]): larger = harder to compare and reason
            "value_bound": (5, 50),
            # Allowed range (window) for indices used in queries (forces subarray focus): smaller window = harder search constraints (REVERSED)
            "window_span": (6, 3),
            # Number of distractor tool calls allowed that reveal misleading summaries; more distractors = harder to avoid traps
            "num_distractors": (0, 4),
            # Limit on building auxiliary tables; fewer tables allowed = harder (REVERSED)
            "aux_budget": (3, 1),
        }

        # Variances
        self.param_variance = {
            "array_length": 2,      # ±2 around interpolated length
            "value_bound": 3,       # ±3
            "window_span": 0,       # discrete small set, keep stable
            "num_distractors": 1,   # ±1
            "aux_budget": 0,        # keep stable to avoid infeasible
        }

        # Placeholder attributes
        self.array_length: int = 0
        self.value_bound: int = 0
        self.window_span: int = 0
        self.num_distractors: int = 0
        self.aux_budget: int = 0

        # State
        self.turn_count: int = 0
        self.episode_over: bool = False
        self.terminated_reason: str = ""
        self.base_array = []
        self.target_metric: Optional[int] = None
        self.window_left: int = 0
        self.window_right: int = 0

        # Workspace
        self.aux_structs: Dict[str, Any] = {}  # e.g., {"prefix": [...], "sorted_prefix": [...]} etc.
        self.aux_used: int = 0
        self.best_known: Dict[str, Any] = {"partial": None}
        self.computed_summaries: Dict[str, Any] = {}
        self.distractors_left: int = 0

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
            # Clamp with reversed support
            lo = min(self.complexity_params[param_name][0], self.complexity_params[param_name][1])
            hi = max(self.complexity_params[param_name][0], self.complexity_params[param_name][1])
            actual_value = max(lo, min(hi, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        return (
            "You are in the AlgoReliquary. Analyze a hidden integer array within a visible index window.\n"
            "Goal: Compute and submit the maximum subarray sum that lies entirely within the current index window.\n"
            "The internal array A has length N. Values are integers and may be negative.\n"
            "You can build auxiliary structures (limited by aux_budget) and query deterministic summaries.\n"
            "Success: Submit the correct scalar value. Failure: Wrong submission or protocol errors.\n"
            "\n"
            "Available actions (use \\boxed{...}):\n"
            "- build name=<id> type=<prefix|dpmax|range_min_prefix>: Consume aux budget to create an auxiliary structure.\n"
            "  prefix: builds prefix sums P where P[0]=0, P[i]=sum(A[0..i-1]).\n"
            "  dpmax: builds Kadane-style DP arrays within window [L,R]: best_end[i], best_any[i].\n"
            "  range_min_prefix: builds RMQ-like sparse minima on prefix sums for O(1) min prefix query (simulated).\n"
            "- query type=<sum|min_prefix|summary> i=<int> j=<int> (when needed):\n"
            "  sum: returns sum(A[i..j]) requiring a built 'prefix'.\n"
            "  min_prefix: returns min(P[k] for k in [i..j]) requiring 'range_min_prefix'.\n"
            "  summary: returns (L,R,N,aux_used,aux_budget,distractors_left).\n"
            "- derive type=<window_max>: derives the maximum subarray sum within [L,R] using built structures.\n"
            "- distract: consume a distractor to receive a misleading but consistent factoid (does not help solve).\n"
            "- submit value=<int>: final answer for the maximum subarray sum inside the window.\n"
            "\n"
            "Rules:\n"
            "- Indices are 0-based and inclusive for i..j; must satisfy L <= i <= j <= R.\n"
            "- Building a structure consumes aux_budget; cannot build duplicates with same name.\n"
            "- Some queries require specific structures; calling them without prerequisites is a protocol violation.\n"
            "- Only 'submit' ends the episode with success/failure, or errors/format issues terminate early.\n"
            "- You have limited turns.\n"
            "\n"
            "Action format:\n"
            "- \\boxed{build name=p type=prefix}\n"
            "- \\boxed{query type=sum i=2 j=5}\n"
            "- \\boxed{derive type=window_max}\n"
            "- \\boxed{distract}\n"
            "- \\boxed{submit value=-3}\n"
            "\n"
            f"Example: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        arr_hint = f"N={len(self.base_array)}; visible window=[{self.window_left},{self.window_right}]; turns_left={max(0, self.max_turns - self.turn_count)}"
        aux_hint = f"aux_used={self.aux_used}/{self.aux_budget}; distractors_left={self.distractors_left}"
        return (
            f"State: {arr_hint}; {aux_hint}. "
            "Use \\boxed{...} to act. Build needed structures, query within the window, then derive/submit."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()

        self.turn_count = 0
        self.episode_over = False
        self.terminated_reason = ""
        self.aux_structs = {}
        self.aux_used = 0
        self.best_known = {"partial": None}
        self.computed_summaries = {}
        self.distractors_left = self.num_distractors

        # Generate array and window
        self.base_array = [random.randint(-self.value_bound, self.value_bound) for _ in range(self.array_length)]
        # Choose window_span, ensure valid
        span = max(1, min(self.window_span, self.array_length))
        start = random.randint(0, self.array_length - span)
        self.window_left = start
        self.window_right = start + span - 1

        # Compute ground truth: maximum subarray sum within [L, R]
        L, R = self.window_left, self.window_right
        current_max = None
        best = -10**9
        running = 0
        for i in range(L, R + 1):
            running = 0
            for j in range(i, R + 1):
                running += self.base_array[j]
                if running > best:
                    best = running
        self.target_metric = best

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self.episode_over:
            return "Episode already ended.", 0.0, True, False, {"suffix": self.get_task_suffix()}

        self.turn_count += 1
        parsed = self._parse_action(action)
        if not parsed:
            self.episode_over = True
            self.terminated_reason = "format_error"
            obs = "INVALID ACTION FORMAT: Use \\boxed{...} with supported parameters."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        name = parsed.get("action", "").lower()

        if name == "build":
            bname = parsed.get("name")
            btype = parsed.get("type")
            if not bname or not btype:
                self.episode_over = True
                self.terminated_reason = "protocol_violation"
                return ("PROTOCOL VIOLATION: build requires name and type.",
                        0.0, True, False, {"suffix": self.get_task_suffix()})
            if self.aux_used >= self.aux_budget:
                self.episode_over = True
                self.terminated_reason = "protocol_violation"
                return ("PROTOCOL VIOLATION: aux_budget exhausted.",
                        0.0, True, False, {"suffix": self.get_task_suffix()})
            if bname in self.aux_structs:
                self.episode_over = True
                self.terminated_reason = "protocol_violation"
                return ("PROTOCOL VIOLATION: duplicate auxiliary name.",
                        0.0, True, False, {"suffix": self.get_task_suffix()})
            if btype == "prefix":
                P = [0]
                acc = 0
                for v in self.base_array:
                    acc += v
                    P.append(acc)
                self.aux_structs[bname] = {"type": "prefix", "data": P}
                self.aux_used += 1
                obs = f"Built prefix as '{bname}'."
            elif btype == "dpmax":
                L, R = self.window_left, self.window_right
                best_end = [0]*(self.array_length)
                best_any = [0]*(self.array_length)
                cur = -10**9
                for i in range(self.array_length):
                    if i < L or i > R:
                        best_end[i] = None
                        best_any[i] = None
                        continue
                    if i == L:
                        cur = self.base_array[i]
                        best_end[i] = cur
                        best_any[i] = cur
                    else:
                        cur = max(self.base_array[i], cur + self.base_array[i])
                        best_end[i] = cur
                        best_any[i] = max(best_any[i-1], cur)
                self.aux_structs[bname] = {"type": "dpmax", "best_end": best_end, "best_any": best_any}
                self.aux_used += 1
                obs = f"Built windowed DP max as '{bname}'."
            elif btype == "range_min_prefix":
                P = [0]
                acc = 0
                for v in self.base_array:
                    acc += v
                    P.append(acc)
                # Simulated RMQ: store sparse blocks minima for deterministic mock
                block = 4
                blocks = []
                for i in range(0, len(P), block):
                    blocks.append(min(P[i:i+block]))
                self.aux_structs[bname] = {"type": "range_min_prefix", "P": P, "block": block, "blocks_min": blocks}
                self.aux_used += 1
                obs = f"Built range_min_prefix as '{bname}'."
            else:
                self.episode_over = True
                self.terminated_reason = "unsupported_action"
                return ("UNSUPPORTED ACTION: unknown build type.",
                        0.0, True, False, {"suffix": self.get_task_suffix()})
            reward = 0.0

        elif name == "query":
            qtype = parsed.get("type")
            if qtype == "summary":
                L, R = self.window_left, self.window_right
                obs = f"SUMMARY: L={L}, R={R}, N={len(self.base_array)}, aux_used={self.aux_used}/{self.aux_budget}, distractors_left={self.distractors_left}"
                reward = 0.0
            elif qtype in ("sum", "min_prefix"):
                try:
                    i = int(parsed.get("i", ""))
                    j = int(parsed.get("j", ""))
                except ValueError:
                    self.episode_over = True
                    self.terminated_reason = "protocol_violation"
                    return ("PROTOCOL VIOLATION: i and j must be integers.",
                            0.0, True, False, {"suffix": self.get_task_suffix()})
                L, R = self.window_left, self.window_right
                if not (L <= i <= j <= R):
                    self.episode_over = True
                    self.terminated_reason = "protocol_violation"
                    return ("PROTOCOL VIOLATION: indices out of window bounds.",
                            0.0, True, False, {"suffix": self.get_task_suffix()})
                if qtype == "sum":
                    # Need prefix
                    prefix_found = None
                    for k, v in self.aux_structs.items():
                        if v.get("type") == "prefix":
                            prefix_found = v["data"]
                            break
                    if prefix_found is None:
                        self.episode_over = True
                        self.terminated_reason = "protocol_violation"
                        return ("PROTOCOL VIOLATION: 'sum' requires a built prefix structure.",
                                0.0, True, False, {"suffix": self.get_task_suffix()})
                    s = prefix_found[j+1] - prefix_found[i]
                    obs = f"SUM[{i},{j}]={s}"
                    reward = 0.0
                else:
                    # min_prefix requires range_min_prefix
                    rmq = None
                    for k, v in self.aux_structs.items():
                        if v.get("type") == "range_min_prefix":
                            rmq = v
                            break
                    if rmq is None:
                        self.episode_over = True
                        self.terminated_reason = "protocol_violation"
                        return ("PROTOCOL VIOLATION: 'min_prefix' requires 'range_min_prefix' structure.",
                                0.0, True, False, {"suffix": self.get_task_suffix()})
                    P = rmq["P"]
                    # Need min of P[i..j] on prefix index domain
                    ii = max(0, min(i, len(P)-1))
                    jj = max(0, min(j, len(P)-1))
                    if ii > jj:
                        ii, jj = jj, ii
                    mval = min(P[ii:jj+1])
                    obs = f"MIN_PREFIX[{ii},{jj}]={mval}"
                    reward = 0.0
            else:
                self.episode_over = True
                self.terminated_reason = "unsupported_action"
                return ("UNSUPPORTED ACTION: unknown query type.",
                        0.0, True, False, {"suffix": self.get_task_suffix()})

        elif name == "derive":
            dtype = parsed.get("type")
            if dtype != "window_max":
                self.episode_over = True
                self.terminated_reason = "unsupported_action"
                return ("UNSUPPORTED ACTION: unknown derive type.",
                        0.0, True, False, {"suffix": self.get_task_suffix()})
            # Try to derive using available structures
            L, R = self.window_left, self.window_right
            # If dpmax exists, we can read best_any[R]
            used_dp = None
            for k, v in self.aux_structs.items():
                if v.get("type") == "dpmax":
                    used_dp = v
                    break
            if used_dp:
                best_any = used_dp["best_any"]
                candidate = best_any[R]
                if candidate is None:
                    # If DP wasn't filled (should not happen), fail gracefully
                    obs = "DERIVE: dpmax incomplete; cannot derive."
                    reward = 0.0
                else:
                    self.best_known["partial"] = candidate
                    obs = f"DERIVE: window_max candidate={candidate}"
                    reward = 0.6
            else:
                # Try fallback with prefix + min_prefix if both exist
                prefix = None
                rmq = None
                for k, v in self.aux_structs.items():
                    if v.get("type") == "prefix":
                        prefix = v["data"]
                    if v.get("type") == "range_min_prefix":
                        rmq = v
                if prefix is not None and rmq is not None:
                    # Compute exact max subarray in [L,R] by scanning with min-prefix up to current:
                    best = -10**9
                    min_pref = prefix[L]
                    for j in range(L, R+1):
                        # best subarray ending at j is prefix[j+1] - min(prefix[L..j+1-1])
                        if prefix[j] < min_pref:
                            min_pref = prefix[j]
                        val = prefix[j+1] - min_pref
                        if val > best:
                            best = val
                    self.best_known["partial"] = best
                    obs = f"DERIVE: window_max candidate={best}"
                    reward = 0.6
                else:
                    obs = "DERIVE: prerequisites missing (need dpmax or prefix+range_min_prefix)."
                    reward = 0.0

        elif name == "distract":
            if self.distractors_left <= 0:
                self.episode_over = True
                self.terminated_reason = "protocol_violation"
                return ("PROTOCOL VIOLATION: no distractors left.",
                        0.0, True, False, {"suffix": self.get_task_suffix()})
            self.distractors_left -= 1
            # Provide a misleading but consistent factoid: e.g., parity of sum outside window
            total_sum = sum(self.base_array)
            outside_sum = total_sum - sum(self.base_array[self.window_left:self.window_right+1])
            parity = "even" if outside_sum % 2 == 0 else "odd"
            obs = f"DISTRACTOR: sum outside window is {parity}."
            reward = 0.0

        elif name == "submit":
            val = parsed.get("value")
            if val is None:
                self.episode_over = True
                self.terminated_reason = "protocol_violation"
                return ("PROTOCOL VIOLATION: submit requires value=<int>.",
                        0.0, True, False, {"suffix": self.get_task_suffix()})
            try:
                ival = int(val)
            except ValueError:
                self.episode_over = True
                self.terminated_reason = "protocol_violation"
                return ("PROTOCOL VIOLATION: submit value must be an integer.",
                        0.0, True, False, {"suffix": self.get_task_suffix()})
            self.episode_over = True
            if ival == self.target_metric:
                obs = f"Success! Correct maximum subarray sum within window is {ival}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Failed! Submitted {ival}, correct was {self.target_metric}."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            self.episode_over = True
            self.terminated_reason = "unsupported_action"
            return ("UNSUPPORTED ACTION: unknown action name.",
                    0.0, True, False, {"suffix": self.get_task_suffix()})

        # Timeout check
        if self.turn_count >= self.max_turns:
            self.episode_over = True
            obs = f"TIMEOUT: Reached max turns ({self.max_turns})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return f"{obs}", reward, False, False, {"suffix": self.get_task_suffix()}

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
        tokens: Dict[str, Any] = {"action": parts[0]}
        for part in parts[1:]:
            if "=" in part:
                k, v = part.split("=", 1)
                tokens[k] = v
        return tokens

    def sample_random_action(self) -> str:
        choices = [
            r"\boxed{build name=p type=prefix}",
            r"\boxed{build name=d type=dpmax}",
            r"\boxed{build name=rm type=range_min_prefix}",
            r"\boxed{query type=summary}",
            r"\boxed{query type=sum i=%d j=%d}" % (self.window_left, self.window_left),
            r"\boxed{derive type=window_max}",
            r"\boxed{distract}",
            r"\boxed{submit value=0}",
        ]
        return random.choice(choices)


class AlgoReliquaryEnvWithFeedback(AlgoReliquaryEnv):
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
            hint = "Wrap your command in \\boxed{...} and include required parameters."
        elif text.startswith("protocol violation"):
            error_type = "ProtocolViolation"
            if "aux_budget exhausted" in text:
                error_detail["violation"] = "aux_budget_exhausted"
                hint = "Avoid duplicate builds; use query/derive or submit."
            elif "duplicate auxiliary name" in text:
                error_detail["violation"] = "duplicate_aux_name"
                hint = "Use a unique name for each build."
            elif "requires a built prefix" in text:
                error_detail["violation"] = "missing_prefix"
                hint = "Build a prefix structure first: \\boxed{build name=p type=prefix}."
            elif "requires 'range_min_prefix'" in text:
                error_detail["violation"] = "missing_range_min_prefix"
                hint = "Build range_min_prefix: \\boxed{build name=rm type=range_min_prefix}."
            elif "indices out of window bounds" in text:
                error_detail["violation"] = "index_out_of_bounds"
                hint = "Keep i and j within the visible window [L,R] from the summary."
            elif "i and j must be integers" in text:
                error_detail["violation"] = "bad_indices"
                hint = "Provide integer indices: e.g., \\boxed{query type=sum i=L j=R}."
            elif "submit requires value" in text:
                error_detail["violation"] = "missing_submit_value"
                hint = "Provide an integer: \\boxed{submit value=...}."
            elif "submit value must be an integer" in text:
                error_detail["violation"] = "non_integer_submit"
                hint = "Submit an integer value (no decimals or text)."
            elif "no distractors left" in text:
                error_detail["violation"] = "distractor_budget_exhausted"
                hint = "Focus on build/query/derive; avoid distract."
            else:
                error_detail["violation"] = "generic_protocol"
                hint = "Check action parameters and prerequisites per instructions."
        elif text.startswith("unsupported action"):
            error_type = "UnsupportedAction"
            if "build type" in text:
                error_detail["unsupported"] = "build_type"
                hint = "Use one of: prefix, dpmax, range_min_prefix."
            elif "query type" in text:
                error_detail["unsupported"] = "query_type"
                hint = "Use one of: sum, min_prefix, summary."
            elif "derive type" in text:
                error_detail["unsupported"] = "derive_type"
                hint = "Use: derive type=window_max."
            else:
                error_detail["unsupported"] = "action_name"
                hint = "Use actions: build, query, derive, distract, submit."
        elif text.startswith("failed!"):
            error_type = "WrongDecision"
            # Try to extract numbers
            got = None
            correct = None
            mg = re.search(r"submitted\s+(-?\d+)", text)
            mc = re.search(r"correct\s+was\s+(-?\d+)", text)
            if mg:
                got = int(mg.group(1))
            if mc:
                correct = int(mc.group(1))
            error_detail["got"] = got
            error_detail["correct"] = correct
            hint = "Use derive after building dpmax or prefix+range_min_prefix to compute the correct value."
        elif "timeout" in text:
            error_type = "Timeout"
            error_detail["turn_limit"] = getattr(self, "max_turns", None)
            hint = "Plan fewer builds and go straight to derive and submit."
        elif text.startswith("success"):
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None
        else:
            error_type = "OK"
            if "built" in text:
                error_detail["progress"] = "build_done"
                hint = "Next, query summaries or attempt derive."
            elif "sum[" in text or "min_prefix" in text or "summary:" in text:
                error_detail["progress"] = "query_done"
                hint = "Leverage results to derive the window max."
            elif "derive: window_max candidate" in text:
                error_detail["progress"] = "candidate_found"
                hint = "If confident, submit your candidate."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["window"] = [getattr(self, "window_left", None), getattr(self, "window_right", None)]
            diagnostic["aux_used"] = getattr(self, "aux_used", None)
            diagnostic["aux_budget"] = getattr(self, "aux_budget", None)
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start by building dpmax or prefix+range_min_prefix, then derive.",
            "turn": 0,
            "window": [getattr(self, "window_left", None), getattr(self, "window_right", None)],
            "aux_used": getattr(self, "aux_used", None),
            "aux_budget": getattr(self, "aux_budget", None),
        }
        return obs, info