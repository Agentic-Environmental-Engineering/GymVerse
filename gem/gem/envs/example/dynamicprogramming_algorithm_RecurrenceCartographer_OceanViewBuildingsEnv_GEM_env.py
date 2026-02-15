from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class RecurrenceCartographerEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 28,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 28

        # Evolvable parameters (DP-native scales)
        self.complexity_params = {
            # Number of items in 0/1 knapsack; more items -> larger state space and deeper recursion -> harder
            "num_items": (4, 18),
            # Capacity of knapsack; larger capacity -> larger second dimension of DP table -> harder
            "capacity": (10, 60),
            # Value magnitude upper bound; larger spread -> less trivial heuristics, requires precise DP -> harder
            "max_value": (8, 50),
            # Weight magnitude upper bound; larger spread -> denser feasible set and trickier transitions -> harder
            "max_weight": (6, 30),
            # REVERSED: number of free base-case reveals allowed; fewer freebies -> harder
            "free_base_reveals": (6, 1),
            # REVERSED: whether table_dump is allowed (1=yes, 0=no); removing dump forces targeted queries -> harder
            "allow_table_dump": (1, 0),
        }

        # Variance settings (adds variety without breaking solvability)
        self.param_variance = {
            "num_items": 1,
            "capacity": 3,
            "max_value": 4,
            "max_weight": 3,
            "free_base_reveals": 1,
            "allow_table_dump": 0,
        }

        # Placeholders set in _apply_complexity_params
        self.num_items: int = 0
        self.capacity: int = 0
        self.max_value: int = 0
        self.max_weight: int = 0
        self.free_base_reveals: int = 0
        self.allow_table_dump: int = 0

        # Domain state
        self.items = []  # list of (weight, value)
        self.turn_count = 0
        self.terminated = False
        self.truncated = False
        self.history = []
        self.revealed = {}  # (i,c) -> value revealed to agent
        self.used_free_reveals = 0
        self.final_answer_submitted = False
        self.reference_opt = 0

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for name, (min_v, max_v) in self.complexity_params.items():
            center = min_v + (max_v - min_v) * normalized
            val = center
            if self.enable_param_randomization:
                var = self.param_variance.get(name, 0)
                if var > 0:
                    val = center + random.uniform(-var, var)
            # Clamp across reversed or normal ranges
            lo = min(min_v, max_v)
            hi = max(min_v, max_v)
            val = max(lo, min(hi, val))
            setattr(self, name, int(round(val)))

    def _build_instance(self):
        # Generate items with constraints; ensure at least one item fits capacity
        self.items = []
        trials = 0
        while True:
            trials += 1
            self.items = []
            for _ in range(self.num_items):
                w = random.randint(1, max(1, self.max_weight))
                v = random.randint(1, max(2, self.max_value))
                self.items.append((w, v))
            if any(w <= self.capacity for w, _ in self.items):
                break
            if trials > 50:
                # Fallback: force one light item
                self.items[0] = (max(1, self.capacity // 3), max(1, self.max_value // 2))
                break

    def _compute_opt(self):
        # Standard 0/1 knapsack DP: V(i,c) = best using first i items with capacity c
        n = self.num_items
        C = self.capacity
        dp = [[0] * (C + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            w, v = self.items[i - 1]
            for c in range(C + 1):
                dp[i][c] = dp[i - 1][c]
                if w <= c:
                    cand = dp[i - 1][c - w] + v
                    if cand > dp[i][c]:
                        dp[i][c] = cand
        self.reference_opt = dp[n][C]
        self._dp_table = dp  # internal exact table for queries

    def _get_instructions(self) -> str:
        return (
            "You are the Recurrence Cartographer, exploring a dynamic program for 0/1 Knapsack.\n"
            "Goal: Determine the optimal achievable value using the first N items within capacity C.\n"
            "Items are (weight, value). The DP state is V(i,c): best value using first i items and capacity c.\n"
            "Base cases: V(0,c)=0 for all c, and V(i,0)=0 for all i.\n"
            "Transition: V(i,c)=max(V(i-1,c), V(i-1,c-w_i)+v_i) if w_i<=c, else V(i-1,c).\n"
            "You may query and compute subproblems and finally SUBMIT the optimal value.\n"
            "Available actions:\n"
            "- prompt: Reprints problem summary.\n"
            "- list_items: Shows all items (index, weight, value).\n"
            "- get i=? c=? : Reveal exact value of V(i,c). i in [0..N], c in [0..C].\n"
            "- eval i=? c=? : Request the recurrence breakdown for V(i,c) (which branch wins and values).\n"
            "- table_shape: Returns dimensions (N+1) x (C+1).\n"
            "- dump_table: If allowed, reveals the entire DP table.\n"
            "- base_free: Consume one free base-case reveal, returns a random base V(0,c) or V(i,0). Limited uses.\n"
            "- submit ans=? : Finalize with your proposed optimal value.\n"
            "Rules:\n"
            "- Actions must be in \\boxed{...}.\n"
            "- Indices must be integers within range.\n"
            "- Episode ends on submit, invalid format, or timeout.\n"
            "Reward: 1.0 for exact correct submit; 0.0 for incorrect or timeout; penalties only for format errors.\n"
            "Example actions:\n"
            f"- {r'\\boxed{list_items}'}\n"
            f"- {r'\\boxed{get i=3 c=7}'}\n"
            f"- {r'\\boxed{eval i=5 c=10}'}\n"
            f"- {r'\\boxed{submit ans=42}'}\n"
        )

    def get_task_suffix(self) -> str:
        status = []
        status.append(f"Turn {self.turn_count}/{self.max_turns}")
        status.append(f"N={self.num_items}, C={self.capacity}")
        status.append(f"Free base reveals used: {self.used_free_reveals}/{self.free_base_reveals}")
        status.append(f"Table dump allowed: {'yes' if self.allow_table_dump==1 else 'no'}")
        return (
            "State summary:\n"
            + "\n".join(status)
            + "\nEnter your action in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()
        self.turn_count = 0
        self.terminated = False
        self.truncated = False
        self.history = []
        self.revealed = {}
        self.used_free_reveals = 0
        self.final_answer_submitted = False

        self._build_instance()
        self._compute_opt()

        intro = self._get_instructions()
        info = {"suffix": self.get_task_suffix()}
        return intro, info

    def _validate_ic(self, i: int, c: int) -> Optional[str]:
        if i < 0 or i > self.num_items:
            return f"ERROR: i out of range [0..{self.num_items}]"
        if c < 0 or c > self.capacity:
            return f"ERROR: c out of range [0..{self.capacity}]"
        return None

    def _dp_value(self, i: int, c: int) -> int:
        return self._dp_table[i][c]

    def _eval_text(self, i: int, c: int) -> str:
        if i == 0 or c == 0:
            return f"V({i},{c}) = 0 (base case)"
        w, v = self.items[i - 1]
        if w > c:
            return f"V({i},{c}) = V({i-1},{c}) = {self._dp_table[i-1][c]} (item {i} too heavy: w={w} > c={c})"
        without = self._dp_table[i - 1][c]
        with_it = self._dp_table[i - 1][c - w] + v
        choice = "with item" if with_it >= without else "without item"
        return (
            f"V({i},{c}) = max(V({i-1},{c})={without}, V({i-1},{c-w}={c-w})+v_i={v} -> {with_it}); "
            f"choose {choice} => {self._dp_table[i][c]}"
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self.terminated or self.truncated:
            return "Episode already ended.", 0.0, True, self.truncated, {"suffix": self.get_task_suffix()}

        self.turn_count += 1
        parsed = self._parse_action(action)
        if not parsed:
            obs = "INVALID ACTION FORMAT: Use \\boxed{...} with a supported action."
            self.terminated = True
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        act = parsed.get("action", "")
        obs = ""
        reward = 0.0
        term = False
        trunc = False

        if act == "prompt":
            obs = self._get_instructions()

        elif act == "list_items":
            lines = []
            for idx, (w, v) in enumerate(self.items, start=1):
                lines.append(f"{idx}: w={w}, v={v}")
            obs = "Items:\n" + ("\n".join(lines) if lines else "(none)")

        elif act == "table_shape":
            obs = f"Table shape: {(self.num_items+1)} rows x {(self.capacity+1)} cols (i=0..N, c=0..C)."

        elif act == "dump_table":
            if self.allow_table_dump == 0:
                obs = "ERROR: table dump not allowed at this complexity."
            else:
                lines = []
                for i in range(self.num_items + 1):
                    row = " ".join(str(self._dp_table[i][c]) for c in range(self.capacity + 1))
                    lines.append(f"i={i}: {row}")
                obs = "DP Table:\n" + "\n".join(lines)

        elif act == "base_free":
            if self.used_free_reveals >= self.free_base_reveals:
                obs = "ERROR: no remaining free base-case reveals."
            else:
                # Reveal a random base case: either (0,c) or (i,0)
                choice = random.choice(["row0", "col0"])
                if choice == "row0":
                    c = random.randint(0, self.capacity)
                    self.revealed[(0, c)] = 0
                    obs = f"Reveal base: V(0,{c})=0"
                else:
                    i = random.randint(0, self.num_items)
                    self.revealed[(i, 0)] = 0
                    obs = f"Reveal base: V({i},0)=0"
                self.used_free_reveals += 1

        elif act == "get":
            try:
                i = int(parsed.get("i"))
                c = int(parsed.get("c"))
            except Exception:
                obs = "ERROR: get requires integer i and c."
            else:
                err = self._validate_ic(i, c)
                if err:
                    obs = err
                else:
                    val = self._dp_value(i, c)
                    self.revealed[(i, c)] = val
                    obs = f"V({i},{c}) = {val}"

        elif act == "eval":
            try:
                i = int(parsed.get("i"))
                c = int(parsed.get("c"))
            except Exception:
                obs = "ERROR: eval requires integer i and c."
            else:
                err = self._validate_ic(i, c)
                if err:
                    obs = err
                else:
                    obs = self._eval_text(i, c)

        elif act == "submit":
            # Finalize with ans=?
            ans_raw = parsed.get("ans")
            if ans_raw is None:
                obs = "ERROR: submit requires ans=<integer>."
                term = True
            else:
                try:
                    ans_val = int(ans_raw)
                except Exception:
                    obs = "ERROR: ans must be an integer."
                    term = True
                else:
                    self.final_answer_submitted = True
                    if ans_val == self.reference_opt:
                        obs = f"Success! Optimal value = {self.reference_opt}."
                        reward = 1.0
                    else:
                        obs = f"Failed! Your answer {ans_val} != optimal {self.reference_opt}."
                        reward = 0.0
                    term = True

        else:
            obs = "ERROR: unsupported action."
            # Unsupported action ends episode as per strict protocol
            term = True

        if not term and not trunc and self.turn_count >= self.max_turns:
            trunc = True
            term = True
            obs = f"TIMEOUT: Reached max turns ({self.max_turns})."

        self.terminated = term
        self.truncated = trunc
        info = {"suffix": self.get_task_suffix()}
        return obs, reward, term, trunc, info

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
        act = parts[0]
        tokens: Dict[str, Any] = {"action": act}
        for p in parts[1:]:
            if "=" in p:
                k, v = p.split("=", 1)
                tokens[k.strip()] = v.strip()
        return tokens

    def sample_random_action(self) -> str:
        choices = [
            r"\boxed{list_items}",
            r"\boxed{table_shape}",
            r"\boxed{get i=1 c=3}",
            r"\boxed{eval i=2 c=5}",
            r"\boxed{base_free}",
        ]
        return random.choice(choices)


class RecurrenceCartographerEnvWithFeedback(RecurrenceCartographerEnv):
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
            error_detail["issue"] = "missing_boxed_format"
            hint = "Wrap your command inside \\boxed{...} and use supported verbs like get, eval, submit."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_command"
            hint = "Use one of: prompt, list_items, get, eval, table_shape, dump_table, base_free, submit."

        elif "error:" in text:
            # Protocol violations or argument issues
            if "out of range" in text:
                error_type = "ProtocolViolation"
                error_detail["violation"] = "index_out_of_range"
                hint = f"Ensure i in [0..{self.num_items}] and c in [0..{self.capacity}]. Use table_shape for dimensions."
            elif "requires integer" in text:
                error_type = "ProtocolViolation"
                error_detail["violation"] = "non_integer_argument"
                hint = "Provide numeric integers for i, c, and ans."
            elif "table dump not allowed" in text:
                error_type = "ProtocolViolation"
                error_detail["violation"] = "forbidden_action"
                hint = "At this difficulty, dump_table is disabled. Query specific V(i,c) or use eval."
            elif "no remaining free" in text:
                error_type = "ProtocolViolation"
                error_detail["violation"] = "exceeded_free_base"
                hint = "Free base reveals exhausted. Query base states explicitly with get i=0 c=k or get i=k c=0."
            elif "submit requires ans" in text or "ans must be an integer" in text:
                error_type = "ProtocolViolation"
                error_detail["violation"] = "bad_submit_format"
                hint = "Submit with \\boxed{submit ans=<integer>} where the value is your final optimal."

        elif "failed!" in text:
            error_type = "WrongDecision"
            # Try to extract numbers if present
            m1 = re.search(r"your answer (\-?\d+)", text)
            m2 = re.search(r"optimal (\-?\d+)", text)
            if m1:
                error_detail["got"] = int(m1.group(1))
            if m2:
                error_detail["expected"] = int(m2.group(1))
            hint = "Use eval on boundary states (i near N, c near C) and get for decisive subproblems before submitting."

        elif "timeout" in text:
            error_type = "Timeout"
            hint = "Prioritize querying V(N,C) via eval/get and avoid unnecessary actions."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["revealed_count"] = len(getattr(self, "revealed", {}))
            diagnostic["free_base_remaining"] = max(0, self.free_base_reveals - self.used_free_reveals)
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start by listing items (list_items), then probe V(N,C) with eval i=N c=C.",
            "turn": 0,
            "revealed_count": 0,
            "free_base_remaining": self.free_base_reveals,
        }
        return obs, info
