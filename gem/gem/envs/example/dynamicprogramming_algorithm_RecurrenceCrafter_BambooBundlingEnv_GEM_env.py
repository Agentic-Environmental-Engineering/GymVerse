from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class RecurrenceCrafterEnv(Env):
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
            'num_items': (3, 12),              # More items increases subproblem space → harder
            'capacity': (10, 50),              # Larger capacity increases table width → harder
            'value_max': (10, 60),             # Higher value range increases numeric variability → slightly harder
            'weight_max': (6, 30),             # Larger weights range increases complexity of feasible combinations → harder
            'auto_revealed_items': (2, 0),     # REVERSED: fewer auto reveals increase informational burden → harder
        }

        # Variance settings
        self.param_variance = {
            'num_items': 1,            # Medium discrete range → ±1
            'capacity': 5,             # Larger range → ±5 (~12% variance)
            'value_max': 5,            # Larger range → ±5 (~10% variance)
            'weight_max': 3,           # Medium range → ±3
            'auto_revealed_items': 0,  # Small range → no randomization
        }

        # Placeholder attributes set in _apply_complexity_params
        self.num_items: int = 0
        self.capacity: int = 0
        self.value_max: int = 0
        self.weight_max: int = 0
        self.auto_revealed_items: int = 0

        # Domain-specific state
        self.items: Dict[int, Tuple[int, int]] = {}        # idx -> (weight, value)
        self.revealed_items: set = set()                   # indices revealed
        self.n_revealed: bool = False
        self.cap_revealed: bool = False

        self.table_created: bool = False
        self.base_set: bool = False
        self.dp: Dict[Tuple[int, int], int] = {}           # computed cells
        self.cells_filled_count: int = 0

        self.ground_truth: int = 0
        self.turn_count: int = 0

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
                    lo = min(min_val, max_val)
                    hi = max(min_val, max_val)
                    actual_value = max(lo, min(hi, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        return (
            "You are RecurrenceCrafter: a dynamic programming environment for 0/1 knapsack.\n"
            "Goal: Submit the optimal total value achievable within the capacity using the given items.\n"
            "The instance (number of items, capacity, each item's weight/value) is partially hidden; use queries.\n"
            "You must follow the protocol to craft the DP solution:\n"
            "1) Reveal n and capacity.\n"
            "2) Create the DP table.\n"
            "3) Set base cases (i=0 row and w=0 column as 0).\n"
            "4) Reveal needed items.\n"
            "5) Compute cells dp[i,w] respecting dependencies.\n"
            "6) Submit the final optimal value.\n"
            "Actions (use \\boxed{...}):\n"
            "- reveal_n\n"
            "- reveal_capacity\n"
            "- reveal_item i=IDX\n"
            "- create_table\n"
            "- set_base\n"
            "- compute i=I w=W\n"
            "- show_cell i=I w=W\n"
            "- submit value=V\n"
            "Rules:\n"
            "- You must reveal n and capacity before create_table.\n"
            "- You must create_table before set_base or compute.\n"
            "- You must set_base before compute.\n"
            "- To compute dp[i,w], item i must be revealed and dp[i-1,w] and dp[i-1,w-weight_i] must already be computed.\n"
            "- Invalid or unsupported actions terminate the episode.\n"
            "Format: Place your action inside \\boxed{...}, with parameters as key=value.\n"
            f"Example: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        total_cells = (self.num_items + 1) * (self.capacity + 1) if (self.table_created and self.n_revealed and self.cap_revealed) else 0
        progress = f"cells_computed={self.cells_filled_count}/{total_cells}" if total_cells > 0 else "cells_computed=0/unknown"
        status = (
            f"n_revealed={self.n_revealed}, cap_revealed={self.cap_revealed}, "
            f"table_created={self.table_created}, base_set={self.base_set}, "
            f"items_revealed={len(self.revealed_items)}/{self.num_items if self.num_items>0 else 'unknown'}, "
            f"{progress}"
        )
        return (
            "Current state:\n"
            f"{status}\n"
            "Enter your action in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        # Generate instance
        self.items = {}
        for i in range(1, self.num_items + 1):
            w = random.randint(1, self.weight_max)
            v = random.randint(1, self.value_max)
            self.items[i] = (w, v)

        # Ensure feasibility: at least one item fits
        if all(self.items[i][0] > self.capacity for i in range(1, self.num_items + 1)):
            pick = random.randint(1, self.num_items)
            self.items[pick] = (max(1, self.capacity // 2), self.items[pick][1])

        # Ground truth via offline DP
        dp_gt = [[0] * (self.capacity + 1) for _ in range(self.num_items + 1)]
        for i in range(1, self.num_items + 1):
            wi, vi = self.items[i]
            for w in range(self.capacity + 1):
                if wi > w:
                    dp_gt[i][w] = dp_gt[i-1][w]
                else:
                    dp_gt[i][w] = max(dp_gt[i-1][w], dp_gt[i-1][w-wi] + vi)
        self.ground_truth = dp_gt[self.num_items][self.capacity]

        # Auto-reveal some items
        self.revealed_items = set()
        auto_count = min(self.auto_revealed_items, self.num_items)
        if auto_count > 0:
            revealed_idxs = random.sample(list(range(1, self.num_items + 1)), auto_count)
            self.revealed_items.update(revealed_idxs)

        # Reset protocol state
        self.n_revealed = False
        self.cap_revealed = False
        self.table_created = False
        self.base_set = False
        self.dp = {}
        self.cells_filled_count = 0
        self.turn_count = 0

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if not parsed:
            obs = "INVALID ACTION FORMAT: Use \\boxed{...} with a supported action."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        act = parsed.get('action', '').strip()

        def ensure_int(token_name: str) -> Optional[int]:
            val = parsed.get(token_name, None)
            if val is None:
                return None
            try:
                return int(val)
            except Exception:
                return None

        # Execute action
        if act == 'reveal_n':
            self.n_revealed = True
            obs = f"n={self.num_items}"
            reward = 0.0

        elif act == 'reveal_capacity':
            self.cap_revealed = True
            obs = f"capacity={self.capacity}"
            reward = 0.0

        elif act == 'reveal_item':
            i = ensure_int('i')
            if i is None or i < 1 or i > self.num_items:
                obs = "PROTOCOL VIOLATION: invalid item index for reveal_item."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            self.revealed_items.add(i)
            w, v = self.items[i]
            obs = f"item[{i}]: weight={w}, value={v}"
            reward = 0.0

        elif act == 'create_table':
            if not (self.n_revealed and self.cap_revealed):
                obs = "PROTOCOL VIOLATION: reveal_n and reveal_capacity required before create_table."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            self.table_created = True
            self.dp = {}
            self.cells_filled_count = 0
            obs = f"TABLE CREATED: size={(self.num_items+1)}x{(self.capacity+1)}"
            reward = 0.0

        elif act == 'set_base':
            if not self.table_created:
                obs = "PROTOCOL VIOLATION: create_table required before set_base."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            # Set i=0 row and w=0 column to 0
            for w in range(self.capacity + 1):
                key = (0, w)
                if key not in self.dp:
                    self.dp[key] = 0
                    self.cells_filled_count += 1
            for i in range(1, self.num_items + 1):
                key = (i, 0)
                if key not in self.dp:
                    self.dp[key] = 0
                    self.cells_filled_count += 1
            self.base_set = True
            obs = "BASE SET: dp[0,w]=0 and dp[i,0]=0"
            reward = 0.0

        elif act == 'compute':
            i = ensure_int('i')
            w = ensure_int('w')
            if i is None or w is None or i < 0 or i > self.num_items or w < 0 or w > self.capacity:
                obs = "PROTOCOL VIOLATION: invalid indices for compute."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            if not self.table_created:
                obs = "PROTOCOL VIOLATION: create_table required before compute."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            if not self.base_set:
                obs = "PROTOCOL VIOLATION: set_base required before compute."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            if i == 0:
                # Base row already defined; computing it again is allowed as a no-op
                val = self.dp.get((0, w), 0)
                if (0, w) not in self.dp:
                    self.dp[(0, w)] = 0
                    self.cells_filled_count += 1
                obs = f"COMPUTE dp[0,{w}]=0"
                reward = 0.0
            else:
                if i not in self.revealed_items:
                    obs = "PROTOCOL VIOLATION: reveal_item i=<idx> required before computing dp[i,w]."
                    return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
                wi, vi = self.items[i]
                # Check dependencies
                dep_a = self.dp.get((i-1, w), None)
                dep_b = None
                if w - wi >= 0:
                    dep_b = self.dp.get((i-1, w - wi), None)
                if dep_a is None or (w - wi >= 0 and dep_b is None):
                    obs = "PROTOCOL VIOLATION: dependency cells not computed (dp[i-1,w] and/or dp[i-1,w-w_i])."
                    return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
                if wi > w:
                    val = dep_a
                    obs = f"COMPUTE dp[{i},{w}] (wi={wi}>w): take dp[{i-1},{w}]={dep_a} -> {val}"
                else:
                    take = dep_b + vi
                    skip = dep_a
                    val = max(skip, take)
                    obs = f"COMPUTE dp[{i},{w}]: max(skip={skip}, take={dep_b}+{vi}={take}) -> {val}"
                if (i, w) not in self.dp:
                    self.dp[(i, w)] = val
                    self.cells_filled_count += 1
                else:
                    self.dp[(i, w)] = val
                reward = 0.0

        elif act == 'show_cell':
            i = ensure_int('i')
            w = ensure_int('w')
            if i is None or w is None or i < 0 or i > self.num_items or w < 0 or w > self.capacity:
                obs = "PROTOCOL VIOLATION: invalid indices for show_cell."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            if not self.table_created:
                obs = "PROTOCOL VIOLATION: create_table required before show_cell."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            val = self.dp.get((i, w), None)
            if val is None:
                obs = f"CELL UNKNOWN: dp[{i},{w}] not computed"
            else:
                obs = f"CELL VALUE: dp[{i},{w}]={val}"
            reward = 0.0

        elif act == 'submit':
            v = ensure_int('value')
            if v is None:
                obs = "PROTOCOL VIOLATION: submit requires numeric value."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            if v == self.ground_truth:
                obs = f"Success! Correct optimal value={v}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Failed! Submitted value={v}, correct={self.ground_truth}."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = "UNSUPPORTED ACTION: use one of [reveal_n, reveal_capacity, reveal_item, create_table, set_base, compute, show_cell, submit]."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        # Check timeout after applying action result
        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns}). Timeout."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

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
        tokens: Dict[str, Any] = {'action': parts[0]}
        for part in parts[1:]:
            if '=' in part:
                k, v = part.split('=', 1)
                tokens[k] = v
        return tokens

    def sample_random_action(self) -> str:
        return r'\boxed{reveal_n}'


class RecurrenceCrafterEnvWithFeedback(RecurrenceCrafterEnv):
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
                hint = 'Use \\boxed{...} and include parameters as key=value (e.g., \\boxed{compute i=2 w=10}).'

            elif "unsupported action" in text:
                error_type = "UnsupportedAction"
                error_detail["supported"] = ["reveal_n", "reveal_capacity", "reveal_item", "create_table", "set_base", "compute", "show_cell", "submit"]
                hint = 'Choose a supported action. Start by \\boxed{reveal_n} and \\boxed{reveal_capacity}.'

            elif "protocol violation" in text:
                error_type = "ProtocolViolation"
                # classify subtypes
                if "reveal_n and reveal_capacity required before create_table" in text:
                    error_detail["violation"] = "create_without_reveals"
                    hint = 'First run \\boxed{reveal_n} and \\boxed{reveal_capacity}, then \\boxed{create_table}.'
                elif "create_table required before set_base" in text:
                    error_detail["violation"] = "base_without_table"
                    hint = 'Call \\boxed{create_table} before \\boxed{set_base}.'
                elif "set_base required before compute" in text:
                    error_detail["violation"] = "compute_without_base"
                    hint = 'Initialize base cases using \\boxed{set_base} before computing cells.'
                elif "invalid indices for compute" in text:
                    error_detail["violation"] = "bad_indices_compute"
                    hint = 'Ensure 0 <= i <= n and 0 <= w <= capacity. Reveal n/capacity to know bounds.'
                elif "invalid item index for reveal_item" in text:
                    error_detail["violation"] = "bad_item_index"
                    hint = 'Pick i in [1..n]. Use \\boxed{reveal_n} to know n.'
                elif "reveal_item i=<idx> required before computing dp[i,w]" in text:
                    error_detail["violation"] = "compute_without_item"
                    hint = 'Reveal the item first using \\boxed{reveal_item i=I}.'
                elif "dependency cells not computed" in text:
                    error_detail["violation"] = "missing_dependencies"
                    hint = 'Compute dp[i-1,w] and dp[i-1,w-w_i] before dp[i,w]. Fill row i-1 left-to-right.'
                elif "create_table required before show_cell" in text:
                    error_detail["violation"] = "show_without_table"
                    hint = 'Create the table with \\boxed{create_table} first.'
                elif "submit requires numeric value" in text:
                    error_detail["violation"] = "submit_non_numeric"
                    hint = 'Provide an integer: \\boxed{submit value=NUMBER}.'
                else:
                    error_detail["violation"] = "generic_protocol_error"
                    hint = 'Follow the sequence: reveal_n → reveal_capacity → create_table → set_base → reveal_item → compute → submit.'

            elif "reached max turns" in text or "timeout" in text:
                error_type = "Timeout"
                error_detail["limit"] = self.max_turns
                hint = 'Be concise: reveal essentials, then compute key dependencies and submit. Avoid unnecessary actions.'

            elif "failed!" in text:
                error_type = "WrongDecision"
                # Try to extract submitted and correct values
                m_sub = re.search(r"submitted value=(\d+)", obs, flags=re.IGNORECASE)
                m_cor = re.search(r"correct=(\d+)", obs, flags=re.IGNORECASE)
                if m_sub:
                    error_detail["got"] = int(m_sub.group(1))
                if m_cor:
                    error_detail["expected"] = int(m_cor.group(1))
                hint = 'Double-check recurrence and capacities. You can verify by computing dp[n,capacity] before submission.'

            elif "success" in text:
                error_type = "OK"
                error_detail["outcome"] = "success"
                hint = None

            diagnostic = {"error_type": error_type}
            if self.feedback_level >= 1:
                diagnostic["error_detail"] = error_detail
                diagnostic["turn"] = getattr(self, "turn_count", None)
                diagnostic["state"] = {
                    "n_revealed": self.n_revealed,
                    "cap_revealed": self.cap_revealed,
                    "table_created": self.table_created,
                    "base_set": self.base_set,
                    "items_revealed": len(self.revealed_items),
                    "num_items": self.num_items,
                    "capacity": self.capacity,
                    "cells_filled": self.cells_filled_count,
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
                "hint": "Start by revealing problem size and capacity: \\boxed{reveal_n} then \\boxed{reveal_capacity}.",
                "turn": 0,
                "state": {
                    "n_revealed": self.n_revealed,
                    "cap_revealed": self.cap_revealed,
                    "table_created": self.table_created,
                    "base_set": self.base_set,
                    "items_revealed": len(self.revealed_items),
                    "num_items": self.num_items,
                    "capacity": self.capacity,
                    "cells_filled": self.cells_filled_count,
                }
            }
            return obs, info