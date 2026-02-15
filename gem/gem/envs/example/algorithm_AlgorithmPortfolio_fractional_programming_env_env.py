from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class AlgorithmPortfolioEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 50,
        **_,
    ):
        super().__init__()
        self.complexity = int(max(1, min(10, complexity)))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 50

        self.complexity_params = {
            # Total number of candidate algorithms: larger → more combinatorics and harder exploration.
            "num_algorithms": (6, 24),
            # Required portfolio size k: larger k → harder combinatorial selection.
            "subset_size_k": (2, 6),
            # REVERSED: Probe budget as % of N; fewer probes → less information and harder.
            "probe_budget_pct": (100, 30),
            # REVERSED: Initial success rate range width (% points): narrower ranges → less uncertainty (easier).
            # We reverse: wider at easy, tighter at hard (hard reveals remain tighter but fewer probes increase difficulty).
            "success_range_width_pct": (20, 4),
            # REVERSED: Initial runtime range width in ms: wider at easy, tighter at hard (but fewer probes make it harder).
            "runtime_range_width_ms": (60, 10),
        }

        self.param_variance = {
            "num_algorithms": 1,          # ±1
            "subset_size_k": 0,           # small range, fix at center for stability
            "probe_budget_pct": 5,        # ±5 percentage points
            "success_range_width_pct": 1, # ±1 percentage point
            "runtime_range_width_ms": 5,  # ±5 ms
        }

        self.num_algorithms: int = 0
        self.subset_size_k: int = 0
        self.probe_budget_pct: int = 0
        self.success_range_width_pct: int = 0
        self.runtime_range_width_ms: int = 0

        self.turn_count: int = 0
        self.algorithms: List[Dict[str, Any]] = []
        self.revealed_count: int = 0
        self.probe_budget: int = 0
        self.remaining_probes: int = 0
        self.optimal_ratio: float = 0.0
        self.optimal_indices: List[int] = []
        self.terminated: bool = False
        self.truncated: bool = False

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for param_name, (min_val, max_val) in self.complexity_params.items():
            center_value = min_val + (max_val - min_val) * normalized
            if self.enable_param_randomization:
                variance = self.param_variance.get(param_name, 0)
                if variance > 0:
                    actual_value = center_value + random.uniform(-variance, variance)
                    lo = min(min_val, max_val)
                    hi = max(min_val, max_val)
                    actual_value = max(lo, min(hi, actual_value))
                else:
                    actual_value = center_value
            else:
                actual_value = center_value
            setattr(self, param_name, int(round(actual_value)))

    def _generate_algorithms(self):
        self.algorithms = []
        N = self.num_algorithms
        # Runtime and success rate generation with a mild negative correlation to induce trade-offs
        # Runtime in [50, 220] ms; success in [0.55, 0.98]
        for i in range(N):
            base_runtime = random.uniform(50, 220)
            # Map runtime to a baseline success (higher runtime -> slightly higher success)
            # Add noise
            corr_strength = 0.3
            noise = random.uniform(-0.06, 0.06)
            base_success = 0.75 + corr_strength * ((base_runtime - 135) / 170) + noise
            s = max(0.55, min(0.98, base_success))
            c = max(10.0, float(round(base_runtime, 2)))
            # Approximate ranges for initial observation
            sr_width = self.success_range_width_pct / 100.0
            rt_width = float(self.runtime_range_width_ms)
            s_lo = max(0.0, s - sr_width / 2.0)
            s_hi = min(1.0, s + sr_width / 2.0)
            c_lo = max(1.0, c - rt_width / 2.0)
            c_hi = c + rt_width / 2.0
            self.algorithms.append({
                "id": f"A{i+1}",
                "s": float(round(s, 4)),
                "c": float(round(c, 2)),
                "s_range": (float(round(s_lo, 2)), float(round(s_hi, 2))),
                "c_range": (int(round(c_lo)), int(round(c_hi))),
                "revealed": False,
            })

    def _compute_optimal_subset(self):
        # Binary search on lambda for fixed-size k portfolio optimization of ratio S/C.
        k = self.subset_size_k
        s_list = [a["s"] for a in self.algorithms]
        c_list = [a["c"] for a in self.algorithms]
        # Upper bound on ratio is max s_i/c_i
        max_ratio = 0.0
        for s, c in zip(s_list, c_list):
            if c > 0:
                max_ratio = max(max_ratio, s / c)
        lo, hi = 0.0, max_ratio if max_ratio > 0 else 1.0
        best_ratio = -1.0
        best_indices = list(range(k))  # placeholder
        # If k == 0 (shouldn't happen), handle
        if k <= 0 or k > len(s_list):
            self.optimal_ratio = 0.0
            self.optimal_indices = []
            return
        for _ in range(60):
            mid = (lo + hi) / 2.0
            # Compute scores s - mid * c
            vals = []
            for idx, (s, c) in enumerate(zip(s_list, c_list)):
                vals.append((s - mid * c, idx))
            vals.sort(key=lambda x: x[0], reverse=True)
            chosen = vals[:k]
            sum_s = 0.0
            sum_c = 0.0
            for v, idx in chosen:
                sum_s += s_list[idx]
                sum_c += c_list[idx]
            f_val = sum_s - mid * sum_c
            # Track best ratio from this selection
            if sum_c > 0:
                r = sum_s / sum_c
                if r > best_ratio:
                    best_ratio = r
                    best_indices = [idx for _, idx in chosen]
            if f_val > 0:
                lo = mid
            else:
                hi = mid
        # After search, finalize best
        self.optimal_ratio = best_ratio
        self.optimal_indices = sorted(best_indices)

    def _get_instructions(self) -> str:
        return (
            "Algorithm Portfolio Selection Game\n"
            "Goal: Pick exactly k algorithms to maximize the ratio (sum of success rates) / (sum of runtimes).\n"
            "You can probe algorithms to reveal exact metrics before making a final selection.\n"
            "Actions:\n"
            "- probe Aidx[,Aidy,...]: Reveal exact metrics of one or more algorithms (counts against probe budget).\n"
            "- list: Show current overview and remaining probes.\n"
            "- help: Show these instructions.\n"
            "- select Aidx[,Aidy,...]: Submit your final set of exactly k algorithms (terminates the episode).\n"
            "- forfeit: Give up (terminates with zero reward).\n"
            "Formatting: Enclose every action in \\boxed{...}.\n"
            f"Examples: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        lines = []
        lines.append(f"N={self.num_algorithms}, required k={self.subset_size_k}, remaining probes={self.remaining_probes}")
        lines.append("Algorithms (approximate ranges; revealed ones show exact):")
        for i, a in enumerate(self.algorithms):
            if a["revealed"]:
                lines.append(
                    f"- {a['id']}: success={a['s']:.4f}, runtime={a['c']:.2f} ms [REVEALED]"
                )
            else:
                s_lo, s_hi = a["s_range"]
                c_lo, c_hi = a["c_range"]
                lines.append(
                    f"- {a['id']}: success in [{s_lo:.2f}, {s_hi:.2f}], runtime in [{c_lo}, {c_hi}] ms"
                )
        lines.append("Submit your next action in \\boxed{...} format.")
        return "\n".join(lines)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()
        self.turn_count = 0
        self.terminated = False
        self.truncated = False
        self._generate_algorithms()
        # Compute probe budget
        self.probe_budget = int(round(self.num_algorithms * self.probe_budget_pct / 100.0))
        self.probe_budget = max(0, min(self.num_algorithms, self.probe_budget))
        self.remaining_probes = self.probe_budget
        self.revealed_count = 0
        self._compute_optimal_subset()
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self.terminated or self.truncated:
            return "Episode already ended.", 0.0, True, self.truncated, {"suffix": self.get_task_suffix()}
        self.turn_count += 1

        parsed = self._parse_action(action)
        if parsed is None:
            obs = "Invalid action format. Use \\boxed{...} with a supported command."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        cmd = parsed.get("cmd")
        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        if cmd == "help":
            obs = self._get_instructions()
            terminated = False

        elif cmd == "list":
            obs = self.get_task_suffix()
            terminated = False

        elif cmd == "forfeit":
            obs = "You forfeited. Episode terminated."
            reward = 0.0
            terminated = True

        elif cmd == "probe":
            ids = parsed.get("ids", [])
            if len(ids) == 0:
                obs = f"No valid IDs to probe. Remaining probes={self.remaining_probes}."
                terminated = False
            else:
                unique_ids = []
                seen = set()
                for idx in ids:
                    if idx not in seen:
                        unique_ids.append(idx)
                        seen.add(idx)
                to_probe = []
                for idx in unique_ids:
                    if 0 <= idx < self.num_algorithms and not self.algorithms[idx]["revealed"]:
                        to_probe.append(idx)
                if self.remaining_probes <= 0:
                    obs_lines = ["Probe limit reached (remaining 0). No new reveals."]
                else:
                    can_reveal = min(self.remaining_probes, len(to_probe))
                    reveal_idxs = to_probe[:can_reveal]
                    for ri in reveal_idxs:
                        self.algorithms[ri]["revealed"] = True
                        self.revealed_count += 1
                    self.remaining_probes -= can_reveal
                    obs_lines = [f"Revealed {can_reveal} algorithm(s). Remaining probes={self.remaining_probes}."]
                    for ri in reveal_idxs:
                        a = self.algorithms[ri]
                        obs_lines.append(f"- {a['id']}: success={a['s']:.4f}, runtime={a['c']:.2f} ms")
                    not_revealed = [idx for idx in to_probe[can_reveal:]]
                    if len(not_revealed) > 0:
                        obs_lines.append(
                            f"Probe request exceeded remaining budget; {len(not_revealed)} request(s) not revealed."
                        )
                obs_lines.append("Use 'list' to view the full overview.")
                obs = "\n".join(obs_lines)
                terminated = False

        elif cmd == "select":
            ids = parsed.get("ids", [])
            # Validate IDs
            ok = True
            reason = ""
            # Remove duplicates but detect them
            seen = set()
            dupe_found = False
            cleaned = []
            for idx in ids:
                if idx in seen:
                    dupe_found = True
                else:
                    seen.add(idx)
                    cleaned.append(idx)
            if dupe_found:
                ok = False
                reason = "Duplicate IDs in selection."
            if ok and len(cleaned) != self.subset_size_k:
                ok = False
                reason = f"Wrong number of algorithms: expected k={self.subset_size_k}, got {len(cleaned)}."
            if ok:
                for idx in cleaned:
                    if not (0 <= idx < self.num_algorithms):
                        ok = False
                        reason = f"Unknown algorithm ID index {idx+1}."
                        break
            if not ok:
                obs = f"Protocol violation: {reason} Episode terminated."
                reward = -0.1
                terminated = True
            else:
                # Compute user ratio
                sum_s = 0.0
                sum_c = 0.0
                for idx in cleaned:
                    a = self.algorithms[idx]
                    sum_s += a["s"]
                    sum_c += a["c"]
                user_ratio = sum_s / sum_c if sum_c > 0 else 0.0
                # Compare to optimal
                tol = 1e-6
                if abs(user_ratio - self.optimal_ratio) <= tol:
                    obs = (
                        "Success! Your selection achieves the optimal ratio.\n"
                        f"Your ratio={user_ratio:.6f}, Optimal ratio={self.optimal_ratio:.6f}."
                    )
                    reward = 1.0
                else:
                    obs = (
                        "Valid but not optimal selection.\n"
                        f"Your ratio={user_ratio:.6f}, Optimal ratio={self.optimal_ratio:.6f}."
                    )
                    reward = 0.0
                terminated = True

        else:
            obs = "Unsupported action. Use help to see valid commands."
            reward = -0.05
            terminated = False

        if not terminated:
            if self.turn_count >= (self.max_turns or 50):
                obs = f"Reached max turns ({self.max_turns}). Episode timed out."
                reward = 0.0
                terminated = True
                truncated = True

        self.terminated = terminated
        self.truncated = truncated
        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        content_l = content.lower().strip()

        def parse_ids(arg_str: str) -> List[int]:
            # Accept comma or space separated A1,A2 or A1 A2
            raw = re.split(r'[,\s]+', arg_str.strip())
            ids = []
            for token in raw:
                if not token:
                    continue
                m = re.match(r'^[aA](\d+)$', token)
                if m:
                    idx = int(m.group(1)) - 1
                    ids.append(idx)
            return ids

        if content_l.startswith("probe"):
            parts = content.split(None, 1)
            if len(parts) == 1:
                return {"cmd": "probe", "ids": []}
            ids = parse_ids(parts[1])
            return {"cmd": "probe", "ids": ids}

        if content_l.startswith("select"):
            parts = content.split(None, 1)
            if len(parts) == 1:
                return {"cmd": "select", "ids": []}
            ids = parse_ids(parts[1])
            return {"cmd": "select", "ids": ids}

        if content_l.strip() == "help":
            return {"cmd": "help"}

        if content_l.strip() == "list":
            return {"cmd": "list"}

        if content_l.strip() == "forfeit":
            return {"cmd": "forfeit"}

        return {"cmd": "unsupported"}

    def sample_random_action(self) -> str:
        if random.random() < 0.5 and self.num_algorithms >= 1:
            idx = random.randint(1, self.num_algorithms)
            return f"\\boxed{{probe A{idx}}}"
        else:
            # Produce a random selection example (may be invalid before reset but for demonstration)
            k = max(1, min(self.subset_size_k or 2, max(1, self.num_algorithms or 4)))
            ids = list(range(1, (self.num_algorithms or 4) + 1))
            random.shuffle(ids)
            chosen = ids[:k]
            return "\\boxed{select " + ",".join(f"A{x}" for x in chosen) + "}"


class AlgorithmPortfolioEnvWithFeedback(AlgorithmPortfolioEnv):
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
            error_detail["issue"] = "missing_or_wrong_boxed_format"
            hint = "Wrap your command like \\boxed{probe A1} or \\boxed{select A1,A2,...}."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_command"
            hint = "Use one of: probe, list, help, select, forfeit."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "duplicate" in text:
                error_detail["violation"] = "duplicate_ids"
                hint = "List each algorithm at most once in your final selection."
            elif "wrong number" in text:
                error_detail["violation"] = "wrong_cardinality"
                k = getattr(self, "subset_size_k", None)
                error_detail["expected_k"] = k
                hint = f"Select exactly k algorithms; k={k}."
            elif "unknown algorithm id" in text:
                error_detail["violation"] = "unknown_id"
                hint = "IDs must be of the form A1..AN where N is the number of algorithms."
            else:
                hint = "Check command syntax and selection rules."

        elif "probe limit reached" in text or "not revealed" in text:
            error_type = "ProbeLimitExceeded"
            error_detail["remaining_probes"] = getattr(self, "remaining_probes", None)
            hint = "Reduce the number of probes in a single action or select now."

        elif "reached max turns" in text or "timed out" in text:
            error_type = "Timeout"
            hint = "Probe only the most informative candidates and decide sooner."

        elif "valid but not optimal selection" in text:
            error_type = "WrongDecision"
            error_detail["expected_ratio"] = getattr(self, "optimal_ratio", None)
            hint = "Focus on maximizing success/runtime. Probe items with uncertain ranges near the trade-off frontier."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        else:
            error_type = "OK"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["remaining_probes"] = getattr(self, "remaining_probes", None)
            diagnostic["k"] = getattr(self, "subset_size_k", None)
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start by probing algorithms that look promising: high success range and low runtime range.",
            "turn": 0,
            "remaining_probes": getattr(self, "remaining_probes", None),
            "k": getattr(self, "subset_size_k", None),
        }
        return obs, info