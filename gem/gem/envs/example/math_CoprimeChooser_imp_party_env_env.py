from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List

class CoprimeChooserEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 25,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 25

        # Evolvable parameters
        self.complexity_params = {
            # pool_size: number of integers presented; larger search space = harder
            "pool_size": (6, 20),
            # number_range: max integer value in pool; larger values introduce more primes but also denser factor overlaps = harder
            "number_range": (20, 200),
            # target_k: size of subset required; larger k makes constraints combinatorially harder
            "target_k": (2, 6),
            # composite_bias: expected count of composite injects per 10 elements; higher bias increases shared factors = harder
            "composite_bias": (3, 8),
        }

        # Variance settings
        self.param_variance = {
            "pool_size": 1,
            "number_range": 10,
            "target_k": 1,
            "composite_bias": 1,
        }

        # Placeholder attributes
        self.pool_size: int = 0
        self.number_range: int = 0
        self.target_k: int = 0
        self.composite_bias: int = 0

        # Other state
        self.turn_count: int = 0
        self.pool: List[int] = []
        self.instance_seed: Optional[int] = None
        self.terminated: bool = False
        self.solution_exists: bool = False
        self.pairwise_coprime_edges: Dict[Tuple[int, int], bool] = {}
        self.last_submission: Optional[List[int]] = None
        self._cached_prime_list: List[int] = []
        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for name, (min_val, max_val) in self.complexity_params.items():
            center_value = min_val + (max_val - min_val) * normalized
            actual_value = center_value
            if self.enable_param_randomization:
                variance = self.param_variance.get(name, 0)
                if variance > 0:
                    actual_value = center_value + random.uniform(-variance, variance)
                    lo = min(min_val, max_val)
                    hi = max(min_val, max_val)
                    actual_value = max(lo, min(hi, actual_value))
            setattr(self, name, int(round(actual_value)))

        # Ensure feasibility-friendly settings do not become impossible
        self.target_k = max(2, min(self.target_k, max(2, self.pool_size // 2)))

    def _sieve_primes(self, n: int) -> List[int]:
        if self._cached_prime_list and self._cached_prime_list[-1] >= n:
            # fast path: return prefix
            return [p for p in self._cached_prime_list if p <= n]
        sieve = [True] * (n + 1)
        sieve[0:2] = [False, False]
        for i in range(2, int(n ** 0.5) + 1):
            if sieve[i]:
                step = i
                start = i * i
                sieve[start:n + 1:step] = [False] * (((n - start) // step) + 1)
        primes = [i for i, is_p in enumerate(sieve) if is_p]
        self._cached_prime_list = primes
        return primes

    def _rand_composite(self, upper: int) -> int:
        # biased composite generator by multiplying small primes
        primes = self._sieve_primes(min(upper, 50))
        # pick 2-3 small primes
        count = random.choice([2, 2, 3])
        val = 1
        for _ in range(count):
            p = random.choice(primes[: min(10, len(primes))]) if primes else 2
            val *= p
        # jitter within range
        val = max(4, min(upper, val + random.randint(0, 5)))
        # ensure composite
        if val <= 3:
            val = 4
        return val

    def _ensure_feasible_pool(self, pool: List[int]) -> bool:
        # Try to greedily build a pairwise coprime set of size target_k
        chosen = []
        for x in pool:
            ok = all(self._gcd(x, y) == 1 for y in chosen)
            if ok:
                chosen.append(x)
                if len(chosen) >= self.target_k:
                    return True
        return False

    def _gcd(self, a: int, b: int) -> int:
        while b:
            a, b = b, a % b
        return abs(a)

    def _build_edges(self):
        self.pairwise_coprime_edges = {}
        for i in range(len(self.pool)):
            for j in range(i + 1, len(self.pool)):
                a, b = self.pool[i], self.pool[j]
                self.pairwise_coprime_edges[(i, j)] = (self._gcd(a, b) == 1)

    def _get_instructions(self) -> str:
        return (
            "You are playing Coprime Chooser.\n"
            "Goal: From the given list of integers, select exactly K distinct indices such that every pair of chosen numbers is coprime (gcd=1).\n"
            "Rules:\n"
            "- Submit your selection as indices into the displayed list (1-based).\n"
            "- Choose exactly K distinct indices; duplicates or out-of-range indices are invalid.\n"
            "- You may submit multiple times up to max turns; only the last submission determines success.\n"
            "Action format:\n"
            "- Use \\boxed{select indices=...} where ... is a comma-separated list of indices.\n"
            "Example: \\boxed{select indices=1,3,5}\n"
        )

    def get_task_suffix(self) -> str:
        numbers_str = ", ".join(f"{i+1}:{v}" for i, v in enumerate(self.pool))
        return (
            f"Instance:\n"
            f"- Target subset size K = {self.target_k}\n"
            f"- Pool size N = {self.pool_size}\n"
            f"- Numbers (index:value): {numbers_str}\n"
            "Enter your action as \\boxed{select indices=i1,i2,...,ik} using 1-based indices."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
            self.instance_seed = seed
        self._apply_complexity_params()
        self.turn_count = 0
        self.terminated = False
        self.last_submission = None
        # Build pool with controlled density of shared factors while ensuring feasibility
        attempts = 0
        while True:
            attempts += 1
            pool = []
            # generate a mix of primes and composites
            target_composites = min(self.pool_size, max(0, int(round(self.composite_bias / 10.0 * self.pool_size))))
            target_primes = self.pool_size - target_composites
            primes = self._sieve_primes(self.number_range)
            # sample primes favoring small ones for coprime richness
            prime_choices = []
            if primes:
                for _ in range(target_primes):
                    choice = random.choice(primes[: max(5, len(primes)//3)])
                    prime_choices.append(choice)
            while len(prime_choices) < target_primes:
                # fallback to random odd numbers
                candidate = random.randint(2, self.number_range)
                if all(candidate % p != 0 for p in range(2, int(candidate ** 0.5) + 1)):
                    prime_choices.append(candidate)
            comp_choices = []
            for _ in range(target_composites):
                comp_choices.append(self._rand_composite(self.number_range))
            pool = prime_choices + comp_choices
            random.shuffle(pool)
            # deduplicate while keeping size
            uniq = []
            seen = set()
            for x in pool:
                if x not in seen:
                    uniq.append(x)
                    seen.add(x)
            while len(uniq) < self.pool_size:
                # fill with random numbers; prefer variety
                candidate = random.randint(2, self.number_range)
                if candidate not in seen:
                    uniq.append(candidate)
                    seen.add(candidate)
            self.pool = uniq[: self.pool_size]
            self._build_edges()
            feasible = self._ensure_feasible_pool(self.pool)
            self.solution_exists = feasible
            if feasible or attempts > 10:
                # after several attempts, accept near-feasible but bias success by rebuilding once
                if not feasible:
                    # small repair: ensure at least target_k primes present
                    primes_all = [x for x in self.pool if x in self._sieve_primes(self.number_range)]
                    deficit = self.target_k - len(primes_all)
                    i = 0
                    while deficit > 0 and i < self.pool_size:
                        if self.pool[i] not in primes_all:
                            # replace with a prime likely new
                            p = random.choice(self._sieve_primes(self.number_range))
                            self.pool[i] = p
                            primes_all.append(p)
                            deficit -= 1
                        i += 1
                    self._build_edges()
                    self.solution_exists = self._ensure_feasible_pool(self.pool)
                break

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self.terminated:
            return "Episode already terminated.", 0.0, True, False, {"suffix": self.get_task_suffix()}
        self.turn_count += 1

        parsed = self._parse_action(action)
        if parsed is None:
            obs = "INVALID ACTION FORMAT: Use \\boxed{select indices=...} with a comma-separated list."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        if parsed.get("action") != "select":
            obs = "UNSUPPORTED ACTION: Only 'select' is supported."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        if "indices" not in parsed:
            obs = "PROTOCOL VIOLATION: Missing 'indices' parameter."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        # Parse indices
        raw = parsed["indices"]
        idx_list = []
        try:
            for part in re.split(r"[,\s]+", raw.strip()):
                if part == "":
                    continue
                idx_list.append(int(part))
        except Exception:
            obs = "INVALID ACTION FORMAT: Indices must be integers separated by commas."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        # Validate indices
        if len(idx_list) != self.target_k:
            obs = f"PROTOCOL VIOLATION: Must select exactly K={self.target_k} indices; got {len(idx_list)}."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        if len(set(idx_list)) != len(idx_list):
            obs = "PROTOCOL VIOLATION: Indices must be distinct; duplicates found."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        if any(i < 1 or i > self.pool_size for i in idx_list):
            obs = "PROTOCOL VIOLATION: One or more indices are out of range."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        # Evaluate coprimality
        indices0 = [i - 1 for i in idx_list]
        values = [self.pool[i] for i in indices0]
        self.last_submission = values
        pairs_total = self.target_k * (self.target_k - 1) // 2
        violations = 0
        violating_pairs = []
        for a_pos in range(len(indices0)):
            for b_pos in range(a_pos + 1, len(indices0)):
                i, j = indices0[a_pos], indices0[b_pos]
                ok = self.pairwise_coprime_edges.get((min(i, j), max(i, j)), True)
                if not ok:
                    violations += 1
                    violating_pairs.append((idx_list[a_pos], idx_list[b_pos], self._gcd(self.pool[i], self.pool[j])))

        if violations == 0:
            obs = (
                "Success! Your selection is pairwise coprime.\n"
                f"Selected indices: {','.join(map(str, idx_list))} -> values {values}\n"
                f"All {pairs_total} pairs have gcd=1."
            )
            self.terminated = True
            return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}

        # Not success; allow multi-turn until max_turns
        detail_pairs = "; ".join(f"({a},{b}) gcd={g}" for a, b, g in violating_pairs[:5])
        more = "" if len(violating_pairs) <= 5 else f" ... and {len(violating_pairs)-5} more"
        obs = (
            f"Failed. {violations} of {pairs_total} pairs are not coprime. "
            f"Violations: {detail_pairs}{more}\n"
            "Try again."
        )
        if self.turn_count >= self.max_turns:
            self.terminated = True
            obs = obs + f"\nTIMEOUT: Reached max turns ({self.max_turns})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}
        return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if action is None:
            return None
        m = re.search(r"\\boxed\{(.+?)\}\s*$", action.strip(), flags=re.DOTALL)
        if not m:
            return None
        inner = m.group(1).strip()
        if not inner:
            return None
        parts = inner.split(None, 1)
        name = parts[0].strip().lower()
        out = {"action": name}
        if len(parts) > 1:
            rest = parts[1].strip()
            # Expect key=value with possible spaces around =
            kv_match = re.findall(r"(\w+)\s*=\s*([^=\n]+?)(?:\s+\w+\s*=|$)", rest + " ")
            if kv_match:
                for k, v in kv_match:
                    out[k.strip().lower()] = v.strip().rstrip(",")
            else:
                # fall back: if looks like indices list without key, accept as indices
                if re.match(r"^\d+(?:\s*,\s*\d+)*$", rest):
                    out["indices"] = rest
        return out

    def sample_random_action(self) -> str:
        if not self.pool:
            return r"\boxed{select indices=1}"
        # heuristic: pick random indices; bias towards primes if possible
        primes_set = set(self._sieve_primes(self.number_range))
        prime_idxs = [i + 1 for i, v in enumerate(self.pool) if v in primes_set]
        choice = []
        if len(prime_idxs) >= self.target_k:
            choice = random.sample(prime_idxs, self.target_k)
        else:
            choice = prime_idxs[:]
            remaining = [i + 1 for i in range(len(self.pool)) if (i + 1) not in choice]
            random.shuffle(remaining)
            need = self.target_k - len(choice)
            choice.extend(remaining[:need])
        choice = sorted(choice)
        return r"\boxed{select indices=" + ",".join(map(str, choice)) + "}"


class CoprimeChooserEnvWithFeedback(CoprimeChooserEnv):
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
            error_detail["issue"] = "boxed_or_indices_parse"
            hint = "Use \\boxed{select indices=i1,i2,...,ik} with integers separated by commas."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["allowed"] = ["select"]
            hint = "Only the 'select' action is supported."
        elif "protocol violation: missing 'indices' parameter" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "missing_indices"
            hint = "Include indices=... after the action name."
        elif "protocol violation: must select exactly" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "wrong_count"
            hint = "Provide exactly K indices as shown in the instance."
        elif "protocol violation: indices must be distinct" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "duplicates"
            hint = "Do not repeat indices; each selection must be unique."
        elif "protocol violation: one or more indices are out of range" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "out_of_range"
            hint = "Use 1-based indices within the pool size."
        elif "success!" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None
        elif "failed." in text and "pairs are not coprime" in text:
            error_type = "WrongDecision"
            # Try to extract first violating pair and gcd
            m = re.search(r"\((\d+),(\d+)\)\s*gcd\s*=\s*(\d+)", text)
            if m:
                error_detail["example_violation"] = {"indices": (int(m.group(1)), int(m.group(2))), "gcd": int(m.group(3))}
            hint = "Avoid selecting numbers sharing small prime factors (like 2, 3, 5). Prefer primes or numbers with distinct factors."
        if "timeout" in text:
            error_type = "Timeout"
            error_detail["outcome"] = "max_turns_reached"
            if hint is None:
                hint = "Use earlier feedback about gcd violations to adjust your selection before running out of turns."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["k"] = getattr(self, "target_k", None)
            diagnostic["pool_size"] = getattr(self, "pool_size", None)
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start by picking primes or numbers that look unrelated (e.g., different parity and not multiples of small primes).",
            "turn": 0,
            "k": getattr(self, "target_k", None),
            "pool_size": getattr(self, "pool_size", None),
        }
        return obs, info