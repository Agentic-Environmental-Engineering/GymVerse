from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List, Set


class MonotoneOracleDeductionEnv(Env):
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
            # number of primitive features defining the secret concept; larger = more hypothesis space
            "num_features": (3, 10),
            # size of the secret conjunction (k-of-N, all must be present); larger k = harder to find
            "secret_size": (1, 5),
            # number of available candidate symbols; larger = more probing space and noise
            "num_symbols": (6, 20),
            # number of allowed "heavy" probes (COUNT queries) per episode; fewer = harder (REVERSED)
            "count_budget": (5, 2),
            # number of allowed "cheap" probes (ASK queries) per episode; fewer = harder (REVERSED)
            "ask_budget": (12, 6),
        }

        # Variance for parameters
        self.param_variance = {
            "num_features": 0,     # small range; keep fixed per level
            "secret_size": 0,      # small discrete; fixed per level
            "num_symbols": 1,      # medium discrete
            "count_budget": 1,     # small range; light jitter
            "ask_budget": 1,       # medium discrete
        }

        # Placeholder attributes set by _apply_complexity_params
        self.num_features: int = 0
        self.secret_size: int = 0
        self.num_symbols: int = 0
        self.count_budget: int = 0
        self.ask_budget: int = 0

        # State
        self.turn_count: int = 0
        self.features: List[str] = []
        self.symbols: List[str] = []
        self.symbol_feature_map: Dict[str, Set[str]] = {}
        self.secret_features: Set[str] = set()
        self.remaining_ask: int = 0
        self.remaining_count: int = 0
        self.terminated: bool = False

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
                    lo, hi = (min_val, max_val) if min_val <= max_val else (max_val, min_val)
                    val = max(lo, min(hi, val))
            setattr(self, name, int(round(val)))

        # Ensure feasibility
        self.secret_size = max(1, min(self.secret_size, self.num_features - 0))
        self.num_features = max(self.num_features, self.secret_size)
        self.num_symbols = max(self.num_symbols, self.secret_size + 1)
        self.remaining_ask = self.ask_budget
        self.remaining_count = self.count_budget

    def _get_instructions(self) -> str:
        intro = []
        intro.append("MonotoneOracleDeduction: Deduce a hidden conjunction of primitive features.")
        intro.append("There is a hidden set of required features F* (size k). A symbol is 'positive' iff it contains all features in F*.")
        intro.append("You may probe the oracle using two tools and finally submit your guess of F*.")
        intro.append("")
        intro.append("Tools:")
        intro.append("- ASK symbol=<id>: returns YES if the symbol is positive (has all secret features), else NO. Costs 1 ASK budget.")
        intro.append("- COUNT features=<comma-separated feature ids>: returns how many symbols that contain ALL listed features are positive. Costs 1 COUNT budget.")
        intro.append("  COUNT serves as an aggregate probe to test necessary co-occurrence against the secret.")
        intro.append("- SUBMIT features=<comma-separated feature ids>: final answer. Ends the episode.")
        intro.append("")
        intro.append("Rules:")
        intro.append(f"- You have limited budgets: ASK={self.remaining_ask}, COUNT={self.remaining_count}. Using beyond budget is a protocol violation.")
        intro.append(f"- Available symbols and their features are listed below. Features are primitive and monotone (no negations).")
        intro.append("- Your objective: output exactly the set of secret features F*.")
        intro.append("")
        intro.append("Action format:")
        intro.append("- Use \\boxed{...}")
        intro.append("- ASK: \\boxed{ASK symbol=s3}")
        intro.append("- COUNT: \\boxed{COUNT features=f1,f4}")
        intro.append("- SUBMIT: \\boxed{SUBMIT features=f2,f3}")
        intro.append("")
        return "\n".join(intro)

    def get_task_suffix(self) -> str:
        lines = []
        lines.append("Instance:")
        lines.append(f"- Features: {', '.join(self.features)}")
        lines.append(f"- Symbols (with their features):")
        for s in self.symbols:
            feats = sorted(self.symbol_feature_map[s])
            lines.append(f"  {s}: {{{', '.join(feats)}}}")
        lines.append(f"- Remaining ASK budget: {self.remaining_ask}")
        lines.append(f"- Remaining COUNT budget: {self.remaining_count}")
        lines.append("")
        lines.append("Enter your action in \\boxed{...} format.")
        return "\n".join(lines)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()
        self.turn_count = 0
        self.terminated = False

        # Create features
        self.features = [f"f{i+1}" for i in range(self.num_features)]

        # Sample secret conjunction
        self.secret_features = set(random.sample(self.features, self.secret_size))

        # Generate symbols with feature sets ensuring solvability (enough diversity)
        self.symbols = [f"s{i+1}" for i in range(self.num_symbols)]
        self.symbol_feature_map = {}

        # Ensure at least some positives and negatives
        # Construct a base positive symbol that includes all secret features plus some extras
        for s in self.symbols:
            # random base size
            base_size = random.randint(self.secret_size, max(self.secret_size, min(self.num_features, self.secret_size + 2)))
            chosen = set(self.secret_features)
            others = [f for f in self.features if f not in chosen]
            if len(chosen) < base_size and others:
                chosen.update(random.sample(others, min(len(others), base_size - len(chosen))))
            # With some probability, remove one secret feature to create negatives, but keep variety
            if random.random() < 0.5:
                # keep as is (likely positive)
                pass
            else:
                if random.random() < 0.5 and len(self.secret_features) > 0:
                    # create a near-miss negative
                    rem = random.choice(list(self.secret_features))
                    if rem in chosen and len(chosen) > 1:
                        chosen.remove(rem)
                        # maybe add a non-secret feature instead
                        nonsec = [f for f in self.features if f not in self.secret_features and f not in chosen]
                        if nonsec:
                            chosen.add(random.choice(nonsec))
            if len(chosen) == 0:
                chosen.add(random.choice(self.features))
            self.symbol_feature_map[s] = chosen

        # Post-fix: ensure at least one positive and one negative exist
        positives = [s for s in self.symbols if self._is_positive(s)]
        if len(positives) == 0:
            # force one symbol to be positive
            target = random.choice(self.symbols)
            self.symbol_feature_map[target] = set(self.secret_features)
            # add some extras to keep ambiguity
            extras = [f for f in self.features if f not in self.secret_features]
            if extras and random.random() < 0.7:
                self.symbol_feature_map[target].add(random.choice(extras))
        negatives = [s for s in self.symbols if not self._is_positive(s)]
        if len(negatives) == 0:
            # force one negative by removing a secret feature
            target = random.choice(self.symbols)
            chosen = set(self.symbol_feature_map[target])
            if len(self.secret_features) > 0:
                rm = random.choice(list(self.secret_features))
                if rm in chosen and len(chosen) > 1:
                    chosen.remove(rm)
                else:
                    # ensure negative by adding irrelevant but not all secrets
                    chosen = set(f for f in chosen if f not in self.secret_features)
                    if len(chosen) == 0:
                        chosen.add(random.choice([f for f in self.features if f not in self.secret_features]))
            self.symbol_feature_map[target] = chosen

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _is_positive(self, symbol_id: str) -> bool:
        feats = self.symbol_feature_map.get(symbol_id, set())
        return self.secret_features.issubset(feats)

    def _count_positive_with_features(self, queried: Set[str]) -> int:
        # Count symbols that both contain all queried features AND are positive
        cnt = 0
        for s in self.symbols:
            feats = self.symbol_feature_map[s]
            if queried.issubset(feats) and self._is_positive(s):
                cnt += 1
        return cnt

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self.terminated:
            return "Episode already ended.", 0.0, True, False, {"suffix": self.get_task_suffix()}

        self.turn_count += 1
        parsed = self._parse_action(action)

        if not parsed:
            obs = "INVALID ACTION FORMAT: Use \\boxed{...} with ASK, COUNT, or SUBMIT."
            self.terminated = True
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        kind = parsed.get("action", "").upper()
        terminated = False
        truncated = False
        reward = 0.0

        if kind not in ("ASK", "COUNT", "SUBMIT"):
            obs = f"UNSUPPORTED ACTION: {kind}. Valid actions are ASK, COUNT, SUBMIT."
            self.terminated = True
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        if kind == "ASK":
            symbol = parsed.get("symbol", "")
            if self.remaining_ask <= 0:
                obs = "PROTOCOL VIOLATION: No ASK budget remaining."
                self.terminated = True
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            if symbol not in self.symbols:
                obs = f"PROTOCOL VIOLATION: Unknown symbol '{symbol}'."
                self.terminated = True
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            self.remaining_ask -= 1
            answer = "YES" if self._is_positive(symbol) else "NO"
            obs = f"ASK RESULT: {symbol} -> {answer}"
            # shaped minor credit for information gathering deep in long puzzles
            reward = 0.0
            if self.max_turns >= 15 and self.remaining_ask + self.remaining_count < (self.ask_budget + self.count_budget):
                reward = 0.0
            info = {"suffix": self.get_task_suffix()}
            # Check timeout after action
            if self.turn_count >= self.max_turns:
                terminated = True
                truncated = True
                obs = f"{obs}\nTIMEOUT: Reached max turns ({self.max_turns})."
                self.terminated = True
            return obs, reward, terminated, truncated, info

        elif kind == "COUNT":
            if self.remaining_count <= 0:
                obs = "PROTOCOL VIOLATION: No COUNT budget remaining."
                self.terminated = True
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            raw = parsed.get("features", "")
            query_set = set([t.strip() for t in raw.split(",") if t.strip()]) if raw else set()
            if len(query_set) == 0:
                obs = "PROTOCOL VIOLATION: COUNT requires features=<list>."
                self.terminated = True
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            if not query_set.issubset(set(self.features)):
                obs = "PROTOCOL VIOLATION: COUNT referenced unknown features."
                self.terminated = True
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            self.remaining_count -= 1
            c = self._count_positive_with_features(query_set)
            obs = f"COUNT RESULT: features={{{{ {', '.join(sorted(query_set))} }}}} -> {c}"
            reward = 0.0
            info = {"suffix": self.get_task_suffix()}
            if self.turn_count >= self.max_turns:
                terminated = True
                truncated = True
                obs = f"{obs}\nTIMEOUT: Reached max turns ({self.max_turns})."
                self.terminated = True
            return obs, reward, terminated, truncated, info

        else:  # SUBMIT
            raw = parsed.get("features", "")
            guess_set = set([t.strip() for t in raw.split(",") if t.strip()]) if raw else set()
            if not guess_set.issubset(set(self.features)):
                obs = "PROTOCOL VIOLATION: SUBMIT includes unknown features."
                self.terminated = True
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            # Evaluate
            correct = guess_set == self.secret_features
            if correct:
                obs = f"Success! Correct secret features: {{{', '.join(sorted(self.secret_features))}}}"
                self.terminated = True
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Failed! Your guess {{{', '.join(sorted(guess_set))}}} is not equal to the secret."
                self.terminated = True
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
        if not parts:
            return None
        tokens: Dict[str, Any] = {}
        tokens["action"] = parts[0].upper()
        for part in parts[1:]:
            if "=" in part:
                k, v = part.split("=", 1)
                tokens[k.strip().lower()] = v.strip()
        return tokens

    def sample_random_action(self) -> str:
        if random.random() < 0.4 and self.symbols:
            return rf"\boxed{{ASK symbol={random.choice(self.symbols)}}}"
        elif random.random() < 0.8 and self.features:
            picks = sorted(random.sample(self.features, k=max(1, min(2, len(self.features)))))  # small feature set
            return rf"\boxed{{COUNT features={','.join(picks)}}}"
        else:
            # random submit guess of plausible size
            k = max(1, min(self.secret_size if hasattr(self, 'secret_size') and self.secret_size > 0 else 1, len(self.features)))
            guess = sorted(random.sample(self.features, k=k))
            return rf"\boxed{{SUBMIT features={','.join(guess)}}}"


class MonotoneOracleDeductionEnvWithFeedback(MonotoneOracleDeductionEnv):
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
            error_detail["issue"] = "missing_boxed_or_params"
            hint = "Wrap your command in \\boxed{...} and use ASK, COUNT, or SUBMIT with proper parameters."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_action"
            hint = "Use one of: ASK symbol=sX, COUNT features=fA,fB, SUBMIT features=fA,fB."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "ask budget" in text:
                error_detail["violation"] = "ask_budget_exhausted"
                hint = "Switch to COUNT or SUBMIT; plan ASK usage earlier."
            elif "count budget" in text:
                error_detail["violation"] = "count_budget_exhausted"
                hint = "Use ASK to refine or proceed to SUBMIT; save COUNT for testing necessary co-occurrence."
            elif "unknown symbol" in text:
                error_detail["violation"] = "unknown_symbol"
                hint = "Only reference symbols listed in the instance (e.g., s1, s2, ...)."
            elif "requires features" in text:
                error_detail["violation"] = "missing_features_param"
                hint = "Provide features for COUNT as comma-separated ids, e.g., COUNT features=f1,f3."
            elif "referenced unknown features" in text:
                error_detail["violation"] = "unknown_features"
                hint = "Use only features listed under Features (e.g., f1..fN)."
            elif "submit includes unknown features" in text:
                error_detail["violation"] = "submit_unknown_features"
                hint = "Submit only known features; cross-check the Features list."
            else:
                error_detail["violation"] = "general_protocol"
                hint = "Check budgets and identifiers; follow the tool formats exactly."

        elif "timeout: reached max turns" in text:
            error_type = "Timeout"
            error_detail["turn_limit"] = self.max_turns
            hint = "Probe efficiently; use COUNT to narrow necessary conjunctions before ASK, then SUBMIT."

        elif "failed!" in text:
            error_type = "WrongDecision"
            # Try to extract guess
            m = re.search(r"your guess \{(.+?)\}", obs, flags=re.IGNORECASE)
            if m:
                guess_str = m.group(1)
                got = [x.strip() for x in guess_str.split(",") if x.strip()]
                error_detail["got"] = got
            # Provide strategic hint without revealing answer
            hint = "Compare positives vs negatives: any feature missing in a positive cannot be required; any feature missing in a negative may be non-essential."

        elif "success!" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "remaining_ask": getattr(self, "remaining_ask", None),
                "remaining_count": getattr(self, "remaining_count", None),
                "num_features": getattr(self, "num_features", None),
                "num_symbols": getattr(self, "num_symbols", None),
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
            "hint": "Start by ASKing a few symbols to separate positives from negatives, then use COUNT to test co-occurring features.",
            "turn": 0,
            "state": {
                "remaining_ask": getattr(self, "remaining_ask", None),
                "remaining_count": getattr(self, "remaining_count", None),
                "num_features": getattr(self, "num_features", None),
                "num_symbols": getattr(self, "num_symbols", None),
            },
        }
        return obs, info