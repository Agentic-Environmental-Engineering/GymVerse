from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List, Set


class LexemeCipherEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 20,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 20

        # Evolvable parameters
        self.complexity_params = {
            # Allowed dictionary size bucket index: higher index = larger lexicon => harder search
            "lexicon_size_idx": (0, 4),
            # Word length: longer words increase branching in edits => harder
            "word_length": (3, 7),
            # Number of permitted operations per episode: more ops allowed increases cognitive load and branching => harder
            "num_ops": (2, 4),
            # Number of clues initially revealed: REVERSED fewer clues => harder
            "initial_clues": (3, 1),
            # Number of query allowances (ask semantic or phonetic questions): REVERSED fewer queries => harder
            "query_budget": (3, 1),
            # Required minimum edit distance threshold for accepting non-exact guesses for partial credit (unused for reward but for feedback richness): higher threshold makes distinguishing closer guesses stricter => harder
            "strict_distance": (2, 4),
        }

        # Variance per parameter
        self.param_variance = {
            "lexicon_size_idx": 0,   # small range of discrete buckets
            "word_length": 1,        # ±1 character
            "num_ops": 1,            # ±1 operation
            "initial_clues": 0,      # keep deterministic at level
            "query_budget": 0,       # fixed to enforce curriculum
            "strict_distance": 1,    # ±1 tolerance shift
        }

        # Placeholder attributes set in _apply_complexity_params
        self.lexicon_size_idx: int = 0
        self.word_length: int = 0
        self.num_ops: int = 0
        self.initial_clues: int = 0
        self.query_budget: int = 0
        self.strict_distance: int = 0

        # Other state
        self.turn_count: int = 0
        self.allowed_ops: List[str] = []
        self.hidden_target: str = ""
        self.current_hint_mask: str = ""
        self.known_positions: Dict[int, str] = {}
        self.known_absent_letters: Set[str] = set()
        self.known_present_letters: Set[str] = set()
        self.remaining_queries: int = 0
        self.lexicon: List[str] = []
        self.semantic_map: Dict[str, Dict[str, Set[str]]] = {}  # word -> {"syn": set(), "ant": set()}
        self.phonetic_map: Dict[str, Dict[str, Any]] = {}  # word -> {"syllables": int, "starts_with_vowel": bool}
        self.game_over: bool = False

        self.reset()

    # Internal helpers
    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for param_name, (min_val, max_val) in self.complexity_params.items():
            center_value = min_val + (max_val - min_val) * normalized
            actual_value = center_value
            if self.enable_param_randomization:
                variance = self.param_variance.get(param_name, 0)
                if variance > 0:
                    actual_value = center_value + random.uniform(-variance, variance)
                    lo, hi = (max_val, min_val) if min_val > max_val else (min_val, max_val)
                    actual_value = max(lo, min(hi, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _make_seed_lexicon(self, target_len: int, size_bucket: int) -> List[str]:
        # Seed pool of root words by length; kept small but diverse across buckets
        base_words = {
            3: ["cat", "bat", "mat", "map", "rap", "sap", "sun", "run", "fun", "gun", "cup", "cap", "car", "bar"],
            4: ["bake", "cake", "lake", "like", "bike", "hike", "time", "tide", "ride", "code", "coda", "mode", "made", "make", "game"],
            5: ["crisp", "crush", "brush", "brash", "flash", "clash", "grape", "shape", "share", "spare", "spark", "stark", "start", "smart"],
            6: ["planet", "planer", "planed", "placer", "placer", "placer", "placerx"],  # will filter invalid length chars later
            7: ["streamy", "dreamer", "creamer", "steamer", "teaming", "searing", "hearing", "bearing", "wearing"],
        }
        # Clean words by length
        pool = [w for w in base_words.get(target_len, []) if len(w) == target_len and w.isalpha()]
        # Expand pool via simple synthetic variations if bucket demands more
        # Buckets: 0 small, 1 medium, 2 larger, 3 big, 4 largest
        bucket_sizes = [12, 18, 24, 30, 36]
        target_size = bucket_sizes[max(0, min(4, size_bucket))]
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        def mutate_one(w):
            i = random.randrange(len(w))
            c = random.choice(alphabet.replace(w[i], ""))
            return w[:i] + c + w[i+1:]
        while len(pool) < target_size and pool:
            candidate = mutate_one(random.choice(pool))
            if len(candidate) == target_len and candidate.isalpha() and candidate not in pool:
                pool.append(candidate)
        # If pool too small, fallback to generating random pronounceable-ish words
        vowels = "aeiou"
        while len(pool) < target_size:
            s = ""
            last_vowel = random.choice([True, False])
            for _ in range(target_len):
                if last_vowel:
                    s += random.choice("bcdfghjklmnpqrstvwxyz")
                else:
                    s += random.choice(vowels)
                last_vowel = not last_vowel
            if s not in pool:
                pool.append(s)
        random.shuffle(pool)
        return pool

    def _build_semantics_phonetics(self, lexicon: List[str]):
        # Pseudo semantic relations by stems (prefixes/suffixes similarity)
        sem_map = {}
        for w in lexicon:
            stem = w[: max(2, len(w)//2)]
            syn = set([x for x in lexicon if x != w and x.startswith(stem)])
            ant = set([x for x in lexicon if x != w and x.endswith(w[-2:]) and not x.startswith(stem)])
            sem_map[w] = {"syn": syn, "ant": ant}

        # Pseudo phonetics by vowel chunks and start letter
        pho_map = {}
        vowels = set("aeiou")
        for w in lexicon:
            syllables = 0
            i = 0
            while i < len(w):
                if w[i] in vowels:
                    syllables += 1
                    while i < len(w) and w[i] in vowels:
                        i += 1
                else:
                    i += 1
            pho_map[w] = {"syllables": max(1, syllables), "starts_with_vowel": w[0] in vowels}
        return sem_map, pho_map

    def _choose_ops(self, k: int) -> List[str]:
        op_pool = ["ADD", "DELETE", "REPLACE", "SWAP", "PROPOSE", "ASK_SEM", "ASK_PHON"]
        # Ensure PROPOSE always available
        must = ["PROPOSE"]
        others = [o for o in op_pool if o not in must]
        chosen = set(must)
        while len(chosen) < k:
            chosen.add(random.choice(others))
        return list(chosen)

    def _make_mask_from_target(self, target: str, reveal_count: int) -> Tuple[str, Dict[int, str]]:
        # Reveal positions spread out
        idxs = list(range(len(target)))
        random.shuffle(idxs)
        show = sorted(idxs[: max(0, min(len(target), reveal_count))])
        known = {i: target[i] for i in show}
        mask_chars = [target[i] if i in known else "_" for i in range(len(target))]
        return "".join(mask_chars), known

    def _distance(self, a: str, b: str) -> int:
        # Levenshtein distance
        n, m = len(a), len(b)
        dp = [[0]*(m+1) for _ in range(n+1)]
        for i in range(n+1):
            dp[i][0] = i
        for j in range(m+1):
            dp[0][j] = j
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = 0 if a[i-1] == b[j-1] else 1
                dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
        return dp[n][m]

    def _valid_word(self, w: str) -> bool:
        return len(w) == self.word_length and w in self.lexicon

    def _format_state(self) -> str:
        ops_desc = ", ".join(self.allowed_ops)
        known_pos = ", ".join([f"{i}:{ch}" for i, ch in sorted(self.known_positions.items())]) or "none"
        present = "".join(sorted(self.known_present_letters)) or "none"
        absent = "".join(sorted(self.known_absent_letters)) or "none"
        return (
            f"Turns: {self.turn_count}/{self.max_turns}\n"
            f"Allowed ops: {ops_desc}\n"
            f"Hint mask: {self.current_hint_mask}\n"
            f"Known positions: {known_pos}\n"
            f"Letters present (unknown spots): {present}\n"
            f"Letters absent: {absent}\n"
            f"Remaining queries: {self.remaining_queries}\n"
            f"Word length: {self.word_length}\n"
        )

    def _get_instructions(self) -> str:
        return (
            "You are playing LexemeCipher.\n"
            "Goal: discover the hidden target word from a constrained lexicon using allowed operations and queries.\n"
            "Rules:\n"
            "- The target is a valid word from the episode's lexicon with a fixed length.\n"
            "- You may only use the operations listed in 'Allowed ops'.\n"
            "- Queries (ASK_SEM, ASK_PHON) consume query budget; proposing a word does not.\n"
            "- To win, PROPOSE the exact target word.\n"
            "- Actions must be provided as \\boxed{...}.\n"
            "Operations:\n"
            "- PROPOSE word=<candidate>: submit a guess (must be in lexicon and correct length).\n"
            "- ADD pos=<i> letter=<c>: add letter c at position i (0-based) if ADD allowed.\n"
            "- DELETE pos=<i>: delete letter at position i if DELETE allowed.\n"
            "- REPLACE pos=<i> letter=<c>: replace letter at i with c if REPLACE allowed.\n"
            "- SWAP i=<a> j=<b>: swap letters at indices a and b if SWAP allowed.\n"
            "- ASK_SEM type=<syn|ant>: get a semantic clue about the target (costs 1 query).\n"
            "- ASK_PHON type=<syllables|starts_with_vowel>: get a phonetic clue (costs 1 query).\n"
            "Formatting:\n"
            "- Example propose: \\boxed{PROPOSE word=grape}\n"
            "- Example replace: \\boxed{REPLACE pos=1 letter=a}\n"
            "- Example ask: \\boxed{ASK_SEM type=syn}\n"
        )

    def get_task_suffix(self) -> str:
        return self._format_state() + "Enter your action in \\boxed{...} format."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()
        self.turn_count = 0
        self.game_over = False

        # Build lexicon and choose target
        self.lexicon = self._make_seed_lexicon(self.word_length, self.lexicon_size_idx)
        self.semantic_map, self.phonetic_map = self._build_semantics_phonetics(self.lexicon)
        self.hidden_target = random.choice(self.lexicon)

        # Allowed operations
        self.allowed_ops = self._choose_ops(self.num_ops)

        # Initialize clues
        self.current_hint_mask, self.known_positions = self._make_mask_from_target(
            self.hidden_target, self.initial_clues
        )
        self.known_absent_letters = set()
        self.known_present_letters = set(ch for ch in set(self.current_hint_mask) if ch != "_")
        self.remaining_queries = self.query_budget

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self.game_over:
            obs = "Episode already ended."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        self.turn_count += 1

        parsed = self._parse_action(action)
        if not parsed:
            obs = "INVALID ACTION FORMAT: Use \\boxed{...} with a supported action."
            self.game_over = True
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        name = parsed.get("action", "").upper()

        if name not in self.allowed_ops:
            # Unsupported or not allowed now
            obs = f"UNSUPPORTED ACTION: '{name}' not in allowed ops."
            self.game_over = True
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        terminated = False
        truncated = False
        reward = 0.0
        obs = ""

        def update_mask_feedback(candidate: str):
            # Update letter presence/absence
            target = self.hidden_target
            for ch in set(candidate):
                if ch in target:
                    if ch not in self.known_present_letters and ch not in self.known_positions.values():
                        self.known_present_letters.add(ch)
                else:
                    self.known_absent_letters.add(ch)
            # Improve known positions if matched
            for i, ch in enumerate(candidate):
                if i < len(target) and target[i] == ch:
                    self.known_positions[i] = ch
            mask = [self.known_positions.get(i, "_") for i in range(len(target))]
            self.current_hint_mask = "".join(mask)

        if name == "PROPOSE":
            cand = parsed.get("word", "")
            if not cand or not cand.isalpha():
                obs = "PROTOCOL VIOLATION: Missing or invalid 'word' parameter."
                self.game_over = True
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            cand = cand.lower()
            if len(cand) != self.word_length:
                obs = f"PROTOCOL VIOLATION: Word length must be {self.word_length}."
                self.game_over = True
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            if not self._valid_word(cand):
                obs = "PROTOCOL VIOLATION: Word not in episode lexicon."
                self.game_over = True
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

            if cand == self.hidden_target:
                obs = f"Success! You discovered the target word '{self.hidden_target}'."
                self.game_over = True
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                dist = self._distance(cand, self.hidden_target)
                update_mask_feedback(cand)
                obs = (
                    f"Incorrect guess. Edit distance: {dist}. "
                    f"Mask now: {self.current_hint_mask}"
                )
                # Continue episode
                terminated = False

        elif name in ["ADD", "DELETE", "REPLACE", "SWAP"]:
            # Transform a working candidate string derived from mask and underscores. Operations are sandboxed;
            # they don't change hidden target, only update hints by probing structure.
            # We construct a temporary work string from mask, treating '_' as placeholder 'a'.
            temp = list(self.current_hint_mask.replace("_", "a"))
            if name == "ADD":
                if "ADD" not in self.allowed_ops:
                    obs = "PROTOCOL VIOLATION: ADD not allowed."
                    self.game_over = True
                    return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
                pos = parsed.get("pos")
                letter = parsed.get("letter")
                if pos is None or letter is None or not str(pos).isdigit() or not letter.isalpha() or len(letter) != 1:
                    obs = "PROTOCOL VIOLATION: Provide pos (int) and letter (single alpha)."
                    self.game_over = True
                    return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
                pos = int(pos)
                if not (0 <= pos <= len(temp)):
                    obs = "PROTOCOL VIOLATION: Position out of range."
                    self.game_over = True
                    return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
                temp.insert(pos, letter.lower())
                candidate = "".join(temp)
                # Trim or pad to target length for comparison feedback
                if len(candidate) > self.word_length:
                    candidate = candidate[:self.word_length]
                elif len(candidate) < self.word_length:
                    candidate = candidate + "a" * (self.word_length - len(candidate))
                update_mask_feedback(candidate)
                obs = f"ADD performed. Probing updated. Mask: {self.current_hint_mask}"

            elif name == "DELETE":
                if "DELETE" not in self.allowed_ops:
                    obs = "PROTOCOL VIOLATION: DELETE not allowed."
                    self.game_over = True
                    return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
                pos = parsed.get("pos")
                if pos is None or not str(pos).isdigit():
                    obs = "PROTOCOL VIOLATION: Provide pos (int)."
                    self.game_over = True
                    return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
                pos = int(pos)
                if not (0 <= pos < len(temp)):
                    obs = "PROTOCOL VIOLATION: Position out of range."
                    self.game_over = True
                    return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
                temp.pop(pos)
                candidate = "".join(temp)
                if len(candidate) > self.word_length:
                    candidate = candidate[:self.word_length]
                elif len(candidate) < self.word_length:
                    candidate = candidate + "a" * (self.word_length - len(candidate))
                update_mask_feedback(candidate)
                obs = f"DELETE performed. Probing updated. Mask: {self.current_hint_mask}"

            elif name == "REPLACE":
                if "REPLACE" not in self.allowed_ops:
                    obs = "PROTOCOL VIOLATION: REPLACE not allowed."
                    self.game_over = True
                    return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
                pos = parsed.get("pos")
                letter = parsed.get("letter")
                if pos is None or letter is None or not str(pos).isdigit() or not letter.isalpha() or len(letter) != 1:
                    obs = "PROTOCOL VIOLATION: Provide pos (int) and letter (single alpha)."
                    self.game_over = True
                    return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
                pos = int(pos)
                if not (0 <= pos < len(temp)):
                    obs = "PROTOCOL VIOLATION: Position out of range."
                    self.game_over = True
                    return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
                temp[pos] = letter.lower()
                candidate = "".join(temp)
                update_mask_feedback(candidate)
                obs = f"REPLACE performed. Probing updated. Mask: {self.current_hint_mask}"

            elif name == "SWAP":
                if "SWAP" not in self.allowed_ops:
                    obs = "PROTOCOL VIOLATION: SWAP not allowed."
                    self.game_over = True
                    return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
                i = parsed.get("i")
                j = parsed.get("j")
                if i is None or j is None or not str(i).isdigit() or not str(j).isdigit():
                    obs = "PROTOCOL VIOLATION: Provide i and j (ints)."
                    self.game_over = True
                    return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
                i = int(i)
                j = int(j)
                if not (0 <= i < len(temp) and 0 <= j < len(temp)):
                    obs = "PROTOCOL VIOLATION: Indices out of range."
                    self.game_over = True
                    return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
                temp[i], temp[j] = temp[j], temp[i]
                candidate = "".join(temp)
                update_mask_feedback(candidate)
                obs = f"SWAP performed. Probing updated. Mask: {self.current_hint_mask}"

        elif name == "ASK_SEM":
            if "ASK_SEM" not in self.allowed_ops:
                obs = "PROTOCOL VIOLATION: ASK_SEM not allowed."
                self.game_over = True
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            if self.remaining_queries <= 0:
                obs = "PROTOCOL VIOLATION: No remaining queries."
                self.game_over = True
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            qtype = parsed.get("type", "")
            if qtype not in ["syn", "ant"]:
                obs = "PROTOCOL VIOLATION: type must be 'syn' or 'ant'."
                self.game_over = True
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            self.remaining_queries -= 1
            rel = self.semantic_map.get(self.hidden_target, {"syn": set(), "ant": set()})
            pool = list(rel[qtype])
            clue = random.choice(pool) if pool else "(no-strong-relations)"
            obs = f"Semantic clue [{qtype}]: {clue}. Remaining queries: {self.remaining_queries}"

        elif name == "ASK_PHON":
            if "ASK_PHON" not in self.allowed_ops:
                obs = "PROTOCOL VIOLATION: ASK_PHON not allowed."
                self.game_over = True
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            if self.remaining_queries <= 0:
                obs = "PROTOCOL VIOLATION: No remaining queries."
                self.game_over = True
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            qtype = parsed.get("type", "")
            if qtype not in ["syllables", "starts_with_vowel"]:
                obs = "PROTOCOL VIOLATION: type must be 'syllables' or 'starts_with_vowel'."
                self.game_over = True
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            self.remaining_queries -= 1
            pho = self.phonetic_map.get(self.hidden_target, {"syllables": 1, "starts_with_vowel": False})
            ans = pho[qtype] if qtype in pho else "(unknown)"
            obs = f"Phonetic clue [{qtype}]: {ans}. Remaining queries: {self.remaining_queries}"

        # Time/turn handling
        if self.turn_count >= self.max_turns and not self.game_over:
            self.game_over = True
            truncated = True
            terminated = True
            obs = f"TIMEOUT: Reached max turns. The target was '{self.hidden_target}'."
            return obs, 0.0, terminated, truncated, {"suffix": self.get_task_suffix()}

        return obs or "OK", reward, terminated, truncated, {"suffix": self.get_task_suffix()}

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
        tokens["action"] = parts[0].upper()
        for part in parts[1:]:
            if "=" in part:
                k, v = part.split("=", 1)
                tokens[k] = v
        return tokens

    def sample_random_action(self) -> str:
        # Prefer a harmless ask if available, else propose a random lexicon word
        if "ASK_PHON" in self.allowed_ops and self.remaining_queries > 0:
            return r"\boxed{ASK_PHON type=syllables}"
        if "ASK_SEM" in self.allowed_ops and self.remaining_queries > 0:
            return r"\boxed{ASK_SEM type=syn}"
        if "REPLACE" in self.allowed_ops:
            return r"\boxed{REPLACE pos=0 letter=a}"
        # fallback propose
        guess = random.choice(self.lexicon) if self.lexicon else "aaa"
        return rf"\boxed{{PROPOSE word={guess}}}"


class LexemeCipherEnvWithFeedback(LexemeCipherEnv):
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
            error_detail["issue"] = "missing_boxed_format"
            hint = "Wrap your command like \\boxed{PROPOSE word=...}."

        elif text.startswith("unsupported action"):
            error_type = "UnsupportedAction"
            error_detail["allowed_ops"] = list(getattr(self, "allowed_ops", []))
            hint = f"Use one of: {', '.join(getattr(self, 'allowed_ops', []))}."

        elif text.startswith("protocol violation"):
            error_type = "ProtocolViolation"
            if "length must be" in text:
                error_detail["violation"] = "wrong_length"
                hint = f"Use a word with exactly {self.word_length} letters."
            elif "not in episode lexicon" in text:
                error_detail["violation"] = "word_not_in_lexicon"
                hint = "Query semantics/phonetics to narrow candidates, then propose a lexicon word."
            elif "provide pos" in text or "indices out of range" in text:
                error_detail["violation"] = "bad_parameters"
                hint = "Ensure numeric indices within current word bounds; provide required key=value pairs."
            elif "not allowed" in text:
                error_detail["violation"] = "op_not_allowed"
                hint = f"Only use allowed ops: {', '.join(self.allowed_ops)}."
            elif "no remaining queries" in text:
                error_detail["violation"] = "query_budget_exceeded"
                hint = "Conserve queries; switch to PROPOSE when out of queries."
            else:
                error_detail["violation"] = "generic_protocol"
                hint = "Check the operation signature and parameters in the instructions."

        elif text.startswith("incorrect guess"):
            error_type = "WrongDecision"
            error_detail["distance_hint"] = "edit_distance_shown"
            error_detail["mask"] = self.current_hint_mask
            present = "".join(sorted(self.known_present_letters)) or "none"
            absent = "".join(sorted(self.known_absent_letters)) or "none"
            error_detail["present_letters"] = present
            error_detail["absent_letters"] = absent
            hint = "Use REPLACE/SWAP to align with the mask; propose a lexicon word matching known letters."

        elif text.startswith("add performed") or text.startswith("delete performed") or text.startswith("replace performed") or text.startswith("swap performed"):
            error_type = "OK"
            error_detail["mask"] = self.current_hint_mask
            hint = "Leverage the updated mask to choose a lexicon word and PROPOSE it."

        elif text.startswith("semantic clue") or text.startswith("phonetic clue"):
            error_type = "OK"
            error_detail["remaining_queries"] = self.remaining_queries
            hint = "Combine clue with mask to shortlist candidates; then PROPOSE."

        elif text.startswith("timeout"):
            error_type = "Timeout"
            error_detail["target"] = getattr(self, "hidden_target", None)
            hint = "Ask strategic clues early, then commit to a PROPOSE before time runs out."

        elif "success! you discovered" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["remaining_queries"] = getattr(self, "remaining_queries", None)
            diagnostic["allowed_ops"] = list(getattr(self, "allowed_ops", []))
            diagnostic["mask"] = getattr(self, "current_hint_mask", "")
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start by asking a clue (ASK_PHON type=syllables) or propose a plausible lexicon word.",
            "turn": 0,
            "remaining_queries": self.remaining_queries,
            "allowed_ops": list(self.allowed_ops),
            "mask": self.current_hint_mask,
        }
        return obs, info