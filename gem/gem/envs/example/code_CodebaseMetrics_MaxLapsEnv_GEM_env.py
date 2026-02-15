from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class CodebaseMetricsEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = None,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization

        # Evolvable parameters
        self.complexity_params = {
            # Number of files: more files increases breadth of exploration and aggregation difficulty
            "num_files": (2, 10),
            # Minimum functions per file: more functions increases per-file analysis workload
            "min_funcs_per_file": (2, 6),
            # Maximum functions per file: larger upper bound increases variability and potential workload
            "max_funcs_per_file": (4, 12),
            # REVERSED: Turn budget (episode interaction limit). Fewer turns → harder due to tight resource constraints
            "turn_budget": (25, 10),
            # Branch upper bound per function: higher bound increases function complexity variance (harder aggregation)
            "branch_max": (3, 9),
            # LOC minimum per function: higher lower bound slightly increases magnitude of values (noise for reasoning)
            "loc_min": (20, 50),
            # LOC maximum per function: higher upper bound expands numeric range (harder to scan/aggregate manually)
            "loc_max": (120, 300),
            # Number of rule predicates: more predicates require more attributes and tighter reasoning
            "rule_predicates": (1, 3),
            # Number of languages present: more languages increases heterogeneity and selection filter complexity
            "num_languages": (1, 3),
            # Modulus base for the branch parity predicate when present: larger base→more nuanced filter
            "branch_mod_base": (2, 3),
            # REVERSED: LOC threshold predicate when present (functions must have LOC <= threshold). Lower threshold→stricter filter, harder reasoning to include/exclude
            "loc_threshold": (200, 80),
        }

        # Randomization variances
        self.param_variance = {
            "num_files": 1,
            "min_funcs_per_file": 1,
            "max_funcs_per_file": 2,
            "turn_budget": 2,
            "branch_max": 1,
            "loc_min": 5,
            "loc_max": 15,
            "rule_predicates": 0,
            "num_languages": 0,
            "branch_mod_base": 0,
            "loc_threshold": 10,
        }

        # Placeholder attributes set in _apply_complexity_params
        self.num_files: int = 0
        self.min_funcs_per_file: int = 0
        self.max_funcs_per_file: int = 0
        self.turn_budget: int = 0
        self.branch_max: int = 0
        self.loc_min: int = 0
        self.loc_max: int = 0
        self.rule_predicates: int = 0
        self.num_languages: int = 0
        self.branch_mod_base: int = 0
        self.loc_threshold: int = 0

        # Fixed parameters and state
        self.turn_count: int = 0
        self.files: List[Dict[str, Any]] = []
        self.languages_pool: List[str] = ["py", "js", "java"]
        self.rule_lang_filter: Optional[str] = None
        self.goal_value: int = 0
        self.last_observation: str = ""
        self.max_turns = max_turns if max_turns is not None else 100

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
            # Clamp within range (supports reversed)
            low, high = (min_val, max_val) if min_val <= max_val else (max_val, min_val)
            actual_value = max(low, min(high, actual_value))
            setattr(self, param_name, int(round(actual_value)))

        # Ensure feasibility constraints
        if self.min_funcs_per_file > self.max_funcs_per_file:
            self.min_funcs_per_file = self.max_funcs_per_file
        # Ensure turn budget sufficient to query RULE + each file summary + ANSWER
        minimal_needed = 2 + self.num_files  # RULE + per-file summary + ANSWER
        self.turn_budget = max(self.turn_budget, minimal_needed)
        # Use budget as episode-level max_turns
        self.max_turns = self.turn_budget

        # Cap num_languages to available pool size
        self.num_languages = max(1, min(self.num_languages, len(self.languages_pool)))

    def _get_instructions(self) -> str:
        return (
            "You are analyzing a synthetic codebase to compute a metric.\n"
            "Goal: Compute the total cyclomatic complexity (branches + 1) of all functions that satisfy the hidden selection rule.\n"
            "You can query the rule and explore the codebase using commands:\n"
            "- RULE: reveal the selection rule\n"
            "- FILES: list files and their languages\n"
            "- FUNCTIONS <file_id>: show how many functions are in a file\n"
            "- FILE_SUMMARY <file_id>: list each function's LOC and branches in a file\n"
            "- FUNC_ATTR <file_id> <func_id> <attr>: get a specific attribute (loc, branches, lang)\n"
            "- FUNC_CYC <file_id> <func_id>: get cyclomatic complexity for one function\n"
            "- FILE_SELECTED_CYC <file_id>: sum of complexities in a file for functions that meet the rule\n"
            "- SUM <int1> <int2> ...: sum provided integers you supply\n"
            "- ANSWER <integer>: submit the final total complexity\n"
            "Invalid or out-of-range commands terminate the episode with a penalty.\n"
            f"Use \\boxed{{...}} to submit actions. Example: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        file_count = len(self.files)
        return (
            f"State: turn {self.turn_count}/{self.max_turns}. Files: {file_count}. "
            "Submit your next action in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()
        self.turn_count = 0
        self.files = []
        self.rule_lang_filter = None
        self.goal_value = 0
        self.last_observation = ""

        allowed_langs = self.languages_pool[:self.num_languages]

        for f_id in range(self.num_files):
            lang = random.choice(allowed_langs)
            num_funcs = random.randint(self.min_funcs_per_file, self.max_funcs_per_file)
            funcs = []
            for j in range(num_funcs):
                loc = random.randint(self.loc_min, self.loc_max)
                branches = random.randint(0, self.branch_max)
                funcs.append({"loc": loc, "branches": branches})
            self.files.append({"id": f_id, "lang": lang, "functions": funcs})

        # Build selection rule
        self.rule_lang_filter = None
        has_loc = True
        has_branch_mod = self.rule_predicates >= 2
        has_lang = self.rule_predicates >= 3
        if has_lang:
            self.rule_lang_filter = random.choice(allowed_langs)

        # Compute ground truth
        total = 0
        for f in self.files:
            for fn in f["functions"]:
                include = True
                if has_loc:
                    include = include and (fn["loc"] <= self.loc_threshold)
                if has_branch_mod:
                    include = include and (fn["branches"] % self.branch_mod_base == 0)
                if has_lang:
                    include = include and (f["lang"] == self.rule_lang_filter)
                if include:
                    total += fn["branches"] + 1
        self.goal_value = total

        obs = self._get_instructions()
        self.last_observation = obs
        return obs, {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if parsed is None:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            self.last_observation = obs
            return (
                obs,
                LanguageGameReward.format_error_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        cmd = parsed["cmd"]
        args = parsed["args"]

        def end_with_protocol_error(msg: str):
            obs = f"Protocol violation: {msg}. Episode terminated."
            self.last_observation = obs
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        if cmd == "HELP":
            obs = self._get_instructions()
            self.last_observation = obs
            if self.turn_count >= self.max_turns:
                return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        elif cmd == "RULE":
            parts = []
            parts.append(f"Predicate 1: LOC <= {self.loc_threshold}")
            if self.rule_predicates >= 2:
                parts.append(f"Predicate 2: branches % {self.branch_mod_base} == 0")
            if self.rule_predicates >= 3 and self.rule_lang_filter is not None:
                parts.append(f"Predicate 3: file language == {self.rule_lang_filter}")
            rule_text = "Selection rule: " + " AND ".join(parts) + "."
            obs = rule_text
            self.last_observation = obs

        elif cmd == "FILES":
            entries = []
            for f in self.files:
                entries.append(f"[id={f['id']}, lang={f['lang']}, functions={len(f['functions'])}]")
            obs = "Files: " + "; ".join(entries) if entries else "Files: none."
            self.last_observation = obs

        elif cmd == "FUNCTIONS":
            if len(args) != 1:
                return end_with_protocol_error("FUNCTIONS expects 1 argument <file_id>")
            try:
                fid = int(args[0])
            except:
                return end_with_protocol_error("FUNCTIONS <file_id> must be an integer")
            if fid < 0 or fid >= len(self.files):
                return end_with_protocol_error("file_id out of range")
            numf = len(self.files[fid]["functions"])
            obs = f"File {fid} has {numf} functions. IDs: 0..{numf - 1}" if numf > 0 else f"File {fid} has 0 functions."
            self.last_observation = obs

        elif cmd == "FILE_SUMMARY":
            if len(args) != 1:
                return end_with_protocol_error("FILE_SUMMARY expects 1 argument <file_id>")
            try:
                fid = int(args[0])
            except:
                return end_with_protocol_error("FILE_SUMMARY <file_id> must be an integer")
            if fid < 0 or fid >= len(self.files):
                return end_with_protocol_error("file_id out of range")
            f = self.files[fid]
            lines = [f"File {fid} (lang={f['lang']}) function summary:"]
            for idx, fn in enumerate(f["functions"]):
                lines.append(f"- func {idx}: LOC={fn['loc']}, branches={fn['branches']}")
            obs = "\n".join(lines) if len(lines) > 1 else f"File {fid} (lang={f['lang']}) has no functions."
            self.last_observation = obs

        elif cmd == "FUNC_ATTR":
            if len(args) != 3:
                return end_with_protocol_error("FUNC_ATTR expects 3 arguments <file_id> <func_id> <attr>")
            try:
                fid = int(args[0])
                fidx = int(args[1])
            except:
                return end_with_protocol_error("FUNC_ATTR <file_id> and <func_id> must be integers")
            attr = args[2].lower()
            if fid < 0 or fid >= len(self.files):
                return end_with_protocol_error("file_id out of range")
            funcs = self.files[fid]["functions"]
            if fidx < 0 or fidx >= len(funcs):
                return end_with_protocol_error("func_id out of range")
            if attr not in {"loc", "branches", "lang"}:
                return end_with_protocol_error("unsupported attr; use loc, branches, or lang")
            if attr == "lang":
                val = self.files[fid]["lang"]
            else:
                val = funcs[fidx][attr]
            obs = f"FUNC_ATTR {fid} {fidx} {attr} => {val}"
            self.last_observation = obs

        elif cmd == "FUNC_CYC":
            if len(args) != 2:
                return end_with_protocol_error("FUNC_CYC expects 2 arguments <file_id> <func_id>")
            try:
                fid = int(args[0]); fidx = int(args[1])
            except:
                return end_with_protocol_error("FUNC_CYC arguments must be integers")
            if fid < 0 or fid >= len(self.files):
                return end_with_protocol_error("file_id out of range")
            funcs = self.files[fid]["functions"]
            if fidx < 0 or fidx >= len(funcs):
                return end_with_protocol_error("func_id out of range")
            cyc = funcs[fidx]["branches"] + 1
            obs = f"FUNC_CYC {fid} {fidx} => {cyc}"
            self.last_observation = obs

        elif cmd == "FILE_SELECTED_CYC":
            if len(args) != 1:
                return end_with_protocol_error("FILE_SELECTED_CYC expects 1 argument <file_id>")
            try:
                fid = int(args[0])
            except:
                return end_with_protocol_error("FILE_SELECTED_CYC <file_id> must be an integer")
            if fid < 0 or fid >= len(self.files):
                return end_with_protocol_error("file_id out of range")
            f = self.files[fid]
            has_loc = True
            has_branch_mod = self.rule_predicates >= 2
            has_lang = self.rule_predicates >= 3
            total = 0
            for fn in f["functions"]:
                include = True
                if has_loc:
                    include = include and (fn["loc"] <= self.loc_threshold)
                if has_branch_mod:
                    include = include and (fn["branches"] % self.branch_mod_base == 0)
                if has_lang:
                    include = include and (f["lang"] == self.rule_lang_filter)
                if include:
                    total += fn["branches"] + 1
            obs = f"FILE_SELECTED_CYC {fid} => {total}"
            self.last_observation = obs

        elif cmd == "SUM":
            if len(args) < 1:
                return end_with_protocol_error("SUM expects at least 1 integer argument")
            try:
                nums = [int(a) for a in args]
            except:
                return end_with_protocol_error("SUM arguments must be integers")
            total = sum(nums)
            obs = f"SUM => {total}"
            self.last_observation = obs

        elif cmd == "ANSWER":
            if len(args) != 1:
                return end_with_protocol_error("ANSWER expects exactly 1 integer argument")
            try:
                val = int(args[0])
            except:
                return end_with_protocol_error("ANSWER argument must be an integer")
            if val == self.goal_value:
                obs = f"Success! Final answer matches {self.goal_value}."
                self.last_observation = obs
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Failed! Submitted answer {val} does not match reference value {self.goal_value}."
                self.last_observation = obs
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = f"Unsupported action: {cmd}. Episode terminated."
            self.last_observation = obs
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns})."
            self.last_observation = obs
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return self.last_observation, 0.0, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        extracted = matches[-1].group(1).strip()
        tokens = extracted.split()
        if len(tokens) == 0:
            return None
        cmd = tokens[0].upper()
        args = tokens[1:]
        return {"cmd": cmd, "args": args}

    def sample_random_action(self) -> str:
        examples = [
            "\\boxed{RULE}",
            "\\boxed{FILES}",
            "\\boxed{FUNCTIONS 0}",
            "\\boxed{FILE_SUMMARY 0}",
            "\\boxed{FUNC_ATTR 0 0 loc}",
            "\\boxed{FUNC_CYC 0 0}",
            "\\boxed{FILE_SELECTED_CYC 0}",
            "\\boxed{SUM 3 5 7}",
            "\\boxed{ANSWER 0}",
        ]
        return random.choice(examples)


class CodebaseMetricsEnvWithFeedback(CodebaseMetricsEnv):
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
            error_detail["issue"] = "missing_boxed_format"
            hint = "Wrap your command in \\boxed{...}, e.g., \\boxed{RULE}."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            # Extract some detail
            m = re.search(r"protocol violation:\s*(.+?)\.", obs, re.IGNORECASE)
            if m:
                error_detail["violation"] = m.group(1)
            hint = "Check command name and argument count. Use FILES and FUNCTIONS to verify indices."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            m = re.search(r"unsupported action:\s*(.+?)\.", obs, re.IGNORECASE)
            if m:
                error_detail["action"] = m.group(1)
            hint = "Use HELP to see valid commands: RULE, FILES, FUNCTIONS, FILE_SUMMARY, FUNC_ATTR, FUNC_CYC, FILE_SELECTED_CYC, SUM, ANSWER."

        elif "failed! submitted answer" in text:
            error_type = "WrongDecision"
            m = re.search(r"failed! submitted answer\s*(\-?\d+)\s*does not match reference value\s*(\-?\d+)", obs, re.IGNORECASE)
            if m:
                error_detail["got"] = int(m.group(1))
                error_detail["expected"] = int(m.group(2))
            hint = "Query RULE, then use FILE_SUMMARY for each file to compute (branches+1) for included functions. Sum results and submit ANSWER."

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Prioritize RULE and FILE_SUMMARY for each file, then compute and ANSWER. Avoid unnecessary per-function queries."

        elif "success!" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["num_files"] = getattr(self, "num_files", None)
            diagnostic["rule_predicates"] = getattr(self, "rule_predicates", None)
            diagnostic["budget"] = getattr(self, "max_turns", None)
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint
        info["diagnostic"] = diagnostic

        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start by querying RULE, then use FILE_SUMMARY for each file.",
            "turn": 0,
            "num_files": getattr(self, "num_files", None),
            "rule_predicates": getattr(self, "rule_predicates", None),
            "budget": getattr(self, "max_turns", None),
        }
        return obs, info