from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class CodeReviewEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 12,
        **_,
    ):
        super().__init__()
        self.complexity = int(max(1, min(10, complexity)))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 12

        self.complexity_params = {
            "repo_files": (2, 8),          # More files increases search space → harder
            "decoy_functions": (0, 5),     # More decoys increase confusion → harder
            "tests_count": (2, 6),         # More tests to consider → more decisions → harder
            "spec_detail": (3, 1),         # REVERSED: Less detailed spec → harder
            "bug_type_count": (2, 5),      # More classification options → harder
            "code_length": (30, 140),      # Larger codebase → harder
            "allowed_checks": (4, 1),      # REVERSED: Fewer analyzer rules available → harder
        }
        self.param_variance = {
            "repo_files": 1,
            "decoy_functions": 1,
            "tests_count": 1,
            "spec_detail": 0,
            "bug_type_count": 1,
            "code_length": 12,
            "allowed_checks": 0,
        }

        self.repo_files: int = 0
        self.decoy_functions: int = 0
        self.tests_count: int = 0
        self.spec_detail: int = 0
        self.bug_type_count: int = 0
        self.code_length: int = 0
        self.allowed_checks: int = 0

        self.turn_count: int = 0
        self.repo: Dict[str, str] = {}
        self.files_order: list = []
        self.categories: list = []
        self.correct_category: str = ""
        self.allowed_rules: list = []
        self.tests: list = []
        self.has_listed_tests: bool = False
        self.last_answer: Optional[str] = None

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
            low, high = (min_val, max_val) if min_val <= max_val else (max_val, min_val)
            val = max(low, min(high, val))
            setattr(self, name, int(round(val)))

    def _build_main_code(self, bug: str) -> str:
        if bug == "Correct":
            code = [
                "def process_list(nums):",
                "    total = 0",
                "    for x in nums:",
                "        if isinstance(x, int):",
                "            total += x",
                "    return total",
            ]
        elif bug == "OffByOne":
            code = [
                "def process_list(nums):",
                "    total = 0",
                "    for i in range(len(nums)-1):",
                "        x = nums[i]",
                "        if isinstance(x, int):",
                "            total += x",
                "    return total",
            ]
        elif bug == "MissingEdgeCase":
            code = [
                "def process_list(nums):",
                "    total = nums[0]",
                "    for x in nums[1:]:",
                "        if isinstance(x, int):",
                "            total += x",
                "    return total",
            ]
        elif bug == "WrongCalculation":
            code = [
                "def process_list(nums):",
                "    ints = [x for x in nums if isinstance(x, int)]",
                "    return max(ints) if ints else 0",
            ]
        elif bug == "Inefficient":
            code = [
                "def process_list(nums):",
                "    total = 0",
                "    for i in range(len(nums)):",
                "        for j in range(i+1):",
                "            x = nums[i]",
                "            if isinstance(x, int):",
                "                total += x",
                "    return total",
            ]
        elif bug == "UnsafeEval":
            code = [
                "def process_list(nums):",
                "    total = 0",
                "    for x in nums:",
                "        if isinstance(x, str):",
                "            try:",
                "                total += eval(x)",
                "            except Exception:",
                "                pass",
                "        elif isinstance(x, int):",
                "            total += x",
                "    return total",
            ]
        else:
            code = [
                "def process_list(nums):",
                "    total = 0",
                "    for x in nums:",
                "        if isinstance(x, int):",
                "            total += x",
                "    return total",
            ]
        return "\n".join(code)

    def _generate_repo(self):
        base_names = ["main.py", "utils.py", "algo.py", "helpers.py", "data.py", "maths.py", "misc.py", "io.py"]
        self.files_order = base_names[: self.repo_files]
        self.repo = {}
        self.repo["main.py"] = self._build_main_code(self.correct_category)
        decoy_templates = [
            "def process_data(data):\n    return [d for d in data if d]\n",
            "def helper_sum(arr):\n    s=0\n    for a in arr:\n        s+=a\n    return s\n",
            "def sum_like(arr):\n    return sum([a for a in arr if isinstance(a,int)])\n",
            "def safe_parse(s):\n    try:\n        return int(s)\n    except:\n        return 0\n",
            "def strange_eval(s):\n    # do not use eval in production\n    try:\n        return eval(s)\n    except:\n        return None\n",
        ]
        decoys_to_add = min(self.decoy_functions, len(decoy_templates))
        extra_lines = self.code_length - len(self.repo["main.py"].splitlines())
        filler_line = "# filler\n"
        for fname in self.files_order:
            if fname == "main.py":
                continue
            content = ""
            for _ in range(3):
                content += filler_line
            for i in range(decoys_to_add):
                content += decoy_templates[i]
            # distribute remaining filler to reach approximate code_length
            need = max(0, extra_lines // max(1, self.repo_files - 1))
            content += (filler_line * need)
            self.repo[fname] = content

    def _build_tests(self):
        all_tests = ["test_empty", "test_positive", "test_negative", "test_mixed", "test_large", "test_types"]
        self.tests = all_tests[: self.tests_count]
        self.has_listed_tests = False

    def _build_rules(self):
        rule_order = ["bounds", "complexity", "safety", "edge"]
        self.allowed_rules = rule_order[: self.allowed_checks]

    def _compute_test_result(self, name: str) -> str:
        bug = self.correct_category
        if name == "test_empty":
            if bug == "MissingEdgeCase":
                return "FAIL (IndexError)"
            elif bug in ["Correct", "OffByOne", "Inefficient", "UnsafeEval", "WrongCalculation"]:
                return "PASS"
        elif name == "test_positive":
            if bug in ["Correct"]:
                return "PASS"
            elif bug in ["OffByOne", "WrongCalculation"]:
                return "FAIL"
            elif bug == "Inefficient":
                return "PASS"
            elif bug == "UnsafeEval":
                return "PASS"
            elif bug == "MissingEdgeCase":
                return "PASS"
        elif name == "test_negative":
            if bug == "Correct":
                return "PASS"
            elif bug in ["OffByOne", "WrongCalculation"]:
                return "FAIL"
            elif bug == "Inefficient":
                return "PASS"
            elif bug == "UnsafeEval":
                return "PASS"
            elif bug == "MissingEdgeCase":
                return "PASS"
        elif name == "test_mixed":
            if bug == "Correct":
                return "PASS"
            elif bug in ["OffByOne", "WrongCalculation"]:
                return "FAIL"
            elif bug == "Inefficient":
                return "PASS"
            elif bug == "UnsafeEval":
                return "FAIL"
            elif bug == "MissingEdgeCase":
                return "PASS"
        elif name == "test_large":
            if bug == "Inefficient":
                return "FAIL (timeout)"
            else:
                return "PASS"
        elif name == "test_types":
            if bug == "UnsafeEval":
                return "FAIL"
            elif bug in ["Correct", "OffByOne", "MissingEdgeCase", "WrongCalculation", "Inefficient"]:
                return "PASS"
        return "PASS"

    def _analyze(self, rule: str) -> str:
        content = self.repo.get("main.py", "")
        bug = self.correct_category
        if rule == "bounds":
            if "range(len(nums)-1)" in content or "nums[0]" in content:
                return "bounds: risk detected"
            else:
                return "bounds: no obvious risk"
        if rule == "complexity":
            if "for j in range(i+1)" in content:
                return "complexity: nested loops observed (likely > O(n))"
            else:
                return "complexity: single loop or simple ops (likely O(n))"
        if rule == "safety":
            if "eval(" in content:
                return "safety: unsafe eval detected"
            else:
                return "safety: no obvious unsafe constructs"
        if rule == "edge":
            if "nums[0]" in content:
                return "edge: missing empty-list guard"
            elif "total = 0" in content:
                return "edge: likely handles empty list"
            else:
                return "edge: uncertain"
        return "unsupported rule"

    def _get_spec_text(self) -> str:
        if self.spec_detail == 3:
            return (
                "Spec: Implement process_list(nums).\n"
                "- Return the sum of all integers in the list.\n"
                "- Ignore non-integer elements.\n"
                "- Handle empty list by returning 0 (no exceptions).\n"
                "- Time complexity must be linear O(n).\n"
                "- Avoid unsafe constructs like eval or dynamic execution."
            )
        elif self.spec_detail == 2:
            return (
                "Spec: process_list(nums) should sum integer elements and be efficient.\n"
                "Avoid unsafe constructs. Handle empty input gracefully."
            )
        else:
            return "Spec: sum integer elements; ignore others; handle empty input."

    def _get_instructions(self) -> str:
        actions = [
            "list",
            "spec",
            "tests",
            "view <filename>",
            "grep <pattern>",
            "analyze <rule>",
            "run <test_name>",
            "answer <category>",
        ]
        rules_str = ", ".join(self.allowed_rules) if self.allowed_rules else "(none)"
        cats_str = ", ".join(self.categories)
        example = self.sample_random_action()
        return (
            "Code Review Game:\n"
            "Your goal is to classify the target implementation's bug type for function process_list.\n"
            "Available categories: " + cats_str + "\n"
            "Allowed analyzer rules: " + rules_str + "\n"
            "Actions:\n- " + "\n- ".join(actions) + "\n"
            "Use \\boxed{...} format for all actions.\n"
            "Example: " + example
        )

    def get_task_suffix(self) -> str:
        files = ", ".join(self.files_order)
        tests = ", ".join(self.tests)
        rules = ", ".join(self.allowed_rules)
        return (
            f"Turn {self.turn_count}/{self.max_turns}\n"
            f"Files known: {files}\n"
            f"Tests available: {tests}\n"
            f"Analyzer rules: {rules}\n"
            "Enter your action in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()
        self.turn_count = 0
        base_cats = ["Correct", "OffByOne", "MissingEdgeCase", "WrongCalculation", "Inefficient", "UnsafeEval"]
        n = max(2, self.bug_type_count)
        self.categories = base_cats[:n]
        self.correct_category = random.choice(self.categories)
        self._build_tests()
        self._build_rules()
        self._generate_repo()
        self.has_listed_tests = False
        self.last_answer = None
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)
        if parsed is None:
            obs = "Invalid action format. Use \\boxed{...}"
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
        terminated = False
        truncated = False
        reward = 0.0
        obs = ""

        atype = parsed["type"]
        arg = parsed.get("arg")

        if atype in ("list", "list_files"):
            obs = "Files: " + ", ".join(self.files_order)
        elif atype == "spec":
            obs = self._get_spec_text()
        elif atype == "tests":
            self.has_listed_tests = True
            obs = "Tests: " + ", ".join(self.tests)
        elif atype == "view":
            fname = arg
            if not fname or fname not in self.repo:
                obs = f"Unsupported action: unknown file '{fname}'"
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            content = self.repo[fname]
            obs = f"Viewing {fname}:\n" + content
        elif atype == "grep":
            pat = arg or ""
            results = []
            for fname, content in self.repo.items():
                for i, line in enumerate(content.splitlines(), start=1):
                    if pat and pat in line:
                        results.append(f"{fname}:{i}: {line.strip()}")
            if results:
                obs = "Grep results:\n" + "\n".join(results)
            else:
                obs = "Grep results: no matches"
        elif atype == "analyze":
            rule = (arg or "").strip()
            if rule not in self.allowed_rules:
                obs = f"Unsupported analyzer rule: '{rule}'"
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            obs = "Analysis: " + self._analyze(rule)
        elif atype == "run":
            if not self.has_listed_tests:
                obs = "Protocol violation: call 'tests' before 'run'"
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            tname = (arg or "").strip()
            if tname not in self.tests:
                obs = f"Unsupported action: unknown test '{tname}'"
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            result = self._compute_test_result(tname)
            obs = f"Test {tname}: {result}"
        elif atype == "answer":
            cat = (arg or "").strip()
            self.last_answer = cat
            if cat not in self.categories:
                obs = f"Unsupported answer category: '{cat}'"
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if cat == self.correct_category:
                obs = f"Success! Correct classification: {cat}."
                reward = 1.0
                terminated = True
            else:
                obs = f"Failed! Wrong classification. The correct bug type was {self.correct_category}."
                reward = -1.0
                terminated = True
        else:
            obs = f"Unsupported action: '{atype}'"
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        if not terminated and self.turn_count >= self.max_turns:
            truncated = True
            terminated = True
            obs = f"Reached max turns ({self.max_turns})"

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Any]:
        if not action:
            return None
        pattern = re.compile(r"\\boxed\{(.+?)\}", re.IGNORECASE | re.DOTALL)
        m = pattern.findall(action)
        if not m:
            return None
        extracted = m[-1].strip()
        tokens = extracted.split()
        if not tokens:
            return None
        head = tokens[0].lower()
        rest = " ".join(tokens[1:]) if len(tokens) > 1 else None
        if head in ["list", "list_files", "spec", "tests"]:
            return {"type": head}
        if head in ["view", "grep", "analyze", "run", "answer"]:
            return {"type": head, "arg": rest}
        return {"type": head}

    def sample_random_action(self) -> str:
        options = []
        options.append("\\boxed{list}")
        options.append("\\boxed{spec}")
        if self.files_order:
            options.append(f"\\boxed{{view {self.files_order[0]}}}")
        if self.tests:
            options.append("\\boxed{tests}")
            options.append(f"\\boxed{{run {self.tests[0]}}}")
        if self.allowed_rules:
            options.append(f"\\boxed{{analyze {self.allowed_rules[0]}}}")
        options.append("\\boxed{grep eval}")
        options.append(f"\\boxed{{answer {self.categories[0] if self.categories else 'Correct'}}}")
        return random.choice(options)


class CodeReviewEnvWithFeedback(CodeReviewEnv):
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
            hint = "Use \\boxed{your_action} exactly, e.g., \\boxed{list}."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "run_test_before_listing"
            hint = "List tests first with \\boxed{tests}, then run a specific test with \\boxed{run <test_name>}."
        elif "unsupported analyzer rule" in text:
            error_type = "UnsupportedAction"
            error_detail["rule"] = True
            hint = f"Use one of: {', '.join(self.allowed_rules)}."
        elif "unsupported action: unknown file" in text:
            error_type = "UnsupportedAction"
            error_detail["file"] = True
            hint = f"List files with \\boxed{{list}} and then view a valid file like \\boxed{{view {self.files_order[0]}}}."
        elif "unsupported action: unknown test" in text:
            error_type = "UnsupportedAction"
            error_detail["test"] = True
            hint = "Call \\boxed{tests} to see available tests, then run one of them."
        elif "unsupported action:" in text or "unsupported action" in text:
            error_type = "UnsupportedAction"
            hint = "Valid actions: list, spec, tests, view <file>, grep <pattern>, analyze <rule>, run <test>, answer <category>."
        elif "unsupported answer category" in text:
            error_type = "UnsupportedAction"
            error_detail["answer_category"] = True
            hint = f"Choose from: {', '.join(self.categories)}."
        elif "failed! wrong classification" in text:
            error_type = "WrongDecision"
            correct = self.correct_category
            got = getattr(self, "last_answer", None)
            error_detail["expected"] = correct
            error_detail["got"] = got
            hint = "Gather evidence: run tests (\\boxed{tests} then \\boxed{run <name>}) and try analyze (\\boxed{analyze <rule>})."
        elif "reached max turns" in text:
            error_type = "Timeout"
            hint = "Plan actions: list files, inspect main.py, run key tests, then answer."
        elif "success! correct classification" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "has_listed_tests": getattr(self, "has_listed_tests", False),
                "allowed_rules": getattr(self, "allowed_rules", []),
                "categories": getattr(self, "categories", []),
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
            "hint": "Start by reading the spec with \\boxed{spec} or listing files with \\boxed{list}.",
            "turn": 0,
            "state": {
                "has_listed_tests": False,
                "allowed_rules": getattr(self, "allowed_rules", []),
                "categories": getattr(self, "categories", []),
            },
        }
        return obs, info