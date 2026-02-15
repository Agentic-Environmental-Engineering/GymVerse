from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class RepoRiskIndexEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 60,
        **_,
    ):
        super().__init__()
        self.complexity = int(max(1, min(10, complexity)))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 60

        self.complexity_params = {
            # num_files: number of files in the repository. More files → larger search space → harder.
            "num_files": (2, 12),
            # max_lines_per_file: upper limit of lines per file. More lines → more scanning → harder.
            "max_lines_per_file": (15, 90),
            # indent_richness: max indentation depth used in generation. Higher → more varied nesting → harder to reason.
            "indent_richness": (1, 6),
            # mutation_level: decoy tokens rate (e.g., TODOX, FIX-ME). Higher → more near-misses → harder.
            "mutation_level": (0, 5),
            # nesting_threshold: minimum indentation depth considered "deep". Higher → fewer deep lines, but more careful analysis required.
            "nesting_threshold": (2, 4),
            # weight_variability: amplitude of variation around base weights. Higher → weights deviate more → harder (must query).
            "weight_variability": (0, 2),
        }

        self.param_variance = {
            "num_files": 1,
            "max_lines_per_file": 5,
            "indent_richness": 1,
            "mutation_level": 1,
            "nesting_threshold": 0,
            "weight_variability": 1,
        }

        self.num_files: int = 0
        self.max_lines_per_file: int = 0
        self.indent_richness: int = 0
        self.mutation_level: int = 0
        self.nesting_threshold: int = 0
        self.weight_variability: int = 0

        self.turn_count: int = 0
        self.files: Dict[str, str] = {}
        self.numeric_artifacts: list = []
        self.base_weights = {"TODO": 2, "FIXME": 3, "RISKY": 4, "DEEP": 1}
        self.weights: Dict[str, int] = {}
        self.target_bri: int = 0
        self.last_action_struct: Optional[Dict[str, Any]] = None

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
                    if min_val > max_val:
                        actual_value = max(max_val, min(min_val, actual_value))
                    else:
                        actual_value = max(min_val, min(max_val, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        return (
            "Repository Risk Index Analysis Game.\n"
            "Goal: Compute the repository-wide Bug Risk Index (BRI) and submit it.\n"
            "BRI = w_TODO * (# of 'TODO') + w_FIXME * (# of 'FIXME') + w_RISKY * (# of risky calls 'eval'/'exec') + w_DEEP * (# of lines with indentation depth >= nesting_threshold).\n"
            "Weights may vary per episode. Query them with WEIGHTS.\n"
            "Available actions (must be in \\boxed{...}):\n"
            "- LIST\n"
            "- SHOW <filename>\n"
            "- GREP <pattern> IN <filename>\n"
            "- COUNT <pattern> IN REPO\n"
            "- COUNT <pattern> IN <filename>\n"
            "- INDENT_STATS <filename>\n"
            "- SUM_ARTIFACTS\n"
            "- CLEAR_ARTIFACTS\n"
            "- WEIGHTS\n"
            "- STATUS\n"
            "- SUBMIT <integer>\n"
            "Patterns are case-sensitive substrings. Risky calls are tokens 'eval' and 'exec'.\n"
            f"Example action: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        files_list = ", ".join(sorted(self.files.keys()))
        return (
            f"Turn {self.turn_count}/{self.max_turns}. Files: {len(self.files)} [{files_list}]. "
            f"Nesting threshold: {self.nesting_threshold}. Numeric artifacts: {len(self.numeric_artifacts)}. "
            "Submit final BRI using \\boxed{SUBMIT <int>}. Use \\boxed{WEIGHTS} to see current weights."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()
        self.turn_count = 0
        self.numeric_artifacts = []
        self.last_action_struct = None

        self._generate_repo()
        self._derive_weights()
        self._compute_target_bri()

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        self.last_action_struct = parsed

        if not parsed:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            return (
                obs,
                LanguageGameReward.format_error_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        cmd = parsed.get("cmd")
        obs = ""
        reward = 0.0

        if cmd == "LIST":
            obs = f"Files: {', '.join(sorted(self.files.keys()))}."
        elif cmd == "SHOW":
            fname = parsed.get("filename")
            if fname not in self.files:
                obs = f"Protocol error: unknown file '{fname}'."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            content = self.files[fname]
            obs = f"Showing {fname}:\n{content}"
        elif cmd == "GREP":
            fname = parsed.get("filename")
            pattern = parsed.get("pattern")
            if fname not in self.files:
                obs = f"Protocol error: unknown file '{fname}'."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            lines = self.files[fname].splitlines()
            matches = []
            for i, line in enumerate(lines, start=1):
                if pattern in line:
                    matches.append(f"{i}: {line}")
            if matches:
                obs = f"GREP '{pattern}' in {fname}:\n" + "\n".join(matches)
            else:
                obs = f"GREP '{pattern}' in {fname}: no matches."
        elif cmd == "COUNT":
            scope = parsed.get("scope")
            pattern = parsed.get("pattern")
            count = 0
            if scope == "REPO":
                for content in self.files.values():
                    count += content.count(pattern)
            else:
                fname = parsed.get("filename")
                if fname not in self.files:
                    obs = f"Protocol error: unknown file '{fname}'."
                    return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
                count = self.files[fname].count(pattern)
            self.numeric_artifacts.append(count)
            obs = f"COUNT '{pattern}' in {scope}: {count} (pushed to artifacts)."
        elif cmd == "INDENT_STATS":
            fname = parsed.get("filename")
            if fname not in self.files:
                obs = f"Protocol error: unknown file '{fname}'."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            lines = self.files[fname].splitlines()
            depth_counts = {}
            deep_count = 0
            for line in lines:
                m = re.match(r"^( +)", line)
                spaces = len(m.group(1)) if m else 0
                depth = spaces // 4
                depth_counts[depth] = depth_counts.get(depth, 0) + 1
                if depth >= self.nesting_threshold:
                    deep_count += 1
            self.numeric_artifacts.append(deep_count)
            parts = [f"depth {d}: {c}" for d, c in sorted(depth_counts.items())]
            obs = (
                f"Indent stats for {fname} (threshold={self.nesting_threshold}): "
                + "; ".join(parts)
                + f". Deep lines: {deep_count} (pushed to artifacts)."
            )
        elif cmd == "SUM_ARTIFACTS":
            s = sum(self.numeric_artifacts) if self.numeric_artifacts else 0
            self.numeric_artifacts.append(s)
            obs = f"SUM_ARTIFACTS: sum={s} (pushed to artifacts)."
        elif cmd == "CLEAR_ARTIFACTS":
            self.numeric_artifacts = []
            obs = "Artifacts cleared."
        elif cmd == "WEIGHTS":
            w = self.weights
            obs = f"Weights: TODO={w['TODO']}, FIXME={w['FIXME']}, RISKY={w['RISKY']}, DEEP={w['DEEP']}."
        elif cmd == "STATUS":
            obs = (
                f"Status: turn {self.turn_count}/{self.max_turns}, artifacts={len(self.numeric_artifacts)}, "
                f"files={len(self.files)}."
            )
        elif cmd == "SUBMIT":
            try:
                val = int(parsed.get("value"))
            except Exception:
                obs = "Submission error: provide an integer with SUBMIT <int>."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            if val == self.target_bri:
                obs = f"Success! Submitted correct Bug Risk Index {val}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Incorrect final answer {val}. Expected a different value."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
        else:
            obs = f"Unsupported action '{cmd}'."
            return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns}). Episode timed out."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action:
            return None
        pattern = re.compile(r"\\boxed\{(.+?)\}", re.IGNORECASE | re.DOTALL)
        m = pattern.search(action)
        if not m:
            return None
        inner = m.group(1).strip()

        tokens = inner.split()
        if not tokens:
            return None

        cmd = tokens[0].upper()
        parsed: Dict[str, Any] = {"cmd": cmd}

        if cmd == "LIST":
            return parsed
        if cmd == "SHOW" and len(tokens) >= 2:
            parsed["filename"] = tokens[1]
            return parsed
        if cmd == "GREP":
            # GREP <pattern> IN <filename> or GREP <pattern> <filename>
            if len(tokens) >= 4 and tokens[2].upper() == "IN":
                parsed["pattern"] = tokens[1]
                parsed["filename"] = tokens[3]
                return parsed
            elif len(tokens) >= 3:
                parsed["pattern"] = tokens[1]
                parsed["filename"] = tokens[2]
                return parsed
            return None
        if cmd == "COUNT":
            # COUNT <pattern> IN REPO or COUNT <pattern> IN <filename>
            if len(tokens) >= 4 and tokens[2].upper() == "IN":
                parsed["pattern"] = tokens[1]
                scope_tok = tokens[3]
                if scope_tok.upper() == "REPO":
                    parsed["scope"] = "REPO"
                    return parsed
                else:
                    parsed["scope"] = "FILE"
                    parsed["filename"] = scope_tok
                    return parsed
            return None
        if cmd == "INDENT_STATS" and len(tokens) >= 2:
            parsed["filename"] = tokens[1]
            return parsed
        if cmd == "SUM_ARTIFACTS":
            return parsed
        if cmd == "CLEAR_ARTIFACTS":
            return parsed
        if cmd == "WEIGHTS":
            return parsed
        if cmd == "STATUS":
            return parsed
        if cmd == "SUBMIT" and len(tokens) >= 2:
            parsed["value"] = tokens[1]
            return parsed

        parsed["cmd"] = cmd
        return parsed

    def sample_random_action(self) -> str:
        choices = [
            r"\boxed{LIST}",
            r"\boxed{WEIGHTS}",
            r"\boxed{COUNT TODO IN REPO}",
            r"\boxed{INDENT_STATS file_1.py}",
            r"\boxed{SHOW file_1.py}",
            r"\boxed{STATUS}",
            r"\boxed{SUM_ARTIFACTS}",
            r"\boxed{SUBMIT 42}",
        ]
        return random.choice(choices)

    def _generate_repo(self):
        self.files = {}
        filenames = [f"file_{i+1}.py" for i in range(self.num_files)]
        # Base probabilities
        p_todo = 0.08
        p_fixme = 0.05
        p_risky = 0.04
        p_decoy = 0.02 * self.mutation_level
        templates = [
            "def func_{k}():",
            "if cond_{k}:",
            "else:",
            "for i in range(n):",
            "print('msg_{k}')",
            "x_{k} = x_{k} + 1",
            "return x_{k}",
        ]
        for fname in filenames:
            n_lines = random.randint(max(5, self.max_lines_per_file // 2), self.max_lines_per_file)
            lines = []
            for i in range(n_lines):
                k = random.randint(1, 999)
                base = random.choice(templates).format(k=k)
                depth = random.randint(0, self.indent_richness)
                indent = " " * (4 * depth)
                line = indent + base
                r = random.random()
                if r < p_todo:
                    line += "  # TODO: handle edge case"
                elif r < p_todo + p_fixme:
                    line += "  # FIXME: bug when x==0"
                elif r < p_todo + p_fixme + p_risky:
                    line += f"  {random.choice(['eval', 'exec'])}('2+2')"
                if random.random() < p_decoy:
                    line += "  # NOTE: TODOX or FIX-ME seen"
                    if random.random() < 0.5:
                        line += "  evaluation()"
                    else:
                        line += "  executor.start()"
                lines.append(line)
            self.files[fname] = "\n".join(lines)

        # Ensure solvability: inject at least one real token if all are zero
        if self._repo_real_token_total() == 0:
            fname = filenames[0]
            self.files[fname] += "\n# TODO: ensure at least one TODO"

    def _repo_real_token_total(self) -> int:
        total = 0
        for content in self.files.values():
            total += len(re.findall(r"\bTODO\b", content))
            total += len(re.findall(r"\bFIXME\b", content))
            total += len(re.findall(r"\beval\b", content))
            total += len(re.findall(r"\bexec\b", content))
        return total

    def _derive_weights(self):
        self.weights = {}
        amp = self.weight_variability
        for k, base in self.base_weights.items():
            offset = random.randint(-amp, amp) if amp > 0 else 0
            w = max(1, base + offset)
            self.weights[k] = w

    def _compute_target_bri(self):
        todo_total = 0
        fixme_total = 0
        risky_total = 0
        deep_total = 0
        for content in self.files.values():
            todo_total += len(re.findall(r"\bTODO\b", content))
            fixme_total += len(re.findall(r"\bFIXME\b", content))
            risky_total += len(re.findall(r"\beval\b|\bexec\b", content))
            lines = content.splitlines()
            for line in lines:
                m = re.match(r"^( +)", line)
                spaces = len(m.group(1)) if m else 0
                depth = spaces // 4
                if depth >= self.nesting_threshold:
                    deep_total += 1
        w = self.weights
        self.target_bri = (
            w["TODO"] * todo_total
            + w["FIXME"] * fixme_total
            + w["RISKY"] * risky_total
            + w["DEEP"] * deep_total
        )


class RepoRiskIndexEnvWithFeedback(RepoRiskIndexEnv):
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
            hint = "Wrap actions in \\boxed{...} exactly."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["cmd"] = (self.last_action_struct or {}).get("cmd")
            hint = "Use LIST, SHOW, GREP, COUNT, INDENT_STATS, SUM_ARTIFACTS, CLEAR_ARTIFACTS, WEIGHTS, STATUS, or SUBMIT."
        elif "protocol error" in text and "unknown file" in text:
            error_type = "ProtocolViolation"
            missing = None
            if self.last_action_struct:
                missing = self.last_action_struct.get("filename")
            error_detail["violation"] = "unknown_file"
            error_detail["filename"] = missing
            hint = "Use LIST to see valid filenames, then reference one of them."
        elif "submission error" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "bad_submit_format"
            hint = "Submit an integer: \\boxed{SUBMIT <int>}."
        elif "incorrect final answer" in text:
            error_type = "WrongDecision"
            submitted = None
            if self.last_action_struct and self.last_action_struct.get("cmd") == "SUBMIT":
                submitted = self.last_action_struct.get("value")
            error_detail["expected"] = self.target_bri
            error_detail["got"] = submitted
            hint = "Query WEIGHTS and use COUNT for 'TODO','FIXME','eval','exec'. Use INDENT_STATS to compute deep lines."
        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = "Well done."

        if "timed out" in text or "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Plan early: LIST files, WEIGHTS, then COUNT and INDENT_STATS to compute totals efficiently."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "num_files": len(self.files),
                "weights": self.weights,
                "artifacts_count": len(self.numeric_artifacts),
                "nesting_threshold": self.nesting_threshold,
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
            "hint": "Start with LIST and WEIGHTS. Then COUNT tokens across REPO and compute INDENT_STATS per file.",
            "turn": 0,
            "state": {
                "num_files": len(self.files),
                "weights": self.weights,
                "artifacts_count": len(self.numeric_artifacts),
                "nesting_threshold": self.nesting_threshold,
            },
        }
        return obs, info