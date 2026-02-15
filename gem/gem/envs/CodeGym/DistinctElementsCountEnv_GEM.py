from typing import Any, Dict, Optional, Tuple
import random
import re
import json
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class DistinctElementsCountEnvGEM(Env):
    def __init__(
        self,
        complexity: int = 5,  # 难度等级 1-10，默认中等
        enable_param_randomization: bool = False,  # 评估时关闭随机化
        max_turns: Optional[int] = 100,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 100

        # 难度参数范围
        self.complexity_params = {
            # 数组长度
            "array_length": (5, 50),
            # 查询数量
            "num_queries": (2, 20),
            # 数值范围（值域）
            "value_range": (10, 10000),
            # 回合上限（难度推荐的预算，最终由 min(self.max_turns, self.turn_limit) 生效）
            "turn_limit": (20, 200),
        }

        # 参数方差（用于启用随机化时的微调）
        self.param_variance = {
            "array_length": 2,
            "num_queries": 1,
            "value_range": 50,
            "turn_limit": 10,
        }

        # 占位属性
        self.array_length: int = 0
        self.num_queries: int = 0
        self.value_range: int = 0
        self.turn_limit: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 任务数据
        self.problem: Dict[str, Any] = {}
        self.arr = []
        self.queries = []
        self.current_query_idx = 0
        self.results = []
        self.last_subarray = None  # 用于便利：若未提供 count 的参数，可使用最近一次提取

        self.reset()

    def _apply_complexity_params(self):
        """根据 complexity 等级计算参数值"""
        normalized = min(1.0, (self.complexity - 1) / 9.0)  # [0, 1]

        for param_name, (min_val, max_val) in self.complexity_params.items():
            center_value = min_val + (max_val - min_val) * normalized
            actual_value = center_value

            if self.enable_param_randomization:
                variance = self.param_variance.get(param_name, 0)
                if variance > 0:
                    actual_value = center_value + random.uniform(-variance, variance)
                    actual_value = max(min_val, min(max_val, actual_value))

            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        return (
            "Distinct Elements Count Task:\n"
            "Given an array and multiple queries [l, r] (1-based), count distinct elements in each subarray.\n"
            "Actions (wrap your command in \\boxed{...}):\n"
            "- observe                        -> Show basic environment info\n"
            "- query                          -> Get current query in format 'l,r' or 'None' if finished\n"
            "- extract L R                    -> Extract subarray with 1-based indices [L, R]\n"
            "- count [JSON_ARRAY]             -> Count distinct elements of the provided subarray\n"
            "- save N                         -> Save current result for the active query\n"
            "- next                           -> Move to the next query\n"
            "- answer [JSON_ARRAY_OF_RESULTS] -> Submit all results to finish\n"
            "Notes:\n"
            "- Indices are 1-based.\n"
            "- You may omit the array for 'count' if you have previously used 'extract'; it will use the last extracted subarray."
        )

    def get_task_suffix(self) -> str:
        remaining_queries = len(self.queries) - self.current_query_idx
        effective_limit = min(self.max_turns, self.turn_limit) if self.turn_limit > 0 else self.max_turns
        return (
            f"Array length: {len(self.arr)} | Current query index: {self.current_query_idx} | "
            f"Remaining queries: {remaining_queries}\n"
            f"Turn: {self.turn_count}/{effective_limit}\n"
            f"Enter action."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # 仅影响实例生成的随机性

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        # 重置状态
        self.turn_count = 0
        self.current_query_idx = 0
        self.results = []
        self.last_subarray = None

        # 绑定便捷引用
        self.arr = self.problem["arr"]
        self.queries = self.problem["queries"]

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        arr = [random.randint(1, self.value_range) for _ in range(self.array_length)]
        queries = []
        for _ in range(self.num_queries):
            l = random.randint(1, max(1, self.array_length))
            r = random.randint(l, max(l, self.array_length))
            queries.append((l, r))
        return {"arr": arr, "queries": queries}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)

        if not parsed:
            obs = f"Format error at turn {self.turn_count}."
            return (
                obs,
                LanguageGameReward.format_error_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        cmd = parsed.get("cmd")
        args = parsed.get("args", [])

        terminated = False
        truncated = False
        reward = 0.0
        obs = ""

        try:
            if cmd == "observe":
                obs = self.Observe()

            elif cmd == "query":
                obs = self.GetCurrentQuery()

            elif cmd == "extract":
                if len(args) < 2:
                    obs = "Error: 'extract' requires two integer indices L and R."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                l, r = int(args[0]), int(args[1])
                if l < 1 or r < 1 or l > r or r > len(self.arr):
                    obs = "Error: indices out of range."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                subarray_str = self.ExtractSubarray(l, r)
                obs = subarray_str
                try:
                    self.last_subarray = json.loads(subarray_str)
                except Exception:
                    self.last_subarray = None

            elif cmd == "count":
                subarray = None
                if len(args) >= 1:
                    # Remaining content as JSON
                    try:
                        # Reconstruct the JSON argument (might contain spaces)
                        json_str = " ".join(args)
                        subarray = json.loads(json_str)
                    except Exception:
                        obs = "Error: invalid JSON for subarray."
                        return (
                            obs,
                            LanguageGameReward.invalid_action_reward,
                            True,
                            False,
                            {"suffix": self.get_task_suffix()},
                        )
                else:
                    # Attempt to use last_subarray
                    if self.last_subarray is None:
                        obs = "Error: no subarray provided and no previous extract."
                        return (
                            obs,
                            LanguageGameReward.invalid_action_reward,
                            True,
                            False,
                            {"suffix": self.get_task_suffix()},
                        )
                    subarray = self.last_subarray
                obs = self.CountDistinct(subarray)

            elif cmd == "save":
                if len(args) < 1:
                    obs = "Error: 'save' requires a result integer."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                try:
                    result = int(args[0])
                except Exception:
                    obs = "Error: 'save' requires an integer."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                obs = self.SaveResult(result)

            elif cmd == "next":
                obs = self.NextQuery()

            elif cmd == "answer":
                if len(args) < 1:
                    obs = "Error: 'answer' requires a JSON array of results."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                # The answer payload may contain spaces; join then parse JSON
                payload = " ".join(args)
                try:
                    answers = json.loads(payload)
                    if not isinstance(answers, list):
                        raise ValueError("answers must be a JSON list")
                except Exception as e:
                    obs = f"Error: invalid JSON for answers: {e}"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                obs = self.Done(answers)
                # 判定奖励
                ref_answer = self.get_ref_answer()
                if answers == ref_answer:
                    reward = 1.0
                else:
                    reward = -1.0
                terminated = True

            else:
                obs = f"Invalid action: {cmd}"
                return (
                    obs,
                    LanguageGameReward.invalid_action_reward,
                    True,
                    False,
                    {"suffix": self.get_task_suffix()},
                )

        except Exception as e:
            obs = f"Error: {str(e)}"
            return (
                obs,
                LanguageGameReward.invalid_action_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        # 超时检查（统一放在 step 结尾）
        effective_limit = min(self.max_turns, self.turn_limit) if self.turn_limit > 0 else self.max_turns
        if not terminated and self.turn_count >= effective_limit:
            obs = f"{obs}\nReached max turns ({effective_limit})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict]:
        if not action:
            return None
        pattern = re.compile(r"\\boxed\{(.+?)\}", re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()

        # Tokenize: first token is command, remaining are args (support commas or spaces for extract)
        # For 'count' and 'answer', args may be JSON that contains spaces; we keep raw split and let step rejoin as needed.
        if not content:
            return None
        # Normalize commas for extract convenience: "extract 1,3" -> ["extract","1","3"]
        if content.lower().startswith("extract"):
            # Replace comma with space to split L and R
            content = content.replace(",", " ")
        tokens = content.strip().split()
        if not tokens:
            return None
        cmd = tokens[0].lower()
        args = tokens[1:]

        # For commands where the rest should be preserved as a single payload (e.g., count, answer),
        # we will handle in step by joining args with spaces again.
        return {"content": content, "cmd": cmd, "args": args}

    def sample_random_action(self) -> str:
        # Provide a simple valid action
        choices = [
            "\\boxed{observe}",
            "\\boxed{query}",
            "\\boxed{next}",
        ]
        return random.choice(choices)

    # -----------------------------
    # 辅助方法（从原环境保留并适配）
    # -----------------------------
    def Observe(self):
        remaining_queries = len(self.queries) - self.current_query_idx
        return f"Array length: {len(self.arr)}, Current query index: {self.current_query_idx}, Remaining queries count: {remaining_queries}"

    def GetCurrentQuery(self):
        if self.current_query_idx < len(self.queries):
            l, r = self.queries[self.current_query_idx]
            return f"{l},{r}"
        return "None"

    def ExtractSubarray(self, l: int, r: int):
        subarray = self.arr[l - 1 : r]
        return json.dumps(subarray)

    def CountDistinct(self, subarray: list):
        distinct_count = len(set(subarray))
        return str(distinct_count)

    def SaveResult(self, result: int):
        self.results.append(result)
        return f"Result saved: {result}, Current results count: {len(self.results)}"

    def NextQuery(self):
        self.current_query_idx += 1
        # Clamp to max length
        if self.current_query_idx > len(self.queries):
            self.current_query_idx = len(self.queries)
        return str(self.current_query_idx)

    def Done(self, answers: list):
        ref_answer = self.get_ref_answer()
        correct = answers == ref_answer
        msg = f"Your answer: {answers}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg

    def get_ref_answer(self):
        result = []
        for l, r in self.queries:
            subarray = self.arr[l - 1 : r]
            result.append(len(set(subarray)))
        return result

    def solve(self) -> str:
        # 一个简单的自动流程，调用内部方法，不通过 step
        results = []
        for (l, r) in self.queries:
            subarray = self.arr[l - 1 : r]
            distinct_count = len(set(subarray))
            results.append(distinct_count)
        return self.Done(results)