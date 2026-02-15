from typing import Any, Dict, Optional, Tuple, List
import random
import re
import json
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class SubgridBeautyEnvGEM(Env):
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

        # 难度参数范围（根据原环境分析）
        # rows/cols: 网格尺寸
        # num_queries: 查询数量
        # value_max: 网格数值的最大范围（min 固定为 0）
        self.complexity_params = {
            "rows": (2, 30),
            "cols": (2, 30),
            "num_queries": (1, 50),
            "value_max": (9, 9999),
        }

        # 参数方差（enable_param_randomization=True 时生效）
        self.param_variance = {
            "rows": 1,
            "cols": 1,
            "num_queries": 2,
            "value_max": 50,
        }

        # 占位属性（由 _apply_complexity_params 填充）
        self.rows: int = 0
        self.cols: int = 0
        self.num_queries: int = 0
        self.value_max: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 问题实例
        self.n: int = 0
        self.m: int = 0
        self.q: int = 0
        self.grid: List[List[int]] = []
        self.queries: List[Tuple[int, int, int, int]] = []
        self.current_query_index: int = 0
        self.results: List[int] = []

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
            "Subgrid Beauty (GEM): Given a grid and a set of rectangle queries, compute beauty = max - min within each subgrid.\n"
            "You can interact with the environment using actions inside \\boxed{...}.\n"
            "Available actions:\n"
            "- \\boxed{observe}                    # 查看当前环境信息\n"
            "- \\boxed{dims}                       # 获取网格尺寸 n,m\n"
            "- \\boxed{count}                      # 获取查询总数 q\n"
            "- \\boxed{get X Y}                    # 获取网格 (X,Y) 的值 (1-based)\n"
            "- \\boxed{query}                      # 获取当前查询坐标 {x1,y1,x2,y2}\n"
            "- \\boxed{calc_max v1 v2 ...}         # 计算最大值\n"
            "- \\boxed{calc_min v1 v2 ...}         # 计算最小值\n"
            "- \\boxed{calc_beauty MAX MIN}        # 计算美丽值\n"
            "- \\boxed{submit B}                   # 提交当前查询的美丽值 B\n"
            "- \\boxed{done}                       # 完成所有查询并验证结果（终止）\n"
            "\n"
            "Note: Provide integers where required. Invalid actions terminate the episode with penalty.\n"
        )

    def get_task_suffix(self) -> str:
        return (
            f"Grid: {self.n}x{self.m} | Queries: {self.q} | Index: {self.current_query_index} | "
            f"Turn: {self.turn_count}/{self.max_turns}\nEnter action."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        # 初始化状态
        self.n = self.problem["n"]
        self.m = self.problem["m"]
        self.q = self.problem["q"]
        self.grid = self.problem["grid"]
        self.queries = self.problem["queries"]
        self.current_query_index = 0
        self.results = []
        self.turn_count = 0

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        n = self.rows
        m = self.cols
        q = self.num_queries
        vmax = max(1, self.value_max)

        grid = [[random.randint(0, vmax) for _ in range(m)] for _ in range(n)]

        queries: List[Tuple[int, int, int, int]] = []
        for _ in range(q):
            x1 = random.randint(1, n)
            x2 = random.randint(1, n)
            y1 = random.randint(1, m)
            y2 = random.randint(1, m)
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            queries.append((x1, y1, x2, y2))

        return {"n": n, "m": m, "q": q, "grid": grid, "queries": queries}

    # ========== 原环境辅助方法（转换后保留） ==========
    def Observe(self) -> str:
        return f"Grid size: {self.n}x{self.m}, total queries: {self.q}, current query index: {self.current_query_index}"

    def GetGridDimensions(self) -> str:
        return json.dumps({"n": self.n, "m": self.m})

    def GetQueryCount(self) -> str:
        return str(self.q)

    def GetGridValue(self, x: int, y: int) -> str:
        if 1 <= x <= self.n and 1 <= y <= self.m:
            return str(self.grid[x - 1][y - 1])
        else:
            return "Error: Coordinates out of grid range"

    def GetCurrentQuery(self) -> str:
        if self.current_query_index < self.q:
            x1, y1, x2, y2 = self.queries[self.current_query_index]
            return json.dumps({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
        else:
            return "{}"

    def CalculateMaxValue(self, values: List[int]) -> str:
        if not values:
            return "Error: empty values"
        return str(max(values))

    def CalculateMinValue(self, values: List[int]) -> str:
        if not values:
            return "Error: empty values"
        return str(min(values))

    def CalculateBeauty(self, max_val: int, min_val: int) -> str:
        return str(max_val - min_val)

    def SubmitQueryAnswer(self, beauty: int) -> str:
        if self.current_query_index < self.q:
            self.results.append(beauty)
            self.current_query_index += 1
            return f"Submitted answer: {beauty}, processed {self.current_query_index}/{self.q} queries"
        else:
            return "All queries have been processed"

    def get_ref_answer(self) -> List[int]:
        def subgrid_beauty(x1, y1, x2, y2):
            subgrid_values = [self.grid[i][j] for i in range(x1 - 1, x2) for j in range(y1 - 1, y2)]
            return max(subgrid_values) - min(subgrid_values)

        results: List[int] = []
        for x1, y1, x2, y2 in self.queries:
            results.append(subgrid_beauty(x1, y1, x2, y2))
        return results

    def Done(self) -> str:
        ref_answer = self.get_ref_answer()
        correct = self.results == ref_answer
        msg = f"Your answers: {self.results}, reference answers: {ref_answer}, result: {'correct' if correct else 'incorrect'}"
        return msg

    # ========== GEM 接口实现 ==========
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

        cmd = parsed["cmd"]
        args = parsed["args"]

        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            if cmd == "observe":
                obs = self.Observe()

            elif cmd == "dims":
                obs = self.GetGridDimensions()

            elif cmd in ("count", "queries"):
                obs = self.GetQueryCount()

            elif cmd == "get":
                if len(args) != 2:
                    obs = "Invalid parameters. Usage: \\boxed{get X Y}"
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                try:
                    x = int(args[0])
                    y = int(args[1])
                except ValueError:
                    obs = "Invalid parameters. X and Y must be integers."
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                obs = self.GetGridValue(x, y)
                if obs.startswith("Error:"):
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}

            elif cmd in ("query", "current"):
                obs = self.GetCurrentQuery()

            elif cmd == "calc_max":
                if len(args) < 1:
                    obs = "Invalid parameters. Usage: \\boxed{calc_max v1 v2 ...}"
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                try:
                    values = [int(a) for a in args]
                except ValueError:
                    obs = "Invalid parameters. Values must be integers."
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                obs = self.CalculateMaxValue(values)
                if obs.startswith("Error:"):
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}

            elif cmd == "calc_min":
                if len(args) < 1:
                    obs = "Invalid parameters. Usage: \\boxed{calc_min v1 v2 ...}"
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                try:
                    values = [int(a) for a in args]
                except ValueError:
                    obs = "Invalid parameters. Values must be integers."
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                obs = self.CalculateMinValue(values)
                if obs.startswith("Error:"):
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}

            elif cmd == "calc_beauty":
                if len(args) != 2:
                    obs = "Invalid parameters. Usage: \\boxed{calc_beauty MAX MIN}"
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                try:
                    max_val = int(args[0])
                    min_val = int(args[1])
                except ValueError:
                    obs = "Invalid parameters. MAX and MIN must be integers."
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                obs = self.CalculateBeauty(max_val, min_val)

            elif cmd == "submit":
                if len(args) != 1:
                    obs = "Invalid parameters. Usage: \\boxed{submit B}"
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                try:
                    beauty = int(args[0])
                except ValueError:
                    obs = "Invalid parameters. B must be an integer."
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                obs = self.SubmitQueryAnswer(beauty)

            elif cmd in ("done", "finish", "answer"):
                # 完成与验证
                obs = self.Done()
                ref_answer = self.get_ref_answer()
                correct = self.results == ref_answer
                reward = 1.0 if correct else -1.0
                terminated = True

            else:
                obs = f"Invalid action: {cmd}"
                return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}

        except Exception as e:
            obs = f"Runtime error: {str(e)}"
            return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}

        # 超时检查（统一放在 step 结尾）
        if not terminated and self.turn_count >= self.max_turns:
            obs = f"{obs}\nReached max turns ({self.max_turns})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action:
            return None
        pattern = re.compile(r"\\boxed\{(.+?)\}", re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        if not content:
            return None
        parts = content.split()
        cmd = parts[0].lower() if parts else ""
        args = parts[1:] if len(parts) > 1 else []
        return {"content": content, "cmd": cmd, "args": args}

    def sample_random_action(self) -> str:
        # 简单的示例动作
        return "\\boxed{observe}"