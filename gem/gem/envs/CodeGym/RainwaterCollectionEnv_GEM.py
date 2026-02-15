from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class RainwaterCollectionEnvGEM(Env):
    def __init__(
        self,
        complexity: int = 5,  # 难度等级 1-10，默认中等
        enable_param_randomization: bool = False,  # 评估时关闭随机化
        max_turns: Optional[int] = 100,
        **_,  # 忽略其他参数
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 100

        # 定义难度参数范围（根据原环境分析：数组长度与数值范围等）
        self.complexity_params = {
            "array_length": (5, 50),  # 建筑数量
            "value_range": (10, 1000),  # 高度数值上限
            "max_delta": (0, 20),  # 相邻建筑高度差的最大值（确保非递减）
        }

        # 参数方差（微调随机性）
        self.param_variance = {
            "array_length": 2,
            "value_range": 100,
            "max_delta": 3,
        }

        # 占位属性
        self.array_length: int = 0
        self.value_range: int = 0
        self.max_delta: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 问题数据
        self.problem: Dict[str, Any] = {}
        self.heights: list[int] = []
        self.water_buffer: list[int] = []

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
            "Rainwater Collection: Given a non-decreasing sequence of building heights, "
            "compute the maximum rainwater that can be collected between consecutive buildings "
            "(defined here as the height difference between building i+1 and i).\n"
            "Available actions (wrap in \\boxed{...}):\n"
            "- Get total buildings: \\boxed{count}\n"
            "- Get height at index i: \\boxed{height i}\n"
            "- Calculate water at index i (heights[i+1] - heights[i], 0 if i >= n-1): \\boxed{calc i}\n"
            "- Find max from a list: \\boxed{max [a, b, c, ...]}\n"
            "- Observe instructions: \\boxed{observe}\n"
            "- Submit final answer N: \\boxed{answer N}\n"
            "Notes:\n"
            "- Index i starts at 0.\n"
            "- Heights are non-decreasing.\n"
        )

    def get_task_suffix(self) -> str:
        return (
            f"Buildings: {len(self.heights)}\n"
            f"Turn: {self.turn_count}/{self.max_turns}\n"
            f"Buffer size: {len(self.water_buffer)}\n"
            "Enter action."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()
        self.heights = self.problem["heights"]

        self.turn_count = 0
        self.water_buffer = []
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        heights = []
        # 初始高度
        start = random.randint(0, max(0, self.value_range // 10))
        heights.append(start)
        for _ in range(self.array_length - 1):
            delta = random.randint(0, self.max_delta)
            next_h = heights[-1] + delta
            next_h = min(next_h, self.value_range)
            heights.append(next_h)
        return {"heights": heights, "size": len(heights)}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)

        if not parsed:
            obs = f"Format error at turn {self.turn_count}. Use \\boxed{{...}} with valid commands."
            return (
                obs,
                LanguageGameReward.format_error_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        terminated = False
        truncated = False
        reward = 0.0
        obs = "Action processed."

        try:
            if parsed["type"] == "count":
                obs = self.GetBuildingCount()
                # Non-terminal informative step

            elif parsed["type"] == "height":
                index = parsed["index"]
                if index < 0 or index >= len(self.heights):
                    obs = "Error: Invalid index"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    obs = self.GetBuildingHeight(index)

            elif parsed["type"] == "calc":
                index = parsed["index"]
                # CalculateWater tolerates out-of-range and returns "0"
                obs = self.CalculateWater(index)
                # Maintain buffer for convenience (optional usage)
                try:
                    self.water_buffer.append(int(obs))
                except Exception:
                    # If obs is not a valid int, ignore buffer append
                    pass

            elif parsed["type"] == "max":
                if parsed.get("use_buffer", False):
                    data = self.water_buffer[:]
                else:
                    data = parsed.get("list", [])
                # Validate list content
                if not isinstance(data, list):
                    obs = "Error: invalid list for max"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    obs = self.FindMaxWater(data)

            elif parsed["type"] == "observe":
                obs = self.Observe()

            elif parsed["type"] == "answer":
                answer_val = parsed["answer"]
                msg = self.Done(answer_val)
                # Check correctness by comparing to ref
                ref = self.get_ref_answer()
                if answer_val == ref:
                    reward = 1.0
                else:
                    reward = -1.0
                obs = msg
                terminated = True

            else:
                obs = f"Invalid action type: {parsed['type']}"
                reward = LanguageGameReward.invalid_action_reward
                terminated = True

        except Exception as e:
            obs = f"Error: {str(e)}"
            reward = LanguageGameReward.invalid_action_reward
            terminated = True

        # 超时检查（统一放在 step 结尾）
        if not terminated and self.turn_count >= self.max_turns:
            obs = f"{obs}\nReached max turns ({self.max_turns})."
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

        # Parse commands
        if content.lower() == "count":
            return {"type": "count"}

        if content.lower() == "observe":
            return {"type": "observe"}

        m = re.match(r"^height\s+(-?\d+)$", content, flags=re.IGNORECASE)
        if m:
            idx = int(m.group(1))
            return {"type": "height", "index": idx}

        m = re.match(r"^calc\s+(-?\d+)$", content, flags=re.IGNORECASE)
        if m:
            idx = int(m.group(1))
            return {"type": "calc", "index": idx}

        # max [a, b, c] or max buffer
        m = re.match(r"^max\s*\[(.*?)\]$", content, flags=re.IGNORECASE)
        if m:
            inner = m.group(1).strip()
            if inner == "":
                lst = []
            else:
                # split by comma or whitespace
                parts = re.split(r"[,\s]+", inner)
                lst = []
                for p in parts:
                    if p == "":
                        continue
                    try:
                        lst.append(int(p))
                    except Exception:
                        return None
            return {"type": "max", "list": lst}

        m = re.match(r"^max\s+buffer$", content, flags=re.IGNORECASE)
        if m:
            return {"type": "max", "use_buffer": True}

        m = re.match(r"^answer\s+(-?\d+)$", content, flags=re.IGNORECASE)
        if m:
            ans = int(m.group(1))
            return {"type": "answer", "answer": ans}

        return None

    def sample_random_action(self) -> str:
        # Random step among available actions
        if not self.heights:
            return "\\boxed{observe}"
        choice = random.choice(["count", "calc", "observe"])
        if choice == "count":
            return "\\boxed{count}"
        elif choice == "calc":
            idx = random.randint(0, max(0, len(self.heights) - 1))
            return f"\\boxed{{calc {idx}}}"
        else:
            return "\\boxed{observe}"

    # ----------------------------
    # 保留原环境的辅助方法并转换
    # ----------------------------

    def GetBuildingCount(self) -> str:
        """
        Get the total number of buildings.
        Returns: str
        """
        return str(len(self.heights))

    def GetBuildingHeight(self, index: int) -> str:
        """
        Get height of building at index.
        Returns: str or error message.
        """
        if 0 <= index < len(self.heights):
            return str(self.heights[index])
        return "Error: Invalid index"

    def CalculateWater(self, index: int) -> str:
        """
        Calculate water collected at index = heights[i+1] - heights[i].
        If index out-of-range (i >= n-1), returns "0".
        """
        if index < 0 or index >= len(self.heights) - 1:
            return "0"
        return str(self.heights[index + 1] - self.heights[index])

    def FindMaxWater(self, water_amounts: list) -> str:
        """
        Find the maximum value from the list of amounts.
        Returns: str
        """
        if not water_amounts:
            return "0"
        try:
            return str(max(int(x) for x in water_amounts))
        except Exception:
            return "0"

    def Observe(self) -> str:
        """
        Return observation/instruction information.
        """
        return "Please use relevant actions to obtain building information and calculate the maximum rainwater volume"

    def Done(self, answer: int) -> str:
        """
        Verify whether the final answer is correct and return result information.
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg

    def get_ref_answer(self) -> int:
        """
        Use the information in the environment to get the reference answer.
        Computes max(heights[i+1] - heights[i]) for i in [0..n-2], 0 if n < 2.
        """
        n = len(self.heights)
        if n < 2:
            return 0
        max_water = 0
        for i in range(n - 1):
            water_collected = self.heights[i + 1] - self.heights[i]
            if water_collected > max_water:
                max_water = water_collected
        return max_water

    def solve(self) -> str:
        """
        Automatically call actions to compute and submit the answer.
        Returns: str result message.
        """
        # Get count
        obs, _, _, _, _ = self.step("\\boxed{count}")
        try:
            building_count = int(obs)
        except Exception:
            building_count = len(self.heights)

        water_amounts = []
        for index in range(building_count):
            obs, _, terminated, _, _ = self.step(f"\\boxed{{calc {index}}}")
            if terminated:
                # If terminated unexpectedly, stop
                break
            try:
                water_amounts.append(int(obs))
            except Exception:
                water_amounts.append(0)

        # Find max
        max_list_str = ", ".join(str(x) for x in water_amounts)
        obs, _, terminated, _, _ = self.step(f"\\boxed{{max [{max_list_str}]}}")
        if terminated:
            return obs
        try:
            max_water = int(obs)
        except Exception:
            max_water = self.get_ref_answer()

        # Submit answer
        final_obs, _, _, _, _ = self.step(f"\\boxed{{answer {max_water}}}")
        return final_obs


# Optional simple test (will only run if module executed directly)
if __name__ == "__main__":
    env = RainwaterCollectionEnvGEM(complexity=5, enable_param_randomization=True, max_turns=50)
    instructions, info = env.reset(seed=42)
    print(instructions)
    print(info["suffix"])
    print(env.solve())