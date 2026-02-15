from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class HistogramMaxAreaEnvGEM(Env):
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
        # 主要影响问题规模和数值范围
        self.complexity_params = {
            "array_length": (5, 50),     # 柱子数量
            "value_range": (2, 100),     # 柱子高度范围 [1, value_range]
            "max_turns_param": (30, 200) # 建议的最大步数（不覆盖 self.max_turns，仅用于显示）
        }

        # 参数方差（仅在 enable_param_randomization=True 时生效）
        self.param_variance = {
            "array_length": 2,
            "value_range": 5,
            "max_turns_param": 10,
        }

        # 占位属性（将由 _apply_complexity_params 填充）
        self.array_length: int = 0
        self.value_range: int = 0
        self.max_turns_param: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 计算状态
        self.heights: list[int] = []
        self.original_heights: list[int] = []
        self.stack: list[int] = []
        self.max_area: int = 0

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
            "Histogram Max Area: Compute the largest rectangle area in a histogram.\n"
            "You may interact with helper actions to simulate the monotonic stack algorithm.\n"
            "Available actions (use the latest \\boxed{...} as action):\n"
            "- Initialize stack: \\boxed{init}\n"
            "- Append trailing zero to heights: \\boxed{append_zero}\n"
            "- Get height at index i: \\boxed{get_height i}\n"
            "- Check stack not empty: \\boxed{stack_not_empty}\n"
            "- Get stack top index: \\boxed{get_stack_top}\n"
            "- Pop from stack (returns height at popped index): \\boxed{pop}\n"
            "- Calculate width with current index i: \\boxed{calc_width i}\n"
            "- Calculate area: \\boxed{calc_area h w}\n"
            "- Update max area: \\boxed{update_max a}\n"
            "- Push index i to stack: \\boxed{push i}\n"
            "- Get current max: \\boxed{get_max}\n"
            "- Observe state: \\boxed{observe}\n"
            "- Submit final answer N: \\boxed{answer N}\n"
        )

    def get_task_suffix(self) -> str:
        bars = len(self.heights) if not self.heights or (self.heights and (len(self.heights) == 0 or self.heights[-1] != 0)) else len(self.heights) - 1
        return (
            f"Bars: {bars} | Turn: {self.turn_count}/{self.max_turns} "
            f"(recommended max turns {self.max_turns_param}) | Current max area: {self.max_area} | Stack size: {len(self.stack)}\n"
            "Enter action."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        # 初始化状态
        self.turn_count = 0
        self.stack = []
        self.max_area = 0

        # 初始化高度数组
        self.heights = self.problem["heights"][:]  # 工作数组（用户可 append_zero）
        self.original_heights = self.problem["heights"][:]  # 用于参考答案

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        heights = [random.randint(1, self.value_range) for _ in range(self.array_length)]
        return {"heights": heights, "size": self.array_length}

    # -----------------------------
    # 辅助方法（从原环境保留并调整）
    # -----------------------------

    @property
    def finished(self) -> bool:
        # GEM 环境通过 step 返回 terminated 控制结束，此属性仅为兼容保留
        return False

    @property
    def reward(self):
        # GEM 环境奖励在 step 返回，此属性仅为兼容保留
        return 0.0

    def get_ref_answer(self):
        """
        使用环境中的数据计算参考答案（最大矩形面积）。
        """
        stack = []
        max_area = 0
        heights = self.original_heights.copy()
        heights.append(0)

        for i in range(len(heights)):
            while stack and heights[i] < heights[stack[-1]]:
                h = heights[stack.pop()]
                w = i if not stack else i - stack[-1] - 1
                max_area = max(max_area, h * w)
            stack.append(i)

        return max_area

    # 行为动作实现（与原环境一致的语义）
    def InitializeStack(self) -> str:
        self.stack = []
        return "Stack has been initialized"

    def AppendZero(self) -> str:
        self.heights.append(0)
        return "Added 0 to the end of the heights array"

    def GetHeight(self, index: int) -> str:
        if index < 0 or index >= len(self.heights):
            return "Error: Index out of range"
        return str(self.heights[index])

    def StackIsNotEmpty(self) -> str:
        return str(len(self.stack) > 0)

    def GetStackTop(self) -> str:
        if not self.stack:
            return "Error: Stack is empty"
        return str(self.stack[-1])

    def PopFromStack(self) -> str:
        if not self.stack:
            return "Error: Stack is empty"
        index = self.stack.pop()
        return str(self.heights[index])

    def CalculateWidth(self, current_index: int) -> str:
        if not self.stack:
            width = current_index
        else:
            width = current_index - self.stack[-1] - 1
        return str(width)

    def CalculateArea(self, height: int, width: int) -> str:
        return str(height * width)

    def UpdateMaxArea(self, area: int) -> str:
        if area > self.max_area:
            self.max_area = area
        return str(self.max_area)

    def PushToStack(self, index: int) -> str:
        self.stack.append(index)
        return f"Pushed index {index} to the stack, top element of the stack is {index}"

    def GetCurrentMax(self) -> str:
        return str(self.max_area)

    def Observe(self) -> str:
        bar_count = len(self.heights) if not self.heights or self.heights[-1] != 0 else len(self.heights) - 1
        return f"The histogram contains {bar_count} bars, current maximum area is {self.max_area}"

    def Done(self, answer: int) -> str:
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg

    def solve(self) -> str:
        """
        自动模拟动作流程并提交答案（使用 \boxed{} 动作）。
        返回最终验证信息字符串。
        """
        # 初始化
        obs, _, _, _, _ = self.step("\\boxed{init}")
        obs, _, _, _, _ = self.step("\\boxed{append_zero}")
        obs, _, _, _, _ = self.step("\\boxed{observe}")

        # 从 observe 文本解析柱子数量
        match = re.search(r'The histogram contains (\d+) bars', obs)
        if not match:
            n = len(self.original_heights)
        else:
            n = int(match.group(1))

        current_index = 0
        while True:
            if current_index > n:
                break
            obs, _, term, trunc, _ = self.step(f"\\boxed{{get_height {current_index}}}")
            if term and trunc:
                break
            try:
                current_height = int(obs)
            except:
                # 若越界或其他错误，终止
                break

            obs, _, term, trunc, _ = self.step("\\boxed{stack_not_empty}")
            if term and trunc:
                break
            stack_not_empty = (obs == "True")

            while stack_not_empty:
                obs, _, term, trunc, _ = self.step("\\boxed{get_stack_top}")
                if term and trunc:
                    break
                try:
                    top_index = int(obs)
                except:
                    break
                obs, _, term, trunc, _ = self.step(f"\\boxed{{get_height {top_index}}}")
                if term and trunc:
                    break
                try:
                    top_height = int(obs)
                except:
                    break

                if current_height < top_height:
                    obs, _, term, trunc, _ = self.step("\\boxed{pop}")
                    if term and trunc:
                        break
                    try:
                        popped_height = int(obs)
                    except:
                        break
                    obs, _, term, trunc, _ = self.step(f"\\boxed{{calc_width {current_index}}}")
                    if term and trunc:
                        break
                    try:
                        width = int(obs)
                    except:
                        break
                    obs, _, term, trunc, _ = self.step(f"\\boxed{{calc_area {popped_height} {width}}}")
                    if term and trunc:
                        break
                    try:
                        area = int(obs)
                    except:
                        break
                    obs, _, term, trunc, _ = self.step(f"\\boxed{{update_max {area}}}")
                    if term and trunc:
                        break
                    obs, _, term, trunc, _ = self.step("\\boxed{stack_not_empty}")
                    if term and trunc:
                        break
                    stack_not_empty = (obs == "True")
                else:
                    break

            obs, _, term, trunc, _ = self.step(f"\\boxed{{push {current_index}}}")
            if term and trunc:
                break
            current_index += 1

        obs, _, term, trunc, _ = self.step("\\boxed{get_max}")
        if term and trunc:
            return obs
        try:
            max_area = int(obs)
        except:
            max_area = self.get_ref_answer()

        obs, reward, terminated, truncated, _ = self.step(f"\\boxed{{answer {max_area}}}")
        return obs

    # -----------------------------
    # GEM 必需方法
    # -----------------------------
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

        cmd = parsed.get("cmd", "")
        args = parsed.get("args", [])

        obs = "Action processed."
        reward = 0.0
        terminated = False
        truncated = False

        try:
            if cmd == "init":
                obs = self.InitializeStack()
            elif cmd == "append_zero":
                obs = self.AppendZero()
            elif cmd == "get_height":
                if len(args) != 1 or not args[0].lstrip("-").isdigit():
                    obs = "Invalid parameters for get_height."
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                index = int(args[0])
                obs = self.GetHeight(index)
            elif cmd == "stack_not_empty":
                obs = self.StackIsNotEmpty()
            elif cmd == "get_stack_top":
                obs = self.GetStackTop()
            elif cmd == "pop":
                obs = self.PopFromStack()
            elif cmd == "calc_width":
                if len(args) != 1 or not args[0].lstrip("-").isdigit():
                    obs = "Invalid parameters for calc_width."
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                current_index = int(args[0])
                obs = self.CalculateWidth(current_index)
            elif cmd == "calc_area":
                if len(args) != 2 or (not args[0].lstrip("-").isdigit()) or (not args[1].lstrip("-").isdigit()):
                    obs = "Invalid parameters for calc_area."
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                height = int(args[0])
                width = int(args[1])
                obs = self.CalculateArea(height, width)
            elif cmd == "update_max":
                if len(args) != 1 or not args[0].lstrip("-").isdigit():
                    obs = "Invalid parameters for update_max."
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                area = int(args[0])
                obs = self.UpdateMaxArea(area)
            elif cmd == "push":
                if len(args) != 1 or not args[0].lstrip("-").isdigit():
                    obs = "Invalid parameters for push."
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                index = int(args[0])
                obs = self.PushToStack(index)
            elif cmd == "get_max":
                obs = self.GetCurrentMax()
            elif cmd == "observe":
                obs = self.Observe()
            elif cmd == "answer":
                if len(args) != 1 or not args[0].lstrip("-").isdigit():
                    obs = "Invalid parameters for answer."
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                answer_val = int(args[0])
                done_msg = self.Done(answer_val)
                ref_answer = self.get_ref_answer()
                correct = (answer_val == ref_answer)
                obs = f"{done_msg}"
                reward = 1.0 if correct else -1.0
                terminated = True
            else:
                obs = f"Invalid action: {cmd}"
                return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
        except Exception as e:
            obs = f"Error: {str(e)}"
            return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}

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
        if not content:
            return None
        tokens = content.split()
        if not tokens:
            return None
        cmd = tokens[0].lower()
        args = tokens[1:]
        return {"content": content, "cmd": cmd, "args": args}

    def sample_random_action(self) -> str:
        # 随机示例动作（非终止）
        actions = [
            "\\boxed{observe}",
            "\\boxed{init}",
            "\\boxed{append_zero}",
            "\\boxed{stack_not_empty}",
            "\\boxed{get_max}",
        ]
        return random.choice(actions)