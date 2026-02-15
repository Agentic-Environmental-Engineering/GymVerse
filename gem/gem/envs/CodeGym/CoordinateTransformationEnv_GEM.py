from typing import Any, Dict, Optional, Tuple
import random
import re
import ast
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class CoordinateTransformationEnvGEM(Env):
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
            "num_commands": (3, 25),   # 指令数量（越多越难）
            "max_units": (2, 15),      # 每条指令的最大步数（越大越难）
            "noise_commands": (0, 6),  # 噪声指令对数（每对指令相互抵消，增加阅读难度）
        }

        # 参数方差（启用随机化时生效）
        self.param_variance = {
            "num_commands": 2,
            "max_units": 2,
            "noise_commands": 1,
        }

        # 占位属性（由 _apply_complexity_params 填充）
        self.num_commands: int = 0
        self.max_units: int = 0
        self.noise_commands: int = 0

        # 状态变量
        self.turn_count: int = 0
        self.current_x: int = 0
        self.current_y: int = 0
        self.problem: Dict[str, Any] = {"commands": []}
        self._last_done_correct: Optional[bool] = None

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
            "Coordinate Transformation: Compute final coordinates from movement commands.\n"
            "You start at (0, 0).\n"
            "Available actions:\n"
            "- Initialize position: \\boxed{init}\n"
            "- Move by direction and units: \\boxed{move D N}  (D in {U, D, L, R}, N is positive integer)\n"
            "- Get current position: \\boxed{pos}\n"
            "- Observe command list: \\boxed{observe}\n"
            "- Submit final answer: \\boxed{answer x y}\n"
        )

    def get_task_suffix(self) -> str:
        total_cmds = len(self.problem["commands"]) if self.problem and "commands" in self.problem else (self.num_commands + 2 * self.noise_commands)
        return f"Commands: {total_cmds}\nTurn: {self.turn_count}/{self.max_turns}\nEnter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响实例生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        # 初始化状态
        self.current_x = 0
        self.current_y = 0
        self.turn_count = 0
        self._last_done_correct = None

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        direction_choices = ["U", "D", "L", "R"]
        commands = []
        # 生成有效指令
        for _ in range(self.num_commands):
            d = random.choice(direction_choices)
            units = random.randint(1, self.max_units)
            commands.append((d, units))
        # 生成噪声成对指令（相互抵消）
        for _ in range(self.noise_commands):
            d = random.choice(direction_choices)
            units = random.randint(1, self.max_units)
            opposite = {"U": "D", "D": "U", "L": "R", "R": "L"}[d]
            commands.append((d, units))
            commands.append((opposite, units))
        # 打乱顺序，增加混淆
        random.shuffle(commands)
        return {"commands": commands}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)

        # 格式错误
        if not parsed:
            obs = f"Format error at turn {self.turn_count}."
            return (
                obs,
                LanguageGameReward.format_error_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        content = parsed["content"]
        verb = parsed["verb"]
        args = parsed.get("args", [])

        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            if verb == "init":
                obs = self.InitializePosition()

            elif verb == "move":
                if len(args) != 2:
                    obs = "Invalid move format. Use: \\boxed{move D N}"
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                direction = args[0].upper()
                try:
                    units = int(args[1])
                except Exception:
                    obs = "Invalid units. N must be an integer."
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                if units <= 0:
                    obs = "Invalid units. N must be a positive integer."
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                # 验证方向合法
                if direction not in {"U", "D", "L", "R"}:
                    obs = f"Error: Invalid direction '{direction}', must be 'U', 'D', 'L', or 'R'"
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                obs = self.ProcessCommand(direction, units)

            elif verb == "pos":
                obs = self.GetCurrentPosition()

            elif verb == "observe":
                obs = self.Observe()

            elif verb == "answer":
                # 解析答案：支持 "answer x y" 或 "answer (x, y)" 或 "answer x, y"
                match = re.match(r"answer\s*\(?\s*(-?\d+)\s*,?\s*(-?\d+)\s*\)?\s*$", content, re.IGNORECASE)
                if not match:
                    obs = "Invalid answer format. Use: \\boxed{answer x y}"
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                x_val = int(match.group(1))
                y_val = int(match.group(2))
                msg = self.Done((x_val, y_val))
                obs = msg
                # 根据正确性给奖励
                if self._last_done_correct is True:
                    reward = 1.0
                else:
                    reward = -1.0
                terminated = True

            else:
                obs = f"Invalid action '{verb}'."
                return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}

        except Exception as e:
            obs = f"Internal error: {str(e)}"
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
        # 简单解析：按空格拆分，verb 为第一个词
        tokens = content.split()
        if not tokens:
            return None
        verb = tokens[0].lower()
        args = tokens[1:]
        return {"content": content, "verb": verb, "args": args}

    def sample_random_action(self) -> str:
        # 随机选择动作示例
        choices = []
        # 高概率返回观察和移动
        choices.extend([
            "\\boxed{observe}",
            "\\boxed{pos}",
            "\\boxed{init}",
            "\\boxed{move U 1}",
            "\\boxed{move R 2}",
            "\\boxed{move D 1}",
            "\\boxed{move L 1}",
        ])
        return random.choice(choices)

    # ----------------------------
    # 原环境的辅助方法（保留并适配）
    # ----------------------------

    @property
    def step_count(self) -> int:
        """保持与原环境的 step_count 接口兼容，映射到 turn_count"""
        return self.turn_count

    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        x, y = 0, 0

        direction_map = {
            'U': (0, 1),
            'D': (0, -1),
            'L': (-1, 0),
            'R': (1, 0)
        }

        for direction, units in self.problem.get("commands", []):
            dx, dy = direction_map[direction]
            x += dx * units
            y += dy * units

        return (x, y)

    def InitializePosition(self):
        r"""
    
        Initialize the current position to the coordinate origin (0, 0).
    
        Args:
            None
    
        Returns:
            str: Prompt message for successful initialization.
    
        Example Output:
            "Position initialized to (0, 0)"
        """
        self.current_x = 0
        self.current_y = 0
        return f"Position initialized to ({self.current_x}, {self.current_y})"

    def ProcessCommand(self, direction: str, units: int):
        r"""
    
        Update the current position based on the given direction and number of units.
    
        Args:
            direction (str): Movement direction, must be 'U', 'D', 'L', or 'R'.
            units (int): Number of movement units, must be a positive integer.
    
        Returns:
            str: The current position after movement.
    
        Example Output:
            "(3, 2)"
        """
        direction_map = {
            'U': (0, 1),
            'D': (0, -1),
            'L': (-1, 0),
            'R': (1, 0)
        }

        if direction not in direction_map:
            return f"Error: Invalid direction '{direction}', must be 'U', 'D', 'L', or 'R'"
        if not isinstance(units, int) or units <= 0:
            return f"Error: Invalid units '{units}', must be a positive integer"

        dx, dy = direction_map[direction]
        self.current_x += dx * units
        self.current_y += dy * units

        return f"({self.current_x}, {self.current_y})"

    def GetCurrentPosition(self):
        r"""
    
        Get the current coordinate position.
    
        Args:
            None
    
        Returns:
            str: The current coordinate position, formatted as "(x, y)".
    
        Example Output:
            "(3, 1)"
        """
        return f"({self.current_x}, {self.current_y})"

    def Observe(self):
        r"""
    
        Get the observation information of the environment, including the command list and current position.
    
        Args:
            None
    
        Returns:
            str: Environment observation information.
    
        Example Output:
            "Command list: [('U', 2), ('R', 3), ('D', 1)], Current position: (0, 0)"
        """
        return f"Command list: {self.problem.get('commands', [])}, Current position: ({self.current_x}, {self.current_y})"

    def Done(self, answer):
        r"""
    
        Verify whether the final answer is correct and return the result information.
    
        Args:
            answer (tuple): The final coordinates submitted by the user, formatted as (x, y).
    
        Returns:
            str: Result information, including correctness and reward information.
    
        Example Output:
            "Your answer: (3, 1), Reference answer: (3, 1), Result: Correct, reward=1"
        """
        ref_answer = self.get_ref_answer()
        # 若为 list，转换为 tuple
        if isinstance(answer, list):
            answer = tuple(answer)
        correct = answer == ref_answer
        self._last_done_correct = correct
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        # 奖励由 step 决定，这里仅返回结果信息
        return msg

    def solve(self):
        r"""
        Automatically call all actions to complete the complete process, and submit the answer for verification. 
    
        Returns:
            str: The result information of the final answer verification. 
        """
        # 初始化
        self.step("\\boxed{init}")
        # 观察
        observe_info, _, _, _, _ = self.step("\\boxed{observe}")
        # 提取命令列表
        try:
            commands_str = observe_info.split("Command list: ")[1].split(", Current position: ")[0]
            commands = ast.literal_eval(commands_str)
        except Exception:
            commands = self.problem.get("commands", [])
        # 执行移动
        for direction, units in commands:
            self.step(f"\\boxed{{move {direction} {units}}}")
        # 获取位置
        current_pos_str, _, _, _, _ = self.step("\\boxed{pos}")
        try:
            current_pos = ast.literal_eval(current_pos_str)
            if isinstance(current_pos, tuple):
                x, y = current_pos
            else:
                # Manual parse if needed
                match = re.match(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", current_pos_str)
                x = int(match.group(1)) if match else 0
                y = int(match.group(2)) if match else 0
        except Exception:
            match = re.match(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", current_pos_str)
            x = int(match.group(1)) if match else 0
            y = int(match.group(2)) if match else 0
        # 提交答案
        final_obs, _, _, _, _ = self.step(f"\\boxed{{answer {x} {y}}}")
        return final_obs