from typing import Any, Dict, Optional, Tuple
import random
import re
import math
import json
import ast

from gem.core import Env
from gem.utils.constants import LanguageGameReward


class CircleOverlapEnvGEM(Env):
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
            # 圆的数量（列表长度）
            "num_circles": (2, 50),
            # 坐标与半径的范围控制：坐标取 [-coord_abs_max, coord_abs_max]
            "coord_abs_max": (10, 1000),
            # 最大半径（半径范围为 [1, max_radius]）
            "max_radius": (1, 50),
        }

        # 参数方差（启用随机化时为中心值添加微扰）
        self.param_variance = {
            "num_circles": 2,
            "coord_abs_max": 50,
            "max_radius": 5,
        }

        # 占位属性
        self.num_circles: int = 0
        self.coord_abs_max: int = 0
        self.max_radius: int = 0

        # 状态
        self.turn_count: int = 0
        self.problem: Dict[str, Any] = {"circles": []}

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
            "Circle Overlap: Determine whether any two circles overlap.\n"
            "You are given n circles, each as [x, y, r]. Your goal is to decide if any pair overlaps.\n"
            "Available actions (use the last \\boxed{...} block in your message):\n"
            "- Observe circles: \\boxed{observe}\n"
            "- Calculate distance between circle i and j (0-based indices): \\boxed{distance i j}\n"
            "- Sum radii of circle i and j: \\boxed{sumr i j}\n"
            "- Compare two numbers (distance and sum_radii): \\boxed{compare D S}\n"
            "- Submit final answer (Yes/No): \\boxed{answer Yes} or \\boxed{answer No}\n"
            "Example workflow: observe -> distance 0 1 -> sumr 0 1 -> compare 5.0 6 -> answer Yes\n"
        )

    def get_task_suffix(self) -> str:
        return f"Circles: {self.num_circles} | Turn: {self.turn_count}/{self.max_turns} | Enter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响实例生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self) -> Dict[str, Any]:
        """根据难度参数随机生成问题实例"""
        n = self.num_circles
        cmax = self.coord_abs_max
        rmax = max(1, self.max_radius)

        circles = []

        # 以 50% 概率确保至少有一对相交，以避免过多全否定样本
        ensure_overlap = random.random() < 0.5 and n >= 2

        if ensure_overlap:
            # 构造第一对相交的圆
            x1 = random.randint(-cmax, cmax)
            y1 = random.randint(-cmax, cmax)
            r1 = random.randint(1, rmax)

            r2 = random.randint(1, rmax)
            # 生成一个长度小于 r1+r2 的向量
            max_d = max(1e-6, (r1 + r2 - 1))  # 留一点余量，避免等边界
            # 随机方向
            theta = random.uniform(0, 2 * math.pi)
            d = random.uniform(0, max_d)
            dx = int(round(math.cos(theta) * d))
            dy = int(round(math.sin(theta) * d))
            x2 = max(-cmax, min(cmax, x1 + dx))
            y2 = max(-cmax, min(cmax, y1 + dy))

            circles.append([x1, y1, r1])
            circles.append([x2, y2, r2])

            # 其余圆随机生成
            for _ in range(2, n):
                x = random.randint(-cmax, cmax)
                y = random.randint(-cmax, cmax)
                r = random.randint(1, rmax)
                circles.append([x, y, r])
        else:
            for _ in range(n):
                x = random.randint(-cmax, cmax)
                y = random.randint(-cmax, cmax)
                r = random.randint(1, rmax)
                circles.append([x, y, r])

        return {"circles": circles}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1

        parsed = self._parse_action(action)
        if not parsed:
            obs = "Format error: Missing or invalid \\boxed{...} action."
            reward = LanguageGameReward.format_error_reward
            terminated = True
            truncated = False
            # 超时检查（统一放在 step 结尾）
            if not terminated and self.turn_count >= self.max_turns:
                obs = f"{obs}\nReached max turns ({self.max_turns})."
                return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

        content = parsed["content"]
        tokens = content.strip().split()
        if len(tokens) == 0:
            obs = "Format error: Empty action."
            reward = LanguageGameReward.format_error_reward
            terminated = True
            truncated = False
            if not terminated and self.turn_count >= self.max_turns:
                obs = f"{obs}\nReached max turns ({self.max_turns})."
                return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

        cmd = tokens[0].lower()
        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            if cmd in ["observe", "obs"]:
                if len(tokens) != 1:
                    obs = "Format error: observe takes no arguments."
                    reward = LanguageGameReward.format_error_reward
                    terminated = True
                else:
                    obs = self.Observe()

            elif cmd in ["distance", "dist"]:
                if len(tokens) != 3:
                    obs = "Format error: distance requires two indices: distance i j"
                    reward = LanguageGameReward.format_error_reward
                    terminated = True
                else:
                    i = int(tokens[1])
                    j = int(tokens[2])
                    if not (0 <= i < len(self.problem["circles"])) or not (0 <= j < len(self.problem["circles"])):
                        obs = f"Format error: indices out of range. Valid: [0, {len(self.problem['circles'])-1}]"
                        reward = LanguageGameReward.format_error_reward
                        terminated = True
                    else:
                        c1 = self.problem["circles"][i]
                        c2 = self.problem["circles"][j]
                        obs = self.CalculateDistance(c1, c2)

            elif cmd in ["sumr", "sum", "sumradii"]:
                if len(tokens) != 3:
                    obs = "Format error: sumr requires two indices: sumr i j"
                    reward = LanguageGameReward.format_error_reward
                    terminated = True
                else:
                    i = int(tokens[1])
                    j = int(tokens[2])
                    if not (0 <= i < len(self.problem["circles"])) or not (0 <= j < len(self.problem["circles"])):
                        obs = f"Format error: indices out of range. Valid: [0, {len(self.problem['circles'])-1}]"
                        reward = LanguageGameReward.format_error_reward
                        terminated = True
                    else:
                        c1 = self.problem["circles"][i]
                        c2 = self.problem["circles"][j]
                        obs = self.SumRadii(c1, c2)

            elif cmd in ["compare", "cmp"]:
                if len(tokens) != 3:
                    obs = "Format error: compare requires two numbers: compare distance sum_radii"
                    reward = LanguageGameReward.format_error_reward
                    terminated = True
                else:
                    d = float(tokens[1])
                    s = float(tokens[2])
                    obs = self.CompareDistanceSum(d, s)

            elif cmd in ["answer", "ans"]:
                if len(tokens) != 2:
                    obs = "Format error: answer requires Yes or No: answer Yes"
                    reward = LanguageGameReward.format_error_reward
                    terminated = True
                else:
                    ans_token = tokens[1].strip().lower()
                    if ans_token in ["yes", "y", "true", "t"]:
                        user_answer = "Yes"
                    elif ans_token in ["no", "n", "false", "f"]:
                        user_answer = "No"
                    else:
                        obs = "Format error: answer must be Yes or No."
                        reward = LanguageGameReward.format_error_reward
                        terminated = True
                        # fall through to timeout check at end
                        if not terminated and self.turn_count >= self.max_turns:
                            obs = f"{obs}\nReached max turns ({self.max_turns})."
                            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}
                        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

                    # 使用 Done 生成结果消息，并根据正确性给予奖励
                    ref = self.get_ref_answer()
                    correct = (user_answer == ref)
                    obs = self.Done(user_answer)
                    reward = 1.0 if correct else -1.0
                    terminated = True

            elif cmd in ["help", "h", "?"]:
                obs = self._get_instructions()

            else:
                obs = f"Invalid action: {tokens[0]}"
                reward = LanguageGameReward.invalid_action_reward
                terminated = True

        except Exception as e:
            obs = f"Format/Runtime error: {str(e)}"
            reward = LanguageGameReward.format_error_reward
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
        return {"content": content}

    def sample_random_action(self) -> str:
        # 简单示例：观察
        return "\\boxed{observe}"

    # -------------------- 以下为原环境的辅助方法（保留并转换） --------------------

    def get_ref_answer(self) -> str:
        """
        使用环境中的信息获取参考答案。
        任意两圆心距 < 半径和 则视为相交（返回 "Yes"），否则 "No"。
        """
        circles = self.problem.get("circles", [])
        n = len(circles)
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if distance < r1 + r2:
                    return "Yes"
        return "No"

    def CalculateDistance(self, circle1: list, circle2: list) -> str:
        """
        计算两个圆心之间的距离，保留 6 位小数。
        输入: circle = [x, y, r]
        返回: 字符串形式的距离，如 "5.000000"
        """
        x1, y1, _ = circle1
        x2, y2, _ = circle2
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return f"{distance:.6f}"

    def SumRadii(self, circle1: list, circle2: list) -> str:
        """
        计算两个圆半径之和。
        返回字符串，如 "4"
        """
        r1 = circle1[2]
        r2 = circle2[2]
        return str(r1 + r2)

    def CompareDistanceSum(self, distance: float, sum_radii: float) -> str:
        """
        比较圆心距离与半径和，返回字符串 "True"/"False"。
        """
        return str(distance < sum_radii)

    def Observe(self) -> str:
        """
        返回当前环境中的圆信息，JSON 字符串：
        {"n": n, "circles": [[x, y, r], ...]}
        """
        observation = {
            "n": len(self.problem.get("circles", [])),
            "circles": self.problem.get("circles", []),
        }
        # 同步 n 到 num_circles（用于 suffix 展示）
        self.num_circles = observation["n"]
        return json.dumps(observation)

    def Done(self, answer: str) -> str:
        """
        验证最终答案是否正确，返回字符串结果信息。
        示例：
        "Your answer: Yes, Reference answer: Yes, Result: Correct, reward=1"
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg + f", reward={1 if correct else 0}"

    def solve(self) -> str:
        """
        自动执行一次完整流程，基于当前 problem 求解并提交答案。
        返回 Done 的字符串消息。
        """
        # 观察
        obs, _, terminated, _, _ = self.step("\\boxed{observe}")
        if terminated:
            return obs

        try:
            obs_data = json.loads(obs)
        except Exception:
            # 若观察失败，直接给出参考答案
            ref = self.get_ref_answer()
            ans = "Yes" if ref == "Yes" else "No"
            done_msg, _, _, _, _ = self.step(f"\\boxed{{answer {ans}}}")
            return done_msg

        n = obs_data.get("n", 0)
        circles = obs_data.get("circles", [])

        # 枚举两两组合
        for i in range(n):
            for j in range(i + 1, n):
                d_obs, _, term1, _, _ = self.step(f"\\boxed{{distance {i} {j}}}")
                if term1:
                    return d_obs
                try:
                    distance = float(d_obs)
                except Exception:
                    continue

                s_obs, _, term2, _, _ = self.step(f"\\boxed{{sumr {i} {j}}}")
                if term2:
                    return s_obs
                try:
                    sum_r = float(s_obs)
                except Exception:
                    continue

                cmp_obs, _, term3, _, _ = self.step(f"\\boxed{{compare {distance} {sum_r}}}")
                if term3:
                    # compare 的错误会结束回合，这里直接返回
                    return cmp_obs

                if cmp_obs.strip() == "True":
                    done_msg, _, _, _, _ = self.step("\\boxed{answer Yes}")
                    return done_msg

        done_msg, _, _, _, _ = self.step("\\boxed{answer No}")
        return done_msg

    # 兼容构造：从字符串创建固定实例（非必须，但保留原风格）
    @staticmethod
    def from_env_str(env_str: str) -> Optional["CircleOverlapEnvGEM"]:
        """
        兼容原始环境的构造方式：
        允许前缀为 'CircleOverlapEnv@' 或 'CircleOverlapEnvGEM@'
        例如: 'CircleOverlapEnv@{"n": 3, "circles": [[1,1,2],[5,1,2],[4,1,2]]}'
        """
        prefixes = ["CircleOverlapEnv@", "CircleOverlapEnvGEM@"]
        if not any(env_str.startswith(p) for p in prefixes):
            return None
        try:
            # 找到第一个 @ 后的字典
            payload = env_str.split("@", 1)[1]
            options = ast.literal_eval(payload)
            n = int(options.get("n", 2))
            circles = options.get("circles", [])
            env = CircleOverlapEnvGEM(enable_param_randomization=False)
            # 重置并覆盖问题
            env.reset()
            env.problem = {"circles": circles}
            env.num_circles = n
            env.turn_count = 0
            return env
        except Exception:
            return None


# 简单自测
if __name__ == "__main__":
    # 固定实例测试
    s1 = "CircleOverlapEnv@{\"n\": 3, \"circles\": [[1, 1, 2], [5, 1, 2], [4, 1, 2]]}"
    env1 = CircleOverlapEnvGEM.from_env_str(s1)
    if env1 is not None:
        print("Test Case 1:")
        print(env1.solve())
        print("turn count:", env1.turn_count)

    s2 = "CircleOverlapEnv@{\"n\": 3, \"circles\": [[1, 1, 1], [4, 4, 1], [8, 8, 1]]}"
    env2 = CircleOverlapEnvGEM.from_env_str(s2)
    if env2 is not None:
        print("\nTest Case 2:")
        print(env2.solve())
        print("turn count:", env2.turn_count)

    # 随机实例测试
    print("\nRandom Case:")
    env3 = CircleOverlapEnvGEM(complexity=7, enable_param_randomization=True, max_turns=50)
    instr, info = env3.reset(seed=42)
    print(instr)
    print(info["suffix"])
    print(env3.solve())
    print("turn count:", env3.turn_count)