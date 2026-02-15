from typing import Any, Dict, Optional, Tuple
import random
import re
import json
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class ProblemCountingEnvGEM(Env):
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

        # 难度参数范围设计
        self.complexity_params = {
            # 问题数量（列表长度）
            "array_length": (5, 50),
            # 数值取值范围上界（难度越高范围越大）
            "value_range": (10, 10000),
            # 步数限制（难度越高允许的步数越多，以容纳更复杂问题）
            "max_turns_param": (20, 200),
        }

        # 参数方差（训练时可启用微随机化）
        self.param_variance = {
            "array_length": 2,
            "value_range": 100,
            "max_turns_param": 10,
        }

        # 占位属性（由 _apply_complexity_params 填充）
        self.array_length: int = 0
        self.value_range: int = 0
        self.max_turns_param: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 当前问题实例
        self.problem: Dict[str, Any] = {}
        self.n: int = 0
        self.threshold: int = 0
        self.problems: list[int] = []

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

        # 将难度参数中的 max_turns_param 应用于环境步数限制
        self.max_turns = int(self.max_turns_param)

    def _get_instructions(self) -> str:
        return (
            "Problem Counting: Given a list of problem difficulties and a threshold, "
            "your task is to count how many problems have difficulty strictly greater than the threshold.\n"
            "Available actions (wrap one command in \\boxed{...}):\n"
            "- Observe the generated instance:\n"
            "  \\boxed{observe}\n"
            "- Locally compute a count for any provided threshold and list:\n"
            "  \\boxed{count threshold=TH problems=[v1, v2, ...]}\n"
            "  Example: \\boxed{count threshold=3000 problems=[2500, 3200, 1500]}\n"
            "- Submit your final answer for the current hidden instance:\n"
            "  \\boxed{answer N}\n"
        )

    def get_task_suffix(self) -> str:
        return f"Turn: {self.turn_count}/{self.max_turns}. Enter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成的实例

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()
        self.n = self.problem["n"]
        self.threshold = self.problem["threshold"]
        self.problems = self.problem["problems"]

        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self) -> Dict[str, Any]:
        """根据难度参数随机生成问题实例"""
        n = self.array_length
        vmax = self.value_range
        # 为了提高多样性，允许阈值在 [0, vmax] 范围
        threshold = random.randint(0, vmax)
        problems = [random.randint(0, vmax) for _ in range(n)]
        return {"n": n, "threshold": threshold, "problems": problems}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)

        # 默认返回
        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        if not parsed:
            obs = f"Format error at turn {self.turn_count}."
            reward = LanguageGameReward.format_error_reward
            terminated = True
        else:
            atype = parsed.get("type", "")
            if atype == "observe":
                msg = self.Observe()
                obs = f"Observed instance:\n{msg}"
                reward = 0.0
                terminated = False
            elif atype == "count":
                th = parsed.get("threshold", None)
                lst = parsed.get("problems", None)
                if th is None or lst is None:
                    obs = "Invalid count command. Expect: count threshold=TH problems=[...]"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    msg = self.CountAboveThreshold(th, lst)
                    obs = f"Local count result: {msg}"
                    reward = 0.0
                    terminated = False
            elif atype == "answer":
                ans = parsed.get("answer", None)
                if ans is None:
                    obs = "Invalid answer command. Expect: answer N"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    msg, correct = self.Done(ans)
                    obs = msg
                    reward = 1.0 if correct else -1.0
                    terminated = True
            else:
                obs = f"Invalid action: {atype}"
                reward = LanguageGameReward.invalid_action_reward
                terminated = True

        # 统一超时检查放在 step 结尾
        if not terminated and self.turn_count >= self.max_turns:
            obs = f"{obs}\nReached max turns ({self.max_turns})."
            reward = 0.0
            terminated = True
            truncated = True

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

        low = content.lower()

        # observe
        if low == "observe":
            return {"type": "observe"}

        # answer N
        if low.startswith("answer"):
            m = re.match(r"(?i)answer\s+(-?\d+)\s*$", content)
            if not m:
                return None
            return {"type": "answer", "answer": int(m.group(1))}

        # count threshold=TH problems=[...]
        if low.startswith("count"):
            # threshold
            thr_m = re.search(r"(?i)threshold\s*=\s*(-?\d+)", content)
            if not thr_m:
                return None
            threshold = int(thr_m.group(1))

            # problems list
            probs_m = re.search(r"(?i)problems\s*=\s*\[([^\]]*)\]", content, re.DOTALL)
            if not probs_m:
                return None
            raw_list = probs_m.group(1).strip()
            probs_list = self._parse_int_list(raw_list)
            if probs_list is None:
                return None

            return {"type": "count", "threshold": threshold, "problems": probs_list}

        return None

    def _parse_int_list(self, s: str) -> Optional[list]:
        if s.strip() == "":
            return []
        parts = [p.strip() for p in s.split(",")]
        result = []
        for p in parts:
            if not p:
                return None
            m = re.match(r"^\s*(-?\d+)\s*$", p)
            if not m:
                return None
            result.append(int(m.group(1)))
        return result

    # 辅助方法（保留并转换）
    @property
    def finished(self) -> bool:
        # GEM 风格用 terminated 标识结束；此处提供兼容属性
        return False  # 在 GEM 中不使用该属性，保留以兼容

    @property
    def reward(self):
        # GEM 标准在 step 返回 reward；此属性仅占位
        return 0.0

    def get_ref_answer(self) -> int:
        """使用环境中的信息获取参考答案"""
        return sum(1 for problem in self.problems if problem > self.threshold)

    def Observe(self) -> str:
        """获取当前环境中的问题数量、阈值和各问题难度"""
        observation = {
            "n": self.n,
            "threshold": self.threshold,
            "problems": self.problems,
        }
        return json.dumps(observation)

    def CountAboveThreshold(self, threshold: int, problems: list) -> str:
        """统计高于给定阈值的元素数量"""
        count = sum(1 for problem in problems if problem > threshold)
        return str(count)

    def Done(self, answer: int) -> Tuple[str, bool]:
        """验证最终答案并返回结果信息"""
        ref_answer = self.get_ref_answer()
        correct = (answer == ref_answer)
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg, correct

    def solve(self) -> str:
        """自动调用动作完成流程并提交答案"""
        # 1) 观察
        obs, _, term, _, _ = self.step("\\boxed{observe}")
        # 从观察结果中提取 JSON（最后一行）
        # 观察返回形如 "Observed instance:\n{json}"
        try:
            json_str = obs.splitlines()[-1]
            observe_data = json.loads(json_str)
            threshold = observe_data["threshold"]
            problems = observe_data["problems"]
        except Exception:
            return "Automatic solve failed: parse observe error."

        if term:
            return obs

        # 2) 计算
        count_action = f"\\boxed{{count threshold={threshold} problems=[{', '.join(str(x) for x in problems)}]}}"
        obs2, _, term2, _, _ = self.step(count_action)
        if term2 and "Local count result" not in obs2:
            return obs2
        try:
            # obs2: "Local count result: X"
            local_count = int(obs2.strip().split(":")[-1].strip())
        except Exception:
            return "Automatic solve failed: parse count error."

        # 3) 提交答案
        ans_action = f"\\boxed{{answer {local_count}}}"
        obs3, _, _, _, _ = self.step(ans_action)
        return obs3

    def sample_random_action(self) -> str:
        # 简单示例：先观察
        return "\\boxed{observe}"