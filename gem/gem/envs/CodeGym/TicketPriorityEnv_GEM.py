from typing import Any, Dict, Optional, Tuple
import random
import re
import json
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class TicketPriorityEnvGEM(Env):
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
            "num_messages": (5, 50),             # 消息数量
            "message_length": (5, 20),           # 每条消息的平均词数（用于生成噪声词）
            "high_ratio_percent": (10, 50),      # 高优先级比例（百分比）
            "medium_ratio_percent": (10, 40),    # 中优先级比例（百分比）
            "ambiguous_percent": (0, 20),        # 同时包含高/中关键字的比例（百分比，参考答案判为HIGH）
            "max_turns_param": (20, 200),        # 最大步数限制
        }

        # 参数方差（启用随机化时，为插值中心值添加微小扰动）
        self.param_variance = {
            "num_messages": 2,
            "message_length": 2,
            "high_ratio_percent": 3,
            "medium_ratio_percent": 3,
            "ambiguous_percent": 2,
            "max_turns_param": 10,
        }

        # 占位属性
        self.num_messages: int = 0
        self.message_length: int = 0
        self.high_ratio_percent: int = 0
        self.medium_ratio_percent: int = 0
        self.ambiguous_percent: int = 0
        self.max_turns_param: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 关键字集合（保留原环境逻辑）
        self.high_keywords = {"immediate", "urgent", "asap"}
        self.medium_keywords = {"soon", "quick", "timely"}

        # 问题实例
        self.problem: Dict[str, Any] = {"messages": []}
        self.messages: list[str] = []

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

        # 将难度参数中的 max_turns 应用到环境步数限制
        self.max_turns = int(self.max_turns_param)

    def _get_instructions(self) -> str:
        return (
            "Ticket Priority: Determine priority (HIGH/MEDIUM/LOW) for each message.\n"
            "Available actions:\n"
            "- Observe messages: \\boxed{observe}\n"
            "- Check HIGH keyword in message i (0-based): \\boxed{check_high i}\n"
            "- Check MEDIUM keyword in message i (0-based): \\boxed{check_medium i}\n"
            "- Submit answers (comma-separated labels for all messages): \\boxed{answer LABEL1,LABEL2,...}\n"
            "Labels must be one of: HIGH, MEDIUM, LOW.\n"
        )

    def get_task_suffix(self) -> str:
        return f"Messages: {self.num_messages}\nTurn: {self.turn_count}/{self.max_turns}\nEnter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响实例生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()
        self.messages = self.problem["messages"]

        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        # 计算各类别数量
        hr = max(0.0, min(1.0, self.high_ratio_percent / 100.0))
        mr = max(0.0, min(1.0, self.medium_ratio_percent / 100.0))
        ar = max(0.0, min(1.0, self.ambiguous_percent / 100.0))

        total = hr + mr + ar
        if total > 0.9:
            scale = 0.9 / total
            hr *= scale
            mr *= scale
            ar *= scale

        n_high = int(round(self.num_messages * hr))
        n_med = int(round(self.num_messages * mr))
        n_amb = int(round(self.num_messages * ar))
        # 调整以避免超过总数
        assigned = n_high + n_med + n_amb
        n_low = max(0, self.num_messages - assigned)
        # 如果分配不足，补到高或中优先级
        while n_high + n_med + n_amb + n_low < self.num_messages:
            n_low += 1

        # 噪声词库
        filler = [
            "please", "respond", "issue", "address", "consider", "note",
            "thanks", "team", "update", "review", "ticket", "request",
            "help", "status", "follow-up", "detail"
        ]

        def make_message(kind: str) -> str:
            words = []
            # 添加关键字
            if kind == "HIGH":
                words.append(random.choice(list(self.high_keywords)))
            elif kind == "MEDIUM":
                words.append(random.choice(list(self.medium_keywords)))
            elif kind == "AMBIGUOUS":
                words.append(random.choice(list(self.high_keywords)))
                words.append(random.choice(list(self.medium_keywords)))
            # 添加噪声词
            n_fill = max(0, self.message_length - len(words))
            for _ in range(n_fill):
                words.append(random.choice(filler))
            # 组句
            msg = " ".join(words)
            # 简单修饰
            msg = msg.capitalize() + "."
            return msg

        messages = []
        messages.extend([make_message("HIGH") for _ in range(n_high)])
        messages.extend([make_message("MEDIUM") for _ in range(n_med)])
        messages.extend([make_message("AMBIGUOUS") for _ in range(n_amb)])
        messages.extend([make_message("LOW") for _ in range(n_low)])

        # 打乱顺序
        random.shuffle(messages)
        return {"messages": messages}

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

            elif cmd == "check_high":
                if len(args) != 1 or not args[0].isdigit():
                    obs = "Invalid check_high usage. Use: \\boxed{check_high i}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    idx = int(args[0])
                    if idx < 0 or idx >= len(self.messages):
                        obs = f"Index out of range: {idx}"
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                    else:
                        obs = self.CheckHighPriority(self.messages[idx])

            elif cmd == "check_medium":
                if len(args) != 1 or not args[0].isdigit():
                    obs = "Invalid check_medium usage. Use: \\boxed{check_medium i}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    idx = int(args[0])
                    if idx < 0 or idx >= len(self.messages):
                        obs = f"Index out of range: {idx}"
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                    else:
                        obs = self.CheckMediumPriority(self.messages[idx])

            elif cmd == "answer":
                if len(args) != 1:
                    obs = "Invalid answer usage. Use: \\boxed{answer LABEL1,LABEL2,...}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    labels_str = args[0]
                    labels = [x.strip().upper() for x in labels_str.split(",") if x.strip()]
                    # 校验标签数量
                    ref = self.get_ref_answer()
                    if len(labels) != len(ref):
                        # 当作失败（提交长度不匹配）
                        result_msg = f"Your answer length={len(labels)} does not match expected length={len(ref)}."
                        done_msg = f"{result_msg} Reference answer: {ref}"
                        obs = done_msg
                        reward = -1.0
                        terminated = True
                    else:
                        msg = self.Done(labels)
                        # Done() 内部按照原逻辑给出结果，不直接使用其奖励以兼容 GEM 规范
                        correct = labels == self.get_ref_answer()
                        reward = 1.0 if correct else -1.0
                        obs = msg
                        terminated = True

            else:
                obs = f"Invalid action: {cmd}"
                reward = LanguageGameReward.invalid_action_reward
                terminated = True

        except Exception as e:
            obs = f"Runtime error: {str(e)}"
            reward = LanguageGameReward.invalid_action_reward
            terminated = True

        # 超时检查（统一放在 step 结尾）
        if not terminated and self.turn_count >= self.max_turns:
            obs = f"{obs}\nReached max turns ({self.max_turns})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r"\\boxed\{(.+?)\}", re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        # 解析命令与参数
        tokens = content.split()
        if not tokens:
            return None
        cmd = tokens[0].lower()
        args = tokens[1:]
        # 特殊处理 answer 的逗号参数
        if cmd == "answer":
            rest = content[len("answer"):].strip()
            if rest.startswith(" "):
                rest = rest.strip()
            args = [rest] if rest else []
        return {"cmd": cmd, "args": args, "content": content}

    def sample_random_action(self) -> str:
        if not self.messages:
            return "\\boxed{observe}"
        # 随机选择一种动作
        choice = random.choice(["observe", "check_high", "check_medium", "answer"])
        if choice == "observe":
            return "\\boxed{observe}"
        elif choice in ("check_high", "check_medium"):
            idx = random.randint(0, len(self.messages) - 1)
            return f"\\boxed{{{choice} {idx}}}"
        else:
            # 生成随机答案
            labels = []
            for _ in range(len(self.messages)):
                labels.append(random.choice(["HIGH", "MEDIUM", "LOW"]))
            return "\\boxed{answer " + ",".join(labels) + "}"

    # ===== 保留原环境的辅助方法并转换 =====

    def get_ref_answer(self):
        """
        Use the information in the environment to get the reference answer.
        """
        high_keywords = {"immediate", "urgent", "asap"}
        medium_keywords = {"soon", "quick", "timely"}

        priorities = []
        for message in self.messages:
            message_lower = message.lower()
            if any(word in message_lower for word in high_keywords):
                priorities.append("HIGH")
            elif any(word in message_lower for word in medium_keywords):
                priorities.append("MEDIUM")
            else:
                priorities.append("LOW")

        return priorities

    def CheckHighPriority(self, message: str):
        """
        Check if the message contains high-priority keywords.
        Returns "True" or "False".
        """
        high_keywords = {"immediate", "urgent", "asap"}
        message_lower = message.lower()
        return "True" if any(word in message_lower for word in high_keywords) else "False"

    def CheckMediumPriority(self, message: str):
        """
        Check if the message contains medium-priority keywords.
        Returns "True" or "False".
        """
        medium_keywords = {"soon", "quick", "timely"}
        message_lower = message.lower()
        return "True" if any(word in message_lower for word in medium_keywords) else "False"

    def Observe(self):
        """
        Get the list of messages in the current environment.
        Returns JSON formatted string.
        """
        return json.dumps(self.messages)

    def Done(self, answers):
        """
        Verify whether the submitted priority answers are correct and return the result information.
        """
        ref_answer = self.get_ref_answer()
        correct = answers == ref_answer
        msg = f"Your answer: {answers}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg