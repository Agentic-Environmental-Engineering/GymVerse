from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class MajorityElementEnvGEM(Env):
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

        # 定义难度参数范围（根据多数元素问题难度设计）
        self.complexity_params = {
            "array_length": (5, 50),            # 投票数组长度
            "distinct_candidates": (2, 20),     # 候选人不同ID数量
            "candidate_id_max": (10, 10000),    # 候选人ID范围上限
            "majority_percent_target": (55, 80) # 若生成多数实例时，多数候选人的占比目标（百分比）
        }

        # 参数方差（用于训练时微调随机性）
        self.param_variance = {
            "array_length": 2,
            "distinct_candidates": 1,
            "candidate_id_max": 50,
            "majority_percent_target": 3,
        }

        # 占位属性（将由 _apply_complexity_params 填充）
        self.array_length: int = 0
        self.distinct_candidates: int = 0
        self.candidate_id_max: int = 0
        self.majority_percent_target: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 问题实例
        self.problem: Dict[str, Any] = {}
        self.votes: list[int] = []

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
            "Majority Element: Determine if there is a candidate with more than half of the votes.\n"
            "Available actions (use the last \\boxed{...} in your message):\n"
            "- Observe votes list: \\boxed{observe}\n"
            "- Count a candidate's votes: \\boxed{count X}  (X is candidate id)\n"
            "- Get total votes: \\boxed{total}\n"
            "- Check majority: \\boxed{check X T}  (X is candidate id, T is total votes)\n"
            "- Submit answer: \\boxed{answer N}  (N is candidate id with majority, or -1 if none)\n"
            "Aliases:\n"
            "- \\boxed{CountVote X}, \\boxed{GetTotalVotes}, \\boxed{CheckMajority X T}, \\boxed{Observe}, \\boxed{Done N}\n"
        )

    def get_task_suffix(self) -> str:
        cur_len = len(self.votes)
        distinct = len(set(self.votes))
        return (
            f"Length: {cur_len} | Distinct candidates: {distinct}\n"
            f"Turn: {self.turn_count}/{self.max_turns}\n"
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
        self.votes = self.problem["votes"]

        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        n = self.array_length
        k = min(self.distinct_candidates, max(1, self.array_length))  # 至少1，最多不超过长度
        # 生成候选人ID集合
        candidates = random.sample(range(1, self.candidate_id_max + 1), k)

        votes = []
        # 以 50% 概率生成有多数的实例
        if random.random() < 0.5 and n > 0:
            majority_candidate = random.choice(candidates)
            # 多数票数至少为 floor(n/2)+1，且接近 majority_percent_target%
            target = int(round(n * self.majority_percent_target / 100.0))
            maj_votes = max(target, (n // 2) + 1)
            maj_votes = min(maj_votes, n)  # 不可超过总票数

            votes.extend([majority_candidate] * maj_votes)
            remaining = n - maj_votes
            for _ in range(remaining):
                # 从其他候选人中随机填充
                other = random.choice([c for c in candidates if c != majority_candidate])
                votes.append(other)
            random.shuffle(votes)
        else:
            # 尽量生成无多数的平衡分布
            if n == 0:
                votes = []
            else:
                base = n // k
                remainder = n - base * k
                counts = [base] * k
                for i in range(remainder):
                    counts[i] += 1
                # 调整确保没有超过一半
                max_allowed = n // 2
                for i in range(k):
                    if counts[i] > max_allowed:
                        # 将超出的部分分配给其他人
                        excess = counts[i] - max_allowed
                        counts[i] = max_allowed
                        j = 0
                        while excess > 0:
                            if j != i:
                                counts[j] += 1
                                excess -= 1
                            j = (j + 1) % k
                # 构造 votes
                for cand, cnt in zip(candidates, counts):
                    votes.extend([cand] * cnt)
                random.shuffle(votes)

        return {"votes": votes, "candidates": candidates}

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

        content = parsed["content"]
        tokens = content.split()
        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            if len(tokens) == 0:
                obs = "Invalid action: empty content."
                reward = LanguageGameReward.invalid_action_reward
                terminated = True
            else:
                cmd = tokens[0].lower()

                if cmd in ["observe", "observe()", "obs"]:
                    obs = self.Observe()
                    # 中间动作不结束
                    terminated = False
                    reward = 0.0

                elif cmd in ["total", "gettotalvotes", "gettotalvotes()"]:
                    obs = self.GetTotalVotes()
                    terminated = False
                    reward = 0.0

                elif cmd in ["count", "countvote", "countvote()"]:
                    # 需要一个整数参数 candidate
                    if len(tokens) < 2:
                        obs = "Invalid action: missing candidate id for count."
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                    else:
                        candidate = self._parse_int(tokens[1])
                        if candidate is None:
                            obs = "Invalid action: candidate id must be integer."
                            reward = LanguageGameReward.invalid_action_reward
                            terminated = True
                        else:
                            obs = self.CountVote(candidate)
                            terminated = False
                            reward = 0.0

                elif cmd in ["check", "checkmajority", "checkmajority()"]:
                    # 需要两个整数参数 candidate total_votes
                    if len(tokens) < 3:
                        obs = "Invalid action: missing candidate or total_votes."
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                    else:
                        candidate = self._parse_int(tokens[1])
                        total_votes = self._parse_int(tokens[2])
                        if candidate is None or total_votes is None:
                            obs = "Invalid action: candidate and total_votes must be integers."
                            reward = LanguageGameReward.invalid_action_reward
                            terminated = True
                        else:
                            obs = self.CheckMajority(candidate, total_votes)
                            terminated = False
                            reward = 0.0

                elif cmd in ["answer", "done", "done()"]:
                    # 需要一个整数参数 answer
                    if len(tokens) < 2:
                        obs = "Invalid action: missing answer."
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                    else:
                        ans = self._parse_int(tokens[1])
                        if ans is None:
                            obs = "Invalid action: answer must be integer."
                            reward = LanguageGameReward.invalid_action_reward
                            terminated = True
                        else:
                            msg, success = self._Done_internal(ans)
                            obs = msg
                            reward = 1.0 if success else -1.0
                            terminated = True

                else:
                    obs = f"Invalid action: unknown command '{cmd}'."
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
        return {"content": content}

    def sample_random_action(self) -> str:
        # 随机示例动作（非终止）
        return "\\boxed{observe}"

    # ===== 辅助方法（保留并转换） =====
    def CountVote(self, candidate: int) -> str:
        """
        Count the number of votes for the specified candidate.
        Returns: str number
        """
        count = self.votes.count(candidate)
        return str(count)

    def GetTotalVotes(self) -> str:
        """
        Get the total number of votes.
        Returns: str total
        """
        total = len(self.votes)
        return str(total)

    def CheckMajority(self, candidate: int, total_votes: int) -> str:
        """
        Check if the specified candidate has obtained more than half of the votes.
        Returns: "True" or "False"
        """
        count = self.votes.count(candidate)
        return str(count > total_votes / 2)

    def Observe(self) -> str:
        """
        Return observation information of the current state, including the vote list.
        Returns: str of list
        """
        return str(self.votes)

    def get_ref_answer(self) -> int:
        """
        Use the information in the environment to get the reference answer.
        Return candidate id or -1
        """
        candidate_count: Dict[int, int] = {}
        n = len(self.votes)

        for vote in self.votes:
            if vote in candidate_count:
                candidate_count[vote] += 1
            else:
                candidate_count[vote] = 1

            if candidate_count[vote] > n / 2:
                return vote

        return -1

    def Done(self, answer: int) -> str:
        """
        Verify whether the final answer is correct and return result information.
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg + f", reward={1 if correct else 0}"

    def _Done_internal(self, answer: int) -> Tuple[str, bool]:
        """
        GEM step内部使用的完成校验，返回消息与是否成功。
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg, correct

    def solve(self) -> str:
        """
        Automatically call actions to solve the environment, and submit the answer.
        Returns final message string.
        """
        # Observe votes
        obs, _, _, _, _ = self.step("\\boxed{observe}")
        try:
            votes = eval(obs)
        except Exception:
            votes = []

        # Get total
        total_str, _, _, _, _ = self.step("\\boxed{total}")
        try:
            total_votes = int(total_str)
        except Exception:
            total_votes = len(self.votes)

        # Try each unique candidate
        unique_candidates = list(set(votes))
        majority_candidate = -1
        for candidate in unique_candidates:
            check_msg, _, _, _, _ = self.step(f"\\boxed{{check {candidate} {total_votes}}}")
            if check_msg == "True":
                majority_candidate = candidate
                break

        final_msg, _, _, _, _ = self.step(f"\\boxed{{answer {majority_candidate}}}")
        return final_msg

    # ===== 工具方法 =====
    @staticmethod
    def _parse_int(token: str) -> Optional[int]:
        try:
            return int(token)
        except Exception:
            return None