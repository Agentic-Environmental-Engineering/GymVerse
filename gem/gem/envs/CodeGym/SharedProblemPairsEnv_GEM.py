from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class SharedProblemPairsEnvGEM(Env):
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
            "num_participants": (5, 50),  # 参与者数量 m
            "num_problems": (5, 50),  # 题目数量 n
            "solve_prob_pct": (25, 75),  # 单题被解的概率（百分比）
            "turn_allowance": (30, 200),  # 步数上限
        }

        # 参数方差（enable_param_randomization=True 时添加微随机）
        self.param_variance = {
            "num_participants": 2,
            "num_problems": 2,
            "solve_prob_pct": 5,
            "turn_allowance": 10,
        }

        # 占位属性
        self.num_participants: int = 0
        self.num_problems: int = 0
        self.solve_prob_pct: int = 0
        self.turn_allowance: int = 0

        # 原环境的核心状态
        self.m: int = 0
        self.n: int = 0
        self.results: list[list[int]] = []

        # 语言游戏状态
        self.turn_count: int = 0
        self.step_count: int = 0  # 兼容原环境
        self._reward: float = 0.0
        self._finished: bool = False

        self.problem: Dict[str, Any] = {}

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

        # 使用难度参数更新 max_turns（依难度控制步数上限）
        self.max_turns = int(round(self.turn_allowance))

    def _get_instructions(self) -> str:
        return (
            "Programming competition result analysis environment: "
            "Find the number of participant pairs that have solved at least one problem together.\n"
            "Available actions:\n"
            "- Observe environment: \\boxed{observe}\n"
            "- Get total participants: \\boxed{get_participant_count}\n"
            "- Get total problems: \\boxed{get_problem_count}\n"
            "- Get solutions of a participant: \\boxed{get_participant_solutions ID}\n"
            "- Check shared problem between two participants: \\boxed{check_shared_problem ID1 ID2}\n"
            "- Submit final answer: \\boxed{answer N}\n"
            "Notes:\n"
            "- IDs start from 0.\n"
            "- Please submit your final numeric answer using the 'answer' action."
        )

    def get_task_suffix(self) -> str:
        return f"Participants: {self.m}, Problems: {self.n}\nTurn: {self.turn_count}/{self.max_turns}\nEnter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        self.turn_count = 0
        self.step_count = 0
        self._reward = 0.0
        self._finished = False

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        self.m = self.num_participants
        self.n = self.num_problems
        prob = max(0.01, min(0.99, self.solve_prob_pct / 100.0))

        results: list[list[int]] = []
        for _ in range(self.m):
            row = [1 if random.random() < prob else 0 for _ in range(self.n)]
            # 轻微修正：避免全 0 行（可提高可玩性）
            if sum(row) == 0 and self.n > 0:
                row[random.randint(0, self.n - 1)] = 1
            results.append(row)
        self.results = results
        return {"m": self.m, "n": self.n, "results": self.results}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        self.step_count = self.turn_count  # 兼容原环境
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
            self._reward = reward
            self._finished = terminated
            return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

        content = parsed["content"]
        tokens = content.split()
        cmd = tokens[0].lower()

        # 处理动作
        if cmd == "observe":
            obs = self.Observe()
        elif cmd == "get_participant_count":
            if len(tokens) != 1:
                obs = "Format error: get_participant_count takes no arguments."
                reward = LanguageGameReward.format_error_reward
                terminated = True
            else:
                obs = self.GetParticipantCount()
        elif cmd == "get_problem_count":
            if len(tokens) != 1:
                obs = "Format error: get_problem_count takes no arguments."
                reward = LanguageGameReward.format_error_reward
                terminated = True
            else:
                obs = self.GetProblemCount()
        elif cmd == "get_participant_solutions":
            if len(tokens) != 2:
                obs = "Format error: get_participant_solutions requires 1 argument: ID."
                reward = LanguageGameReward.format_error_reward
                terminated = True
            else:
                try:
                    participant_id = int(tokens[1])
                except Exception:
                    obs = "Format error: ID must be an integer."
                    reward = LanguageGameReward.format_error_reward
                    terminated = True
                else:
                    if not (0 <= participant_id < self.m):
                        obs = "Invalid action: Invalid participant_id"
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                    else:
                        obs = self.GetParticipantSolutions(participant_id)
        elif cmd == "check_shared_problem":
            if len(tokens) != 3:
                obs = "Format error: check_shared_problem requires 2 arguments: ID1 ID2."
                reward = LanguageGameReward.format_error_reward
                terminated = True
            else:
                try:
                    p1 = int(tokens[1])
                    p2 = int(tokens[2])
                except Exception:
                    obs = "Format error: IDs must be integers."
                    reward = LanguageGameReward.format_error_reward
                    terminated = True
                else:
                    if not (0 <= p1 < self.m) or not (0 <= p2 < self.m):
                        obs = "Invalid action: Invalid participant_id"
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                    else:
                        obs = self.CheckSharedProblem(p1, p2)
        elif cmd == "answer":
            if len(tokens) != 2:
                obs = "Format error: answer requires 1 integer argument."
                reward = LanguageGameReward.format_error_reward
                terminated = True
            else:
                try:
                    answer_val = int(tokens[1])
                except Exception:
                    obs = "Format error: answer must be an integer."
                    reward = LanguageGameReward.format_error_reward
                    terminated = True
                else:
                    msg, is_correct = self._process_answer(answer_val)
                    obs = msg
                    reward = 1.0 if is_correct else -1.0
                    terminated = True
        else:
            obs = f"Invalid action: {cmd}"
            reward = LanguageGameReward.invalid_action_reward
            terminated = True

        # 记录奖励与结束标记
        self._reward = reward if terminated else 0.0
        self._finished = terminated

        # 超时检查（统一放在 step 结尾）
        if not terminated and self.turn_count >= self.max_turns:
            obs = f"{obs}\nReached max turns ({self.max_turns})."
            reward = 0.0
            terminated = True
            truncated = True
            self._reward = reward
            self._finished = terminated

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
        return {"content": content}

    def sample_random_action(self) -> str:
        if self.m > 0:
            return "\\boxed{get_participant_solutions 0}"
        return "\\boxed{observe}"

    # --------------------------
    # 保留原环境的辅助方法并转换
    # --------------------------
    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def reward(self):
        return float(self._reward)

    def get_ref_answer(self) -> int:
        count = 0
        for i in range(self.m):
            for j in range(i + 1, self.m):
                for k in range(self.n):
                    if self.results[i][k] == 1 and self.results[j][k] == 1:
                        count += 1
                        break
        return count

    def GetParticipantCount(self) -> str:
        r"""
        Get the total number of participants.

        Returns:
            str: The total number of participants.
        Example Output:
            "3"
        """
        return str(self.m)

    def GetProblemCount(self) -> str:
        r"""
        Get the total number of problems.

        Returns:
            str: The total number of problems.
        Example Output:
            "3"
        """
        return str(self.n)

    def GetParticipantSolutions(self, participant_id: int) -> str:
        r"""
        Get the list of problems solved by the specified participant.

        Args:
            participant_id (int): Participant ID, starting from 0.

        Returns:
            str: A list containing the participant's solution status, where 1 indicates solved and 0 indicates unsolved.
        Example Output:
            "[1, 0, 1]"
        """
        if 0 <= participant_id < self.m:
            return str(self.results[participant_id])
        else:
            return "Error: Invalid participant_id"

    def CheckSharedProblem(self, participant1_id: int, participant2_id: int) -> str:
        r"""
        Check if two participants have any commonly solved problems.

        Args:
            participant1_id (int): ID of the first participant, starting from 0.
            participant2_id (int): ID of the second participant, starting from 0.

        Returns:
            str: "True" means there are commonly solved problems, "False" means there are none.
        Example Output:
            "True"
        """
        if 0 <= participant1_id < self.m and 0 <= participant2_id < self.m:
            for k in range(self.n):
                if self.results[participant1_id][k] == 1 and self.results[participant2_id][k] == 1:
                    return "True"
            return "False"
        else:
            return "Error: Invalid participant_id"

    def Observe(self) -> str:
        r"""
        Return basic information about the current environment.

        Returns:
            str: A prompt message describing the state of the environment.
        Example Output:
            "Programming competition result analysis environment: Please find the number of participant pairs that have solved at least one problem together"
        """
        return (
            "Programming competition result analysis environment: "
            "Please find the number of participant pairs that have solved at least one problem together"
        )

    def Done(self, answer: int) -> str:
        r"""
        Verify whether the final answer is correct and return result information.

        Args:
            answer (int): The answer submitted by the user, i.e., the number of participant pairs that meet the conditions.

        Returns:
            str: Result information, including whether it is correct and reward information.
        Example Output:
            "Your answer: 3, Reference answer: 3, Result: Correct, reward=1"
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        # 在 GEM 框架中，奖励由 step 控制，这里仅返回描述
        return msg + f", reward={'1' if correct else '0'}"

    def _process_answer(self, answer: int) -> Tuple[str, bool]:
        """内部处理 'answer' 指令并返回消息与是否正确"""
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg, correct

    @staticmethod
    def from_env_str(env_str: str, **kwargs) -> "SharedProblemPairsEnvGEM":
        """
        从旧环境字符串构建 GEM 环境，并用该字符串中的 m, n, results 初始化问题实例。
        兼容旧格式：'SharedProblemPairsEnv@{"m": M, "n": N, "results": [[...], ...]}'。
        """
        prefix = "SharedProblemPairsEnv@"
        if not env_str.startswith(prefix):
            raise ValueError("Invalid env_str prefix.")
        import ast
        options = ast.literal_eval(env_str.split("@")[1])
        env = SharedProblemPairsEnvGEM(**kwargs)
        # 直接覆盖生成的问题实例
        env.m = int(options.get("m", 0))
        env.n = int(options.get("n", 0))
        env.results = options.get("results", [])
        env.problem = {"m": env.m, "n": env.n, "results": env.results}
        # 重置计数但不重新生成
        env.turn_count = 0
        env.step_count = 0
        env._reward = 0.0
        env._finished = False
        return env

    def solve(self) -> str:
        """
        自动调用动作来完成流程，并提交答案进行验证。
        返回最终答案验证的结果信息（字符串）。
        """
        # 获取参与者数量
        obs, _, term, _, _ = self.step("\\boxed{get_participant_count}")
        if term:
            return obs
        try:
            participant_count = int(obs)
        except Exception:
            return "Error: Failed to parse participant count."

        count = 0
        for i in range(participant_count):
            for j in range(i + 1, participant_count):
                obs, _, term, _, _ = self.step(f"\\boxed{ { 'check_shared_problem ' + str(i) + ' ' + str(j) } }")
                # 修正生成格式
                obs, _, term, _, _ = self.step(f"\\boxed{{check_shared_problem {i} {j}}}")
                if term and (obs.startswith("Format error") or obs.startswith("Invalid action")):
                    return obs
                if obs == "True":
                    count += 1

        obs, _, _, _, _ = self.step(f"\\boxed{{answer {count}}}")
        return obs