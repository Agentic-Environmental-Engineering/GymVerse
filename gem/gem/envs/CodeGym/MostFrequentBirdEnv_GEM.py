from typing import Any, Dict, Optional, Tuple
import random
import re
import json
import ast
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class MostFrequentBirdEnvGEM(Env):
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

        # 难度参数范围设计（与原环境相关）
        # - array_length: 鸟类观测列表长度
        # - value_range: 鸟类 ID 值域（1..value_range）
        # - turn_budget: 最大可用步数预算（受 complexity 控制）
        self.complexity_params = {
            "array_length": (5, 50),
            "value_range": (3, 100),
            "turn_budget": (20, 200),
        }

        # 参数方差（enable_param_randomization=True 时生效）
        self.param_variance = {
            "array_length": 2,
            "value_range": 3,
            "turn_budget": 10,
        }

        # 占位属性（将由 _apply_complexity_params 填充）
        self.array_length: int = 0
        self.value_range: int = 0
        self.turn_budget: int = 0

        # 状态变量
        self.turn_count: int = 0
        self.sightings: list = []
        self.problem: Dict[str, Any] = {}

        # 与原环境一致的内部状态（用于辅助方法）
        self._reward: float = 0.0
        self._done: bool = False

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

        # 应用复杂度驱动的步数预算到 max_turns（取更紧的限制）
        self.max_turns = min(self.max_turns, self.turn_budget)

    def _get_instructions(self) -> str:
        return (
            "Most Frequent Bird: Find the smallest bird ID with the highest sighting frequency.\n"
            "You can inspect the sightings and compute frequencies.\n"
            "Available actions (use the latest \\boxed{...} block):\n"
            "- Observe current list: \\boxed{observe}\n"
            "- List unique IDs: \\boxed{list}\n"
            "- Count a specific ID: \\boxed{count ID}\n"
            "- Compute max frequency from counts: \\boxed{maxfreq counts=[c1,c2,...]}\n"
            "- Find min ID among those with max freq: \\boxed{minid maxfreq=F ids=[i1,...] counts=[c1,...]}\n"
            "- Submit final answer: \\boxed{answer N}\n"
        )

    def get_task_suffix(self) -> str:
        return (
            f"Sightings length: {len(self.sightings)}\n"
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
        self.sightings = self.problem["sightings"]

        self.turn_count = 0
        self._reward = 0.0
        self._done = False
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        # 生成随机鸟类观测列表
        sightings = [random.randint(1, self.value_range) for _ in range(self.array_length)]
        return {"sightings": sightings}

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
        obs = "Action processed."
        reward = 0.0
        terminated = False
        truncated = False

        try:
            # 解析命令
            lowered = content.strip().lower()

            if lowered.startswith("observe"):
                obs = self.Observe()

            elif lowered.startswith("list"):
                obs = self.GetAllBirdIds()

            elif lowered.startswith("count"):
                m = re.match(r"count\s+(-?\d+)", content, re.IGNORECASE)
                if not m:
                    obs = "Error: Missing or invalid parameter for 'count'. Usage: \\boxed{count ID}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    bird_id = int(m.group(1))
                    obs = self.CountBirdSightings(bird_id)

            elif lowered.startswith("maxfreq"):
                counts = self._parse_list_param(content, "counts")
                if counts is None or len(counts) == 0:
                    obs = "Error: 'counts' parameter is missing or empty for maxfreq."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    obs = self.FindMaxFrequency(counts)

            elif lowered.startswith("minid"):
                max_freq = self._parse_int_param(content, "maxfreq")
                bird_ids = self._parse_list_param(content, "ids")
                counts = self._parse_list_param(content, "counts")
                if max_freq is None or bird_ids is None or counts is None:
                    obs = "Error: 'maxfreq', 'ids', or 'counts' parameter is missing for minid."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                elif len(bird_ids) != len(counts):
                    obs = "Error: Length mismatch between 'ids' and 'counts'."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    obs = self.FindMinIdWithMaxFrequency(bird_ids, counts, max_freq)

            elif lowered.startswith("answer"):
                m = re.match(r"answer\s+(-?\d+)", content, re.IGNORECASE)
                if not m:
                    obs = "Error: Missing or invalid parameter for 'answer'. Usage: \\boxed{answer N}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    answer = int(m.group(1))
                    ref_answer = self.get_ref_answer()
                    obs = self.Done(answer)
                    reward = 1.0 if answer == ref_answer else -1.0
                    terminated = True

            else:
                obs = f"Invalid action: '{content}'."
                reward = LanguageGameReward.invalid_action_reward
                terminated = True

        except Exception as e:
            obs = f"Error: {str(e)}"
            reward = LanguageGameReward.invalid_action_reward
            terminated = True

        # 超时检查（统一放在 step 结尾）
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
        return {"content": content}

    def _parse_list_param(self, content: str, name: str) -> Optional[list]:
        """解析形如 name=[a,b,c] 的参数为整数列表"""
        m = re.search(rf"{name}\s*=\s*\[([^\]]*)\]", content, re.IGNORECASE)
        if not m:
            return None
        inner = m.group(1).strip()
        if inner == "":
            return []
        try:
            items = [int(x.strip()) for x in inner.split(",") if x.strip() != ""]
            return items
        except Exception:
            return None

    def _parse_int_param(self, content: str, name: str) -> Optional[int]:
        """解析形如 name=42 的整数参数"""
        m = re.search(rf"{name}\s*=\s*(-?\d+)", content, re.IGNORECASE)
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    def sample_random_action(self) -> str:
        if not self.sightings:
            return "\\boxed{observe}"
        choice = random.choice(["observe", "list", "count", "answer"])
        if choice == "count":
            bird_id = random.randint(1, self.value_range)
            return f"\\boxed{{count {bird_id}}}"
        elif choice == "list":
            return "\\boxed{list}"
        elif choice == "answer":
            # naive guess: pick a random id
            return f"\\boxed{{answer {random.randint(1, self.value_range)}}}"
        else:
            return "\\boxed{observe}"

    # ------------------------
    # 以下为原环境的辅助方法（保留并转换）
    # ------------------------
    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        bird_count = {}
        for bird in self.sightings:
            if bird in bird_count:
                bird_count[bird] += 1
            else:
                bird_count[bird] = 1
        return min([k for k, v in bird_count.items() if v == max(bird_count.values())])

    def CountBirdSightings(self, bird_id: int):
        r"""
        Count the number of sightings for a specific bird ID.
        Returns:
            str: The number of sightings of the bird ID.
        Example Output:
            "3"
        """
        count = self.sightings.count(bird_id)
        return str(count)

    def GetAllBirdIds(self):
        r"""
        Get a list of all bird IDs that have appeared, without duplicates and sorted in ascending order.
        Returns:
            str: A list of all bird IDs that have appeared.
        Example Output:
            "[1, 2, 3]"
        """
        unique_ids = sorted(list(set(self.sightings)))
        return json.dumps(unique_ids)

    def FindMaxFrequency(self, counts: list):
        r"""
        Find the maximum frequency value from the count list.
        Returns:
            str: The maximum frequency value.
        Example Output:
            "2"
        """
        max_freq = max(counts)
        return str(max_freq)

    def FindMinIdWithMaxFrequency(self, bird_ids: list, counts: list, max_freq: int):
        r"""
        Find the smallest bird ID with the maximum frequency.
        Returns:
            str: The smallest bird ID with the maximum frequency.
        Example Output:
            "1"
        """
        candidates = [bird_ids[i] for i in range(len(bird_ids)) if counts[i] == max_freq]
        return str(min(candidates))

    def Observe(self):
        r"""
        Return observation information of the current state, including the list of bird sightings.
        Returns:
            str: A prompt message describing the current state.
        Example Output:
            "Current bird sightings list: [1, 1, 2, 2, 3, 3]"
        """
        return f"Current bird sightings list: {self.sightings}"

    def Done(self, answer):
        r"""
        Verify whether the final answer is correct and return result information.
        Args:
            answer (int): The answer submitted by the user, i.e., the most frequently occurring bird ID.
        Returns:
            str: Result information, including whether it is correct and reward information.
        Example Output:
            "Your answer: 1, Reference answer: 1, Result: Correct, reward=1"
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        self._reward = 1 if correct else 0
        self._done = True
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg + f", reward={self._reward}"

    def solve(self) -> str:
        r"""
        Automatically call all actions to complete the process, and submit the answer for verification. 
        Returns:
            str: The result information of the final answer verification. 
        """
        # List unique IDs
        obs, _, _, _, _ = self.step("\\boxed{list}")
        try:
            bird_ids = ast.literal_eval(obs)
        except Exception:
            try:
                bird_ids = json.loads(obs)
            except Exception:
                # Fallback: parse digits
                bird_ids = []

        # Count each ID
        counts = []
        for bird_id in bird_ids:
            obs, _, _, _, _ = self.step(f"\\boxed{{count {bird_id}}}")
            try:
                counts.append(int(obs))
            except Exception:
                counts.append(0)

        # Find max frequency
        obs, _, _, _, _ = self.step(f"\\boxed{{maxfreq counts={[int(c) for c in counts]}}}")
        try:
            max_freq = int(obs)
        except Exception:
            max_freq = max(counts) if counts else 0

        # Find min ID among those with max frequency
        obs, _, _, _, _ = self.step(
            f"\\boxed{{minid maxfreq={max_freq} ids={bird_ids} counts={[int(c) for c in counts]}}}"
        )
        try:
            answer = int(obs)
        except Exception:
            answer = self.get_ref_answer()

        # Submit final answer
        obs, _, _, _, _ = self.step(f"\\boxed{{answer {answer}}}")
        return obs