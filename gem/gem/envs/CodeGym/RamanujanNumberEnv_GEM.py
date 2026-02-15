from typing import Any, Dict, Optional, Tuple
import random
import re
import math
from collections import defaultdict
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class RamanujanNumberEnvGEM(Env):
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

        # 根据原环境的结构设计难度参数
        # value_range: N 的数值范围上限（越大越难）
        # search_space: i, j 的搜索上限（越大越难）
        # predicate_terms: 需要的不同分解数量（Ramanujan数要求至少 2）
        self.complexity_params = {
            "value_range": (1000, 50000),
            "search_space": (20, 300),
            "predicate_terms": (2, 2),
        }

        # 参数方差（在 enable_param_randomization=True 时微扰中心值）
        self.param_variance = {
            "value_range": 2000,
            "search_space": 10,
            "predicate_terms": 0,
        }

        # 占位属性（由 _apply_complexity_params 填充）
        self.value_range: int = 0
        self.search_space: int = 0
        self.predicate_terms: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 原环境中的状态
        self.N: int = 0
        self.cubes: Dict[int, list] = defaultdict(list)
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

    def _get_instructions(self) -> str:
        return (
            "Ramanujan Number: Determine whether target N can be expressed as a sum of two cubes in at least two distinct ways.\n"
            "Available actions:\n"
            "- Observe current state: \\boxed{observe}\n"
            "- Calculate cube sum: \\boxed{calc i j}  (returns i^3 + j^3)\n"
            "- Store a pair for N: \\boxed{store sum i j}  (stores (i,j) if sum == N and sum <= N)\n"
            "- Check Ramanujan: \\boxed{check}  (returns 'Yes a b c d' or 'No')\n"
            "- Submit final answer: \\boxed{answer <Yes a b c d|No>}\n"
        )

    def get_task_suffix(self) -> str:
        return f"N: {self.N}\nTurn: {self.turn_count}/{self.max_turns}\nStored sums: {len(self.cubes)}\nEnter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响实例生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        problem = self._generate_random_problem()
        self.N = problem["N"]

        # 重置原环境状态
        self.cubes = defaultdict(list)
        self._reward = 0.0
        self._done = False

        # 状态变量
        self.turn_count = 0

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        # 限制搜索空间以便构造候选 N
        max_i = min(int(round(self.value_range ** (1.0 / 3.0))) + 2, self.search_space)

        sum_map: Dict[int, list] = defaultdict(list)
        for i in range(1, max_i + 1):
            i3 = i * i * i
            for j in range(i, max_i + 1):
                s = i3 + j * j * j
                if s > self.value_range:
                    break
                sum_map[s].append((i, j))

        # 找出具有至少两组分解的候选（Ramanujan数）
        ramanujan_candidates = [s for s, pairs in sum_map.items() if len(pairs) >= self.predicate_terms]

        # 以一定概率生成 Ramanujan 或非 Ramanujan 的 N
        choose_ramanujan = True if not sum_map else (random.random() < 0.6 and len(ramanujan_candidates) > 0)

        if choose_ramanujan and ramanujan_candidates:
            N = random.choice(ramanujan_candidates)
        else:
            # 选择一个不满足条件的数（没有两种分解）
            non_candidates = [s for s, pairs in sum_map.items() if len(pairs) < self.predicate_terms]
            if non_candidates:
                N = random.choice(non_candidates)
            else:
                # 回退：直接随机选择值范围内的数
                N = random.randint(max(10, self.value_range // 10), self.value_range)

        return {"N": int(N)}

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
        tokens = content.strip().split()
        if len(tokens) == 0:
            obs = f"Invalid action at turn {self.turn_count}."
            return (
                obs,
                LanguageGameReward.invalid_action_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        cmd = tokens[0].lower()
        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            if cmd == "observe":
                obs = self.Observe()
                reward = 0.0

            elif cmd == "calc":
                if len(tokens) != 3:
                    obs = f"Invalid calc syntax. Use: \\boxed{{calc i j}}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    i = int(tokens[1])
                    j = int(tokens[2])
                    obs = self.CalculateCubeSum(i, j)
                    reward = 0.0

            elif cmd == "store":
                # Expect: store sum i j
                if len(tokens) != 4 or tokens[1].lower() != "sum":
                    obs = f"Invalid store syntax. Use: \\boxed{{store sum i j}}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    i = int(tokens[2])
                    j = int(tokens[3])
                    # For store, calculate sum first then store
                    sum_val = i**3 + j**3
                    obs = self.StoreCubeSum(sum_val, i, j)
                    reward = 0.0

            elif cmd == "check":
                if len(tokens) != 1:
                    obs = f"Invalid check syntax. Use: \\boxed{{check}}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    obs = self.CheckRamanujan()
                    reward = 0.0

            elif cmd == "answer":
                # Accept everything after 'answer ' as raw answer string
                if len(tokens) < 2:
                    obs = f"Invalid answer syntax. Use: \\boxed{{answer <Yes a b c d|No>}}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    raw_answer = content[len("answer"):].strip()
                    result_msg = self.Done(raw_answer)
                    # Evaluate correctness from Done()
                    ref_answer = self.get_ref_answer()
                    correct = (raw_answer == ref_answer)
                    reward = 1.0 if correct else -1.0
                    obs = result_msg
                    terminated = True

            else:
                obs = f"Invalid action '{cmd}'."
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
        choice = random.choice(["observe", "check", "calc", "store"])
        if choice == "observe":
            return "\\boxed{observe}"
        elif choice == "check":
            return "\\boxed{check}"
        elif choice == "calc":
            i = random.randint(1, max(2, min(self.search_space, int(round(self.value_range ** (1.0 / 3.0))))))
            j = random.randint(i, max(2, min(self.search_space, int(round(self.value_range ** (1.0 / 3.0))))))
            return f"\\boxed{{calc {i} {j}}}"
        else:
            i = random.randint(1, max(2, min(self.search_space, int(round(self.value_range ** (1.0 / 3.0))))))
            j = random.randint(i, max(2, min(self.search_space, int(round(self.value_range ** (1.0 / 3.0))))))
            return f"\\boxed{{store sum {i} {j}}}"

    # ----------------------------
    # 以下为原环境的辅助方法（已转换为当前环境使用）
    # ----------------------------
    @property
    def finished(self) -> bool:
        return self._done

    @property
    def reward(self):
        return float(self._reward)

    def CalculateCubeSum(self, i: int, j: int):
        r"""
        Calculate the sum of the cubes of i and j.

        Args:
            i (int): The first integer
            j (int): The second integer

        Returns:
            str: The calculation result of i³ + j³

        Example Output:
            "1729"
        """
        sum_val = i**3 + j**3
        return str(sum_val)

    def StoreCubeSum(self, sum_val: int, i: int, j: int):
        r"""
        Store the cube sum and its corresponding pair of numbers.

        Args:
            sum_val (int): The cube sum
            i (int): The first integer
            j (int): The second integer

        Returns:
            str: A message indicating successful storage, including the cube sum and the number pair

        Example Output:
            "Stored: 1729 -> (1, 12)"
        """
        if sum_val <= self.N and sum_val == self.N:
            self.cubes[sum_val].append((i, j))
            return f"Stored: {sum_val} -> ({i}, {j})"
        if sum_val < self.N:
            return f"Not stored: {sum_val} < {self.N}"
        return f"Not stored: {sum_val} > {self.N}"

    def CheckRamanujan(self):
        r"""
        Check if there exists a combination of Ramanujan numbers among the currently stored cube sums.

        Args:
            None

        Returns:
            str: The check result, returns a string containing four integers if they exist, otherwise returns "No"

        Example Output:
            "Yes 1 12 9 10" or "No"
        """
        if self.N in self.cubes and len(self.cubes[self.N]) > 1:
            a, b = self.cubes[self.N][0]
            c, d = self.cubes[self.N][1]
            return f"Yes {a} {b} {c} {d}"
        else:
            return "No"

    def Observe(self):
        r"""
        Return the observation information of the current environment, including the number N to be checked and the quantity of stored cube sums.

        Args:
            None

        Returns:
            str: A description of the current environment state

        Example Output:
            "Currently checked number: 1729, Number of stored cube sums: 5"
        """
        return f"Currently checked number: {self.N}, Number of stored cube sums: {len(self.cubes)}"

    def Done(self, answer):
        r"""
        Submit the final answer and verify its correctness.

        Args:
            answer (str): The submitted answer, in the format "Yes a b c d" or "No"

        Returns:
            str: Result information, including whether it is correct and reward information

        Example Output:
            "Your answer: Yes 1 12 9 10, Reference answer: Yes 1 12 9 10, Result: Correct, reward=1"
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        self._reward = 1 if correct else 0
        self._done = True
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg + f", reward={self._reward}"

    def get_ref_answer(self):
        cubes = defaultdict(list)
        max_i = int(pow(self.N, 1/3)) + 1

        for i in range(1, max_i):
            for j in range(i, max_i):
                sum_of_cubes = i**3 + j**3
                if sum_of_cubes > self.N:
                    break
                cubes[sum_of_cubes].append((i, j))

        if self.N in cubes and len(cubes[self.N]) > 1:
            a, b = cubes[self.N][0]
            c, d = cubes[self.N][1]
            return f"Yes {a} {b} {c} {d}"
        else:
            return "No"

    def solve(self) -> str:
        r"""
        Automatically calls actions to determine whether N is a Ramanujan number, and submits the answer for verification.

        Returns:
            str: The result information of the final answer verification.
        """
        observe_info, _, _, _, _ = self.step("\\boxed{observe}")
        import re as _re
        n_match = _re.search(r'Currently checked number: (\d+)', observe_info)
        N = int(n_match.group(1)) if n_match else 0

        max_num = int(N ** (1/3)) + 2

        for i in range(1, max_num + 1):
            for j in range(i, max_num + 1):  # i <= j to reduce duplicate calculations and storage
                calc_obs, _, _, _, _ = self.step(f"\\boxed{{calc {i} {j}}}")
                try:
                    sum_val = int(calc_obs.strip())
                except:
                    continue
                if sum_val == N:
                    self.step(f"\\boxed{{store sum {i} {j}}}")

        check_result, _, _, _, _ = self.step("\\boxed{check}")
        final_obs, _, _, _, _ = self.step(f"\\boxed{{answer {check_result}}}")
        return final_obs