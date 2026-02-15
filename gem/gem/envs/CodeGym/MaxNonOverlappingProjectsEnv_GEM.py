from typing import Any, Dict, Optional, Tuple
import random
import re
import json
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class MaxNonOverlappingProjectsEnvGEM(Env):
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
        # array_length: 项目数量
        # value_range: 时间范围（start/end 的取值范围）
        # overlap_factor: 重叠偏好（越大平均持续时间越长、起始时间更集中，重叠更容易）
        # num_clusters: 起始时间聚类数（越少越集中，重叠更容易）
        self.complexity_params = {
            "array_length": (5, 50),
            "value_range": (10, 1000),
            "overlap_factor": (1, 5),
            "num_clusters": (1, 5),
        }

        # 参数方差（enable_param_randomization=True 时引入轻微扰动）
        self.param_variance = {
            "array_length": 3,
            "value_range": 50,
            "overlap_factor": 1,
            "num_clusters": 1,
        }

        # 占位属性
        self.array_length: int = 0
        self.value_range: int = 0
        self.overlap_factor: int = 0
        self.num_clusters: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 任务数据
        self.projects: list = []
        self.sorted_projects: Optional[list] = None
        self.last_end_time: float = float("-inf")
        self.selected_count: int = 0

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
            "Max Non-Overlapping Projects: Each project is [start, end] with start < end.\n"
            "Goal: Select maximum number of non-overlapping projects.\n"
            "Available actions:\n"
            "- Sort projects by end time: \\boxed{sort}\n"
            "- Try select project at index i (in sorted list): \\boxed{select i}\n"
            "- Observe current state: \\boxed{observe}\n"
            "- Submit your answer N: \\boxed{answer N}\n"
            "Notes:\n"
            "- You should sort before selecting.\n"
            "- Index i starts from 0.\n"
        )

    def get_task_suffix(self) -> str:
        sorted_status = "sorted" if self.sorted_projects is not None else "unsorted"
        return (
            f"Projects: {len(self.projects)} ({sorted_status}), "
            f"Selected: {self.selected_count}, "
            f"Turn: {self.turn_count}/{self.max_turns}\nEnter action."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        # 初始化状态
        self.projects = self.problem["projects"]
        self.sorted_projects = None
        self.last_end_time = float("-inf")
        self.selected_count = 0

        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成项目区间列表"""
        n = self.array_length
        T = self.value_range
        k = max(1, self.num_clusters)
        overlap = max(1, self.overlap_factor)

        # 确定持续时间范围，overlap 越大，平均持续时间越长
        # 使得更容易发生重叠
        base_frac_min = 0.03
        base_frac_max = 0.15
        # 放大上界以增加重叠
        dur_frac_max = base_frac_max + 0.06 * (overlap - 1)  # up to ~0.39 when overlap=5
        dur_min = max(1, int(T * base_frac_min))
        dur_max = max(dur_min + 1, int(T * dur_frac_max))

        # 生成聚类中心
        centers = []
        for _ in range(k):
            centers.append(random.randint(0, T - 1))

        projects = []
        for _ in range(n):
            # 选择一个簇
            c = random.choice(centers)
            # 生成起始时间（围绕簇中心），加入噪声并裁剪到范围内
            noise = int(random.gauss(0, max(1, T * 0.05)))  # 5% 范围标准差
            start = c + noise
            # 生成持续时间
            duration = random.randint(dur_min, dur_max)
            start = max(0, min(T - 1, start))
            end = start + duration
            if end >= T:
                # 如超出范围，尝试调整 start
                start = max(0, T - duration - 1)
                end = start + duration
            # 确保 start < end
            if end <= start:
                end = start + 1
            projects.append([int(start), int(end)])

        # 打乱以保证原列表非排序
        random.shuffle(projects)
        return {"projects": projects, "time_range": T, "clusters": k, "overlap_factor": overlap}

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

        cmd = parsed["command"]
        args = parsed.get("args", [])
        terminated = False
        truncated = False
        reward = 0.0
        obs = ""

        try:
            if cmd == "sort":
                obs = self.SortProjectsByEnd()

            elif cmd == "select":
                if len(args) != 1:
                    obs = "Error: select requires one integer index, e.g., \\boxed{select 3}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    try:
                        idx = int(args[0])
                    except Exception:
                        obs = "Error: index must be an integer."
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                    else:
                        msg = self.SelectNextProject(idx)
                        # 如果返回错误信息，视为无效动作
                        if msg.startswith("Error:"):
                            obs = msg
                            reward = LanguageGameReward.invalid_action_reward
                            terminated = True
                        else:
                            obs = msg

            elif cmd == "observe":
                obs = self.Observe()

            elif cmd == "answer":
                if len(args) != 1:
                    obs = "Error: answer requires one integer N, e.g., \\boxed{answer 4}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    try:
                        count = int(args[0])
                    except Exception:
                        obs = "Error: answer value must be an integer."
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                    else:
                        msg, is_correct = self.Done(count, return_correct_flag=True)
                        obs = msg
                        reward = 1.0 if is_correct else -1.0
                        terminated = True

            else:
                obs = f"Invalid action: {cmd}"
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
        if not content:
            return None
        tokens = content.split()
        cmd = tokens[0].lower()
        args = tokens[1:]
        return {"command": cmd, "args": args}

    def sample_random_action(self) -> str:
        return "\\boxed{sort}"

    # ===== 原环境辅助方法（转换后保留） =====

    def get_ref_answer(self) -> int:
        """
        使用环境中的项目信息获取参考答案（最大不重叠项目数量）。
        """
        if not self.projects:
            return 0

        sorted_projects = sorted(self.projects, key=lambda x: x[1])
        count = 0
        end_time = float("-inf")

        for project in sorted_projects:
            if project[0] > end_time:
                count += 1
                end_time = project[1]

        return count

    def SortProjectsByEnd(self) -> str:
        """
        按项目结束时间排序，并返回排序后的列表字符串（JSON）。
        """
        self.sorted_projects = sorted(self.projects, key=lambda x: x[1])
        return json.dumps(self.sorted_projects)

    def SelectNextProject(self, index: int) -> str:
        """
        尝试选择排序列表中的指定索引项目；若与已选项目不重叠则选取。
        """
        if self.sorted_projects is None:
            return "Error: Please sort the projects first"

        if index < 0 or index >= len(self.sorted_projects):
            return "Error: Index out of range"

        project = self.sorted_projects[index]
        if project[0] > self.last_end_time:
            self.selected_count += 1
            self.last_end_time = project[1]
            return f"Successfully selected project {project}, currently selected {self.selected_count} projects"
        else:
            return (
                f"Cannot select project {project}, it overlaps with selected projects, "
                f"currently selected {self.selected_count} projects"
            )

    def Observe(self) -> str:
        """
        返回当前环境的观察信息，包括原始项目列表、排序状态以及当前选择数量。
        """
        sorted_status = "projects are sorted" if self.sorted_projects is not None else "projects are not sorted"
        return f"Original project list: {self.projects}, {sorted_status}, number of selected projects: {self.selected_count}"

    def Done(self, count: int, return_correct_flag: bool = False):
        """
        提交最终答案并验证其正确性。
        返回消息及（可选）是否正确的布尔值。
        """
        ref_answer = self.get_ref_answer()
        correct = count == ref_answer
        msg = f"Your answer: {count}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        if return_correct_flag:
            return msg, correct
        return msg

    def solve(self) -> str:
        """
        自动调用动作完成任务并提交答案进行验证。
        """
        # sort
        obs, _, _, _, _ = self.step("\\boxed{sort}")
        # 解析排序结果长度
        try:
            sorted_projects = json.loads(obs)
            project_count = len(sorted_projects)
        except Exception:
            # 如果解析失败，使用内部状态兜底
            sorted_projects = self.sorted_projects or []
            project_count = len(sorted_projects)

        # select all in order
        for i in range(project_count):
            self.step(f"\\boxed{{select {i}}}")

        # observe to read selected count
        observe_info, _, _, _, _ = self.step("\\boxed{observe}")
        count_match = re.search(r'number of selected projects: (\d+)', observe_info)
        if count_match:
            max_count = int(count_match.group(1))
        else:
            # 兜底使用内部计数
            max_count = self.selected_count

        # answer
        final_obs, _, _, _, _ = self.step(f"\\boxed{{answer {max_count}}}")
        return final_obs