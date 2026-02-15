from typing import Any, Dict, Optional, Tuple, List
import random
import re
import json
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class TaskManagerEnvGEM(Env):
    def __init__(
        self,
        complexity: int = 5,  # 难度等级 1-10，默认中等
        enable_param_randomization: bool = False,  # 评估时关闭随机化
        max_turns: Optional[int] = 100,
        **_,  # 忽略其他参数
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        # max_turns 将在 _apply_complexity_params 中根据难度设置，如需要也可被入参覆盖
        self.max_turns = max_turns if max_turns is not None else 100

        # 定义难度参数范围（根据原环境分析）
        # - num_ops: 操作数量（影响任务规模）
        # - max_task_id: 任务ID范围上限
        # - max_priority: 优先级范围上限（1..max_priority）
        # - complete_percentage: 完成操作的比例（百分比，影响负样例与动态性）
        # - max_turns_param: 步数限制（与 DungeonScout 建议保持一致）
        self.complexity_params = {
            "num_ops": (5, 50),
            "max_task_id": (10, 1000),
            "max_priority": (3, 50),
            "complete_percentage": (10, 60),  # 10% - 60%
            "max_turns_param": (20, 200),
        }

        # 参数方差（可选，用于微调随机性）
        self.param_variance = {
            "num_ops": 2,
            "max_task_id": 20,
            "max_priority": 2,
            "complete_percentage": 5,
            "max_turns_param": 10,
        }

        # 占位属性
        self.num_ops: int = 0
        self.max_task_id: int = 0
        self.max_priority: int = 0
        self.complete_percentage: int = 0
        self.max_turns_param: int = 0

        # 原环境状态
        self.task_operations: List[Dict[str, Any]] = []
        self.tasks: List[Tuple[int, int, int]] = []  # (priority, taskId, timestamp)
        self.timestamp: int = 0

        # GEM 状态
        self.turn_count: int = 0

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

        # 根据难度参数更新 max_turns
        self.max_turns = int(round(self.max_turns_param))

    def _get_instructions(self) -> str:
        return (
            "Task Manager: Maintain tasks with priorities and submit final ordering.\n"
            "Available actions (wrap in \\boxed{...}):\n"
            "- Observe operations: \\boxed{observe}\n"
            "- Add a task: \\boxed{add TASK_ID PRIORITY}\n"
            "- Complete a task: \\boxed{complete TASK_ID}\n"
            "- Get current tasks by priority: \\boxed{get}\n"
            "- Submit final answer: \\boxed{answer [id1, id2, ...]}\n"
            "Notes:\n"
            "- PRIORITY is a positive integer (higher means higher priority).\n"
            "- Tasks with same priority are ordered by addition timestamp.\n"
        )

    def get_task_suffix(self) -> str:
        return (
            f"Ops: {len(self.task_operations)} | "
            f"Tasks: {len(self.tasks)} | "
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

        # 初始化状态
        self.tasks = []
        self.timestamp = 0
        self.turn_count = 0

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        ops: List[Dict[str, Any]] = []
        existing_ids = set()
        p_complete = self.complete_percentage / 100.0

        for _ in range(self.num_ops):
            if random.random() < p_complete:
                # complete 操作
                if existing_ids and random.random() < 0.8:
                    tid = random.choice(list(existing_ids))
                else:
                    # 20% 概率完成不存在的任务（生成负样例）
                    tid = random.randint(1, self.max_task_id)
                ops.append({"action": "complete", "taskId": tid})
                # 本地集合只在生成时用于选择，实际处理在运行时执行
                if tid in existing_ids:
                    existing_ids.remove(tid)
            else:
                # add 操作
                # 生成不一定唯一的 taskId（真实环境可出现重复 add 的情况，由 Complete 操作处理）
                tid = random.randint(1, self.max_task_id)
                prio = random.randint(1, self.max_priority)
                ops.append({"action": "add", "taskId": tid, "priority": prio})
                existing_ids.add(tid)

        self.task_operations = ops
        return {"ops": ops, "num_ops": len(ops)}

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

        cmd = parsed.get("cmd", "").lower()
        terminated = False
        truncated = False
        reward = 0.0
        obs = ""

        try:
            if cmd in ["observe"]:
                obs = self.Observe()

            elif cmd in ["add", "addtask"]:
                tid = parsed.get("taskId", None)
                prio = parsed.get("priority", None)
                if tid is None or prio is None:
                    obs = "Error: 'add' requires TASK_ID and PRIORITY."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    obs = self.AddTask(int(tid), int(prio))

            elif cmd in ["complete", "completetask"]:
                tid = parsed.get("taskId", None)
                if tid is None:
                    obs = "Error: 'complete' requires TASK_ID."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    obs = self.CompleteTask(int(tid))

            elif cmd in ["get", "gettasksbypriority"]:
                obs = self.GetTasksByPriority()

            elif cmd in ["answer", "done"]:
                ans_list = parsed.get("answer", None)
                if ans_list is None or not isinstance(ans_list, list):
                    obs = "Format error: 'answer' requires a JSON-like list, e.g., \\boxed{answer [1,2,3]}."
                    reward = LanguageGameReward.format_error_reward
                    terminated = True
                else:
                    msg, correct = self.Done(ans_list)
                    obs = msg
                    reward = 1.0 if correct else -1.0
                    terminated = True

            else:
                obs = f"Invalid action: {cmd}"
                reward = LanguageGameReward.invalid_action_reward
                terminated = True

        except Exception as e:
            obs = f"Error: {str(e)}"
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

        # Support multiple formats:
        # - "observe"
        # - "add 5 3" or "AddTask 5 3"
        # - "complete 5" or "CompleteTask 5"
        # - "get" or "GetTasksByPriority"
        # - "answer [1,2,3]" or "done [1,2,3]"
        tokens = content.split()
        if not tokens:
            return None

        cmd = tokens[0].lower()
        parsed: Dict[str, Any] = {"cmd": cmd}

        if cmd in ["observe", "get", "gettasksbypriority"]:
            return parsed

        if cmd in ["add", "addtask"]:
            if len(tokens) < 3:
                return None
            try:
                parsed["taskId"] = int(tokens[1])
                parsed["priority"] = int(tokens[2])
                return parsed
            except Exception:
                return None

        if cmd in ["complete", "completetask"]:
            if len(tokens) < 2:
                return None
            try:
                parsed["taskId"] = int(tokens[1])
                return parsed
            except Exception:
                return None

        if cmd in ["answer", "done"]:
            # Extract list portion after the command
            rest = content[len(tokens[0]):].strip()
            # Expect something like "[1,2,3]"
            try:
                # Allow JSON-like parsing
                ans = json.loads(rest) if rest.startswith("[") else json.loads(rest[rest.find("["):])
                if isinstance(ans, list):
                    parsed["answer"] = ans
                    return parsed
                return None
            except Exception:
                return None

        return None

    # ---------- 原环境动作实现（保留并适配） ----------

    def AddTask(self, taskId: int, priority: int) -> str:
        """
        Add a task with the specified task ID and priority.
        """
        self.tasks.append((priority, taskId, self.timestamp))
        self.timestamp += 1
        return f"Task {taskId} added with priority {priority}"

    def CompleteTask(self, taskId: int) -> str:
        """
        Complete and remove a task by its task ID.
        """
        initial_count = len(self.tasks)
        self.tasks = [task for task in self.tasks if task[1] != taskId]

        if len(self.tasks) < initial_count:
            return f"Task {taskId} completed and removed"
        else:
            return f"Task {taskId} not found"

    def GetTasksByPriority(self) -> str:
        """
        Retrieve the list of task IDs sorted by priority in descending order.
        Tasks with the same priority are sorted by their addition order.
        """
        sorted_tasks = sorted(self.tasks, key=lambda x: (-x[0], x[2]))
        task_ids = [task[1] for task in sorted_tasks]
        return json.dumps(task_ids)

    def Observe(self) -> str:
        """
        Get the current list of task operations that need to be processed.
        """
        return json.dumps(self.task_operations)

    def get_ref_answer(self) -> List[int]:
        """
        Calculate the reference answer based on current environment state (task_operations).
        """
        temp_tasks = []
        temp_timestamp = 0

        for op in self.task_operations:
            if op["action"] == "add":
                temp_tasks.append((op["priority"], op["taskId"], temp_timestamp))
                temp_timestamp += 1
            elif op["action"] == "complete":
                temp_tasks = [t for t in temp_tasks if t[1] != op["taskId"]]

        temp_tasks.sort(key=lambda x: (-x[0], x[2]))
        return [taskId for _, taskId, _ in temp_tasks]

    def Done(self, answer: List[int]) -> Tuple[str, bool]:
        """
        Submit the final answer and check against the reference answer.
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = (
            f"Your answer: {answer}, Reference answer: {ref_answer}, "
            f"Result: {'Correct' if correct else 'Incorrect'}"
        )
        return msg, correct

    def sample_random_action(self) -> str:
        # 示例动作：观察操作
        return "\\boxed{observe}"

    # ---------- 可选：自动求解（保留并适配 GEM 风格） ----------
    def solve(self) -> str:
        """
        Automatically process operations and submit the answer for verification.
        """
        # Observe operations
        obs, _, _, _, _ = self.step("\\boxed{observe}")
        try:
            operations = json.loads(obs)
        except Exception:
            operations = []

        # Process operations
        for op in operations:
            if op.get("action") == "add":
                self.step(f"\\boxed{{add {op['taskId']} {op['priority']}}}")
            elif op.get("action") == "complete":
                self.step(f"\\boxed{{complete {op['taskId']}}}")

        # Get sorted tasks
        sorted_tasks_json, _, _, _, _ = self.step("\\boxed{get}")
        try:
            sorted_tasks = json.loads(sorted_tasks_json)
        except Exception:
            sorted_tasks = []

        # Submit answer
        obs, _, _, _, _ = self.step(f"\\boxed{{answer {json.dumps(sorted_tasks)}}}")
        return obs