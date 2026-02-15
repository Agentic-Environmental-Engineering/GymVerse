from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class TreeCheckEnvGEM(Env):
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
            "num_nodes": (5, 50),           # 节点数量
            "edge_weight_max": (10, 10000), # 边权上限
            "non_tree_types": (1, 4),       # 非树构造类型的可用种类数（1-4种）
        }

        # 参数方差（仅在 enable_param_randomization=True 时生效）
        self.param_variance = {
            "num_nodes": 3,
            "edge_weight_max": 100,
            "non_tree_types": 1,
        }

        # 占位属性（由 _apply_complexity_params 填充）
        self.num_nodes: int = 0
        self.edge_weight_max: int = 0
        self.non_tree_types: int = 0

        # 状态变量
        self.turn_count: int = 0
        self._terminated: bool = False

        # 问题实例相关
        self.n: int = 0
        self.edges: list = []
        self.parent: list = []

        # 内部统计/缓存
        self._last_obs: str = ""

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
            "Tree Check: Determine if the given undirected graph (weights ignored) is a tree.\n"
            "A graph is a tree iff it is connected and has exactly n-1 edges (no cycles).\n"
            "Available actions (wrap in \\boxed{...}):\n"
            "- Observe problem: \\boxed{observe}\n"
            "- Check edge count equals M: \\boxed{check M}\n"
            "- Initialize Union-Find with size N: \\boxed{inituf N}\n"
            "- Find root of node X: \\boxed{find X}\n"
            "- Union nodes X and Y: \\boxed{union X Y}\n"
            "- Submit final answer: \\boxed{answer YES} or \\boxed{answer NO}\n"
            "- Show help: \\boxed{help}\n"
        )

    def get_task_suffix(self) -> str:
        return f"Nodes: {self.n}, Edges: {len(self.edges)}, Turn: {self.turn_count}/{self.max_turns}\nEnter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 仅影响实例生成，不影响难度参数

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()
        self.n = self.problem["n"]
        self.edges = self.problem["edges"]

        self.parent = []  # 需用户通过 inituf 初始化
        self.turn_count = 0
        self._terminated = False
        self._last_obs = self._get_instructions()
        return self._last_obs, {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self) -> Dict[str, Any]:
        """根据难度参数随机生成问题实例"""
        n = max(1, int(self.num_nodes))
        # 先生成一棵随机生成树
        edges = []
        if n >= 2:
            for v in range(2, n + 1):
                u = random.randint(1, v - 1)
                w = random.randint(1, max(1, self.edge_weight_max))
                edges.append((u, v, w))

        # 难度越高，越可能生成非树
        prob_non_tree = min(1.0, (self.complexity - 1) / 9.0)
        make_non_tree = random.random() < prob_non_tree

        if make_non_tree:
            # 可用的非树类型
            types = ["cycle", "disconnected", "duplicate", "selfloop"]
            types = types[: max(1, self.non_tree_types)]
            chosen = random.choice(types)

            if chosen == "cycle":
                # 在树上加一条额外边，形成环
                # 随机选一对不同节点，且尽量不与已有边重复
                attempts = 0
                existing = {tuple(sorted((u, v))) for u, v, _ in edges}
                while attempts < 50:
                    u = random.randint(1, n)
                    v = random.randint(1, n)
                    if u != v and tuple(sorted((u, v))) not in existing:
                        w = random.randint(1, max(1, self.edge_weight_max))
                        edges.append((u, v, w))
                        break
                    attempts += 1
                if attempts >= 50 and n >= 2:
                    # 兜底：重复边（也会形成非树）
                    u, v, _ = random.choice(edges)
                    w = random.randint(1, max(1, self.edge_weight_max))
                    edges.append((u, v, w))

            elif chosen == "disconnected":
                # 移除一条边，导致非连通（少于 n-1 条边）
                if edges:
                    remove_idx = random.randrange(len(edges))
                    edges.pop(remove_idx)
                # 也可能添加一条边在同一分量内部（保持非连通不可保证，这里不添加也可）
                # 只需不是 n-1 条边即可确保不是树

            elif chosen == "duplicate":
                # 添加一条重复边（同一对端点），会导致并查集发现环
                if edges:
                    u, v, _ = random.choice(edges)
                    w = random.randint(1, max(1, self.edge_weight_max))
                    edges.append((u, v, w))
                elif n >= 2:
                    u, v = 1, 2
                    w = random.randint(1, max(1, self.edge_weight_max))
                    edges.append((u, v, w))
                    edges.append((u, v, w))

            elif chosen == "selfloop":
                # 添加自环，必非树
                x = random.randint(1, n)
                w = random.randint(1, max(1, self.edge_weight_max))
                edges.append((x, x, w))

        return {"n": n, "edges": edges}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        # 如果已经结束，阻止继续交互
        if self._terminated:
            obs = "Episode has finished. Please reset to start a new game."
            return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}

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

        content: str = parsed["content"]
        tokens = content.strip().split()
        if not tokens:
            obs = "Empty action."
            return (
                obs,
                LanguageGameReward.invalid_action_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        cmd = tokens[0].lower()
        terminated = False
        truncated = False
        reward = 0.0
        obs = ""

        try:
            if cmd == "help":
                obs = self._get_instructions()

            elif cmd == "observe":
                obs = self.Observe()

            elif cmd == "check":
                if len(tokens) != 2 or not tokens[1].isdigit():
                    obs = "Invalid 'check' usage. Use: \\boxed{check M}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    m = int(tokens[1])
                    obs = self.CheckEdgeCount(m)

            elif cmd in ("inituf", "init"):
                if len(tokens) != 2 or not tokens[1].isdigit():
                    obs = "Invalid 'inituf' usage. Use: \\boxed{inituf N}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    size = int(tokens[1])
                    obs = self.InitializeUnionFind(size)

            elif cmd == "find":
                if len(tokens) != 2 or not tokens[1].isdigit():
                    obs = "Invalid 'find' usage. Use: \\boxed{find X}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    x = int(tokens[1])
                    obs = self.FindRoot(x)

            elif cmd == "union":
                if len(tokens) != 3 or (not tokens[1].isdigit()) or (not tokens[2].isdigit()):
                    obs = "Invalid 'union' usage. Use: \\boxed{union X Y}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    x = int(tokens[1])
                    y = int(tokens[2])
                    obs = self.UnionNodes(x, y)

            elif cmd == "answer":
                if len(tokens) != 2 or tokens[1].upper() not in ("YES", "NO"):
                    obs = "Invalid 'answer' usage. Use: \\boxed{answer YES} or \\boxed{answer NO}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    user_answer = tokens[1].upper()
                    ref_answer = self.get_ref_answer()
                    correct = user_answer == ref_answer
                    obs = f"Your answer: {user_answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
                    reward = 1.0 if correct else -1.0
                    terminated = True
                    self._terminated = True

            else:
                obs = f"Invalid action: {cmd}"
                reward = LanguageGameReward.invalid_action_reward
                terminated = True

        except Exception as e:
            obs = f"Runtime error: {str(e)}"
            reward = LanguageGameReward.invalid_action_reward
            terminated = True

        # 统一超时检查（放在 step 结尾）
        if not terminated and self.turn_count >= self.max_turns:
            obs = f"{obs}\nReached max turns ({self.max_turns})."
            reward = 0.0
            terminated = True
            truncated = True
            self._terminated = True

        self._last_obs = obs
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
        return "\\boxed{observe}"

    # ======================
    # 辅助方法（从原环境转换/保留）
    # ======================

    @property
    def reward(self):
        # GEM 风格中 reward 由 step 返回，这里仅作为兼容占位
        return 0.0

    def get_ref_answer(self) -> str:
        """
        使用环境中的信息得到参考答案。
        判断是否为树：边数为 n-1 且无环。
        """
        if self.n == 1:
            return "YES" if len(self.edges) == 0 else "NO"

        if len(self.edges) != self.n - 1:
            return "NO"

        parent = list(range(self.n + 1))

        def find(x):
            if parent[x] == x:
                return x
            parent[x] = find(parent[x])  # 路径压缩
            return parent[x]

        def union(x, y):
            rootX = find(x)
            rootY = find(y)
            if rootX != rootY:
                parent[rootX] = rootY
                return True
            return False

        for u, v, w in self.edges:
            if u < 1 or v < 1 or u > self.n or v > self.n:
                return "NO"
            if not union(u, v):
                return "NO"

        return "YES"

    def CheckEdgeCount(self, expected_count: int) -> str:
        """
        检查边数量是否等于期望值。
        返回 "True" 或 "False"
        """
        return str(len(self.edges) == expected_count)

    def InitializeUnionFind(self, size: int) -> str:
        """
        初始化并查集。
        """
        if size < 0:
            return "Error: size must be non-negative"
        self.parent = list(range(size + 1))
        return f"Union-Find structure initialized, size: {size}"

    def UnionNodes(self, x: int, y: int) -> str:
        """
        合并两个节点所在集合。成功合并返回 "True"，否则返回 "False"。
        """
        if not self.parent:
            return "Error: Union-Find not initialized."
        if x < 0 or y < 0 or x >= len(self.parent) or y >= len(self.parent):
            return "Error: node index out of range."
        rootX = self.FindRoot(x, return_str=False)
        rootY = self.FindRoot(y, return_str=False)

        if isinstance(rootX, str) or isinstance(rootY, str):
            # 传递错误消息
            return "Error: invalid root query."

        if rootX != rootY:
            self.parent[rootX] = rootY
            return "True"
        return "False"

    def FindRoot(self, x: int, return_str: bool = True):
        """
        查找节点的根，并进行路径压缩。
        返回根节点的字符串形式（默认），或数值（当 return_str=False）。
        """
        if not self.parent:
            result = "Error: Union-Find not initialized."
            return result if return_str else result
        if x < 0 or x >= len(self.parent):
            result = "Error: node index out of range."
            return result if return_str else result

        # 迭代 + 路径压缩
        def _find(val: int) -> int:
            while self.parent[val] != val:
                self.parent[val] = self.parent[self.parent[val]]
                val = self.parent[val]
            return val

        result = _find(x)
        return str(result) if return_str else result

    def Observe(self) -> str:
        """
        返回当前实例的节点数量与边列表。
        """
        return f"Number of nodes: {self.n}, Edge list: {str(self.edges)}"

    def Done(self, answer: str) -> str:
        """
        验证最终答案（兼容原环境的方法）。GEM 中请使用 \\boxed{answer YES/NO}。
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg