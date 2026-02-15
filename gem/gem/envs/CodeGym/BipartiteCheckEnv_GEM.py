from typing import Any, Dict, Optional, Tuple
import random
import re
import json
from collections import deque
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class BipartiteCheckEnvGEM(Env):
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

        # 难度参数范围设计：控制图大小、组件数、边密度、以及允许步数
        self.complexity_params = {
            "num_vertices": (5, 50),        # 顶点数量（数组长度）
            "edge_factor": (1, 5),          # 边密度因子（近似每个组件边数 ~ edge_factor * size）
            "num_components": (1, 5),       # 连通分量数量
            "turn_allowance": (20, 200),    # 步数上限（难度控制）
        }

        # 参数方差（用于微调随机性）
        self.param_variance = {
            "num_vertices": 2,
            "edge_factor": 1,
            "num_components": 1,
            "turn_allowance": 10,
        }

        # 占位属性（由 _apply_complexity_params 填充）
        self.num_vertices: int = 0
        self.edge_factor: int = 0
        self.num_components: int = 0
        self.turn_allowance: int = 0

        # 问题实例状态
        self.n: int = 0
        self.m: int = 0
        self.edges: list[tuple[int, int]] = []

        # 运行时状态
        self.turn_count: int = 0
        self.max_turns_current: int = self.max_turns

        # 交互过程中生成的中间产物
        self.adj: Optional[list[list[int]]] = None
        self.colors: Optional[list[int]] = None

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

        # 动态步数上限为难度控制值和构造传入上限的较小者
        self.max_turns_current = int(min(self.turn_allowance, self.max_turns))

    def _get_instructions(self) -> str:
        return (
            "Bipartite Graph Check: Determine if the generated graph is bipartite.\n"
            "Available actions (use the latest \\boxed{...} content):\n"
            "- Observe problem: \\boxed{observe}\n"
            "- Build adjacency list: \\boxed{build_adj}\n"
            "- Initialize colors: \\boxed{init_colors}\n"
            "- BFS check from a start vertex: \\boxed{bfs start=K}\n"
            "- Check all components: \\boxed{check_all}\n"
            "- Submit final answer (Yes/No): \\boxed{answer Yes} or \\boxed{answer No}\n"
            "Aliases supported:\n"
            "- \\boxed{CreateAdjacencyList}, \\boxed{InitializeColors}, \\boxed{BFSCheck start=K}, \\boxed{CheckAllComponents}, \\boxed{Done Yes}\n"
        )

    def get_task_suffix(self) -> str:
        return f"Vertices: {self.n}, Edges: {self.m}\nTurn: {self.turn_count}/{self.max_turns_current}\nEnter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()
        self.n = self.problem["n"]
        self.edges = self.problem["edges"]
        self.m = len(self.edges)

        # 清理中间产物
        self.adj = None
        self.colors = None

        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成图问题实例"""
        n = self.num_vertices
        c = min(self.num_components, n) if n > 0 else 0
        c = max(1, c)

        # 组件大小分配，保证每个组件至少 1 个顶点
        sizes = [1 for _ in range(c)]
        remaining = max(0, n - c)
        for _ in range(remaining):
            sizes[random.randint(0, c - 1)] += 1

        # 非二分的概率随难度增加
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        non_bip_ratio = 0.2 + 0.6 * normalized  # 20% ~ 80%

        global_edges = []
        base = 1

        for size in sizes:
            if size <= 0:
                continue
            vertices = list(range(base, base + size))

            # 决定该组件是否非二分
            is_non_bip = random.random() < non_bip_ratio

            # 目标边数（近似）
            target_edges = max(0, min(size * self.edge_factor, size * (size - 1) // 2))

            # 构造边集合
            edges_set = set()

            if is_non_bip and size >= 3:
                # 构造一个奇环保证非二分
                # 选择奇环长度
                if size % 2 == 1:
                    odd_len = size
                else:
                    odd_len = size - 1 if size - 1 >= 3 else 3
                cycle_nodes = vertices[:odd_len]
                for i in range(odd_len):
                    u = cycle_nodes[i]
                    v = cycle_nodes[(i + 1) % odd_len]
                    edge = (min(u, v), max(u, v))
                    edges_set.add(edge)
            else:
                # 构造二分：分成两侧，仅跨侧连边
                left_size = size // 2
                left = vertices[:left_size] if left_size > 0 else []
                right = vertices[left_size:] if left_size < size else []
                # 若某侧为空，退化为线性链
                if not left or not right:
                    chain_nodes = vertices
                    for i in range(len(chain_nodes) - 1):
                        u, v = chain_nodes[i], chain_nodes[i + 1]
                        edge = (min(u, v), max(u, v))
                        edges_set.add(edge)
                else:
                    # 随机跨侧连边
                    possible = []
                    for u in left:
                        for v in right:
                            possible.append((min(u, v), max(u, v)))
                    random.shuffle(possible)
                    for e in possible[:target_edges]:
                        edges_set.add(e)

            # 填充到目标边数（不添加自环、不重复）
            attempts = 0
            max_attempts = 5 * target_edges + 50
            while len(edges_set) < target_edges and attempts < max_attempts:
                u = random.choice(vertices)
                v = random.choice(vertices)
                if u == v:
                    attempts += 1
                    continue
                e = (min(u, v), max(u, v))
                edges_set.add(e)
                attempts += 1

            global_edges.extend(list(edges_set))
            base += size

        return {"n": n, "edges": global_edges}

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

        name = parsed["name"]
        params = parsed.get("params", {})

        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        # 动作处理
        try:
            if name in ("observe", "Observe"):
                obs = self.Observe()
                reward = 0.0
                terminated = False

            elif name in ("build_adj", "CreateAdjacencyList"):
                self.adj = json.loads(self.CreateAdjacencyList(self.n, self.edges))
                obs = f"Adjacency list created: {json.dumps(self.adj)}"
                reward = 0.0
                terminated = False

            elif name in ("init_colors", "InitializeColors"):
                self.colors = json.loads(self.InitializeColors(self.n))
                obs = f"Colors initialized: {json.dumps(self.colors)}"
                reward = 0.0
                terminated = False

            elif name in ("bfs", "BFSCheck"):
                start = params.get("start", None)
                if start is None or not isinstance(start, int) or start < 1 or start > self.n:
                    obs = "Invalid or missing parameter: start"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                if self.adj is None or self.colors is None:
                    obs = "Prerequisite missing: build_adj and init_colors required before bfs."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                result = json.loads(self.BFSCheck(start, self.adj, self.colors))
                self.colors = result["colors"]
                obs = json.dumps({"is_bipartite": result["is_bipartite"], "colors": result["colors"]})
                reward = 0.0
                terminated = False

            elif name in ("check_all", "CheckAllComponents"):
                if self.adj is None or self.colors is None:
                    obs = "Prerequisite missing: build_adj and init_colors required before check_all."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                # 使用一个拷贝，避免修改当前 colors
                colors_copy = list(self.colors)
                result = self.CheckAllComponents(self.n, self.adj, colors_copy)
                obs = result
                reward = 0.0
                terminated = False

            elif name in ("answer", "Done"):
                ans = params.get("answer", None)
                # 支持直接传 content "answer Yes" 的解析（见 _parse_action）
                if ans is None and isinstance(params.get("content_answer", None), str):
                    ans = params["content_answer"]

                if ans not in ("Yes", "No"):
                    obs = "Invalid or missing parameter: answer must be 'Yes' or 'No'."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                msg = self.Done(ans)
                obs = msg
                ref = self.get_ref_answer()
                reward = 1.0 if ans == ref else -1.0
                terminated = True

            else:
                obs = f"Invalid action: {name}"
                return (
                    obs,
                    LanguageGameReward.invalid_action_reward,
                    True,
                    False,
                    {"suffix": self.get_task_suffix()},
                )

        except Exception as e:
            obs = f"Execution error: {str(e)}"
            return (
                obs,
                LanguageGameReward.invalid_action_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        # 超时检查（统一放在 step 结尾）
        if not terminated and self.turn_count >= self.max_turns_current:
            obs = f"{obs}\nReached max turns ({self.max_turns_current})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action:
            return None
        pattern = re.compile(r"\\boxed\{(.+?)\}", re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()

        # 基础解析，支持 "cmd", "cmd arg=value", 以及别名
        lower = content.lower()

        # observe
        if lower == "observe":
            return {"name": "observe", "params": {}}

        # build_adj / CreateAdjacencyList
        if lower in ("build_adj", "createadjacencylist", "create adjacencylist"):
            return {"name": "build_adj", "params": {}}

        # init_colors / InitializeColors
        if lower in ("init_colors", "initializecolors", "initialize colors"):
            return {"name": "init_colors", "params": {}}

        # bfs / BFSCheck
        bfs_match = re.match(r"^(bfs|bf s|bfscheck)\s+start\s*=\s*(\d+)\s*$", lower)
        if bfs_match:
            start_val = int(bfs_match.group(2))
            # 保留原名兼容
            name = "bfs" if bfs_match.group(1) == "bfs" else "BFSCheck"
            return {"name": name, "params": {"start": start_val}}

        # check_all / CheckAllComponents
        if lower in ("check_all", "checkallcomponents", "check allcomponents", "check all"):
            return {"name": "check_all", "params": {}}

        # answer / Done
        ans_match = re.match(r"^(answer|done)\s+(yes|no)\s*$", lower)
        if ans_match:
            ans = ans_match.group(2).capitalize()
            name = "answer" if ans_match.group(1) == "answer" else "Done"
            return {"name": name, "params": {"answer": ans, "content_answer": ans}}

        return None

    def sample_random_action(self) -> str:
        choices = [
            "\\boxed{observe}",
            "\\boxed{build_adj}",
            "\\boxed{init_colors}",
            "\\boxed{bfs start=1}",
            "\\boxed{check_all}",
            "\\boxed{answer Yes}",
            "\\boxed{answer No}",
        ]
        return random.choice(choices)

    # ---------------------------
    # 保留原环境的辅助方法并转换
    # ---------------------------
    def CreateAdjacencyList(self, n: int, edges: list):
        """
        Create an adjacency list based on the number of vertices and edge list.
        Returns a serialized adjacency list JSON string.
        """
        adj = [[] for _ in range(n + 1)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return json.dumps(adj)

    def InitializeColors(self, n: int):
        """
        Initialize the color array (-1 means uncolored).
        Returns a serialized color array JSON string.
        """
        colors = [-1] * (n + 1)
        return json.dumps(colors)

    def BFSCheck(self, start: int, adj: list, colors: list):
        """
        Perform BFS coloring check starting from the specified vertex to determine bipartiteness.
        Returns JSON string containing the check result and updated color array.
        """
        queue = deque([start])
        colors[start] = 0
        is_bipartite = True

        while queue and is_bipartite:
            u = queue.popleft()
            for v in adj[u]:
                if colors[v] == -1:
                    colors[v] = 1 - colors[u]
                    queue.append(v)
                elif colors[v] == colors[u]:
                    is_bipartite = False
                    break

        return json.dumps({"is_bipartite": is_bipartite, "colors": colors})

    def CheckAllComponents(self, n: int, adj: list, colors: list):
        """
        Check if all connected components in the graph are bipartite.
        Returns: "Yes" or "No".
        """
        for i in range(1, n + 1):
            if colors[i] == -1:
                result = json.loads(self.BFSCheck(i, adj, colors))
                if not result["is_bipartite"]:
                    return "No"
                colors = result["colors"]
        return "Yes"

    def Observe(self):
        """
        Return basic information about the current graph.
        """
        return f"Number of vertices: {self.n}, Number of edges: {self.m}, Edge list: {[(u, v) for (u, v) in self.edges]}"

    def get_ref_answer(self):
        """
        Use the current graph information to get the reference answer ("Yes"/"No").
        """
        adj = [[] for _ in range(self.n + 1)]
        for u, v in self.edges:
            adj[u].append(v)
            adj[v].append(u)

        colors = [-1] * (self.n + 1)

        def bfs_check(start):
            queue = deque([start])
            colors[start] = 0

            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if colors[v] == -1:
                        colors[v] = 1 - colors[u]
                        queue.append(v)
                    elif colors[v] == colors[u]:
                        return False
            return True

        for i in range(1, self.n + 1):
            if colors[i] == -1:
                if not bfs_check(i):
                    return "No"
        return "Yes"

    def Done(self, answer: str):
        """
        Verify if the final answer is correct and return the result information.
        Does not change internal GEM termination here; step() manages reward/termination.
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg