from typing import Any, Dict, Optional, Tuple
import random
import re
import json
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class MaximumSpanningTreeEnvGEM(Env):
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

        # 定义难度参数范围（根据原环境分析）
        # num_nodes: 节点数量（图规模）
        # num_edges_factor: 额外边数量的倍数（控制图密度）
        # weight_max: 边权重最大值（数值范围）
        # turn_limit: 步数限制（游戏时限）
        self.complexity_params = {
            "num_nodes": (4, 50),
            "num_edges_factor": (1, 6),
            "weight_max": (10, 10000),
            "turn_limit": (20, 200),
        }

        # 参数方差（用于微调随机性）
        self.param_variance = {
            "num_nodes": 2,
            "num_edges_factor": 1,
            "weight_max": 100,
            "turn_limit": 10,
        }

        # 占位属性
        self.num_nodes: int = 0
        self.num_edges_factor: int = 0
        self.weight_max: int = 0
        self.turn_limit: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 问题实例数据
        self.problem: Dict[str, Any] = {}
        # 运行时状态（保留原环境的辅助结构）
        self.sorted_edges: list = []
        self.mst_edges: list = []

        # 游戏结束标志
        self._answered: bool = False

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

        # 将复杂度控制的 turn_limit 应用于环境的 max_turns
        self.max_turns = int(self.turn_limit)

    def _get_instructions(self) -> str:
        return (
            "Maximum Spanning Tree (MST): Build the maximum spanning tree by inspecting edges.\n"
            "Available actions (use \\boxed{...} format):\n"
            "- Sort edges by descending weight: \\boxed{sort}\n"
            "- Find root in a union-find parent array: \\boxed{find node=I parent=[...]}  (returns root index)\n"
            "- Union two sets: \\boxed{union x=I y=J parent=[...] rank=[...]}  (returns updated parent and rank JSON)\n"
            "- Add an edge weight to MST: \\boxed{add w=W}\n"
            "- Calculate total MST weight: \\boxed{total}\n"
            "- Observe environment: \\boxed{observe}\n"
            "- Submit final answer: \\boxed{answer N}\n"
            "Notes:\n"
            "- Nodes are 0-based indices.\n"
            "- parent and rank parameters must be JSON arrays when using find/union.\n"
        )

    def get_task_suffix(self) -> str:
        n = self.problem.get("n", 0)
        m = len(self.problem.get("edges", []))
        return (
            f"Nodes: {n}, Edges: {m}, MST edges: {len(self.mst_edges)}\n"
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

        # 清理运行状态
        self.turn_count = 0
        self.sorted_edges = []
        self.mst_edges = []
        self._answered = False

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例，确保图连通"""
        n = self.num_nodes
        max_possible_edges = n * (n - 1) // 2
        # 目标边数：至少保证连通（n-1），再加上密度因子
        target_edges = min(max_possible_edges, (n - 1) + self.num_edges_factor * n)
        # 构建初始生成树以保证连通
        edges_set = set()
        edges_list = []

        # 生成一个随机生成树（链式或随机连接）
        nodes_order = list(range(n))
        random.shuffle(nodes_order)
        for i in range(1, n):
            u = nodes_order[i]
            v = nodes_order[random.randint(0, i - 1)]
            if u > v:
                u, v = v, u
            if (u, v) not in edges_set:
                w = random.randint(1, self.weight_max)
                edges_set.add((u, v))
                edges_list.append((u, v, w))

        # 添加额外边
        remaining = target_edges - (n - 1)
        while remaining > 0 and len(edges_set) < max_possible_edges:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if (u, v) in edges_set:
                continue
            w = random.randint(1, self.weight_max)
            edges_set.add((u, v))
            edges_list.append((u, v, w))
            remaining -= 1

        return {"n": n, "edges": edges_list}

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
        args = parsed.get("args", {})

        if self._answered and cmd != "observe":
            obs = "Game already finished. No further actions allowed."
            return (
                obs,
                LanguageGameReward.invalid_action_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            if cmd == "sort":
                obs = self.SortEdgesByWeight()

            elif cmd == "find":
                if "node" not in args or "parent" not in args:
                    obs = "Error: 'node' and 'parent' required for find."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    node = self._decode_value(args["node"])
                    parent = self._decode_value(args["parent"])
                    if not isinstance(node, int) or not isinstance(parent, list):
                        obs = "Error: Invalid 'node' or 'parent' format."
                        reward = LanguageGameReward.format_error_reward
                        terminated = True
                    else:
                        obs = self.FindRoot(node, parent)

            elif cmd == "union":
                required = ["x", "y", "parent", "rank"]
                if not all(k in args for k in required):
                    obs = "Error: Missing parameters for union (x,y,parent,rank)."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    x = self._decode_value(args["x"])
                    y = self._decode_value(args["y"])
                    parent = self._decode_value(args["parent"])
                    rank = self._decode_value(args["rank"])
                    if not (isinstance(x, int) and isinstance(y, int) and isinstance(parent, list) and isinstance(rank, list)):
                        obs = "Error: Invalid parameter types for union."
                        reward = LanguageGameReward.format_error_reward
                        terminated = True
                    else:
                        new_parent, new_rank = self.UnionSets(x, y, parent, rank)
                        obs = json.dumps({"parent": new_parent, "rank": new_rank})

            elif cmd == "add":
                w = args.get("w", args.get("weight"))
                if w is None:
                    obs = "Error: Missing 'w' (weight) for add."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    weight = self._decode_value(w)
                    if not isinstance(weight, int):
                        obs = "Error: weight must be an integer."
                        reward = LanguageGameReward.format_error_reward
                        terminated = True
                    else:
                        obs = self.AddEdgeToMST(weight)

            elif cmd == "total":
                obs = self.CalculateTotalWeight()

            elif cmd == "observe":
                obs = self.Observe()

            elif cmd == "answer":
                if "answer" in args:
                    ans_val = args["answer"]
                else:
                    # allow plain number as second token: \\boxed{answer N}
                    # already parsed args includes any key=val; if no key, try to parse the last segment
                    # fallback: try to parse content split
                    content = parsed.get("raw", "")
                    parts = content.split()
                    if len(parts) >= 2 and parts[0].lower() == "answer":
                        ans_val = parts[1]
                    else:
                        ans_val = None
                if ans_val is None:
                    obs = "Error: Missing answer value."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    user_answer = self._decode_value(ans_val)
                    if not isinstance(user_answer, int):
                        obs = "Error: Answer must be an integer."
                        reward = LanguageGameReward.format_error_reward
                        terminated = True
                    else:
                        ref_answer = self.get_ref_answer()
                        correct = (user_answer == ref_answer)
                        self._answered = True
                        terminated = True
                        truncated = False
                        reward = 1.0 if correct else -1.0
                        obs = f"Your answer: {user_answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"

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
        # First token is command; subsequent tokens are key=value pairs or positional value
        parts = content.split()
        if not parts:
            return None
        cmd = parts[0].strip()
        args: Dict[str, Any] = {}
        for tok in parts[1:]:
            if "=" in tok:
                k, v = tok.split("=", 1)
                args[k.strip()] = v.strip()
            else:
                # store positional for possible answer handling
                # we won't overwrite if already present
                if "positional" not in args:
                    args["positional"] = tok.strip()
        return {"cmd": cmd, "args": args, "raw": content}

    def _decode_value(self, val: Any) -> Any:
        """Decode a value token into int or list via JSON if applicable."""
        if isinstance(val, (int, list, dict)):
            return val
        if not isinstance(val, str):
            return val
        s = val.strip()
        # Try JSON for arrays/objects/numbers
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                return json.loads(s)
            except Exception:
                return s
        # Try integer (support negative)
        if re.fullmatch(r"-?\d+", s):
            try:
                return int(s)
            except Exception:
                return s
        return s

    # ----------------------
    # 保留原环境的辅助方法并转换
    # ----------------------

    def SortEdgesByWeight(self) -> str:
        """
        Sort the roads in descending order of weight.
        Returns JSON string of edges list: [[u,v,w], ...] sorted descending by w.
        """
        self.sorted_edges = sorted(self.problem.get("edges", []), key=lambda item: item[2], reverse=True)
        return json.dumps(self.sorted_edges)

    def FindRoot(self, node: int, parent: list) -> str:
        """
        Find the root node of the node in the union-find data structure.
        Returns the index of the root node as string.
        """
        if parent[node] == node:
            return str(node)
        else:
            return self.FindRoot(parent[node], parent)

    def UnionSets(self, x: int, y: int, parent: list, rank: list):
        """
        Merge two sets.
        Returns (parent, rank) after union.
        """
        if rank[x] < rank[y]:
            parent[x] = y
        elif rank[x] > rank[y]:
            parent[y] = x
        else:
            parent[y] = x
            rank[x] += 1
        return parent, rank

    def AddEdgeToMST(self, weight: int) -> str:
        """
        Add the weight of the edge to the MST.
        Returns MST weight list JSON string.
        """
        self.mst_edges.append(weight)
        return json.dumps(self.mst_edges)

    def CalculateTotalWeight(self) -> str:
        """
        Calculate the total weight of all edges in the MST.
        Returns total weight as string.
        """
        return str(sum(self.mst_edges) if self.mst_edges else 0)

    def Observe(self) -> str:
        """
        Obtain the current environmental state information.
        Returns description string.
        """
        n = self.problem.get("n", 0)
        edges = self.problem.get("edges", [])
        return f"Number of cities: {n}, Number of roads: {len(edges)}, Number of edges in MST: {len(self.mst_edges)}"

    def Done(self, answer: int) -> str:
        """
        Submit the final answer and verify if it is correct. (Not used directly in GEM step; kept for compatibility)
        Returns verification result string with reward info appended.
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg + f", reward={1 if correct else 0}"

    def get_ref_answer(self) -> int:
        """
        Compute the reference maximum spanning tree total weight using Kruskal (descending).
        """
        n = self.problem.get("n", 0)
        edges = list(self.problem.get("edges", []))
        edges.sort(key=lambda item: item[2], reverse=True)

        parent = [i for i in range(n)]
        rank = [0] * n
        result_weights = []

        def find(parent_arr, i):
            if parent_arr[i] == i:
                return i
            else:
                return find(parent_arr, parent_arr[i])

        def union(parent_arr, rank_arr, x, y):
            rx = find(parent_arr, x)
            ry = find(parent_arr, y)
            if rx == ry:
                return
            if rank_arr[rx] < rank_arr[ry]:
                parent_arr[rx] = ry
            elif rank_arr[rx] > rank_arr[ry]:
                parent_arr[ry] = rx
            else:
                parent_arr[ry] = rx
                rank_arr[rx] += 1

        e = 0
        i = 0
        while e < n - 1 and i < len(edges):
            u, v, w = edges[i]
            i += 1
            x = find(parent, u)
            y = find(parent, v)
            if x != y:
                e += 1
                result_weights.append(w)
                union(parent, rank, x, y)

        return sum(result_weights) if result_weights else 0

    def sample_random_action(self) -> str:
        # 提供简单示例动作
        return "\\boxed{observe}"