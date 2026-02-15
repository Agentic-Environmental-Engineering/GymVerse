from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional

class AlgorithmPlannerEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 6,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 6

        # Evolvable parameters
        self.complexity_params = {
            # Number of nodes: larger graphs increase algorithmic cost and decision difficulty
            'num_nodes': (15, 350),
            # Average degree: denser graphs increase edges → harder due to higher E
            'avg_degree': (2, 20),
            # REVERSED: operation budget (ops_limit). Lower budget = harder constraint
            'ops_limit': (3500000, 900000),
            # REVERSED: probability (%) of negative edges. Fewer negatives at high complexity to keep solvable under tight budgets
            'negative_edge_pct': (30, 5),
            # REVERSED: probability (%) of negative cycle. High at low complexity (small V → BF fits), low at high complexity
            'negative_cycle_pct': (20, 2),
            # REVERSED: grid-likeness (%) affecting A* applicability; less grid structure at high complexity → harder for A*
            'grid_likelihood_pct': (80, 20),
            # REVERSED: heuristic quality (%) for A*; lower quality at high complexity → less guidance
            'heuristic_quality_pct': (90, 55),
            # SPFA badness (%) scaling factor; larger → worse expected performance
            'spfa_badness_pct': (5, 60),
        }

        # Variance settings
        self.param_variance = {
            'num_nodes': 25,
            'avg_degree': 1,
            'ops_limit': 150000,
            'negative_edge_pct': 3,
            'negative_cycle_pct': 2,
            'grid_likelihood_pct': 5,
            'heuristic_quality_pct': 3,
            'spfa_badness_pct': 5,
        }

        # Placeholder attributes
        self.num_nodes: int = 0
        self.avg_degree: int = 0
        self.ops_limit: int = 0
        self.negative_edge_pct: int = 0
        self.negative_cycle_pct: int = 0
        self.grid_likelihood_pct: int = 0
        self.heuristic_quality_pct: int = 0
        self.spfa_badness_pct: int = 0

        # State
        self.turn_count: int = 0
        self.selected_alg: Optional[str] = None
        self.selected_ds: Optional[str] = None
        self.edges_estimate: int = 0
        self.weights_type: str = "nonnegative"  # 'unweighted', 'nonnegative', 'negative'
        self.has_negative_edges: bool = False
        self.has_negative_cycle: bool = False
        self.grid_like: bool = False
        self.heuristic_admissible: bool = False
        self.oracle_best: Optional[Tuple[str, Optional[str], int]] = None  # (alg, ds, cost)
        self._last_eval: Dict[str, Any] = {}
        self._applicability_cache: Dict[str, Dict[str, Any]] = {}

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for param_name, (min_val, max_val) in self.complexity_params.items():
            center_value = min_val + (max_val - min_val) * normalized
            variance = self.param_variance.get(param_name, 0)
            if self.enable_param_randomization and variance > 0:
                center_value += random.uniform(-variance, variance)
            # clamp for reversed ranges as well
            lo = min(min_val, max_val)
            hi = max(min_val, max_val)
            center_value = max(lo, min(hi, center_value))
            setattr(self, param_name, int(round(center_value)))

    def _bitlog2(self, n: int) -> int:
        return max(1, n.bit_length())

    def _estimate_costs_and_oracle(self):
        V = self.num_nodes
        E = self.edges_estimate
        logV = self._bitlog2(V)

        # coefficients
        c_bfs_v, c_bfs_e = 3, 5
        c_d_bin = 4
        c_d_arr = 2
        c_bf = 1
        c_spfa = 1
        c_fw = 1
        c_astar = 4
        q = max(0.2, 1.0 - 0.5 * (self.heuristic_quality_pct / 100.0))
        spfa_factor = 1.0 + (self.spfa_badness_pct / 100.0) * (max(1, self.avg_degree) / 20.0)

        def applicable(alg: str) -> bool:
            if alg == "BFS":
                return (self.weights_type == "unweighted") and not self.has_negative_cycle
            if alg == "Dijkstra":
                return not self.has_negative_edges and not self.has_negative_cycle
            if alg == "BellmanFord":
                return True
            if alg == "SPFA":
                return (self.has_negative_edges and not self.has_negative_cycle) or (not self.has_negative_edges and not self.has_negative_cycle)
            if alg == "FloydWarshall":
                return True
            if alg == "AStar":
                return (not self.has_negative_edges) and self.grid_like
            return False

        def guarantees_optimal(alg: str) -> bool:
            if alg in ("BFS", "Dijkstra", "BellmanFord", "FloydWarshall", "SPFA"):
                return True
            if alg == "AStar":
                return self.heuristic_admissible
            return False

        def cost(alg: str, ds: Optional[str]) -> int:
            if alg == "BFS":
                return int(c_bfs_v * V + c_bfs_e * E)
            if alg == "Dijkstra":
                if ds == "Array":
                    return int(c_d_arr * (V * V + E))
                else:
                    return int(c_d_bin * (E + V) * logV)
            if alg == "BellmanFord":
                return int(c_bf * V * E)
            if alg == "SPFA":
                # expected-ish
                return int(c_spfa * E * spfa_factor * logV)
            if alg == "FloydWarshall":
                return int(c_fw * V * V * V)
            if alg == "AStar":
                base = int(c_astar * (E + V) * logV)
                return int(max(1, base * q))
            return 10**9

        candidates = []
        algs = ["BFS", "Dijkstra", "BellmanFord", "SPFA", "FloydWarshall", "AStar"]
        for alg in algs:
            if alg == "Dijkstra":
                for ds in ("BinaryHeap", "Array"):
                    candidates.append((alg, ds, applicable(alg), guarantees_optimal(alg), cost(alg, ds)))
            else:
                candidates.append((alg, None, applicable(alg), guarantees_optimal(alg), cost(alg, None)))

        self._applicability_cache = {}
        for alg, ds, app, opt, c in candidates:
            key = alg if ds is None else f"{alg}({ds})"
            self._applicability_cache[key] = {"applicable": app, "guarantees_optimal": opt, "cost": c}

        # Determine oracle: minimal cost among applicable & optimal
        feasible_optimal = [(alg, ds, c) for (alg, ds, app, opt, c) in candidates if app and opt]
        if feasible_optimal:
            best = min(feasible_optimal, key=lambda x: x[2])
            self.oracle_best = best
        else:
            self.oracle_best = None

        # Ensure solvability by adjusting ops_limit if needed
        if self.oracle_best is not None:
            min_cost = self.oracle_best[2]
            if self.ops_limit < min_cost:
                self.ops_limit = int(min_cost * 1.05)

    def _get_instructions(self) -> str:
        example = self.sample_random_action()
        return (
            "Algorithm Planner Game: choose a shortest-path solver for the described graph.\n"
            "Goal: select an applicable algorithm (and data structure if needed) that fits within the operation budget. "
            "Success yields +1 if your choice is optimal (minimal estimated cost among applicable optimal solvers). "
            "Valid but suboptimal yields 0. Inapplicable or over budget yields -1.\n"
            "Actions (one per turn, in \\boxed{...} format):\n"
            "- ALG=<BFS|Dijkstra|BellmanFord|SPFA|FloydWarshall|AStar>\n"
            "- DS=<BinaryHeap|Array>  (only relevant for Dijkstra)\n"
            "- COMMIT  (evaluate your current choice)\n"
            f"Example: {example}\n"
        )

    def get_task_suffix(self) -> str:
        s_alg = self.selected_alg if self.selected_alg else "None"
        s_ds = self.selected_ds if self.selected_ds else "None"
        return (
            f"State:\n"
            f"- Nodes: {self.num_nodes}, AvgDegree: {self.avg_degree}, ApproxEdges: {self.edges_estimate}\n"
            f"- Weights: {self.weights_type}, NegativeEdges: {self.has_negative_edges}, NegativeCycle: {self.has_negative_cycle}\n"
            f"- GridLike: {self.grid_like}, HeuristicQuality%: {self.heuristic_quality_pct}, HeuristicAdmissible: {self.heuristic_admissible}\n"
            f"- OpsBudget: {self.ops_limit}\n"
            f"- Selection: ALG={s_alg}, DS={s_ds}\n"
            "Enter action as \\boxed{ALG=...} or \\boxed{DS=...} or \\boxed{COMMIT}."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        self.turn_count = 0
        self.selected_alg = None
        self.selected_ds = None

        # Sample instance
        self.edges_estimate = int(self.num_nodes * self.avg_degree)
        neg_edge = random.random() < (self.negative_edge_pct / 100.0)
        neg_cycle = False
        if neg_edge:
            neg_cycle = random.random() < (self.negative_cycle_pct / 100.0)
        self.has_negative_edges = neg_edge
        self.has_negative_cycle = neg_cycle

        # weights type
        if self.has_negative_edges:
            self.weights_type = "negative"
        else:
            # chance of truly unweighted
            unweighted_chance = max(0.1, 0.35 - 0.02 * max(0, self.complexity - 1))
            self.weights_type = "unweighted" if random.random() < unweighted_chance else "nonnegative"

        self.grid_like = random.random() < (self.grid_likelihood_pct / 100.0)
        self.heuristic_admissible = self.grid_like and (self.heuristic_quality_pct >= 70)

        self._estimate_costs_and_oracle()

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if parsed is None:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        messages = []
        reward = 0.0

        for item in parsed:
            if item['type'] == 'ALG':
                alg = item['value']
                if alg not in {"BFS", "Dijkstra", "BellmanFord", "SPFA", "FloydWarshall", "AStar"}:
                    messages.append(f"Unsupported action: algorithm '{alg}' is unknown.")
                else:
                    self.selected_alg = alg
                    messages.append(f"Selected algorithm: {alg}.")
                    # reset DS if switch away from Dijkstra
                    if alg != "Dijkstra":
                        self.selected_ds = None
            elif item['type'] == 'DS':
                ds = item['value']
                if ds not in {"BinaryHeap", "Array"}:
                    messages.append(f"Unsupported action: data structure '{ds}' is unknown.")
                else:
                    self.selected_ds = ds
                    messages.append(f"Selected data structure: {ds}.")
            elif item['type'] == 'COMMIT':
                if self.selected_alg is None:
                    obs = "Protocol violation: COMMIT without selecting ALG."
                    return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
                # Default DS for Dijkstra
                ds = self.selected_ds if self.selected_alg == "Dijkstra" else None
                if self.selected_alg == "Dijkstra" and ds is None:
                    ds = "BinaryHeap"

                key = self.selected_alg if ds is None else f"{self.selected_alg}({ds})"
                info = self._applicability_cache.get(key, {"applicable": False, "guarantees_optimal": False, "cost": 10**9})
                applicable = info["applicable"]
                guarantees = info["guarantees_optimal"]
                cost = info["cost"]
                self._last_eval = {"key": key, "applicable": applicable, "guarantees_optimal": guarantees, "cost": cost}

                if not applicable:
                    obs = f"Failed! Inapplicable choice: {key} for weights='{self.weights_type}', negative_edges={self.has_negative_edges}, negative_cycle={self.has_negative_cycle}."
                    return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

                if cost > self.ops_limit:
                    obs = f"Failed! Budget exceeded: choice cost={cost} > ops_limit={self.ops_limit}."
                    return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

                # Suboptimal vs optimal
                best = self.oracle_best
                if guarantees:
                    if best is not None and key == (best[0] if best[1] is None else f"{best[0]}({best[1]})"):
                        obs = f"Success! Optimal choice: {key} (cost={cost} within budget)."
                        return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
                    else:
                        if best is not None:
                            bkey = best[0] if best[1] is None else f"{best[0]}({best[1]})"
                            obs = f"Valid but suboptimal: {key} (cost={cost}). Best is {bkey} (cost={best[2]})."
                        else:
                            obs = f"Valid but suboptimal: {key} (cost={cost})."
                        return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
                else:
                    obs = f"Valid but heuristic may be suboptimal: {key} (cost={cost}) within budget."
                    return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                messages.append("Unsupported action: token not recognized.")

        if self.turn_count >= (self.max_turns or 0):
            obs = f"Reached max turns ({self.max_turns})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        obs = "; ".join(messages) if messages else f"At turn {self.turn_count}, no state change."
        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Any]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        m = pattern.findall(action)
        if not m:
            return None
        content = m[-1].strip()
        tokens = [t.strip() for t in content.split(";") if t.strip()]
        parsed = []
        for t in tokens:
            if "=" in t:
                k, v = t.split("=", 1)
                k = k.strip().upper()
                v = v.strip()
                if k == "ALG":
                    parsed.append({"type": "ALG", "value": v})
                elif k == "DS":
                    parsed.append({"type": "DS", "value": v})
                else:
                    parsed.append({"type": "UNKNOWN", "value": t})
            else:
                if t.strip().upper() == "COMMIT":
                    parsed.append({"type": "COMMIT"})
                else:
                    parsed.append({"type": "UNKNOWN", "value": t})
        return parsed if parsed else None

    def sample_random_action(self) -> str:
        algs = ["BFS", "Dijkstra", "BellmanFord", "SPFA", "FloydWarshall", "AStar"]
        a = random.choice(algs)
        if a == "Dijkstra":
            ds = random.choice(["BinaryHeap", "Array"])
            return f"\\boxed{{ALG={a}; DS={ds}}}"
        else:
            return f"\\boxed{{ALG={a}}}"


class AlgorithmPlannerEnvWithFeedback(AlgorithmPlannerEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        text = obs.lower()

        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint: Optional[str] = None

        if "invalid action format" in text or "use \\boxed" in text:
            error_type = "FormatError"
            error_detail["issue"] = "missing_boxed_format"
            hint = "Use \\boxed{ALG=...}, optional \\boxed{DS=...}, then \\boxed{COMMIT}."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "commit_without_algorithm"
            hint = "Select an algorithm first: for unweighted graphs use BFS; for nonnegative weights use Dijkstra; with negative edges use BellmanFord."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["allowed_algorithms"] = ["BFS", "Dijkstra", "BellmanFord", "SPFA", "FloydWarshall", "AStar"]
            error_detail["allowed_data_structures"] = ["BinaryHeap", "Array"]
            hint = "Use allowed tokens: ALG=<one of listed>; DS=<BinaryHeap|Array>; COMMIT."

        elif "failed!" in text:
            error_type = "WrongDecision"
            if "inapplicable" in text:
                error_detail["reason"] = "inapplicable_algorithm"
                error_detail["weights_type"] = self.weights_type
                error_detail["negative_edges"] = self.has_negative_edges
                error_detail["negative_cycle"] = self.has_negative_cycle
                hint = "Match algorithm to graph: BFS only for unweighted; Dijkstra for nonnegative; BellmanFord or FloydWarshall for negative edges or cycles."
            elif "budget exceeded" in text:
                error_detail["reason"] = "budget_exceeded"
                error_detail["ops_limit"] = self.ops_limit
                last = getattr(self, "_last_eval", {})
                error_detail["cost"] = last.get("cost")
                hint = "Choose a faster solver: BFS for unweighted; Dijkstra(BinaryHeap) for nonnegative; avoid FloydWarshall/BellmanFord on large graphs."

        elif "valid but suboptimal" in text or "heuristic may be suboptimal" in text:
            error_type = "WrongDecision"
            last = getattr(self, "_last_eval", {})
            error_detail["chosen"] = last.get("key")
            if self.oracle_best is not None:
                best_key = self.oracle_best[0] if self.oracle_best[1] is None else f"{self.oracle_best[0]}({self.oracle_best[1]})"
                error_detail["best"] = best_key
                error_detail["best_cost"] = self.oracle_best[2]
            hint = "Aim for the minimal-cost applicable optimal solver. Consider the oracle-best shown if available."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["max_turns"] = self.max_turns
            hint = "Commit earlier after selecting ALG (and DS if Dijkstra)."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "num_nodes": self.num_nodes,
                "avg_degree": self.avg_degree,
                "edges_estimate": self.edges_estimate,
                "weights_type": self.weights_type,
                "negative_edges": self.has_negative_edges,
                "negative_cycle": self.has_negative_cycle,
                "grid_like": self.grid_like,
                "ops_limit": self.ops_limit,
                "selected_alg": self.selected_alg,
                "selected_ds": self.selected_ds,
            }
            if self.oracle_best is not None:
                diagnostic["state"]["oracle_best"] = {
                    "alg": self.oracle_best[0],
                    "ds": self.oracle_best[1],
                    "cost": self.oracle_best[2],
                }
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        hint = "Start by choosing ALG. Use BFS if unweighted; Dijkstra if weights are nonnegative; BellmanFord if negative edges may appear."
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": hint if self.feedback_level >= 2 else None,
            "turn": 0,
            "state": {
                "num_nodes": self.num_nodes,
                "avg_degree": self.avg_degree,
                "edges_estimate": self.edges_estimate,
                "weights_type": self.weights_type,
                "negative_edges": self.has_negative_edges,
                "negative_cycle": self.has_negative_cycle,
                "grid_like": self.grid_like,
                "ops_limit": self.ops_limit,
            } if self.feedback_level >= 1 else None,
        }
        return obs, info