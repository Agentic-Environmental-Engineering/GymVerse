from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional

class ArmadaTallyEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 50,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 50

        # Evolvable parameters with explanations:
        self.complexity_params = {
            "num_ships": (5, 18),                 # More ships increases exploration and computation
            "max_degree": (2, 5),                 # Higher degree yields denser links → harder neighborhood logic
            "k_hops_max": (1, 3),                 # Larger hop radius expands neighborhood size → harder
            "weight_threshold_range": (4, 14),    # Larger possible threshold increases path-cost variety → harder
            "aggregator_variety": (1, 3),         # 1: sum, 2: sum/avg, 3: sum/avg/max → more operators = harder
            "rule_variety": (1, 2),               # 1: hops-only, 2: hops or weight → more rule types = harder
            "precision_decimals_max": (0, 2),     # More decimal precision demands exact rounding → harder
            "tool_access_level": (2, 1),          # REVERSED: 2 includes neighborhood(); 1 locks it → harder
        }

        # Randomization variances matched to ranges
        self.param_variance = {
            "num_ships": 1,
            "max_degree": 1,
            "k_hops_max": 0,
            "weight_threshold_range": 2,
            "aggregator_variety": 0,
            "rule_variety": 0,
            "precision_decimals_max": 0,
            "tool_access_level": 0,
        }

        # Placeholder attributes
        self.num_ships: int = 0
        self.max_degree: int = 0
        self.k_hops_max: int = 0
        self.weight_threshold_range: int = 0
        self.aggregator_variety: int = 0
        self.rule_variety: int = 0
        self.precision_decimals_max: int = 0
        self.tool_access_level: int = 0

        # Domain state
        self.turn_count: int = 0
        self.ship_names: list = []
        self.attributes: Dict[str, Dict[str, int]] = {}
        self.graph: Dict[str, list] = {}
        self.reference_ship: Optional[str] = None
        self.rule_type: Optional[str] = None  # "hops" or "weight"
        self.rule_param: Optional[int] = None
        self.target_attribute: Optional[str] = None  # "shield", "hull", "cargo"
        self.aggregator: Optional[str] = None  # "sum", "avg", "max"
        self.precision_decimals: int = 0
        self.correct_value: Optional[float] = None
        self.paths_hops: Dict[str, int] = {}
        self.paths_cost: Dict[str, int] = {}
        self.scanned_ships: set = set()
        self.last_submitted_value: Optional[float] = None

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for param_name, (min_val, max_val) in self.complexity_params.items():
            center_value = min_val + (max_val - min_val) * normalized
            actual_value = center_value
            if self.enable_param_randomization:
                variance = self.param_variance.get(param_name, 0)
                if variance > 0:
                    actual_value = center_value + random.uniform(-variance, variance)
            # clamp range, support reversed params
            if min_val <= max_val:
                actual_value = max(min_val, min(max_val, actual_value))
            else:
                actual_value = max(max_val, min(min_val, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _generate_names(self, n: int) -> list:
        base = [
            "Aster", "Boreal", "Cinder", "Dione", "Eclipse", "Fjord", "Gale",
            "Halcyon", "Ion", "Javelin", "Kestrel", "Lumen", "Mistral", "Nadir",
            "Orion", "Pulsar", "Quasar", "Rime", "Solace", "Turbine", "Umbra",
            "Vanguard", "Warden", "Xenon", "Yarrow", "Zephyr"
        ]
        random.shuffle(base)
        names = []
        i = 0
        while len(names) < n:
            if i < len(base):
                candidate = base[i]
            else:
                candidate = f"Nova-{i}"
            names.append(candidate)
            i += 1
        return names

    def _build_graph(self):
        self.graph = {name: [] for name in self.ship_names}
        degree = {name: 0 for name in self.ship_names}
        # Ensure connectivity by forming a tree
        for i in range(1, len(self.ship_names)):
            a = self.ship_names[i]
            b = random.choice(self.ship_names[:i])
            w = random.randint(1, max(2, self.weight_threshold_range // 2))
            self.graph[a].append((b, w))
            self.graph[b].append((a, w))
            degree[a] += 1
            degree[b] += 1
        # Add extra edges to approach max_degree
        attempts = 0
        max_attempts = len(self.ship_names) * self.max_degree * 2
        while attempts < max_attempts:
            a, b = random.sample(self.ship_names, 2)
            if a == b:
                attempts += 1
                continue
            if any(n == b for n, _ in self.graph[a]):
                attempts += 1
                continue
            if degree[a] >= self.max_degree or degree[b] >= self.max_degree:
                attempts += 1
                continue
            w = random.randint(1, max(2, self.weight_threshold_range))
            self.graph[a].append((b, w))
            self.graph[b].append((a, w))
            degree[a] += 1
            degree[b] += 1
            attempts += 1

    def _compute_hops(self, src: str) -> Dict[str, int]:
        dist = {name: float("inf") for name in self.ship_names}
        dist[src] = 0
        q = [src]
        while q:
            u = q.pop(0)
            for v, _ in self.graph[u]:
                if dist[v] > dist[u] + 1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return {k: int(d) if d != float("inf") else 9999 for k, d in dist.items()}

    def _compute_costs(self, src: str) -> Dict[str, int]:
        # Dijkstra without heapq (linear scan)
        visited = {name: False for name in self.ship_names}
        dist = {name: float("inf") for name in self.ship_names}
        dist[src] = 0
        for _ in range(len(self.ship_names)):
            # find unvisited node with smallest distance
            u = None
            best = float("inf")
            for name in self.ship_names:
                if not visited[name] and dist[name] < best:
                    best = dist[name]
                    u = name
            if u is None:
                break
            visited[u] = True
            for v, w in self.graph[u]:
                if dist[v] > dist[u] + w:
                    dist[v] = dist[u] + w
        return {k: int(d) if d != float("inf") else 999999 for k, d in dist.items()}

    def _choose_rule(self):
        if self.rule_variety == 1:
            self.rule_type = "hops"
        else:
            self.rule_type = random.choice(["hops", "weight"])
        if self.rule_type == "hops":
            self.rule_param = random.randint(1, self.k_hops_max)
        else:
            # pick threshold ensuring at least one neighbor aside from reference
            possible = sorted(set(self.paths_cost.values()))
            possible = [p for p in possible if p != 0 and p <= self.weight_threshold_range]
            if not possible:
                self.rule_param = min(self.weight_threshold_range, max(self.paths_cost.values()))
            else:
                self.rule_param = random.choice(possible)

    def _compute_neighborhood(self) -> list:
        dmap = self.paths_hops if self.rule_type == "hops" else self.paths_cost
        neighbors = [name for name in self.ship_names if dmap[name] <= self.rule_param]
        return neighbors

    def _compute_ground_truth(self):
        members = self._compute_neighborhood()
        vals = [self.attributes[name][self.target_attribute] for name in members]
        if self.aggregator == "sum":
            val = sum(vals)
            self.correct_value = float(val)
        elif self.aggregator == "max":
            val = max(vals) if vals else 0
            self.correct_value = float(val)
        else:
            # avg
            val = sum(vals) / max(1, len(vals))
            if self.precision_decimals > 0:
                fmt = "{:0." + str(self.precision_decimals) + "f}"
                self.correct_value = float(fmt.format(val))
            else:
                self.correct_value = float(round(val))

    def _get_instructions(self) -> str:
        functions = [
            "- list_all(): list all ship names",
            "- rule(): show the current query specification",
            "- list_links(ship=\"Name\"): neighbors of a ship with link weights",
            "- paths(mode=\"hops\"|\"weight\"): shortest distances from the reference ship",
            "- peek(ship=\"Name\"): reveal a ship's attributes (shield, hull, cargo)",
            "- neighborhood(): list ships within the rule (may be locked at higher complexities)",
            "- submit(value=NUMBER): submit your final aggregated numeric answer",
            "- help(): reprint instructions",
        ]
        precision_note = f"Precision: if aggregator is average, round to {self.precision_decimals} decimal(s)."
        return (
            "ArmadaTally: You command a sky armada and must compute a tactical tally.\n"
            "Goal: Aggregate the target attribute over the neighborhood of a reference ship.\n"
            "Neighborhood selection uses either hop distance or path cost threshold, includes the reference ship.\n"
            f"{precision_note}\n"
            "Available functions:\n" + "\n".join(functions) + "\n"
            "Format actions as: <action>[func_name(param=value, ...)]</action>\n"
            "Example:\n" + self.sample_random_action() + "\n"
        )

    def get_task_suffix(self) -> str:
        members_scanned = len(self.scanned_ships.intersection(set(self._compute_neighborhood())))
        total_members = len(self._compute_neighborhood())
        return (
            f"Formation ships: {', '.join(self.ship_names)}\n"
            f"Reference: {self.reference_ship}\n"
            f"Target attribute: {self.target_attribute}\n"
            f"Aggregator: {self.aggregator}\n"
            f"Rule: {self.rule_type} <= {self.rule_param}\n"
            f"Progress: scanned {members_scanned}/{total_members} neighborhood ships\n"
            "Enter action: <action>[function_name(param=value,...)]</action>"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()

        self.turn_count = 0
        self.last_submitted_value = None

        self.ship_names = self._generate_names(self.num_ships)
        self.attributes = {}
        # Attribute scale increases with complexity indirectly via num_ships and thresholds
        base_max = 20 + self.num_ships * 2
        for name in self.ship_names:
            self.attributes[name] = {
                "shield": random.randint(5, base_max),
                "hull": random.randint(5, base_max),
                "cargo": random.randint(0, base_max),
            }

        self._build_graph()
        # Reference ship
        self.reference_ship = random.choice(self.ship_names)
        self.paths_hops = self._compute_hops(self.reference_ship)
        self.paths_cost = self._compute_costs(self.reference_ship)

        # Rule type and param
        self._choose_rule()

        # Aggregator selection
        if self.aggregator_variety == 1:
            self.aggregator = "sum"
        elif self.aggregator_variety == 2:
            self.aggregator = random.choice(["sum", "avg"])
        else:
            self.aggregator = random.choice(["sum", "avg", "max"])

        # Precision decimals
        self.precision_decimals = random.randint(0, self.precision_decimals_max) if self.aggregator == "avg" else 0

        # Target attribute
        self.target_attribute = random.choice(["shield", "hull", "cargo"])

        # Scanned set
        self.scanned_ships = set()

        # Compute ground truth
        self._compute_ground_truth()

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)
        if not parsed:
            obs = "Invalid action format. Use <action>[function_name(param=value,...)]</action>."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        name = parsed["name"]
        params = parsed["parameters"]
        terminated = False
        truncated = False
        reward = 0.0
        obs = ""

        if name == "list_all":
            obs = "Ships: " + ", ".join(self.ship_names)

        elif name == "help":
            obs = self._get_instructions()

        elif name == "rule":
            obs = f"Reference={self.reference_ship}; Attribute={self.target_attribute}; Aggregator={self.aggregator}; Rule={self.rule_type}<={self.rule_param}; Precision={self.precision_decimals}"

        elif name == "list_links":
            ship = params.get("ship")
            if not ship or ship not in self.ship_names:
                obs = "Protocol violation: 'list_links' requires a valid ship name."
            else:
                links = ", ".join([f"{n}(w={w})" for n, w in self.graph[ship]])
                obs = f"Links for {ship}: {links if links else 'none'}"

        elif name == "paths":
            mode = params.get("mode", self.rule_type)
            if mode not in ["hops", "weight"]:
                obs = "Protocol violation: 'paths' mode must be 'hops' or 'weight'."
            else:
                if mode == "hops":
                    pairs = [f"{n}:{self.paths_hops[n]}" for n in self.ship_names]
                    obs = "Hops from reference: " + ", ".join(pairs)
                else:
                    pairs = [f"{n}:{self.paths_cost[n]}" for n in self.ship_names]
                    obs = "Path cost from reference: " + ", ".join(pairs)

        elif name == "peek":
            ship = params.get("ship")
            if not ship or ship not in self.ship_names:
                obs = "Protocol violation: 'peek' requires ship=\"Name\"."
            else:
                self.scanned_ships.add(ship)
                attrs = self.attributes[ship]
                obs = f"{ship} attributes: shield={attrs['shield']}, hull={attrs['hull']}, cargo={attrs['cargo']}"
                # shaped reward milestones
                nbhd = set(self._compute_neighborhood())
                scanned_in_nbhd = len(self.scanned_ships.intersection(nbhd))
                total_nbhd = len(nbhd)
                if scanned_in_nbhd >= total_nbhd and total_nbhd > 0:
                    reward = 0.7
                elif scanned_in_nbhd >= max(1, total_nbhd // 2):
                    reward = 0.4
                elif scanned_in_nbhd > 0:
                    reward = 0.2

        elif name == "neighborhood":
            if self.tool_access_level < 2:
                obs = "Function locked: 'neighborhood' is unavailable at this complexity. Use paths() + peek()."
            else:
                members = self._compute_neighborhood()
                obs = "Neighborhood: " + ", ".join(members)

        elif name == "submit":
            value = params.get("value")
            if value is None:
                obs = "Protocol violation: 'submit' requires value=NUMBER."
                terminated = False
            else:
                try:
                    submitted = float(value)
                except Exception:
                    obs = "Protocol violation: 'submit' value must be numeric."
                    submitted = None
                if submitted is not None:
                    self.last_submitted_value = submitted
                    # compare with required precision
                    target = self.correct_value
                    equal = False
                    if self.aggregator == "avg" and self.precision_decimals > 0:
                        fmt = "{:0." + str(self.precision_decimals) + "f}"
                        sub_fmt = float(fmt.format(submitted))
                        equal = sub_fmt == target
                    else:
                        equal = float(submitted) == float(target)
                    if equal:
                        obs = "Success! Your submitted value matches the required aggregation."
                        reward = 1.0
                        terminated = True
                    else:
                        obs = "Failed! Submitted value does not match the required aggregation."
                        reward = 0.0
                        terminated = True

        else:
            obs = f"Unknown function '{name}'. Use help() to see available functions."

        if not terminated:
            if self.turn_count >= self.max_turns:
                truncated = True
                terminated = True
                obs = f"Reached max turns ({self.max_turns})."

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action or "<action>" not in action or "</action>" not in action:
            return None
        m = re.search(r"<action>(.*?)</action>", action, flags=re.DOTALL)
        if not m:
            return None
        content = m.group(1).strip()
        if not content.startswith("[") or not content.endswith("]"):
            return None
        call = content[1:-1].strip()
        fmatch = re.match(r"^([A-Za-z_]\w*)\((.*)\)$", call, flags=re.DOTALL)
        if not fmatch:
            return None
        fname = fmatch.group(1)
        pstr = fmatch.group(2).strip()
        params: Dict[str, Any] = {}
        if pstr:
            pairs = re.findall(r'(\w+)\s*=\s*([^,]+?)(?:,|$)', pstr, flags=re.DOTALL)
            for k, v in pairs:
                v = v.strip()
                if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                    params[k] = v[1:-1]
                elif re.match(r"^-?\d+$", v):
                    params[k] = int(v)
                elif re.match(r"^-?\d+\.\d+$", v):
                    params[k] = float(v)
                elif v.lower() in ["true", "false"]:
                    params[k] = True if v.lower() == "true" else False
                else:
                    params[k] = v
        return {"name": fname, "parameters": params}

    def sample_random_action(self) -> str:
        if not self.ship_names:
            return '<action>[help()]</action>'
        choice = random.choice(["peek", "list_all", "rule", "list_links", "paths"])
        if choice == "peek":
            ship = random.choice(self.ship_names)
            return f'<action>[peek(ship="{ship}")]</action>'
        elif choice == "list_links":
            ship = random.choice(self.ship_names)
            return f'<action>[list_links(ship="{ship}")]</action>'
        elif choice == "paths":
            mode = random.choice(["hops", "weight"])
            return f'<action>[paths(mode="{mode}")]</action>'
        elif choice == "rule":
            return '<action>[rule()]</action>'
        else:
            return '<action>[list_all()]</action>'


class ArmadaTallyEnvWithFeedback(ArmadaTallyEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        text = obs.lower()
        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint: Optional[str] = None

        if "invalid action format" in text:
            error_type = "FormatError"
            error_detail["issue"] = "bad_action_tags_or_brackets"
            hint = "Use <action>[function_name(param=value,...)]</action> exactly."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "list_links" in text:
                error_detail["violation"] = "missing_or_invalid_ship_name"
                hint = "Call list_all() to see valid names, then list_links(ship=\"Name\")."
            elif "paths" in text:
                error_detail["violation"] = "invalid_mode"
                hint = "Use paths(mode=\"hops\") or paths(mode=\"weight\")."
            elif "peek" in text:
                error_detail["violation"] = "missing_or_invalid_ship_name"
                hint = "Provide a valid ship: peek(ship=\"Name\")."
            elif "submit" in text:
                error_detail["violation"] = "missing_or_non_numeric_value"
                hint = "Provide a number: submit(value=123) or submit(value=12.5)."
            else:
                error_detail["violation"] = "unspecified_protocol"
                hint = "Use help() and follow parameter requirements."

        elif "function locked" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "locked_function_neighborhood"
            hint = "Use paths() to find ships within rule and peek() to gather attributes."

        elif "unknown function" in text:
            error_type = "UnsupportedAction"
            error_detail["name"] = re.findall(r"unknown function '([^']+)'", obs, flags=re.IGNORECASE)
            hint = "Use help() to see supported functions."

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Start with rule() and list_all(), then peek() neighborhood ships before submit()."

        elif "failed! submitted value does not match" in text:
            error_type = "WrongDecision"
            error_detail["expected"] = self.correct_value
            error_detail["got"] = self.last_submitted_value
            hint = "Verify the neighborhood via paths() and recheck rounding: if average, round to the specified precision."

        elif "success! your submitted value matches" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "reference": self.reference_ship,
                "attribute": self.target_attribute,
                "aggregator": self.aggregator,
                "rule": f"{self.rule_type}<={self.rule_param}",
                "precision": self.precision_decimals,
            }
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Begin with rule() to understand the task, then list_all() and paths() to identify the neighborhood.",
            "turn": 0,
            "state": {
                "reference": self.reference_ship,
                "attribute": self.target_attribute,
                "aggregator": self.aggregator,
                "rule": f"{self.rule_type}<={self.rule_param}",
                "precision": self.precision_decimals,
            },
        }
        return obs, info