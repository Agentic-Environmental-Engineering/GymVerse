from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional

class ForgeBenchOrchestratorEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        max_turns: Optional[int] = 100,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.max_turns = max_turns if max_turns is not None else 100
        self.min_required_steps = self.complexity
        self.max_required_steps = self.complexity * 2
        self._init_database()
        self.reset()

    def _init_database(self):
        self.tools = {
            "load_blueprint": {
                "description": "Load the target blueprint by id.",
                "parameters": [{"name": "id", "type": "string"}],
                "returns": "Blueprint loaded"
            },
            "select_stock": {
                "description": "Select a stock piece by id. Must match required material.",
                "parameters": [{"name": "id", "type": "string"}],
                "returns": "Current piece set from stock"
            },
            "cut_to": {
                "description": "Cut the current piece to target length (cannot increase length).",
                "parameters": [{"name": "length", "type": "float"}],
                "returns": "Piece length adjusted"
            },
            "drill": {
                "description": "Drill evenly spaced holes along the piece.",
                "parameters": [{"name": "count", "type": "int"}, {"name": "diameter", "type": "float"}],
                "returns": "Holes drilled"
            },
            "sand": {
                "description": "Sand the surface with a grit value (80, 120, 220). Reduces roughness.",
                "parameters": [{"name": "grit", "type": "int"}],
                "returns": "Surface roughness reduced"
            },
            "coat": {
                "description": "Apply a coating of a given color.",
                "parameters": [{"name": "color", "type": "string"}],
                "returns": "Coating applied"
            },
            "validate": {
                "description": "Validate the current piece against the loaded blueprint.",
                "parameters": [],
                "returns": "Validation result"
            }
        }
        materials = ["oak", "pine", "aluminum", "steel", "acrylic"]
        num_stocks = 8 + self.complexity
        self.stocks = {}
        for i in range(num_stocks):
            sid = f"S{i+1:03d}"
            mat = random.choice(materials)
            length = round(random.uniform(20.0, 120.0), 1)
            self.stocks[sid] = {
                "id": sid,
                "material": mat,
                "length": length,
            }
        self.grit_effect = {80: 30, 120: 20, 220: 15}
        self.blueprints = {}

    def _generate_task_requiring_n_steps(self, required_steps: int) -> Dict[str, Any]:
        baseline = 3  # load + select + validate
        ops_needed = max(required_steps - baseline, 0)
        include_cut = True
        include_drill = True if ops_needed >= 2 else (ops_needed >= 1 and random.choice([True, False]))
        include_coat = True if ops_needed >= 3 else (random.choice([True, False]) if ops_needed > 0 else False)
        fixed_ops = int(include_cut) + int(include_drill) + int(include_coat)
        min_sand_steps = max(ops_needed - fixed_ops, 0)

        target_material = random.choice(list({s["material"] for s in self.stocks.values()}))
        candidate_stocks = [s for s in self.stocks.values() if s["material"] == target_material]
        if not candidate_stocks:
            any_stock = random.choice(list(self.stocks.values()))
            target_material = any_stock["material"]
            candidate_stocks = [any_stock]

        chosen_stock = random.choice(candidate_stocks)
        start_length = chosen_stock["length"]
        target_length = round(random.uniform(10.0, max(10.0, start_length - 5.0)), 1)
        if target_length >= start_length:
            target_length = max(5.0, round(start_length - random.uniform(5.0, 15.0), 1))

        holes = None
        if include_drill:
            holes = {"count": random.randint(2, 6), "diameter": round(random.uniform(0.3, 1.2), 2)}

        coat = None
        if include_coat:
            coat = random.choice(["red", "blue", "black", "clear", "green"])

        start_roughness = 100
        max_reduction_per_sand = max(self.grit_effect.values())
        target_roughness = start_roughness - max_reduction_per_sand * min_sand_steps
        if target_roughness < 0:
            target_roughness = 0
        blueprint_id = f"BP{random.randint(1000,9999)}"
        bp = {
            "id": blueprint_id,
            "material": target_material,
            "target_length": target_length,
            "holes": holes,
            "coat": coat,
            "max_roughness": target_roughness if min_sand_steps > 0 else 100,
            "tolerances": {"length": 0.5, "diameter": 0.1},
            "min_sand_steps": min_sand_steps,
            "fixed_ops": {"cut": include_cut, "drill": include_drill, "coat": include_coat}
        }
        self.blueprints[blueprint_id] = bp
        return {
            "required_steps": required_steps,
            "blueprint_id": blueprint_id,
            "start_stock_id": chosen_stock["id"],
        }

    def _get_instructions(self) -> str:
        tools_list = []
        for name, meta in self.tools.items():
            params = ", ".join([p["name"] for p in meta["parameters"]])
            tools_list.append(f"- {name}({params}): {meta['description']}")
        tools_desc = "\n".join(tools_list)
        return (
            "You are operating a fabrication workbench. Use one tool per turn to transform a stock into an artifact "
            "that matches the blueprint. Tools are stateful and have prerequisites. The episode succeeds only when "
            "you call validate() and the artifact meets all blueprint specs. Invalid action format or unsupported tools end the episode.\n"
            "Available tools:\n" + tools_desc
        )

    def get_task_suffix(self) -> str:
        bp_id = self.task["blueprint_id"]
        bp = self.blueprints[bp_id]
        stock_id = self.task["start_stock_id"]
        stock = self.stocks[stock_id]
        pieces = []
        pieces.append(f"Target blueprint {bp_id}: material={bp['material']}, target_length={bp['target_length']}±{bp['tolerances']['length']}")
        if bp["holes"]:
            pieces.append(f"holes=count {bp['holes']['count']}, diameter {bp['holes']['diameter']}±{bp['tolerances']['diameter']}")
        else:
            pieces.append("holes=none")
        pieces.append(f"max_roughness={bp['max_roughness']}")
        pieces.append(f"coat={'none' if bp['coat'] is None else bp['coat']}")
        pieces.append(f"required_min_steps={self.task['required_steps']}")
        pieces.append(f"candidate_stock={stock_id} material={stock['material']} length={stock['length']}")
        state_line = " | ".join(pieces)
        fmt = "Respond with a single tool call in \\boxed{tool_name(param1=value,param2=value)}. Numbers may be integers or floats; no quotes."
        return state_line + "\n" + fmt

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        req_steps = random.randint(self.min_required_steps, self.max_required_steps)
        self.task = self._generate_task_requiring_n_steps(req_steps)
        self.turn_count = 0
        self.steps_taken = 0
        self.blueprint = None
        self.current_piece = None
        self.execution_state = {
            "blueprint_loaded": False,
            "stock_selected": False,
            "sand_count": 0,
            "last_result": None,
        }
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        if not isinstance(action, str):
            return None
        m = re.search(r"\\boxed\{(.*)\}", action)
        if not m:
            return None
        inner = m.group(1).strip()
        if "(" in inner and inner.endswith(")"):
            tool_name = inner[:inner.index("(")].strip()
            params_str = inner[inner.index("(")+1:-1].strip()
            args = {}
            if params_str:
                parts = [p.strip() for p in params_str.split(",") if p.strip()]
                for part in parts:
                    if "=" not in part:
                        return None
                    k, v = [x.strip() for x in part.split("=", 1)]
                    if re.fullmatch(r"-?\d+", v):
                        args[k] = int(v)
                    elif re.fullmatch(r"-?\d+\.\d+", v):
                        args[k] = float(v)
                    else:
                        args[k] = v
            return (tool_name, args)
        else:
            tool_name = inner.strip()
            return (tool_name, {})

    def sample_random_action(self) -> str:
        if not self.execution_state["blueprint_loaded"]:
            return f"\\boxed{{load_blueprint(id={self.task['blueprint_id']})}}"
        if not self.execution_state["stock_selected"]:
            return f"\\boxed{{select_stock(id={self.task['start_stock_id']})}}"
        bp = self.blueprints[self.task["blueprint_id"]]
        if bp["fixed_ops"]["cut"] and (abs((self.current_piece['length'] if self.current_piece else self.stocks[self.task['start_stock_id']]['length']) - bp["target_length"]) > bp["tolerances"]["length"]):
            return f"\\boxed{{cut_to(length={bp['target_length']})}}"
        if bp["fixed_ops"]["drill"] and (not self.current_piece or len(self.current_piece.get("holes", [])) != (bp['holes']['count'] if bp['holes'] else 0)):
            if bp["holes"]:
                return f"\\boxed{{drill(count={bp['holes']['count']},diameter={bp['holes']['diameter']})}}"
        if bp["max_roughness"] < (self.current_piece['roughness'] if self.current_piece else 100):
            return "\\boxed{sand(grit=120)}"
        if bp["coat"] and (not self.current_piece or self.current_piece.get("coat") != bp["coat"]):
            return f"\\boxed{{coat(color={bp['coat']})}}"
        return "\\boxed{validate()}"

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        if tool_name == "load_blueprint":
            bp_id = str(args.get("id"))
            if bp_id not in self.blueprints:
                raise ValueError(f"Blueprint not found: {bp_id}")
            self.blueprint = self.blueprints[bp_id]
            self.execution_state["blueprint_loaded"] = True
            return f"Loaded blueprint {bp_id}"
        if tool_name == "select_stock":
            if not self.execution_state["blueprint_loaded"]:
                raise RuntimeError("Must load blueprint before selecting stock.")
            sid = str(args.get("id"))
            if sid not in self.stocks:
                raise ValueError(f"Stock not found: {sid}")
            stock = self.stocks[sid]
            if stock["material"] != self.blueprint["material"]:
                raise RuntimeError(f"Material mismatch: stock material {stock['material']} vs blueprint {self.blueprint['material']}")
            self.current_piece = {
                "material": stock["material"],
                "length": stock["length"],
                "holes": [],
                "roughness": 100,
                "coat": None,
            }
            self.execution_state["stock_selected"] = True
            self.execution_state["sand_count"] = 0
            return f"Selected stock {sid} material={stock['material']} length={stock['length']}"
        if tool_name == "cut_to":
            if not self.execution_state["stock_selected"]:
                raise RuntimeError("Must select stock before cutting.")
            target_len = float(args.get("length"))
            if target_len > self.current_piece["length"]:
                raise RuntimeError("Cannot increase length when cutting.")
            self.current_piece["length"] = round(target_len, 1)
            return f"Cut piece to length {self.current_piece['length']}"
        if tool_name == "drill":
            if not self.execution_state["stock_selected"]:
                raise RuntimeError("Must select stock before drilling.")
            count = int(args.get("count"))
            diameter = float(args.get("diameter"))
            if count <= 0 or diameter <= 0:
                raise ValueError("Invalid hole parameters.")
            self.current_piece["holes"] = [{"diameter": diameter} for _ in range(count)]
            return f"Drilled {count} holes with diameter {round(diameter,2)}"
        if tool_name == "sand":
            if not self.execution_state["stock_selected"]:
                raise RuntimeError("Must select stock before sanding.")
            grit = int(args.get("grit"))
            if grit not in self.grit_effect:
                raise ValueError("Unsupported grit. Use 80, 120, or 220.")
            effect = self.grit_effect[grit]
            before = self.current_piece["roughness"]
            after = max(0, before - effect)
            self.current_piece["roughness"] = after
            self.execution_state["sand_count"] += 1
            return f"Sanded with grit {grit}: roughness {before} -> {after}"
        if tool_name == "coat":
            if not self.execution_state["stock_selected"]:
                raise RuntimeError("Must select stock before coating.")
            color = str(args.get("color"))
            self.current_piece["coat"] = color
            return f"Applied coat color {color}"
        if tool_name == "validate":
            if not self.execution_state["blueprint_loaded"]:
                return "Validation failed: no blueprint loaded"
            if not self.execution_state["stock_selected"] or self.current_piece is None:
                return "Validation failed: no stock selected"
            bp = self.blueprint
            piece = self.current_piece
            reasons = []
            tol_len = bp["tolerances"]["length"]
            if abs(piece["length"] - bp["target_length"]) > tol_len:
                reasons.append(f"length {piece['length']} vs target {bp['target_length']}±{tol_len}")
            if bp["holes"]:
                expected_count = bp["holes"]["count"]
                expected_diam = bp["holes"]["diameter"]
                actual_count = len(piece["holes"])
                if actual_count != expected_count:
                    reasons.append(f"hole_count {actual_count} vs {expected_count}")
                elif actual_count > 0:
                    actual_diams = set(round(h["diameter"], 2) for h in piece["holes"])
                    if len(actual_diams) != 1:
                        reasons.append("hole diameters inconsistent")
                    else:
                        actual_diam = list(actual_diams)[0]
                        if abs(actual_diam - expected_diam) > bp["tolerances"]["diameter"]:
                            reasons.append(f"hole_diameter {actual_diam} vs {expected_diam}±{bp['tolerances']['diameter']}")
            else:
                if len(piece["holes"]) > 0:
                    reasons.append("holes present but blueprint requires none")
            if piece["roughness"] > bp["max_roughness"]:
                reasons.append(f"roughness {piece['roughness']} > max {bp['max_roughness']}")
            if bp["coat"]:
                if piece["coat"] != bp["coat"]:
                    reasons.append(f"coat {piece['coat']} vs required {bp['coat']}")
            result = "Validation OK" if not reasons else "Validation failed: " + "; ".join(reasons)
            return result
        raise ValueError(f"Unsupported tool: {tool_name}")

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)
        if not parsed:
            obs = "Invalid action format. Use \\boxed{tool_name(param=value,...)}."
            return obs, float(LanguageGameReward.format_error_reward), True, False, {"suffix": self.get_task_suffix()}
        tool_name, args = parsed
        if tool_name not in self.tools:
            obs = f"Unsupported tool: {tool_name}. Episode terminated."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
        try:
            result = self._execute_tool(tool_name, args)
            self.steps_taken += 1
            self.execution_state["last_result"] = result
            terminated = False
            truncated = False
            reward = 0.0
            extra = []
            if tool_name == "validate":
                if result.startswith("Validation OK"):
                    terminated = True
                    reward = 1.0
                    extra.append(f"Success: artifact matches {self.blueprint['id']}. Steps_taken={self.steps_taken}, required_min={self.task['required_steps']}")
                else:
                    extra.append("Validation did not pass. Continue fixing discrepancies.")
            if self.turn_count >= self.max_turns and not terminated:
                truncated = True
                terminated = True
                extra.append("Timeout: max turns reached.")
            piece_state = self._render_piece_state()
            obs = f"Tool {tool_name} executed.\nResult: {result}\nState: {piece_state}"
            if extra:
                obs += "\n" + "\n".join(extra)
            return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}
        except RuntimeError as e:
            obs = f"Protocol violation: {str(e)}. Episode terminated."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
        except Exception as e:
            obs = f"Execution error: {str(e)}. Episode terminated."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

    def _render_piece_state(self) -> str:
        bp_id = self.task["blueprint_id"]
        loaded = "yes" if self.execution_state["blueprint_loaded"] else "no"
        selected = "yes" if self.execution_state["stock_selected"] else "no"
        if not self.execution_state["stock_selected"] or self.current_piece is None:
            return f"blueprint_loaded={loaded} | stock_selected={selected} | piece=none"
        p = self.current_piece
        holes = len(p["holes"])
        diam = round(p["holes"][0]["diameter"], 2) if holes > 0 else "-"
        return f"blueprint_loaded={loaded}({bp_id}) | stock_selected={selected} | length={p['length']} | holes={holes} diam={diam} | roughness={p['roughness']} | coat={p['coat']}"

class ForgeBenchOrchestratorEnvWithFeedback(ForgeBenchOrchestratorEnv):
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
            error_detail["issue"] = "missing_or_malformed_boxed"
            hint = "Use \\boxed{tool_name(param=value,...)} with correct parameters."

        elif "unsupported tool" in text:
            error_type = "UnsupportedAction"
            error_detail["tool"] = self._extract_tool_name(action)
            hint = "Choose a tool from the provided list: load_blueprint, select_stock, cut_to, drill, sand, coat, validate."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "load blueprint" in text:
                error_detail["violation"] = "missing_blueprint_before_action"
                hint = f"First call load_blueprint(id={self.task['blueprint_id']})."
            elif "select stock" in text:
                error_detail["violation"] = "missing_stock_before_action"
                hint = f"Call select_stock(id={self.task['start_stock_id']}) after loading the blueprint."
            elif "increase length" in text:
                error_detail["violation"] = "cut_increase_length"
                hint = "cut_to cannot increase length; provide a target less than or equal to current length."
            else:
                error_detail["violation"] = "general_protocol_error"
                hint = "Check tool prerequisites in the instructions."

        elif "execution error" in text:
            error_type = "UnsupportedAction"
            error_detail["message"] = obs
            hint = "Verify parameters and tool availability."

        elif "validation failed" in text:
            error_type = "WrongDecision"
            mismatches = []
            if "length" in text:
                mismatches.append("length")
            if "hole_count" in text or "holes present" in text:
                mismatches.append("holes")
            if "hole_diameter" in text or "diameters inconsistent" in text:
                mismatches.append("hole_diameter")
            if "roughness" in text:
                mismatches.append("roughness")
            if "coat" in text:
                mismatches.append("coat")
            error_detail["mismatches"] = mismatches
            hint = self._build_hint_for_mismatches(mismatches)

        elif "success: artifact matches" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        elif "timeout" in text:
            error_type = "Timeout"
            error_detail["outcome"] = "max_turns_reached"
            hint = "Prioritize essential steps: load_blueprint, select_stock, cut_to, drill, sand as needed, coat, then validate."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = self._snapshot_state()
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint
        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": f"Start with \\boxed{{load_blueprint(id={self.task['blueprint_id']})}}, then \\boxed{{select_stock(id={self.task['start_stock_id']})}}.",
            "turn": 0,
            "state": self._snapshot_state(),
        }
        return obs, info

    def _extract_tool_name(self, action: str) -> Optional[str]:
        m = re.search(r"\\boxed\{([a-zA-Z_][a-zA-Z0-9_]*)", action)
        return m.group(1) if m else None

    def _build_hint_for_mismatches(self, mismatches):
        if not mismatches:
            return "Review the blueprint and validate again after adjustments."
        tips = []
        if "length" in mismatches:
            tips.append("Use cut_to(length=target_length) within tolerance.")
        if "holes" in mismatches:
            tips.append("Call drill(count=...,diameter=...) to match expected count.")
        if "hole_diameter" in mismatches:
            tips.append("Ensure drill diameter matches blueprint within tolerance.")
        if "roughness" in mismatches:
            tips.append("Use sand(grit=80|120|220) until roughness ≤ max_roughness.")
        if "coat" in mismatches:
            tips.append("Apply coat(color=required_color).")
        return " ".join(tips)

    def _snapshot_state(self):
        bp_id = self.task["blueprint_id"]
        bp = self.blueprints[bp_id]
        piece = None
        if getattr(self, "current_piece", None) is not None:
            piece = {
                "length": self.current_piece["length"],
                "holes": len(self.current_piece["holes"]),
                "diameter": round(self.current_piece["holes"][0]["diameter"], 2) if self.current_piece["holes"] else None,
                "roughness": self.current_piece["roughness"],
                "coat": self.current_piece["coat"],
            }
        return {
            "blueprint_id": bp_id,
            "required_min_steps": self.task["required_steps"],
            "blueprint_specs": {
                "material": bp["material"],
                "target_length": bp["target_length"],
                "holes": bp["holes"],
                "coat": bp["coat"],
                "max_roughness": bp["max_roughness"],
                "tolerances": bp["tolerances"],
                "min_sand_steps": bp["min_sand_steps"],
            },
            "piece": piece,
            "turn_count": getattr(self, "turn_count", None),
        }