from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List

class SignalHarnessMapperEnv(Env):
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
            "init_project": {
                "description": "Initialize a harness mapping project.",
                "parameters": [{"name": "name", "type": "string"}],
                "returns": "OK if initialized",
                "example": "\\boxed{init_project|name=SHM-001}",
            },
            "list_pins": {
                "description": "List sensor pins and signal types.",
                "parameters": [],
                "returns": "List of pins",
                "example": "\\boxed{list_pins}",
            },
            "list_channels": {
                "description": "List logger channels and accepted signal types.",
                "parameters": [],
                "returns": "List of channels",
                "example": "\\boxed{list_channels}",
            },
            "reveal_clue": {
                "description": "Reveal a clue about the hidden target channel for a specific pin.",
                "parameters": [{"name": "pin", "type": "string"}],
                "returns": "Clue text",
                "example": "\\boxed{reveal_clue|pin=S1}",
            },
            "test_match": {
                "description": "Check basic type compatibility of a pin-channel pair.",
                "parameters": [{"name": "pin", "type": "string"}, {"name": "ch", "type": "string"}],
                "returns": "True/False",
                "example": "\\boxed{test_match|pin=S2;ch=L3}",
            },
            "confirm_pair": {
                "description": "Confirm that a compatible pin-channel pair is considered for final mapping.",
                "parameters": [{"name": "pin", "type": "string"}, {"name": "ch", "type": "string"}],
                "returns": "Recorded confirmation or rejection",
                "example": "\\boxed{confirm_pair|pin=S2;ch=L3}",
            },
            "finalize_mapping": {
                "description": "Submit final mapping pairs in format pin->channel separated by commas.",
                "parameters": [{"name": "pairs", "type": "string"}],
                "returns": "Success or failure",
                "example": "\\boxed{finalize_mapping|pairs=S1->L3,S2->L1,S3->L2}",
            },
            "help": {
                "description": "Return tool catalog and usage.",
                "parameters": [],
                "returns": "Tool list",
                "example": "\\boxed{help}",
            },
        }
        self.signal_types = [
            "analog",
            "digital",
            "thermocouple",
            "pressure",
            "frequency",
            "current",
            "voltage",
        ]

    def _get_instructions(self) -> str:
        lines = []
        lines.append("You are assembling a signal harness that maps SENSOR pins to LOGGER channels.")
        lines.append("Your objective: produce an injective mapping from pins to channels that exactly preserves signal types.")
        lines.append("Hidden reference mapping exists. You must uncover it using tools and submit it.")
        lines.append("Protocol prerequisites before finalization:")
        lines.append("- Call init_project once.")
        lines.append("- Call list_pins and list_channels at least once.")
        lines.append("- Confirm each submitted pair via confirm_pair after ensuring type compatibility.")
        lines.append(f"- Do not finalize until you have made at least {self.required_steps} valid tool calls.")
        lines.append("")
        lines.append("Available tools and format:")
        lines.append("- Use \\boxed{tool|arg1=value1;arg2=value2} for actions.")
        lines.append("- Examples:")
        lines.append("  \\boxed{init_project|name=SHM-001}")
        lines.append("  \\boxed{list_pins}")
        lines.append("  \\boxed{list_channels}")
        lines.append("  \\boxed{reveal_clue|pin=S1}")
        lines.append("  \\boxed{test_match|pin=S1;ch=L3}")
        lines.append("  \\boxed{confirm_pair|pin=S1;ch=L3}")
        lines.append("  \\boxed{finalize_mapping|pairs=S1->L3,S2->L1,...}")
        lines.append("")
        lines.append("Clue system:")
        lines.append("- Each pin has clues about its target channel index (e.g., parity, modulo, threshold).")
        lines.append("- Multiple clues may be needed to deduce the exact mapping.")
        lines.append("")
        lines.append("Constraints:")
        lines.append("- Mapping must be injective (each pin maps to a unique channel).")
        lines.append("- Signal type must match (analog->analog, etc.).")
        lines.append("")
        lines.append(self._task_overview_text())
        return "\n".join(lines)

    def _task_overview_text(self) -> str:
        pcount = len(self.sensor_pins)
        lines = []
        lines.append(f"Task overview: {pcount} pins (S1..S{pcount}) and {pcount} channels (L1..L{pcount}).")
        lines.append("Each pin and channel has a signal_type attribute. The hidden mapping is a permutation.")
        return "\n".join(lines)

    def get_task_suffix(self) -> str:
        pcount = len(self.sensor_pins)
        confirmed_count = len(self.confirmed_pairs)
        lines = []
        lines.append(f"State: turns={self.turn_count}/{self.max_turns}, tool_calls={self.steps_taken}, required_calls={self.required_steps}.")
        lines.append(f"Pins={pcount}, confirmed_pairs={confirmed_count}. Project_initialized={'yes' if self.project_initialized else 'no'}.")
        lines.append("Respond with a single action using \\boxed{...} as specified.")
        return "\n".join(lines)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self.required_steps = random.randint(self.min_required_steps, self.max_required_steps)

        pcount = random.randint(max(3, self.complexity + 1), min(5 + self.complexity, 20))
        pin_types = [random.choice(self.signal_types) for _ in range(pcount)]
        ch_types = pin_types[:]  # same multiset of types
        random.shuffle(ch_types)

        self.sensor_pins = {}
        self.logger_channels = {}
        for i in range(1, pcount + 1):
            pid = f"S{i}"
            self.sensor_pins[pid] = {"id": pid, "signal_type": pin_types[i - 1], "meta": {"label": f"Pin {i}"}}
        for j in range(1, pcount + 1):
            cid = f"L{j}"
            self.logger_channels[cid] = {"id": cid, "signal_type": ch_types[j - 1], "meta": {"label": f"Channel {j}"}}

        # Build a hidden mapping that preserves types
        # For each pin type, find a channel with same type ensuring permutation injectively
        available_by_type: Dict[str, List[str]] = {}
        for cid, c in self.logger_channels.items():
            available_by_type.setdefault(c["signal_type"], []).append(cid)
        for t in available_by_type:
            random.shuffle(available_by_type[t])

        self.hidden_mapping = {}
        for pid, p in self.sensor_pins.items():
            t = p["signal_type"]
            cid = available_by_type[t].pop(0)
            self.hidden_mapping[pid] = cid

        # Precompute clues for each pin based on target channel index
        self.pin_clues: Dict[str, List[str]] = {}
        for pid, cid in self.hidden_mapping.items():
            idx = int(cid[1:])  # numeric index
            clues = []
            parity = "even" if idx % 2 == 0 else "odd"
            clues.append(f"Parity clue: target channel index is {parity}.")
            modk = random.choice([3, 4, 5])
            clues.append(f"Modulo clue: target index ≡ {idx % modk} (mod {modk}).")
            threshold = random.randint(1, max(1, pcount - 1))
            comp = ">" if idx > threshold else "<= or =="
            clues.append(f"Threshold clue: target index {comp} {threshold}.")
            self.pin_clues[pid] = clues

        self.project_initialized = False
        self.seen_pins = False
        self.seen_channels = False
        self.confirmed_pairs: Dict[str, str] = {}
        self.steps_taken = 0
        self.turn_count = 0

        instructions = self._get_instructions()
        info = {"suffix": self.get_task_suffix()}
        return instructions, info

    def _parse_action(self, action: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        if not isinstance(action, str):
            return None
        m = re.search(r"\\boxed\{(.*)\}", action.strip(), flags=re.DOTALL)
        if not m:
            return None
        inner = m.group(1).strip()
        parts = inner.split("|", 1)
        tool = parts[0].strip()
        args: Dict[str, Any] = {}
        if len(parts) > 1:
            arg_str = parts[1].strip()
            # Special handling for pairs list
            if tool == "finalize_mapping" and arg_str.startswith("pairs="):
                pairs_str = arg_str[len("pairs="):].strip()
                args["pairs_raw"] = pairs_str
                # Also parse into list
                pairs_list = []
                if pairs_str:
                    for tok in pairs_str.split(","):
                        tok = tok.strip()
                        mt = re.match(r"(S\d+)\s*->\s*(L\d+)", tok)
                        if mt:
                            pairs_list.append((mt.group(1), mt.group(2)))
                args["pairs"] = pairs_list
            else:
                for kv in arg_str.split(";"):
                    kv = kv.strip()
                    if not kv:
                        continue
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        args[k.strip()] = v.strip()
        return (tool, args)

    def sample_random_action(self) -> str:
        choices = [
            "\\boxed{help}",
            "\\boxed{init_project|name=SHM-001}",
            "\\boxed{list_pins}",
            "\\boxed{list_channels}",
        ]
        # Add a random pin
        if self.sensor_pins:
            pin = random.choice(list(self.sensor_pins.keys()))
            choices += [
                f"\\boxed{{reveal_clue|pin={pin}}}",
            ]
            # Add a random channel for tests
            if self.logger_channels:
                ch = random.choice(list(self.logger_channels.keys()))
                choices += [
                    f"\\boxed{{test_match|pin={pin};ch={ch}}}",
                    f"\\boxed{{confirm_pair|pin={pin};ch={ch}}}",
                ]
        # Add a sample finalize with empty
        choices += ["\\boxed{finalize_mapping|pairs=S1->L1}"]
        return random.choice(choices)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1

        parsed = self._parse_action(action)
        info: Dict[str, Any] = {"suffix": self.get_task_suffix()}
        if parsed is None:
            obs = "Invalid action format: Use \\boxed{tool|arg1=value1;arg2=value2}."
            reward = LanguageGameReward.format_error_reward
            return obs, reward, True, False, info

        tool, args = parsed

        if tool not in self.tools:
            obs = f"Unsupported tool: {tool}. Episode terminated."
            return obs, 0.0, True, False, info  # Fixed: was -0.05, failures should be 0.0

        terminated = False
        truncated = False
        reward = 0.0
        obs_lines = []

        def require_project():
            if not self.project_initialized:
                return "Protocol violation: init_project must be called first."
            return None

        if tool == "help":
            obs_lines.append("Tool catalog:")
            for name, meta in self.tools.items():
                obs_lines.append(f"- {name}: {meta['description']}")
            self.steps_taken += 1

        elif tool == "init_project":
            n = args.get("name", None)
            self.project_initialized = True
            obs_lines.append(f"Project initialized: {n if n else 'unnamed'}.")
            self.steps_taken += 1

        elif tool == "list_pins":
            pv = require_project()
            if pv:
                obs_lines.append(pv)
            else:
                self.seen_pins = True
                obs_lines.append("Sensor pins:")
                for pid, p in sorted(self.sensor_pins.items(), key=lambda kv: int(kv[0][1:])):
                    obs_lines.append(f"  {pid}: type={p['signal_type']}")
                self.steps_taken += 1

        elif tool == "list_channels":
            pv = require_project()
            if pv:
                obs_lines.append(pv)
            else:
                self.seen_channels = True
                obs_lines.append("Logger channels:")
                for cid, c in sorted(self.logger_channels.items(), key=lambda kv: int(kv[0][1:])):
                    obs_lines.append(f"  {cid}: type={c['signal_type']}")
                self.steps_taken += 1

        elif tool == "reveal_clue":
            pv = require_project()
            if pv:
                obs_lines.append(pv)
            else:
                pid = args.get("pin")
                if pid not in self.sensor_pins:
                    obs_lines.append(f"Protocol violation: unknown pin '{pid}'.")
                else:
                    clues = self.pin_clues.get(pid, [])
                    if not clues:
                        obs_lines.append(f"No more clues available for {pid}.")
                    else:
                        obs_lines.append(f"Clue for {pid}: {clues.pop(0)}")
                    self.steps_taken += 1

        elif tool == "test_match":
            pv = require_project()
            if pv:
                obs_lines.append(pv)
            else:
                pid = args.get("pin")
                cid = args.get("ch")
                if pid not in self.sensor_pins or cid not in self.logger_channels:
                    obs_lines.append("Protocol violation: invalid pin or channel.")
                else:
                    ok = self.sensor_pins[pid]["signal_type"] == self.logger_channels[cid]["signal_type"]
                    obs_lines.append(f"Compatibility test for {pid} and {cid}: {'compatible' if ok else 'incompatible'}.")
                    self.steps_taken += 1

        elif tool == "confirm_pair":
            pv = require_project()
            if pv:
                obs_lines.append(pv)
            else:
                pid = args.get("pin")
                cid = args.get("ch")
                if pid not in self.sensor_pins or cid not in self.logger_channels:
                    obs_lines.append("Protocol violation: invalid pin or channel.")
                else:
                    ok = self.sensor_pins[pid]["signal_type"] == self.logger_channels[cid]["signal_type"]
                    if not ok:
                        obs_lines.append(f"Rejection: {pid}({self.sensor_pins[pid]['signal_type']}) cannot map to {cid}({self.logger_channels[cid]['signal_type']}).")
                    else:
                        self.confirmed_pairs[pid] = cid
                        obs_lines.append(f"Confirmed consideration: {pid} -> {cid}.")
                    self.steps_taken += 1

        elif tool == "finalize_mapping":
            pv = require_project()
            if pv:
                obs_lines.append(pv)
            elif not self.seen_pins or not self.seen_channels:
                obs_lines.append("Protocol violation: must list_pins and list_channels before finalization.")
            elif self.steps_taken < self.required_steps:
                obs_lines.append(f"Protocol violation: too early to finalize. Required calls={self.required_steps}, current={self.steps_taken}.")
            else:
                pairs = args.get("pairs", [])
                provided_map: Dict[str, str] = {}
                injective_ok = True
                used_channels = set()
                # Validate format
                valid_pairs = True
                for pid, cid in pairs:
                    if pid not in self.sensor_pins or cid not in self.logger_channels:
                        valid_pairs = False
                        break
                    provided_map[pid] = cid
                    if cid in used_channels:
                        injective_ok = False
                    used_channels.add(cid)

                if not valid_pairs:
                    obs_lines.append("Failure: finalize_mapping contains unknown pin or channel.")
                    reward = 0.0
                    terminated = True
                else:
                    # Check injectivity and size
                    pcount = len(self.sensor_pins)
                    if len(provided_map) != pcount or not injective_ok:
                        obs_lines.append("Failure: mapping must be injective and cover all pins exactly once.")
                        reward = 0.0
                        terminated = True
                    else:
                        # Check confirmation prerequisite
                        missing_conf = []
                        for pid, cid in provided_map.items():
                            if self.confirmed_pairs.get(pid) != cid:
                                missing_conf.append(pid)
                        if missing_conf:
                            obs_lines.append(f"Protocol violation: the following pairs are not confirmed: {', '.join(missing_conf)}.")
                            # No termination; allow correction
                        else:
                            # Check correctness
                            if all(self.hidden_mapping.get(pid) == cid for pid, cid in provided_map.items()):
                                obs_lines.append("Success: Correct harness mapping achieved.")
                                reward = 1.0
                                terminated = True
                            else:
                                obs_lines.append("Failure: mapping incorrect.")
                                reward = 0.0
                                terminated = True
                self.steps_taken += 1

        # Timeout check
        if not terminated and self.turn_count >= self.max_turns:
            obs_lines.append("Timeout: maximum turns reached.")
            truncated = True
            terminated = True
            reward = 0.0

        obs = "\n".join(obs_lines) if obs_lines else "OK."
        info["suffix"] = self.get_task_suffix()
        return obs, reward, terminated, truncated, info


class SignalHarnessMapperEnvWithFeedback(SignalHarnessMapperEnv):
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
            error_detail["issue"] = "missing_or_bad_boxed_format"
            hint = "Use \\boxed{tool|arg=value} with the correct tool and parameters."

        elif "unsupported tool" in text:
            error_type = "UnsupportedAction"
            error_detail["tool"] = "unknown"
            hint = "Call help to see valid tools, then choose one."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "init_project" in text:
                error_detail["violation"] = "missing_init_project"
                hint = "Start with \\boxed{init_project|name=...}."
            elif "list_pins and list_channels" in text:
                error_detail["violation"] = "missing_listing"
                hint = "Call \\boxed{list_pins} and \\boxed{list_channels} before finalize."
            elif "too early to finalize" in text:
                error_detail["violation"] = "insufficient_steps"
                hint = "Perform more investigative tool calls (reveal_clue, test_match, confirm_pair) before finalizing."
            elif "invalid pin or channel" in text:
                error_detail["violation"] = "bad_identifier"
                hint = "Use IDs like S1..S{n} and L1..L{n}. Check with list_pins/list_channels."
            elif "not confirmed" in text:
                error_detail["violation"] = "missing_confirmations"
                hint = "Confirm each pair via \\boxed{confirm_pair|pin=Sx;ch=Ly} after testing compatibility."

        elif "failure: mapping incorrect" in text:
            error_type = "WrongDecision"
            error_detail["expected_pairs_count"] = len(self.sensor_pins)
            error_detail["note"] = "submitted mapping does not match hidden reference"
            hint = "Gather more clues for ambiguous pins and ensure injective mapping consistent with type constraints."

        elif "failure: mapping must be injective" in text:
            error_type = "WrongDecision"
            error_detail["issue"] = "non_injective_or_incomplete"
            hint = "Map each pin exactly once to a unique channel and cover all pins."

        elif "failure: finalize_mapping contains unknown pin or channel" in text:
            error_type = "WrongDecision"
            error_detail["issue"] = "unknown_ids_in_submission"
            hint = "Verify IDs via list_pins and list_channels before submitting."

        elif "timeout" in text:
            error_type = "Timeout"
            error_detail["outcome"] = "max_turns_reached"
            hint = "Plan actions to complete within the turn limit; start with init_project, then inspect, probe, confirm, finalize."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["required_steps"] = getattr(self, "required_steps", None)
            diagnostic["steps_taken"] = getattr(self, "steps_taken", None)
            diagnostic["pins_count"] = len(getattr(self, "sensor_pins", {}))
            diagnostic["confirmed_count"] = len(getattr(self, "confirmed_pairs", {}))
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start with \\boxed{init_project|name=...}, then \\boxed{list_pins} and \\boxed{list_channels}. Use \\boxed{reveal_clue|pin=Sx} and \\boxed{test_match|pin=Sx;ch=Ly} before confirming and finalizing.",
            "turn": 0,
            "required_steps": getattr(self, "required_steps", None),
            "steps_taken": 0,
            "pins_count": len(getattr(self, "sensor_pins", {})),
            "confirmed_count": 0,
        }
        return obs, info
