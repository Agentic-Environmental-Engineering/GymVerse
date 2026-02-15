from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class BeaconCorridorNavigatorEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 28,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 28

        # Evolvable parameters
        self.complexity_params = {
            # Workspace radius (meters): bigger map increases search and path length ⇒ harder
            "world_radius": (12, 35),
            # Number of circular obstacles: more obstacles ⇒ harder routing and pushing
            "num_obstacles": (1, 7),
            # Obstacle max radius (cm): larger obstacles close corridors ⇒ harder
            "obstacle_max_radius_cm": (60, 220),
            # Rover max thrust per step (cm): REVERSED, smaller thrust ⇒ finer but harder to progress
            "thrust_cm": (35, 12),
            # Rotation per step (degrees): REVERSED, smaller rotation ⇒ harder to orient precisely
            "rotation_deg": (35, 10),
            # Required beacon-to-rover contact distance to push (cm): REVERSED, smaller tolerance ⇒ harder
            "push_contact_cm": (30, 12),
            # Extraction radius (cm): REVERSED, tighter goal region ⇒ harder
            "extract_radius_cm": (80, 28),
            # Start separation rover→beacon (m): larger separation ⇒ harder
            "start_sep_m": (2, 9),
        }

        self.param_variance = {
            "world_radius": 2,               # ±2m within 12-35m
            "num_obstacles": 1,              # ±1 obstacle
            "obstacle_max_radius_cm": 15,    # ±15cm
            "thrust_cm": 3,                  # ±3cm
            "rotation_deg": 3,               # ±3°
            "push_contact_cm": 2,            # ±2cm
            "extract_radius_cm": 3,          # ±3cm
            "start_sep_m": 1,                # ±1m
        }

        # Placeholder attributes populated in _apply_complexity_params
        self.world_radius: int = 0
        self.num_obstacles: int = 0
        self.obstacle_max_radius_cm: int = 0
        self.thrust_cm: int = 0
        self.rotation_deg: int = 0
        self.push_contact_cm: int = 0
        self.extract_radius_cm: int = 0
        self.start_sep_m: int = 0

        # State
        self.turn_count: int = 0
        self.rover_x: float = 0.0
        self.rover_y: float = 0.0
        self.rover_theta: float = 0.0  # degrees
        self.beacon_x: float = 0.0
        self.beacon_y: float = 0.0
        self.extract_x: float = 0.0
        self.extract_y: float = 0.0
        self.obstacles = []  # list of (x, y, r)
        self.invalid_reason: Optional[str] = None

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for name, (min_v, max_v) in self.complexity_params.items():
            center = min_v + (max_v - min_v) * normalized
            val = center
            if self.enable_param_randomization:
                var = self.param_variance.get(name, 0)
                if var > 0:
                    val = center + random.uniform(-var, var)
                    lo, hi = (max_v, min_v) if min_v > max_v else (min_v, max_v)
                    val = max(lo, min(hi, val))
            setattr(self, name, int(round(val)))

    def _within_world(self, x: float, y: float) -> bool:
        return (x**2 + y**2) <= (self.world_radius ** 2)

    def _collides_obstacles(self, x: float, y: float) -> bool:
        for ox, oy, r in self.obstacles:
            if ((x - ox) ** 2 + (y - oy) ** 2) <= (r ** 2):
                return True
        return False

    def _point_in_extract(self, x: float, y: float) -> bool:
        return ((x - self.extract_x) ** 2 + (y - self.extract_y) ** 2) <= ((self.extract_radius_cm/100.0) ** 2)

    def _spawn_obstacles(self):
        self.obstacles = []
        tries = 0
        target = self.num_obstacles
        while len(self.obstacles) < target and tries < 400:
            tries += 1
            r = random.uniform(self.obstacle_max_radius_cm*0.4, self.obstacle_max_radius_cm) / 100.0
            angle = random.uniform(0, 360)
            dist = random.uniform(0.3*self.world_radius, 0.9*self.world_radius)
            x = dist * (random.choice([-1,1])) * random.random()
            y = dist * (random.choice([-1,1])) * random.random()
            if not self._within_world(x, y):
                continue
            # Avoid placing obstacles too close to extraction or start
            if ((x - self.extract_x) ** 2 + (y - self.extract_y) ** 2) < (r + self.extract_radius_cm/100.0 + 0.8) ** 2:
                continue
            if self._distance(x, y, self.beacon_x, self.beacon_y) < r + 0.6:
                continue
            if self._distance(x, y, self.rover_x, self.rover_y) < r + 0.6:
                continue
            # Avoid overlap with existing
            overlap = False
            for ox, oy, orad in self.obstacles:
                if self._distance(x, y, ox, oy) < (r + orad + 0.25):
                    overlap = True
                    break
            if overlap:
                continue
            self.obstacles.append((x, y, r))
        # If failed to place enough, reduce count to ensure feasibility
        if len(self.obstacles) < target:
            self.obstacles = self.obstacles[:len(self.obstacles)]

    def _distance(self, x1, y1, x2, y2) -> float:
        return ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5

    def _random_edge_point(self, margin: float = 1.0) -> Tuple[float, float]:
        # near boundary but within world
        r = random.uniform(self.world_radius - margin, self.world_radius - 0.2)
        ang = random.uniform(0, 360)
        rad = ang * 3.1415926535 / 180.0
        return r * float.__mul__(1.0, (1.0)) * (1.0) * (1.0) * (1.0) * (1.0) if False else (r * (random.choice([1,-1]) * random.random()), r * (random.choice([1,-1]) * random.random()))

    def _safe_place_point(self, min_clear: float) -> Tuple[float, float]:
        # place within world and away from obstacles by min_clear
        for _ in range(300):
            x = random.uniform(-self.world_radius+0.5, self.world_radius-0.5)
            y = random.uniform(-self.world_radius+0.5, self.world_radius-0.5)
            if not self._within_world(x, y):
                continue
            if self._collides_obstacles(x, y):
                continue
            ok = True
            for ox, oy, r in self.obstacles:
                if self._distance(x, y, ox, oy) < r + min_clear:
                    ok = False
                    break
            if ok:
                return x, y
        # fallback center
        return 0.0, 0.0

    def _get_instructions(self) -> str:
        return (
            "You are piloting a rover to herd a beacon into the extraction circle.\n"
            "Goal: move the beacon so its position lies inside the extraction circle.\n"
            "Space: continuous 2D world (meters). Obstacles are circular; entering them is invalid.\n"
            "Rover can rotate and thrust; to push the beacon, you must be within contact distance and facing it.\n"
            "Available actions:\n"
            "- rotate deg=INT   (positive = clockwise, negative = counter-clockwise; limited by rotation_deg)\n"
            "- thrust cm=INT    (moves rover forward along heading; limited by thrust_cm per step)\n"
            "- push cm=INT      (when in contact and facing beacon within 30° cone, displaces beacon forward relative to rover heading; limited by thrust_cm per push)\n"
            "Rules:\n"
            "- Any move causing rover or beacon to overlap an obstacle or leave the world is invalid and ends the episode.\n"
            "- Beacon only moves via push. Thrust moves rover only. Rotate changes heading only.\n"
            "- Episode ends with success when beacon is inside extraction circle.\n"
            "- Max turns apply.\n"
            "Format your action in \\boxed{...}. Examples:\n"
            f"{self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        state = (
            f"Turn {self.turn_count}/{self.max_turns}\n"
            f"Rover: pos=({self.rover_x:.2f},{self.rover_y:.2f}) heading={self.rover_theta:.1f}°\n"
            f"Beacon: pos=({self.beacon_x:.2f},{self.beacon_y:.2f})\n"
            f"Extraction: center=({self.extract_x:.2f},{self.extract_y:.2f}) radius={self.extract_radius_cm/100.0:.2f}m\n"
            f"Obstacles: {len(self.obstacles)} circles\n"
            f"Limits: rotate<=±{self.rotation_deg}°, thrust<= {self.thrust_cm}cm, push<= {self.thrust_cm}cm, contact<= {self.push_contact_cm}cm\n"
            "Enter your action as \\boxed{rotate deg=...} or \\boxed{thrust cm=...} or \\boxed{push cm=...}"
        )
        return state

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        # Apply complexity-driven parameters (within bounds) and randomization
        self._apply_complexity_params()

        self.turn_count = 0
        self.invalid_reason = None
        self.obstacles = []

        # Place extraction at origin
        self.extract_x, self.extract_y = 0.0, 0.0

        # Place beacon at a safe distance along +X away from extraction
        beacon_dist = max(1.5, min(self.start_sep_m, self.world_radius - 2))
        self.beacon_x, self.beacon_y = beacon_dist, 0.0

        # Place rover further along +X so it can push beacon toward origin
        rover_offset = max(self.push_contact_cm / 100.0 + 0.5, 1.0)
        self.rover_x, self.rover_y = min(self.world_radius - 0.5, self.beacon_x + rover_offset), 0.0
        self.rover_theta = 180.0  # facing negative X toward beacon/extraction

        # Spawn obstacles after setting key positions
        self._spawn_obstacles()

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)
        if not parsed:
            obs = "INVALID ACTION FORMAT: Use \\boxed{...} with rotate|thrust|push and parameters."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        name = parsed.get("action", "")
        terminated = False
        truncated = False
        reward = 0.0
        message = ""

        def deg2rad(d): return d * 3.1415926535 / 180.0

        if name not in ("rotate", "thrust", "push"):
            message = "UNSUPPORTED ACTION: Allowed actions are rotate, thrust, push."
            return message, 0.0, True, False, {"suffix": self.get_task_suffix()}

        if name == "rotate":
            try:
                val = int(float(parsed.get("deg", "0")))
            except:
                message = "INVALID ACTION FORMAT: rotate requires deg=INT."
                return message, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
            if abs(val) > self.rotation_deg:
                message = f"PROTOCOL VIOLATION: rotation exceeds limit ±{self.rotation_deg}°."
                return message, 0.0, True, False, {"suffix": self.get_task_suffix()}
            self.rover_theta = (self.rover_theta + val) % 360.0
            message = f"Rotated by {val}°, new heading {self.rover_theta:.1f}°."

        elif name == "thrust":
            try:
                cm = int(float(parsed.get("cm", "0")))
            except:
                message = "INVALID ACTION FORMAT: thrust requires cm=INT."
                return message, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
            if cm < -self.thrust_cm or cm > self.thrust_cm:
                message = f"PROTOCOL VIOLATION: thrust exceeds limit ±{self.thrust_cm}cm."
                return message, 0.0, True, False, {"suffix": self.get_task_suffix()}
            dx = (cm/100.0) * (round(float(__import__('math').cos(deg2rad(self.rover_theta))), 10) if False else __import__('math').cos(deg2rad(self.rover_theta)))
            dy = (cm/100.0) * (__import__('math').sin(deg2rad(self.rover_theta)))
            nx = self.rover_x + dx
            ny = self.rover_y + dy
            if not self._within_world(nx, ny) or self._collides_obstacles(nx, ny):
                message = "INVALID MOVE: rover collided or left world."
                return message, 0.0, True, False, {"suffix": self.get_task_suffix()}
            self.rover_x, self.rover_y = nx, ny
            message = f"Thrust {cm}cm to ({self.rover_x:.2f},{self.rover_y:.2f})."

        elif name == "push":
            try:
                cm = int(float(parsed.get("cm", "0")))
            except:
                message = "INVALID ACTION FORMAT: push requires cm=INT."
                return message, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
            if cm < -self.thrust_cm or cm > self.thrust_cm:
                message = f"PROTOCOL VIOLATION: push exceeds limit ±{self.thrust_cm}cm."
                return message, 0.0, True, False, {"suffix": self.get_task_suffix()}
            # Check contact and facing
            dist = self._distance(self.rover_x, self.rover_y, self.beacon_x, self.beacon_y)
            if dist > (self.push_contact_cm / 100.0):
                message = "PROTOCOL VIOLATION: not in contact range to push."
                return message, 0.0, True, False, {"suffix": self.get_task_suffix()}
            # Facing constraint: vector rover->beacon aligned within 30°
            vx = self.beacon_x - self.rover_x
            vy = self.beacon_y - self.rover_y
            if dist == 0:
                facing_ok = True
            else:
                heading = deg2rad(self.rover_theta)
                hx, hy = __import__('math').cos(heading), __import__('math').sin(heading)
                dot = (vx*hx + vy*hy) / max(1e-9, dist)
                dot = max(-1.0, min(1.0, dot))
                angle = __import__('math').degrees(__import__('math').acos(dot))
                facing_ok = (angle <= 30.0)
            if not facing_ok:
                message = "PROTOCOL VIOLATION: rover not facing beacon within 30°."
                return message, 0.0, True, False, {"suffix": self.get_task_suffix()}
            # Displace beacon forward
            heading = deg2rad(self.rover_theta)
            bnx = self.beacon_x + (cm/100.0) * __import__('math').cos(heading)
            bny = self.beacon_y + (cm/100.0) * __import__('math').sin(heading)
            if not self._within_world(bnx, bny) or self._collides_obstacles(bnx, bny):
                message = "INVALID MOVE: beacon collided or left world during push."
                return message, 0.0, True, False, {"suffix": self.get_task_suffix()}
            self.beacon_x, self.beacon_y = bnx, bny
            message = f"Pushed beacon {cm}cm to ({self.beacon_x:.2f},{self.beacon_y:.2f})."

        if self._point_in_extract(self.beacon_x, self.beacon_y):
            obs = f"Success: beacon reached extraction at ({self.beacon_x:.2f},{self.beacon_y:.2f})."
            return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            obs = f"TIMEOUT: reached max turns {self.max_turns} without success."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        obs = f"OK: {message}"
        return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if action is None:
            return None
        # Support multiple boxed segments; take the last one.
        matches = list(re.finditer(r"\\boxed\{(.+?)\}", str(action), flags=re.DOTALL))
        if not matches:
            return None
        inner = matches[-1].group(1).strip()
        parts = inner.split()
        if not parts:
            return None
        tokens = {"action": parts[0]}
        for part in parts[1:]:
            if "=" in part:
                k, v = part.split("=", 1)
                tokens[k.strip()] = v.strip()
        return tokens

    def sample_random_action(self) -> str:
        choice = random.choice(["rotate", "thrust", "push"])
        if choice == "rotate":
            deg = random.choice([-self.rotation_deg, -int(self.rotation_deg/2), int(self.rotation_deg/2), self.rotation_deg])
            return rf"\boxed{{rotate deg={deg}}}"
        if choice == "thrust":
            cm = random.choice([-self.thrust_cm, -int(self.thrust_cm/2), int(self.thrust_cm/2), self.thrust_cm])
            return rf"\boxed{{thrust cm={cm}}}"
        cm = random.choice([-int(self.thrust_cm/2), int(self.thrust_cm/2)])
        return rf"\boxed{{push cm={cm}}}"


class BeaconCorridorNavigatorEnvWithFeedback(BeaconCorridorNavigatorEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        text = obs.lower()
        error_type = "OK"
        error_detail = {}
        hint = None

        if "invalid action format" in text:
            error_type = "FormatError"
            error_detail["issue"] = "bad_format"
            hint = "Use \\boxed{rotate deg=..}, \\boxed{thrust cm=..}, or \\boxed{push cm=..}."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["allowed"] = ["rotate", "thrust", "push"]
            hint = "Pick one of the supported actions."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "rotation exceeds" in text:
                error_detail["violation"] = "rotate_limit"
                hint = "Use a smaller absolute deg within the allowed limit."
            elif "thrust exceeds" in text:
                error_detail["violation"] = "thrust_limit"
                hint = "Use a cm value within ±thrust limit."
            elif "not in contact range" in text:
                error_detail["violation"] = "push_contact"
                hint = "Move the rover near the beacon (within contact distance) before pushing."
            elif "not facing beacon" in text:
                error_detail["violation"] = "push_facing"
                hint = "Rotate to face the beacon (within 30°) before pushing."
            else:
                error_detail["violation"] = "rule_violation"
                hint = "Respect action limits and preconditions."
        elif "invalid move: rover collided" in text or "invalid move: beacon collided" in text:
            error_type = "WrongDecision"
            if "rover collided" in text:
                error_detail["who"] = "rover"
                hint = "Check obstacle positions and adjust rotation before thrust."
            else:
                error_detail["who"] = "beacon"
                hint = "Push along a clear corridor; rotate to a safer heading."
        elif "timeout" in text:
            error_type = "Timeout"
            error_detail["outcome"] = "max_turns"
            hint = "Plan a shorter path; avoid wasted rotations."
        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None
        elif "unsupported" in text:
            error_type = "UnsupportedAction"
            hint = "Only rotate, thrust, push are valid."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "rover": (round(self.rover_x, 2), round(self.rover_y, 2), round(self.rover_theta, 1)),
                "beacon": (round(self.beacon_x, 2), round(self.beacon_y, 2)),
                "extract_center": (round(self.extract_x, 2), round(self.extract_y, 2)),
                "extract_radius_m": round(self.extract_radius_cm/100.0, 2),
                "obstacle_count": len(self.obstacles),
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
            "hint": "Rotate to face the beacon, thrust to approach, then push toward extraction.",
            "turn": 0,
        }
        return obs, info
