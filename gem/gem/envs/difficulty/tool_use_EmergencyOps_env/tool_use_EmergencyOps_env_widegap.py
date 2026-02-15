from gem.core import Env
from .tool_use_EmergencyOps_env import EmergencyOpsEnv, EmergencyOpsEnvWithFeedback


def _step_range_wide(level: int):
    base = 3 * level - 1  # level1:2-3 ... level10:29-30
    return base, base + 1


class EmergencyOpsWideGapEnv(EmergencyOpsEnv):
    """Wide-gap版：步数 3N-1~3N，其他逻辑沿用基类。"""

    def __init__(self, complexity: int = 1, max_turns: int = 270, **kwargs):
        self.complexity = max(1, min(10, int(complexity)))
        self.max_turns = max_turns if max_turns is not None else 270
        self.min_required_steps, self.max_required_steps = _step_range_wide(self.complexity)
        Env.__init__(self)
        self._init_database()
        self.reset()

    def evolve(self, step_success_rate: float, **kwargs) -> int:
        old_complexity = self.complexity
        new_complexity = Env.evolve(self, step_success_rate, **kwargs)
        if new_complexity != old_complexity:
            self.min_required_steps, self.max_required_steps = _step_range_wide(new_complexity)
            self._init_database()
        return new_complexity

    def set_complexity(self, complexity: int):
        self.complexity = max(1, min(10, int(complexity)))
        self.min_required_steps, self.max_required_steps = _step_range_wide(self.complexity)
        self._init_database()


class EmergencyOpsWideGapEnvWithFeedback(EmergencyOpsEnvWithFeedback, EmergencyOpsWideGapEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        EmergencyOpsWideGapEnv.__init__(self, **kwargs)
