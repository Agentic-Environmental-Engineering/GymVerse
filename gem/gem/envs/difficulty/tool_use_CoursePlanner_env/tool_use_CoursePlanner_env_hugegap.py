from gem.core import Env
from .tool_use_CoursePlanner_env import CoursePlannerEnv, CoursePlannerEnvWithFeedback


def _step_range_huge(level: int):
    base = 5 * level - 3  # level1:2-3 ... level10:47-48
    return base, base + 1


class CoursePlannerHugeGapEnv(CoursePlannerEnv):
    """Huge-gap版：步数 5N-3~5N-2，其他逻辑沿用基类。"""

    def __init__(self, complexity: int = 1, max_turns: int = 360, **kwargs):
        self.complexity = max(1, min(10, int(complexity)))
        self.max_turns = max_turns if max_turns is not None else 360
        self.min_required_steps, self.max_required_steps = _step_range_huge(self.complexity)
        Env.__init__(self)
        self._init_database()
        self.reset()

    def evolve(self, step_success_rate: float, **kwargs) -> int:
        old_complexity = self.complexity
        new_complexity = Env.evolve(self, step_success_rate, **kwargs)
        if new_complexity != old_complexity:
            self.min_required_steps, self.max_required_steps = _step_range_huge(new_complexity)
            self._init_database()
        return new_complexity

    def set_complexity(self, complexity: int):
        self.complexity = max(1, min(10, int(complexity)))
        self.min_required_steps, self.max_required_steps = _step_range_huge(self.complexity)
        self._init_database()


class CoursePlannerHugeGapEnvWithFeedback(CoursePlannerEnvWithFeedback, CoursePlannerHugeGapEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        CoursePlannerHugeGapEnv.__init__(self, **kwargs)

