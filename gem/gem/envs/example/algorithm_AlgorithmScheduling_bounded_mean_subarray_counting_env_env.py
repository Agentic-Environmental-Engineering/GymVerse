from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class AlgorithmSchedulingEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = None,
        **_,
    ):
        super().__init__()
        self.complexity = int(max(1, min(10, complexity)))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 100

        # Evolvable parameters
        self.complexity_params = {
            # Number of jobs: more jobs increases branching and sequence length → harder
            'num_jobs': (3, 14),
            # Max processing time: larger tasks increase time horizon and tie potential → harder
            'proc_time_max': (3, 9),
            # Release time spread: larger spread increases idling/availability reasoning → harder
            'release_spread': (0, 15),
            # REVERSED: Minimum slack margin added to each job's deadline; less slack = tighter feasibility → harder
            'min_slack': (7, 2),
            # Number of tie groups: more ties in processing times demand proper tie-breaking → harder
            'tie_groups': (0, 3),
        }

        # Variance settings
        self.param_variance = {
            'num_jobs': 1,
            'proc_time_max': 1,
            'release_spread': 2,
            'min_slack': 1,
            'tie_groups': 1,
        }

        # Placeholders (set in _apply_complexity_params)
        self.num_jobs: int = 0
        self.proc_time_max: int = 0
        self.release_spread: int = 0
        self.min_slack: int = 0
        self.tie_groups: int = 0

        # State
        self.jobs: list = []
        self.expected_order: list = []
        self.current_index: int = 0
        self.current_time: int = 0
        self.turn_count: int = 0
        self.terminated: bool = False

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for param_name, (min_val, max_val) in self.complexity_params.items():
            # Handle reversed parameters (min_val > max_val means reversed difficulty)
            if min_val <= max_val:
                center_value = min_val + (max_val - min_val) * normalized
            else:
                center_value = min_val - (min_val - max_val) * normalized
            actual_value = center_value
            if self.enable_param_randomization:
                variance = self.param_variance.get(param_name, 0)
                if variance > 0:
                    actual_value = center_value + random.uniform(-variance, variance)
            # Clamp to bounds
            low = min(min_val, max_val)
            high = max(min_val, max_val)
            actual_value = max(low, min(high, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _compute_spt_order(self, jobs):
        remaining = [dict(j) for j in jobs]
        order = []
        t = 0
        finish_times = {}
        while remaining:
            available = [j for j in remaining if j['r'] <= t]
            if available:
                # SPT: sort by p, then by r (earlier release), then id
                available.sort(key=lambda x: (x['p'], x['r'], x['id']))
                chosen = available[0]
                t = max(t, chosen['r']) + chosen['p']
                order.append(chosen['id'])
                finish_times[chosen['id']] = t
                remaining = [j for j in remaining if j['id'] != chosen['id']]
            else:
                t = min(j['r'] for j in remaining)
        return order, finish_times

    def _generate_jobs(self):
        n = self.num_jobs
        jobs = []
        for i in range(1, n + 1):
            p = random.randint(1, self.proc_time_max)
            r = random.randint(0, self.release_spread) if self.release_spread > 0 else 0
            jobs.append({'id': i, 'p': p, 'r': r, 'd': 0})

        # Introduce tie groups in processing times
        groups = max(0, min(self.tie_groups, n // 2))
        for _ in range(groups):
            if n < 2:
                break
            base_idx = random.randint(0, n - 1)
            partner_idx = random.randint(0, n - 1)
            if partner_idx == base_idx:
                partner_idx = (partner_idx + 1) % n
            jobs[partner_idx]['p'] = jobs[base_idx]['p']

        # Compute expected order and finish times; set deadlines with slack
        order, finish_times = self._compute_spt_order(jobs)
        jobs_with_deadlines = []
        for j in jobs:
            slack = self.min_slack + random.randint(0, 4)
            d = finish_times[j['id']] + slack
            jobs_with_deadlines.append({'id': j['id'], 'p': j['p'], 'r': j['r'], 'd': d})
        return jobs_with_deadlines, order

    def _get_instructions(self) -> str:
        return (
            "Algorithm Scheduling Game\n"
            "Goal: Construct the exact non-preemptive schedule produced by the Shortest Processing Time (SPT) rule with release times.\n"
            "Policy: At any moment, among jobs whose release time is <= current time, choose the job with the smallest processing time. "
            "Tie-break by earlier release time, then by smaller job ID. If no jobs are available, the schedule idles until the earliest release.\n"
            "Deadlines are provided for feasibility and are derived from the target schedule; you must still pick according to SPT.\n"
            "Actions:\n"
            "- Use \\boxed{schedule X} to schedule job X next (X is an integer job ID).\n"
            "- After scheduling all jobs, use \\boxed{finish} to submit the final answer.\n"
            "Constraints:\n"
            "- Do not schedule a job twice.\n"
            "- Do not finish before all jobs are scheduled.\n"
            "Rewards:\n"
            "- Success: 1.0 when the final schedule is exactly correct.\n"
            "- Wrong decision: -1.0 terminates the episode.\n"
            "- Protocol violations (e.g., duplicate scheduling, premature finish): -0.5 terminates.\n"
            "- Format error (missing \\boxed{...}): small negative penalty.\n"
            "Example action: " + self.sample_random_action() + "\n"
        )

    def get_task_suffix(self) -> str:
        lines = []
        lines.append("Current state:")
        lines.append(f"- Time: {self.current_time}")
        lines.append(f"- Jobs scheduled so far: {self.current_index}/{len(self.jobs)}")
        lines.append("- Jobs:")
        for j in sorted(self.jobs, key=lambda x: x['id']):
            marker = "scheduled" if j['id'] in self.expected_order[:self.current_index] else "pending"
            lines.append(f"  ID {j['id']}: p={j['p']}, r={j['r']}, d={j['d']} ({marker})")
        lines.append("Enter your action using \\boxed{schedule X} or \\boxed{finish}.")
        return "\n".join(lines)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        self.turn_count = 0
        self.current_time = 0
        self.current_index = 0
        self.terminated = False

        self.jobs, self.expected_order = self._generate_jobs()

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1

        parsed = self._parse_action(action)
        if parsed is None:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{schedule X}} or \\boxed{{finish}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        if parsed['type'] == 'finish':
            if self.current_index < len(self.jobs):
                obs = f"Protocol violation: attempted to finish with {len(self.jobs) - self.current_index} jobs remaining."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            obs = "Success: final schedule accepted."
            return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}

        if parsed['type'] != 'schedule':
            obs = f"Unsupported action: {parsed['type']}."
            return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}

        job_id = parsed.get('id', None)
        all_ids = {j['id'] for j in self.jobs}
        if job_id is None or job_id not in all_ids:
            obs = f"Protocol violation: unknown job ID '{job_id}'."
            return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}

        if job_id in self.expected_order[:self.current_index]:
            obs = f"Protocol violation: job {job_id} was already scheduled."
            return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}

        expected_next = self.expected_order[self.current_index]
        if job_id != expected_next:
            obs = f"Wrong decision: expected job {expected_next} next, got {job_id}."
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        job = next(j for j in self.jobs if j['id'] == job_id)
        self.current_time = max(self.current_time, job['r']) + job['p']
        self.current_index += 1

        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns}). Episode truncated."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        if self.current_index == len(self.jobs):
            obs = (f"All jobs scheduled correctly. Current time={self.current_time}. "
                   "Submit \\boxed{finish} to complete.")
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        obs = (f"Step accepted: scheduled job {job_id}. Current time={self.current_time}. "
               f"{len(self.jobs) - self.current_index} jobs remaining.")
        return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Any]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip().lower()
        if content == "finish":
            return {'type': 'finish'}
        m = re.match(r'(schedule)\s*[: ]\s*(\d+)', content)
        if m:
            cmd = m.group(1)
            jid = int(m.group(2))
            return {'type': cmd, 'id': jid}
        return None

    def sample_random_action(self) -> str:
        example_id = 1
        if self.jobs:
            # suggest an action that is either the next expected or some valid id
            example_id = self.expected_order[self.current_index] if self.current_index < len(self.expected_order) else self.jobs[0]['id']
        return f"\\boxed{{schedule {example_id}}}"


class AlgorithmSchedulingEnvWithFeedback(AlgorithmSchedulingEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        text = obs.lower()

        error_type = "OK"
        error_detail = {}
        hint = None

        if "invalid action format" in text or "use \\boxed" in text:
            error_type = "FormatError"
            error_detail["issue"] = "missing_or_bad_boxed_format"
            hint = "Use \\boxed{schedule X} for scheduling or \\boxed{finish} to submit."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "already scheduled" in text:
                error_detail["violation"] = "duplicate_job"
                hint = "Pick a job that has not been scheduled yet."
            elif "finish" in text and "remaining" in text:
                error_detail["violation"] = "premature_finish"
                hint = "Schedule all jobs before sending \\boxed{finish}."
            elif "unknown job" in text:
                error_detail["violation"] = "unknown_job_id"
                hint = "Use a valid job ID shown in the state."
            else:
                error_detail["violation"] = "general_protocol_violation"
                hint = "Follow the SPT policy and valid command formats."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["action"] = "unknown_command"
            hint = "Only schedule jobs using \\boxed{schedule X} or complete with \\boxed{finish}."

        elif "wrong decision" in text:
            error_type = "WrongDecision"
            m_exp = re.search(r'expected job (\d+)', text)
            m_got = re.search(r'got (\d+)', text)
            if m_exp:
                error_detail["expected"] = int(m_exp.group(1))
            if m_got:
                error_detail["got"] = int(m_got.group(1))
            hint = "Choose the smallest processing time among available jobs; break ties by earliest release, then smallest ID."

        elif "reached max turns" in text and "truncated" in text:
            error_type = "Timeout"
            error_detail["limit"] = getattr(self, "max_turns", None)
            hint = "Act decisively and avoid unnecessary steps."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            remaining = len(self.jobs) - getattr(self, "current_index", 0)
            diagnostic["remaining_jobs"] = remaining
            nxt = None
            if hasattr(self, "expected_order") and self.current_index < len(self.expected_order):
                nxt = self.expected_order[self.current_index]
            diagnostic["next_expected_job"] = nxt
            diagnostic["current_time"] = getattr(self, "current_time", None)

        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "At each turn, choose the job with the smallest processing time among those with release <= current time; tie-break by earlier release, then by smaller ID.",
            "turn": 0,
            "remaining_jobs": len(self.jobs),
            "next_expected_job": self.expected_order[0] if self.expected_order else None,
            "current_time": self.current_time,
        }
        return obs, info