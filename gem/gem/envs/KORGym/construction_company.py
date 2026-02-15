# Copyright 2025 AxonRL Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Construction Company environment - Project scheduling optimization."""

import heapq
import random
import string
from collections import defaultdict
from typing import Optional, Tuple, Dict, Any, List

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class ConstructionCompanyEnv(Env):
    """
    Construction Company scheduling environment.

    The agent must calculate the minimum total time to complete all projects
    given company capabilities and concurrent execution constraints.

    This is a single-turn environment with sparse rewards.
    """

    def __init__(
        self,
        min_companies: int = 5,
        max_companies: int = 50,
        min_tasks_per_company: int = 5,
        max_tasks_per_company: int = 20,
        min_total_tasks: int = 20,
        max_total_tasks: int = 100,
        min_concurrent: int = 2,
        max_concurrent: int = 4,
        **_,
    ):
        """
        Initialize Construction Company environment.

        Args:
            min_companies: Minimum number of companies
            max_companies: Maximum number of companies
            min_tasks_per_company: Minimum tasks each company can handle
            max_tasks_per_company: Maximum tasks each company can handle
            min_total_tasks: Minimum total tasks in city plan
            max_total_tasks: Maximum total tasks in city plan
            min_concurrent: Minimum concurrent tasks allowed
            max_concurrent: Maximum concurrent tasks allowed
        """
        super().__init__()
        self.min_companies = min_companies
        self.max_companies = max_companies
        self.min_tasks_per_company = min_tasks_per_company
        self.max_tasks_per_company = max_tasks_per_company
        self.min_total_tasks = min_total_tasks
        self.max_total_tasks = max_total_tasks
        self.min_concurrent = min_concurrent
        self.max_concurrent = max_concurrent
        self.correct_answer = None

    def _calculate_total_time(
        self, plan: List[Tuple[int, str]], companies: List[Dict[str, int]], task_allowed: int
    ) -> int:
        """Calculate minimum total time using heap-based scheduling."""
        heap = []
        max_end = 0

        for company_idx, project in plan:
            duration = companies[company_idx - 1][project]

            # If all slots are occupied, wait for the earliest one to finish
            if len(heap) >= task_allowed:
                start = heapq.heappop(heap)
            else:
                start = 0

            end = start + duration
            heapq.heappush(heap, end)
            max_end = max(max_end, end)

        return max_end

    def _get_instructions(self) -> str:
        """Return game instructions."""
        return (
            "You are a good game player. I'll give you a game board and rules.\n"
            "Your task is:\n"
            "- First, give your answer according to the game board and rules.\n"
            "- Second, output the answer in the required format. The last line of your response "
            "should be: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final "
            "answer to the question, e.g., 'Answer: 12'\n\n"
            "Alternatively, you can use \\boxed{number} format, e.g., \\boxed{12}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a new puzzle instance.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        # Random parameters
        company_num = random.randint(self.min_companies, self.max_companies)
        task_num = random.randint(self.min_tasks_per_company, self.max_tasks_per_company)
        total_task_num = random.randint(self.min_total_tasks, self.max_total_tasks)
        task_allowed = random.randint(self.min_concurrent, self.max_concurrent)

        # Generate project pool
        project_pool = {}
        total_projects = company_num * task_num * 2
        while len(project_pool) < total_projects:
            proj_name = ''.join(random.choices(string.ascii_lowercase, k=6))
            if proj_name not in project_pool:
                project_pool[proj_name] = 0

        # Assign projects to companies
        companies = []
        project_to_companies = defaultdict(list)

        for company_id in range(1, company_num + 1):
            available_projects = list(project_pool.keys())
            selected_projects = random.sample(available_projects, task_num)
            company = {}
            for proj in selected_projects:
                duration = random.randint(1, 10)  # Duration in years
                company[proj] = duration
            companies.append(company)

            for proj in selected_projects:
                project_to_companies[proj].append(company_id)

        # Generate city plan
        plan = []
        valid_projects = list(project_to_companies.keys())
        do_remove = len(valid_projects) > total_task_num

        for _ in range(total_task_num):
            proj = random.choice(valid_projects)
            company_id = random.choice(project_to_companies[proj])
            plan.append((company_id, proj))
            if do_remove:
                valid_projects.remove(proj)

        # Build question text
        companies_info = "\n".join(
            f"Company {idx} can handle:" +
            "".join(f"\n  {proj}: {dur} year{'s' if dur != 1 else ''}"
                    for proj, dur in company.items())
            for idx, company in enumerate(companies, 1)
        )

        plan_details = " -> ".join(f"({c}, {p})" for c, p in plan)

        question = f"""[Construction Company Scheduling Game Rules]
1. Game Objective:
Calculate the total time required to complete all projects in the city's plan, considering:
- Projects must be executed in the order listed.
- A maximum of {task_allowed} projects can run simultaneously.

2. Company Capabilities:
{companies_info}

3. City Project Plan (in strict order; data format is (Company ID, Project Name)):
{plan_details}

4. Rules:
- Projects start immediately when a slot is available.
- Time is measured in years.
- If all concurrent slots are occupied, new projects must wait.
- The total duration is from the start of the first project to the completion of the last project.
- Each company can only undertake projects they are capable of.
- When projects are repeated, they must be completed each time.

Please calculate the minimum possible total time to complete all projects.
"""

        # Calculate correct answer
        self.correct_answer = self._calculate_total_time(plan, companies, task_allowed)

        observation = f"{self._get_instructions()}{question}"
        return observation, {"suffix": "Calculate the minimum total time."}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Validate the agent's answer.

        Args:
            action: Agent's response containing the number

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Try to extract answer using \boxed{} format first
        parsed_answer = extract_last_boxed_answer(action)

        # If not found, try "Answer:" format
        if parsed_answer is None:
            import re
            match = re.search(r'Answer:\s*(.+?)(?:\n|$)', action, re.IGNORECASE | re.MULTILINE)
            if match:
                parsed_answer = match.group(1).strip()

        if parsed_answer is None:
            obs = (
                "Your response did not contain a valid answer. "
                "Please use 'Answer: number' or \\boxed{number} format."
            )
            return obs, 0.0, True, False, {}

        # Try to parse as integer
        try:
            user_answer = int(parsed_answer)
        except ValueError:
            obs = f"Failed to parse '{parsed_answer}' as an integer."
            return obs, 0.0, True, False, {}

        # Check if answer is correct
        is_correct = user_answer == self.correct_answer
        reward = 1.0 if is_correct else 0.0

        if is_correct:
            obs = f"Correct! The minimum total time is {self.correct_answer} years."
        else:
            obs = f"Incorrect. Your answer was {user_answer}, but the correct answer is {self.correct_answer} years."

        return obs, reward, True, False, {}

    def sample_random_action(self) -> str:
        """
        Sample a random action.

        Returns:
            Random number (for testing, returns correct answer)
        """
        if self.correct_answer is not None:
            return f"\\boxed{{{self.correct_answer}}}"
        else:
            return f"\\boxed{{{random.randint(10, 100)}}}"
