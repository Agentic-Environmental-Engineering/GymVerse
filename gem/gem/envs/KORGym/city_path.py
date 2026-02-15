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

"""CityPath environment - Shortest path finding in city networks."""

import heapq
import random
from typing import Optional, Tuple, Dict, Any, List

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class CityPathEnv(Env):
    """
    CityPath environment.

    Players are given a network of cities connected by roads with distances.
    They must calculate the shortest distance from a start city to a target city.

    This is a single-turn environment with sparse rewards.
    """

    def __init__(
        self,
        min_cities: int = 70,
        max_cities: int = 200,
        **_,
    ):
        """
        Initialize CityPath environment.

        Args:
            min_cities: Minimum number of cities
            max_cities: Maximum number of cities
        """
        super().__init__()
        self.min_cities = min_cities
        self.max_cities = max_cities
        self.question = None
        self.answer = None

    def _dijkstra(
        self,
        graph: Dict[str, Dict[str, int]],
        start: str,
        target: str,
        cities: List[str]
    ) -> int:
        """
        Calculate shortest path using Dijkstra's algorithm.

        Args:
            graph: Adjacency dictionary
            start: Start city
            target: Target city
            cities: List of all cities

        Returns:
            Shortest distance or float('inf') if unreachable
        """
        distances = {city: float('inf') for city in cities}
        distances[start] = 0
        visited = set()
        pq = [(0, start)]

        while pq:
            cur_dist, cur_city = heapq.heappop(pq)
            if cur_city in visited:
                continue
            visited.add(cur_city)
            if cur_city == target:
                break

            for neighbor, d in graph[cur_city].items():
                if neighbor in visited:
                    continue
                new_dist = cur_dist + d
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor))

        return distances[target]

    def _generate_puzzle(self, seed: int) -> Dict[str, Any]:
        """
        Generate a city path puzzle.

        Args:
            seed: Random seed

        Returns:
            Dictionary with question and answer
        """
        random.seed(seed)

        # Generate cities
        num = random.randint(self.min_cities, self.max_cities)
        cities = [f"City{i}" for i in range(num)]

        # Generate spanning tree to ensure connectivity
        edges = []
        shuffled_cities = cities[:]
        random.shuffle(shuffled_cities)
        for i in range(len(shuffled_cities) - 1):
            d = random.randint(1, 20)
            edges.append((shuffled_cities[i], shuffled_cities[i + 1], d))

        # Add extra random edges
        extra_edges = num
        for _ in range(extra_edges):
            city_a = random.choice(cities)
            city_b = random.choice(cities)
            if city_a == city_b:
                continue

            # Check if edge already exists
            exists = False
            for (a, b, _) in edges:
                if (a == city_a and b == city_b) or (a == city_b and b == city_a):
                    exists = True
                    break
            if exists:
                continue

            d = random.randint(1, 20)
            edges.append((city_a, city_b, d))

        # Choose start and target cities
        start_city = random.choice(cities)
        target_city = random.choice(cities)
        while target_city == start_city:
            target_city = random.choice(cities)

        # Build adjacency graph (take minimum distance if multiple edges)
        graph = {city: {} for city in cities}
        for (a, b, d) in edges:
            if b not in graph[a] or d < graph[a][b]:
                graph[a][b] = d
            if a not in graph[b] or d < graph[b][a]:
                graph[b][a] = d

        # Calculate shortest distance using Dijkstra
        shortest_distance = self._dijkstra(graph, start_city, target_city, cities)

        # Build question
        board = "City Network Information:\n"
        board += "Cities: " + ", ".join(cities) + "\n"
        board += "Roads (format: CityA - CityB (distance)):\n"
        for a, b, d in edges:
            board += f"{a} - {b} ({d}), "
        board = board.rstrip(", ") + "\n"
        board += f"Start City: {start_city}\nTarget City: {target_city}\n"
        board += "Question: What is the shortest distance from the start city to the target city?"

        return {
            "question": board,
            "answer": int(shortest_distance) if shortest_distance != float('inf') else None
        }

    def _get_instructions(self) -> str:
        """Return game instructions."""
        return (
            "You are an expert in city navigation. Below is the information about a network "
            "of cities and roads.\n"
            "Your task:\n"
            "- Read the information carefully.\n"
            "- Calculate the shortest distance from the start city to the target city.\n"
            "- Provide your answer in the following format: 'Answer: $YOUR_ANSWER' (without quotes).\n\n"
            "Alternatively, you can use \\boxed{NUMBER} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a new city path puzzle.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        # Generate puzzle
        puzzle = self._generate_puzzle(seed if seed else random.randint(0, 1000000))
        self.question = puzzle["question"]
        self.answer = puzzle["answer"]

        # Build observation
        observation = f"{self._get_instructions()}{self.question}"

        return observation, {
            "suffix": f"Find shortest path between cities."
        }

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Validate the agent's answer.

        Args:
            action: Agent's response containing the shortest distance

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Try to extract answer
        parsed_answer = extract_last_boxed_answer(action)

        if parsed_answer is None:
            import re
            match = re.search(r'Answer:\s*(\d+)', action, re.IGNORECASE)
            if match:
                parsed_answer = match.group(1).strip()

        if parsed_answer is None:
            obs = (
                "Your response did not contain a valid answer. "
                "Please use 'Answer: NUMBER' or \\boxed{NUMBER} format."
            )
            return obs, 0.0, True, False, {}

        # Parse number
        try:
            user_answer = int(parsed_answer)
        except ValueError:
            obs = f"Failed to parse answer as integer: '{parsed_answer}'"
            return obs, 0.0, True, False, {}

        # Check answer
        if user_answer == self.answer:
            obs = f"Correct! The shortest distance is {self.answer}."
            return obs, 1.0, True, False, {}
        else:
            obs = f"Incorrect. You answered {user_answer}, but the correct answer is {self.answer}."
            return obs, 0.0, True, False, {}

    def sample_random_action(self) -> str:
        """
        Sample a random action.

        Returns:
            The correct answer
        """
        if self.answer is not None:
            return f"\\boxed{{{self.answer}}}"
        else:
            return "\\boxed{0}"
