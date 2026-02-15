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

"""City environment - Cities and connections graph reasoning game."""

import random
import string
from typing import Optional, Tuple, Dict, Any, List

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class CityEnv(Env):
    """
    City environment.

    A graph reasoning game where players must analyze a network of cities
    with connections and attributes, then answer questions about the k-nearest
    neighbors of a reference city.

    This is a single-turn environment with sparse reward.
    """

    def __init__(
        self,
        min_cities: int = 100,
        max_cities: int = 200,
        min_edges: int = 200,
        max_edges: int = 1000,
        **_,
    ):
        """
        Initialize City environment.

        Args:
            min_cities: Minimum number of cities
            max_cities: Maximum number of cities
            min_edges: Minimum number of edges
            max_edges: Maximum number of edges
        """
        super().__init__()
        self.min_cities = min_cities
        self.max_cities = max_cities
        self.min_edges = min_edges
        self.max_edges = max_edges
        self.answer = None

    def _generate_puzzle(self, seed: int) -> Tuple[str, int]:
        """Generate city network puzzle."""
        random.seed(seed)

        # Generate random parameters
        n = random.randint(self.min_cities, self.max_cities)
        e = random.randint(self.min_edges, self.max_edges)

        # Edges cannot exceed n*(n-1)/2
        max_edges = n * (n - 1) // 2
        if e > max_edges:
            e = max_edges

        # Generate cities
        city_list = []
        existing_names = set()

        for _ in range(n):
            # Generate unique city name
            while True:
                name = ''.join(random.choices(string.ascii_uppercase, k=10))
                if name not in existing_names:
                    existing_names.add(name)
                    break

            # Random coastal or inland (1=coastal, 0=inland)
            location = random.choice([0, 1])
            city = {
                'name': name,
                'coastal cities': location,
                'inland cities': 1 - location,
                'population': random.randint(100, 10000) * 10000,
                'lumber mills': random.randint(0, 100),
                'hospitals': random.randint(0, 100),
                'churches': random.randint(0, 100),
                'banks': random.randint(0, 100),
                'stadiums': random.randint(0, 100),
                'restaurants': random.randint(0, 100),
                'mines': random.randint(0, 100),
                'factories': random.randint(0, 100),
                'research centers': random.randint(0, 100),
            }
            city_list.append(city)

        # Generate graph using simple approach (avoid networkx dependency)
        edges = set()
        graph = {i: [] for i in range(n)}

        # Generate random edges
        attempts = 0
        max_attempts = e * 10
        while len(edges) < e and attempts < max_attempts:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v and (u, v) not in edges and (v, u) not in edges:
                distance = random.randint(1, 100)
                edges.add((u, v))
                graph[u].append((v, distance))
                graph[v].append((u, distance))
            attempts += 1

        edge_list = []
        for u, v in edges:
            # Find distance
            for neighbor, dist in graph[u]:
                if neighbor == v:
                    edge_list.append((city_list[u]['name'], city_list[v]['name'], dist))
                    break

        # Select a city with edges as reference
        nodes_with_edges = [node for node in graph.keys() if len(graph[node]) > 0]
        if not nodes_with_edges:
            a_node = 0
        else:
            a_node = random.choice(nodes_with_edges)
        A_name = city_list[a_node]['name']

        # Select attribute to calculate
        items_list = ['coastal cities', 'inland cities', 'population', 'lumber mills', 'hospitals',
                      'churches', 'banks', 'stadiums', 'restaurants', 'mines', 'factories', 'research centers']
        selected_item = random.choice(items_list)

        # Get neighbors of reference city
        neighbors = graph[a_node]
        if not neighbors:
            answer = 0
            k = 0
        else:
            # Sort neighbors by distance
            sorted_neighbors = sorted(neighbors, key=lambda x: x[1])
            # Select k nearest neighbors
            k = random.randint(1, len(sorted_neighbors))
            selected_neighbors = sorted_neighbors[:k]
            answer = sum(city_list[n][selected_item] for (n, d) in selected_neighbors)

        # Generate city details
        city_details = "\n".join([
            f"City Name={c['name']}: Type={'Coastal' if c['coastal cities'] == 1 else 'Inland'}, Population={c['population']}, "
            f"Lumber Mills={c['lumber mills']}, Hospitals={c['hospitals']}, "
            f"Churches={c['churches']}, Banks={c['banks']}, Stadiums={c['stadiums']}, "
            f"Restaurants={c['restaurants']}, Mines={c['mines']}, "
            f"Factories={c['factories']}, Research Centers={c['research centers']}"
            for c in city_list
        ])

        connection_details = "\n".join([
            f"City {u} is connected to City {v} (Distance: {d}km)"
            for u, v, d in edge_list
        ])

        # Generate question
        question = (
            "Given the following cities and their connections:\n\n"
            "── City Details ──\n"
            f"{city_details}\n\n"
            "── Connections ──\n"
            f"{connection_details}\n\n"
            f"Question: What is the total number of {selected_item} in the {k} nearest cities directly adjacent to {A_name}?\n"
            "Please provide your answer in the following format: 'Answer: $YOUR_ANSWER' (without quotes), "
            "where $YOUR_ANSWER is your final answer.\n\n"
            "Alternatively, you can use \\boxed{$YOUR_ANSWER} format."
        )

        return question, answer

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Start a new City puzzle.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        # Generate puzzle
        puzzle_seed = seed if seed is not None else random.randint(0, 1000000)
        question, self.answer = self._generate_puzzle(puzzle_seed)

        return question, {
            "suffix": f"Cities: {self.min_cities}-{self.max_cities}, Edges: {self.min_edges}-{self.max_edges}."
        }

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Process player's answer.

        Args:
            action: Player's numeric answer

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Parse action
        parsed_action = extract_last_boxed_answer(action)

        if parsed_action is None:
            import re
            match = re.search(r'Answer:\s*(.+)', action, re.IGNORECASE | re.DOTALL)
            if match:
                parsed_action = match.group(1).strip()

        if parsed_action is None:
            obs = (
                "Your response did not contain a valid answer. "
                "Please use 'Answer: NUMBER' or \\boxed{NUMBER} format."
            )
            return obs, 0.0, True, False, {}

        # Parse numeric answer
        try:
            user_answer = int(str(parsed_action).strip())
        except ValueError:
            obs = f"Failed to parse numeric answer: {parsed_action}"
            return obs, 0.0, True, False, {}

        # Check answer
        if user_answer == self.answer:
            obs = f"Correct! The answer is {self.answer}."
            return obs, 1.0, True, False, {}
        else:
            obs = f"Incorrect. Your answer: {user_answer}, Correct answer: {self.answer}."
            return obs, 0.0, True, False, {}

    def sample_random_action(self) -> str:
        """
        Sample the correct answer.

        Returns:
            Correct answer as string
        """
        return f"\\boxed{{{self.answer}}}"
