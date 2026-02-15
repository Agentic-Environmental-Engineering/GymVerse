"""
Configuration for example environments (NetworkPlanningGame, etc.)

These environments are demonstration/example environments with their own
complexity systems.
"""

# Example environment difficulty configurations
EXAMPLE_CONFIGS = {
    "NetworkPlanningGame": {
        "parameters": ["complexity_level"],
        "easy": {"complexity_level": 1},
        "hard": {"complexity_level": 10},
        "scaling_strategy": "linear",
        "description": "Network planning with RLVE-style complexity evolution (1-10)"
    }
}
