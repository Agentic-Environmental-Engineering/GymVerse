"""
Game Difficulty Configuration System.

This module defines difficulty parameter ranges for game environments
and provides interpolation functions to generate progressive difficulty levels.
"""

from typing import Dict, Any, Optional, Callable
import re


# Interpolation functions
def linear_interpolate(easy_val: float, hard_val: float, progress: float) -> float:
    """
    Linear interpolation between easy and hard values.

    Args:
        easy_val: Value at easiest difficulty (progress=0.0)
        hard_val: Value at hardest difficulty (progress=1.0)
        progress: Difficulty progress from 0.0 (easiest) to 1.0 (hardest)

    Returns:
        Interpolated value
    """
    return easy_val + (hard_val - easy_val) * progress


def stepped_interpolate(easy_val: Any, hard_val: Any, progress: float, threshold: float = 0.5) -> Any:
    """
    Stepped interpolation for discrete parameters.

    Args:
        easy_val: Value for easier instances
        hard_val: Value for harder instances
        progress: Difficulty progress from 0.0 to 1.0
        threshold: Progress threshold for switching (default: 0.5)

    Returns:
        easy_val if progress < threshold, else hard_val
    """
    return easy_val if progress < threshold else hard_val


def boolean_interpolate(easy_val: bool, hard_val: bool, progress: float) -> bool:
    """
    Boolean interpolation.

    Args:
        easy_val: Boolean value for easier instances
        hard_val: Boolean value for harder instances
        progress: Difficulty progress from 0.0 to 1.0

    Returns:
        If both same: return that value. Otherwise switch at 50% progress.
    """
    if easy_val == hard_val:
        return easy_val
    return hard_val if progress >= 0.5 else easy_val


def tuple_interpolate(easy_val: tuple, hard_val: tuple, progress: float) -> tuple:
    """
    Interpolate tuple values element-wise using linear interpolation.

    Args:
        easy_val: Tuple of values at easiest difficulty
        hard_val: Tuple of values at hardest difficulty
        progress: Difficulty progress from 0.0 to 1.0

    Returns:
        Tuple with interpolated values
    """
    return tuple(
        int(linear_interpolate(e, h, progress))
        for e, h in zip(easy_val, hard_val)
    )


# Game difficulty configurations
GAME_CONFIGS = {
    "GuessTheNumber": {
        "parameters": ["min_number", "max_number", "max_turns"],
        "easy": {"min_number": 1, "max_number": 10, "max_turns": 4},
        "hard": {"min_number": 1, "max_number": 50, "max_turns": 7},
        "scaling_strategy": "linear"
    },

    "Mastermind": {
        "parameters": ["code_length", "num_numbers", "max_turns", "duplicate_numbers"],
        "easy": {"code_length": 2, "num_numbers": 6, "max_turns": 10, "duplicate_numbers": False},
        "hard": {"code_length": 4, "num_numbers": 8, "max_turns": 30, "duplicate_numbers": False},
        "scaling_strategy": "linear",
        "duplicate_numbers_strategy": "boolean"
    },

    "Minesweeper": {
        "parameters": ["rows", "cols", "max_turns"],
        "easy": {"rows": 5, "cols": 5, "max_turns": 25},
        "hard": {"rows": 8, "cols": 8, "max_turns": 64},
        "scaling_strategy": "linear",
        "derived_params": {
            "num_mines": lambda params: int(params["rows"] * params["cols"] * 0.2)
        }
    },

    "Wordle": {
        "parameters": ["word_length", "max_turns"],
        "easy": {"word_length": 3, "max_turns": 15},
        "hard": {"word_length": 5, "max_turns": 25},
        "scaling_strategy": "linear"
    },

    "FifteenPuzzle": {
        "parameters": ["num_rows", "max_turns"],
        "easy": {"num_rows": 2, "max_turns": 10},
        "hard": {"num_rows": 4, "max_turns": 50},
        "scaling_strategy": "linear"
    },

    "Hangman": {
        "parameters": ["word_length", "max_turns"],
        "easy": {"word_length": 3, "max_turns": 10},
        "hard": {"word_length": 7, "max_turns": 20},
        "scaling_strategy": "linear"
    },

    "Sudoku": {
        "parameters": ["scale", "clues", "max_turns"],
        "easy": {"scale": 4, "clues": 10, "max_turns": 15},
        "hard": {"scale": 9, "clues": 50, "max_turns": 50},
        "scaling_strategy": "linear",
        "scale_strategy": "stepped"  # scale must be discrete: 4 or 9
    },

    "TowerofHanoi": {
        "parameters": ["num_disks", "max_turns"],
        "easy": {"num_disks": 3, "max_turns": 10},
        "hard": {"num_disks": 5, "max_turns": 35},
        "scaling_strategy": "linear"
    },

    "Game2048": {
        "parameters": ["target_tile", "max_turns"],
        "easy": {"target_tile": 64, "max_turns": 50},
        "hard": {"target_tile": 512, "max_turns": 50},
        "scaling_strategy": "exponential",  # target_tile scales exponentially
        "target_tile_strategy": "exponential"
    },

    "Sokoban": {
        "parameters": ["dim_room", "num_boxes", "max_turns"],
        "easy": {"dim_room": (6, 6), "num_boxes": 2, "max_turns": 20},
        "hard": {"dim_room": (8, 8), "num_boxes": 4, "max_turns": 50},
        "scaling_strategy": "linear",
        "dim_room_strategy": "tuple"
    },

    "Crosswords": {
        "parameters": ["num_words", "max_turns", "hardcore"],
        "easy": {"num_words": 3, "max_turns": 30, "hardcore": False},
        "hard": {"num_words": 3, "max_turns": 40, "hardcore": True},
        "scaling_strategy": "linear",
        "hardcore_strategy": "boolean"
    },

    "WordSearch": {
        "parameters": ["num_words", "max_turns", "hardcore"],
        "easy": {"num_words": 5, "max_turns": 20, "hardcore": False},
        "hard": {"num_words": 5, "max_turns": 20, "hardcore": True},
        "scaling_strategy": "linear",
        "hardcore_strategy": "boolean"
    }
}


def exponential_interpolate(easy_val: float, hard_val: float, progress: float) -> float:
    """
    Exponential interpolation for parameters that scale exponentially.

    Args:
        easy_val: Value at easiest difficulty
        hard_val: Value at hardest difficulty
        progress: Difficulty progress from 0.0 to 1.0

    Returns:
        Exponentially interpolated value
    """
    if easy_val <= 0 or hard_val <= 0:
        return linear_interpolate(easy_val, hard_val, progress)
    return easy_val * ((hard_val / easy_val) ** progress)


# Map strategy names to functions
INTERPOLATION_STRATEGIES: Dict[str, Callable] = {
    "linear": linear_interpolate,
    "stepped": stepped_interpolate,
    "boolean": boolean_interpolate,
    "exponential": exponential_interpolate,
    "tuple": tuple_interpolate
}


def extract_game_name(env_id: str) -> str:
    """
    Extract game/environment name from environment ID.

    Supports multiple environment types:
    - game:Mastermind-v0 → Mastermind
    - example:NetworkPlanningGame → NetworkPlanningGame
    - rlve:Fibonacci → Fibonacci
    - envsyn:algorithm_AlgorithmCost_MaxLapsEnv_GEM → algorithm_AlgorithmCost_MaxLapsEnv_GEM

    Args:
        env_id: Environment ID (e.g., "game:Mastermind-v0-easy")

    Returns:
        Environment name (e.g., "Mastermind", "NetworkPlanningGame", "Fibonacci", "algorithm_AlgorithmCost_MaxLapsEnv_GEM")

    Raises:
        ValueError: If env_id format is invalid
    """
    # Split by colon
    if ':' not in env_id:
        raise ValueError(f"Invalid env_id format (missing ':' separator): {env_id}")

    prefix, name_part = env_id.split(':', 1)

    # Validate prefix
    if prefix not in ['game', 'example', 'rlve', 'envsyn', 'bfcl', 'difficulty']:
        raise ValueError(f"Unsupported environment prefix '{prefix}' in: {env_id}")

    # For game: remove version suffix (-v0, -v0-easy, etc.)
    if prefix == 'game':
        match = re.match(r"^([A-Za-z0-9]+)", name_part)
        if not match:
            raise ValueError(f"Cannot extract game name from: {env_id}")
        return match.group(1)

    # For example:, rlve:, envsyn:, and bfcl:, use name as-is (no version suffix)
    return name_part



def calculate_difficulty_params(
    game_name: str,
    instance_idx: int,
    total_instances: int = 32,
    custom_config: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Calculate difficulty parameters for a specific instance.

    Args:
        game_name: Name of the game (e.g., "Mastermind")
        instance_idx: Instance index from 0 (easiest) to total_instances-1 (hardest)
        total_instances: Total number of instances (default: 32)
        custom_config: Optional custom configuration to override defaults

    Returns:
        Dictionary of difficulty parameters for gem.make()

    Raises:
        ValueError: If game_name is not configured or instance_idx is invalid
    """
    # Validate instance_idx
    if not (0 <= instance_idx < total_instances):
        raise ValueError(
            f"instance_idx must be between 0 and {total_instances-1}, got {instance_idx}"
        )

    # Get configuration
    config = custom_config if custom_config else GAME_CONFIGS.get(game_name)
    if not config:
        raise ValueError(
            f"Game '{game_name}' not configured. "
            f"Available games: {list(GAME_CONFIGS.keys())}"
        )

    # Calculate progress (0.0 to 1.0)
    progress = instance_idx / (total_instances - 1) if total_instances > 1 else 0.0

    # Build difficulty parameters
    params = {}
    for param_name in config["parameters"]:
        easy_val = config["easy"][param_name]
        hard_val = config["hard"][param_name]

        # Determine interpolation strategy for this parameter
        param_strategy_key = f"{param_name}_strategy"
        if param_strategy_key in config:
            strategy_name = config[param_strategy_key]
        else:
            strategy_name = config.get("scaling_strategy", "linear")

        # Get interpolation function
        interpolate_fn = INTERPOLATION_STRATEGIES.get(strategy_name, linear_interpolate)

        # Interpolate value
        if strategy_name in ["linear", "exponential"]:
            # For numeric parameters, convert to int if original was int
            interpolated = interpolate_fn(easy_val, hard_val, progress)
            if isinstance(easy_val, int) and isinstance(hard_val, int):
                params[param_name] = int(interpolated)
            else:
                params[param_name] = interpolated
        elif strategy_name == "tuple":
            params[param_name] = interpolate_fn(easy_val, hard_val, progress)
        else:
            # For stepped, boolean, etc.
            params[param_name] = interpolate_fn(easy_val, hard_val, progress)

    # Handle derived parameters
    if "derived_params" in config:
        for derived_name, derive_fn in config["derived_params"].items():
            params[derived_name] = derive_fn(params)

    return params


def get_supported_games() -> list:
    """
    Get list of supported game names.

    Returns:
        List of game names that have difficulty configurations
    """
    return list(GAME_CONFIGS.keys())


def validate_game_support(env_id: str) -> tuple[bool, str]:
    """
    Check if a game environment is supported for progressive difficulty evaluation.

    Args:
        env_id: Environment ID

    Returns:
        Tuple of (is_supported: bool, message: str)
    """
    try:
        game_name = extract_game_name(env_id)
        if game_name in GAME_CONFIGS:
            return True, f"Game '{game_name}' is supported"
        else:
            return False, (
                f"Game '{game_name}' is not configured. "
                f"Supported games: {', '.join(get_supported_games())}"
            )
    except ValueError as e:
        return False, str(e)
