"""
Unified Difficulty Configuration System

This module merges difficulty configurations from multiple environment types:
- Game environments (Mastermind, Sudoku, etc.)
- Example environments (NetworkPlanningGame)
- RLVE environments (Fibonacci, Binario, etc.)

All configurations follow the same format and can be accessed through
the unified ALL_DIFFICULTY_CONFIGS dictionary.
"""

from .game_configs import GAME_CONFIGS
from .example_configs import EXAMPLE_CONFIGS
from .rlve_configs import RLVE_CONFIGS

# Merge all configurations into a single dictionary
ALL_DIFFICULTY_CONFIGS = {
    **GAME_CONFIGS,
    **EXAMPLE_CONFIGS,
    **RLVE_CONFIGS
}

# Backward compatibility: GAME_DIFFICULTY_CONFIGS points to merged configs
GAME_DIFFICULTY_CONFIGS = ALL_DIFFICULTY_CONFIGS

# Export interpolation functions from game_configs
from .game_configs import (
    linear_interpolate,
    stepped_interpolate,
    boolean_interpolate,
    tuple_interpolate,
    extract_game_name,
    INTERPOLATION_STRATEGIES
)

# Export exponential_interpolate from rlve_configs
from .rlve_configs import exponential_interpolate

# Override get_supported_games to return all environments
def get_supported_games() -> list:
    """
    Get list of all supported environment names across all types.

    Returns:
        List of environment names from GAME, EXAMPLE, and RLVE configs
    """
    return list(ALL_DIFFICULTY_CONFIGS.keys())

# New validate_game_support that uses ALL_DIFFICULTY_CONFIGS
def validate_game_support(env_id: str) -> tuple:
    """
    Check if an environment is supported for progressive difficulty evaluation.

    Args:
        env_id: Environment ID (e.g., "game:Mastermind-v0", "example:NetworkPlanningGame", "rlve:Fibonacci", "envsyn:...")

    Returns:
        Tuple of (is_supported: bool, message: str)
    """
    try:
        # EnvSyn environments are always supported (they may not have difficulty configs)
        if env_id.startswith('envsyn:'):
            return True, f"EnvSyn environment '{env_id}' is supported"

        if env_id.startswith('difficulty:'):
            return True, f"Difficulty environment '{env_id}' is supported"

        # BFCL environments are supported, but do not use the standard difficulty configs.
        if env_id.startswith("bfcl:"):
            return True, f"BFCL environment '{env_id}' is supported"
        
        game_name = extract_game_name(env_id)
        if game_name in ALL_DIFFICULTY_CONFIGS:
            return True, f"Environment '{game_name}' is supported"
        else:
            return False, (
                f"Environment '{game_name}' is not configured. "
                f"Supported environments: {', '.join(get_supported_games())}"
            )
    except ValueError as e:
        return False, str(e)

# New calculate_difficulty_params that uses ALL_DIFFICULTY_CONFIGS
def calculate_difficulty_params(
    game_name: str,
    instance_idx: int,
    total_instances: int = 32,
    custom_config: dict = None
) -> dict:
    """
    Calculate difficulty parameters for a specific instance.

    Args:
        game_name: Name of the environment (e.g., "Mastermind", "NetworkPlanningGame", "Fibonacci")
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
    config = custom_config if custom_config else ALL_DIFFICULTY_CONFIGS.get(game_name)
    if not config:
        raise ValueError(
            f"Environment '{game_name}' not configured. "
            f"Available environments: {list(ALL_DIFFICULTY_CONFIGS.keys())}"
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

    return params

__all__ = [
    # Configuration dictionaries
    'ALL_DIFFICULTY_CONFIGS',
    'GAME_DIFFICULTY_CONFIGS',  # Backward compatibility
    'GAME_CONFIGS',
    'EXAMPLE_CONFIGS',
    'RLVE_CONFIGS',

    # Interpolation functions
    'linear_interpolate',
    'stepped_interpolate',
    'boolean_interpolate',
    'tuple_interpolate',
    'exponential_interpolate',

    # Utility functions
    'calculate_difficulty_params',
    'extract_game_name',
    'validate_game_support',
    'get_supported_games',
]
