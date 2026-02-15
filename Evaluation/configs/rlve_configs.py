"""
Configuration for RLVE environments (Reasoning Learning with Value Evolution)

RLVE environments are typically single-turn mathematical/algorithmic problem environments.
Each environment has different difficulty parameters that control problem complexity.
"""

# RLVE environment difficulty configurations
RLVE_CONFIGS = {
    # ===== Phase 1: Core Mathematical Environments =====

    "Fibonacci": {
        "parameters": ["max_n", "modulo"],
        "easy": {"max_n": 100, "modulo": 100},
        "hard": {"max_n": 1000000, "modulo": 10000},
        "scaling_strategy": "exponential",
        "max_n_strategy": "exponential",  # max_n grows exponentially
        "modulo_strategy": "linear",       # modulo grows linearly
        "description": "Linear recurrence: A[n] = p*A[n-1] + q*A[n-2] mod m"
    },

    "Binario": {
        "parameters": ["max_n_m", "sparsity"],
        "easy": {"max_n_m": 6, "sparsity": 0.7},      # Small matrix, many hints
        "hard": {"max_n_m": 10, "sparsity": 0.3},     # Larger matrix, fewer hints
        "scaling_strategy": "linear",
        "max_n_m_strategy": "linear",
        "sparsity_strategy": "linear",  # Sparsity decreases as difficulty increases
        "description": "Binary fill puzzle (conservative max_n_m due to generation complexity)"
    },

    "AdditionTable": {
        "parameters": ["min_N", "max_N"],
        "easy": {"min_N": 3, "max_N": 8},
        "hard": {"min_N": 10, "max_N": 26},
        "scaling_strategy": "linear",
        "description": "Addition table in base N"
    },

    "CongruentEquation": {
        "parameters": ["max_a_b"],
        "easy": {"max_a_b": 100},
        "hard": {"max_a_b": 10000},
        "scaling_strategy": "exponential",
        "description": "Solve linear congruence equations: A*x ≡ 1 (mod B)"
    },

    "GaussianElimination": {
        "parameters": ["N", "M"],
        "easy": {"N": 3, "M": 2},
        "hard": {"N": 10, "M": 10},
        "scaling_strategy": "linear",
        "description": "Solve N linear equations with M variables"
    },

    "Crt": {  # Note: gem registers as "rlve:Crt" not "rlve:CRT"
        "parameters": ["max_x", "M"],
        "easy": {"max_x": 100, "M": 3},
        "hard": {"max_x": 10000, "M": 7},
        "scaling_strategy": "exponential",
        "max_x_strategy": "exponential",
        "M_strategy": "linear",
        "description": "Chinese Remainder Theorem with M moduli"
    },

    "BezoutIdentity": {
        "parameters": ["N", "MAX_A"],
        "easy": {"N": 2, "MAX_A": 50},
        "hard": {"N": 5, "MAX_A": 1000},
        "scaling_strategy": "linear",
        "N_strategy": "linear",
        "MAX_A_strategy": "exponential",
        "description": "Find Bezout coefficients for N numbers"
    },

    "DiscreteLogarithm": {
        "parameters": ["max_z"],
        "easy": {"max_z": 100},
        "hard": {"max_z": 1000000},
        "scaling_strategy": "exponential",
        "description": "Solve discrete logarithm: a^x ≡ b (mod p)"
    },

    "CatalanNumberMod": {
        "parameters": ["max_n", "max_mod"],
        "easy": {"max_n": 10, "max_mod": 1000},
        "hard": {"max_n": 1000, "max_mod": 1000000000},
        "scaling_strategy": "exponential",
        "description": "Compute nth Catalan number modulo m"
    },

    "FactorialTrailingZeroCount": {
        "parameters": ["max_n_k"],
        "easy": {"max_n_k": 10},
        "hard": {"max_n_k": 1000},
        "scaling_strategy": "exponential",
        "description": "Count trailing zeros in n! in base k"
    },

    # ===== Additional Common Environments (can be added later) =====
    # Placeholder for Phase 2 expansion
}


# Exponential interpolation function for RLVE configs
def exponential_interpolate(easy_val: float, hard_val: float, progress: float, base: float = 2.0) -> float:
    """
    Exponential interpolation for parameters that should grow exponentially.

    Args:
        easy_val: Value at easiest difficulty (progress=0.0)
        hard_val: Value at hardest difficulty (progress=1.0)
        progress: Difficulty progress from 0.0 to 1.0
        base: Base for exponential growth (default: 2.0)

    Returns:
        Interpolated value with exponential scaling
    """
    import math
    # Use logarithmic scale for exponential growth
    log_easy = math.log(max(1, easy_val))
    log_hard = math.log(max(1, hard_val))
    log_val = log_easy + (log_hard - log_easy) * progress
    return math.exp(log_val)
