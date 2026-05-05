import numpy as np


def beta_mean(alpha, beta):
    return alpha / (alpha + beta)

def reward_function(R, W, level, function_type='linear'):
    """
    R: reward baseline
    W: reward growth rate
    level: current skill level (0-based)
    function_type: 'linear'
    """
    if function_type == 'linear':
        return R + W * level
    else:
        raise ValueError("Invalid reward function type")


def skill_selection(expected_return, selection_type = 'greedy'):
    """
    expected_return: list of expected returns for each skill
    selection_type: 'greedy' returns the skill with the highest expected return
    """
    if selection_type == 'greedy':
        return np.argmax(expected_return)
    else:
        raise ValueError("Invalid selection type")