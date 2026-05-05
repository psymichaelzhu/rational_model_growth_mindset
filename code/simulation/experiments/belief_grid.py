"""
Experiment: Belief Grid
Sweeps over all combinations of prior mean (M0_m) and prior concentration (M0_c), as well as the skill malleability M
to examine how prior beliefs about malleability affect decision behavior.
"""
from datetime import datetime
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from simulation.core.runner import run_simulation_process

n_sim = 500

# default parameters for the environment
R = 51
W = 7.5
T = 45
# M is not specified in the default parameter, cause each condition in env_dict has M

default_param = {
    'n_skills': 2,
    'T': T,
    'R': [R * 4/3, R * 2/3],
    'W': [W * 2/3, W * 4/3],
}

# both skills share the same malleability M
# we consider three levels of skill malleability: low (0.3), medium (0.5), and high (0.7)
env_dict = {
    'M=0.3': {'M': [0.3, 0.3]},
    'M=0.5': {'M': [0.5, 0.5]},
    'M=0.7': {'M': [0.7, 0.7]},
}

# Build agent with varying prior beliefs about malleability
agent_dict = {}
for M0_m in [round(x * 0.1, 1) for x in range(11)]:  # prior mean: 0.0, 0.1, ..., 1.0
    for M0_c in range(10, 61, 10):  # prior concentration: 10, 20, ..., 60
        alpha = M0_m * M0_c
        beta  = (1 - M0_m) * M0_c
        agent_dict[f'{M0_m}_{M0_c}'] = {'M_prior': [(alpha, beta), (alpha, beta)]}

experiment_id = 'belief_grid' + datetime.now().strftime('_%Y%m%d_%H%M%S')

run_simulation_process(
    experiment_id=experiment_id,
    n_sim=n_sim,
    default_param=default_param,
    env_dict=env_dict,
    agent_dict=agent_dict,
)

print(experiment_id)

