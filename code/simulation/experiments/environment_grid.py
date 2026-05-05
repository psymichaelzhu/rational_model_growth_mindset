"""
Experiment: Environment Grid
Compares growth-mindset (gm) vs fixed-mindset (fm) agents across
different values of R (reward baseline), W (reward growth rate), and T (time horizon).
"""
from datetime import datetime
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from simulation.core.runner import run_simulation_process

n_sim = 500

gm_prior = (6, 4) # gm: high-M prior Beta(6,4)
fm_prior  = gm_prior[::-1] # fm: low-M prior Beta(4,6)

agent_dict = {
    'gm': {'M_prior': [gm_prior, gm_prior]},
    'fm': {'M_prior': [fm_prior, fm_prior]},
}

# default values for the environment
R = 51
W = 7.5
T = 45
M = 0.5

default_param = {
    'n_skills': 2,
    'T': T,
    'R': [R * 4/3, R * 2/3], # Easy skill: high baseline, low growth rate
    'W': [W * 2/3, W * 4/3], # Challenging skill: low baseline, high growth rate
    'M': [M, M],
}

env_dict = {
    # R sweep
    'R=3.0':  {'R': [3  * 4/3, 3  * 2/3]},
    'R=51.0': {'R': [51 * 4/3, 51 * 2/3]},
    'R=99.0': {'R': [99 * 4/3, 99 * 2/3]},
    # W sweep
    'W=0.9':  {'W': [0.9  * 2/3, 0.9  * 4/3]},
    'W=7.5':  {'W': [7.5  * 2/3, 7.5  * 4/3]},
    'W=14.1': {'W': [14.1 * 2/3, 14.1 * 4/3]},
    # T sweep
    'T=10.0': {'T': 10},
    'T=45.0': {'T': 45},
    'T=80.0': {'T': 80},
}

experiment_id = 'environment_grid' + datetime.now().strftime('_%Y%m%d_%H%M%S')

run_simulation_process(
    experiment_id=experiment_id,
    n_sim=n_sim,
    default_param=default_param,
    env_dict=env_dict,
    agent_dict=agent_dict,
)

print(experiment_id)
