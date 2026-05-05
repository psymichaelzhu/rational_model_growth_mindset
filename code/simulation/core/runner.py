import os
import json
import pandas as pd
from tqdm import tqdm
from simulation.core.environment import Environment
from simulation.core.agents import Agent_unknown_malleability

SKILL_LABELS = {0: 'Easy', 1: 'Challenging'}


def run_simulation_process(experiment_id, n_sim=30,
                            default_param=None, env_dict=None,
                            agent_dict=None):
    """
    Run simulations across all combinations of environment, malleability, and agent configs.
    Saves per-trial data as CSV and experiment config as JSON.

    Note that this function is tailored for the task in the paper
    where we have two skills, Easy and Challenging; for each skill, the agent decides to cultivate or harvest at each step.
    It only considers agents with malleability uncertainty, not agents with full knowledge of the environment parameters.

    param keys:
      experiment_id: [str]: unique identifier for the whole experiment simulation
      n_sim: [int]: number of simulations to run
      default_param: [dict]: default parameters for the environment
      env_dict: [dict]: dictionary containing the environment parameters to update for each condition
      agent_dict: [dict]: dictionary containing the agent parameters to update for each agent type
    """
    get_val = lambda arr, idx: arr[idx] if (arr is not None and len(arr) > idx) else None

    all_trials = []

    for env_name, env_update in env_dict.items():
        # for example:
        # env_name = 'test_condition'
        # env_update = {'R': [4, 2], 'W': [2, 3]}
        # this means the R and W parameters will be replaced with [4, 2] and [2, 3] for this condition
        # while the remaining parameters will use the default parameter
        for agent_name, agent_update in agent_dict.items():
            # for example:
            # agent_name = 'growth_mindset'
            # agent_update = {'M_prior': [(7, 3), (7, 3)]}
            # this means the M_prior parameter will be replaced with [(7, 3), (7, 3)] for this condition
            # while the remaining parameters will use the default parameter
            env_param = default_param.copy() # copy the default parameter to environment
            env_param.update(env_update) # update the parameter with new value given in env_update

            agent_param = env_param.copy() # agent have full knolwedge of the environment parameters (except M)
            # note, for agents in class Agent_unknown_malleability:
            # they won't use true M values for decision-making, but use the estimated M values instead
            # the M passed to agents will be overridden by the estimated M values in the initialization
            agent_param.update(agent_update) # update the parameter with new value given in agent_update, mainly for M_prior

            for sim_id in tqdm(list(range(n_sim)),
                               desc=f'{agent_name} in {env_name}'):
                env_param.update({'env_seed': sim_id}) # use sim_id as seed, so that each simulation has a unique seed
                # initialize environment and agent
                env = Environment(env_param)
                agent = Agent_unknown_malleability(agent_param)

                while not env.finish:
                    agent.act(env)
                    env.step(agent)

                    intended_skill  = agent.intended_skill
                    action          = agent.action
                    M               = agent.M
                    switch_points   = agent.logging.get('switch_points', None)
                    returns         = agent.logging.get('returns', None)
                    cultivation_outcome = env.logging.get('cultivation_outcome', None)
                    reward              = env.logging.get('reward', None)
                    new_skill_levels    = env.logging.get('new_skill_levels', None)

                    trial_record = {
                        'sim':     sim_id,
                        'trial':   env.current_step,

                        'env':     env_name,
                        'M_level': env.M[0],
                        'agent':   agent_name,

                        'skill1_Mhat':         get_val(M, 0),
                        'skill1_switch_point': get_val(switch_points, 0),
                        'skill1_return':       get_val(returns, 0),

                        'skill2_Mhat':         get_val(M, 1),
                        'skill2_switch_point': get_val(switch_points, 1),
                        'skill2_return':       get_val(returns, 1),

                        'intended_skill':      SKILL_LABELS.get(intended_skill, None),
                        'action':              action,
                        'cultivation_outcome': cultivation_outcome,
                        'reward':              reward,

                        'skill1_new_level': get_val(new_skill_levels, 0),
                        'skill2_new_level': get_val(new_skill_levels, 1),
                    }
                    all_trials.append(trial_record)

    summary_df = pd.DataFrame(all_trials)

    # save results
    directory = f'../../../results/simulation/{experiment_id}'
    os.makedirs(directory, exist_ok=True)

    summary_df.to_csv(os.path.join(directory, f'trial_results.csv'), index=False)

    # Append this run's config to a shared info.json
    json_filepath = os.path.join(directory, 'info.json')
    runs = []
    if os.path.exists(json_filepath):
        with open(json_filepath, 'r') as f:
            runs = json.load(f)

    runs.append({
        'experiment_id':     experiment_id,
        'n_sim':             n_sim,
        'default_param':     default_param,
        'env_dict':          env_dict,
        'agent_dict':        agent_dict,
    })
    with open(json_filepath, 'w') as f:
        json.dump(runs, f, indent=2)
