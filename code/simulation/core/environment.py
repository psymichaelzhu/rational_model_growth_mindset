import numpy as np
from .utils import reward_function


class Environment:
    f"""
    Multi-skill environment for the cultivate-harvest dilemma.

    At each step the agent picks one skill and either:
      - cultivates: 
        - advances skill level with a probability M, p(s_i+1|s_i, a_c) = M, p(s_i|s_i, a_c) = 1 - M
        - no reward, r(s_i, a_c) = 0
      - harvests: 
        - skill level remains unchanged, p(s_i|s_i, a_h) = 1
        - collects reward based on current skill level, r(s_i, a_h) = R + W * i (i is the skill level)

    Each skill is defined by three parameters:
      M  — malleability: success probability of cultivation
      W  — reward growth rate: slope of the harvest reward function
      R  — reward baseline: intercept of the harvest reward function

    Note. The above description is based on linear reward function.

    Time horizon T and reward function type apply to all skills

    params keys:
      n_skills: [int]: number of skills
      M: [list [float]]: malleability for each skill, each element is a float between 0 and 1
      W: [list [float]]: reward growth rate for each skill
      R: [list [float]]: reward baseline for each skill
      T: [int]: time horizon
      env_seed: [int]: random seed for reproducibility
    """

    def __init__(self, params):
        self.n_skills = params['n_skills']
        self.M = params['M'].copy()
        self.W = params['W'].copy()
        self.R = params['R'].copy()
        self.T = params['T']
        self.reward_function_type = 'linear'
        self.env_seed = params.get('env_seed', None)

        # validate parameters
        if len(self.M) != self.n_skills:
            raise ValueError("Length of M must match n_skills")
        if len(self.W) != self.n_skills:
            raise ValueError("Length of W must match n_skills")
        if len(self.R) != self.n_skills:
            raise ValueError("Length of R must match n_skills")
        if any(m <= 0 or m >= 1 for m in self.M):
            raise ValueError("Each M must be between 0 and 1")

        self.initialize()

    def initialize(self):
        self._generate_outcomes() # pre-generate cultivation outcome sequences

        self.finish = False # whether the time horizon is reached
        self.current_skill_levels = [0] * self.n_skills # current skill levels for all skills, all skills start at level 0
        self.current_step = 0 # current time step index
        self.chosen_skill = None # which skill was chosen in the most recent step
        self.cultivation_outcome = None # outcome of the cultivation attempt: 1 for success, 0 for failure, or None if harvest

        self.logging = {} # data collector

    def _generate_outcomes(self):
        """Pre-generate cultivation outcome sequences for all skills."""
        self.rng = np.random.default_rng(self.env_seed)
        self.outcomes = [
            (self.rng.random(self.T) < self.M[idx_skill]).astype(int).tolist()
            for idx_skill in range(self.n_skills)
        ]

    def step(self, agent):
        # agent's decision
        intended_skill = agent.intended_skill # which skill the agent intends to engage with in the current step
        action = agent.action # whether the agent intends to cultivate or harvest in the current step

        if intended_skill not in range(self.n_skills):
            raise ValueError("Invalid skill index")

        if action == 'cultivate':
            cultivation_outcome = self.outcomes[intended_skill][self.current_step] # improve the skill level with a probability M
            if cultivation_outcome == 1:
                self.current_skill_levels[intended_skill] += 1
            reward = 0 # no reward
        elif action == 'harvest':
            cultivation_outcome = None # no cultivation outcome, no improvement of the skill level
            reward = reward_function( # reward is based on current skill level
                self.R[intended_skill], self.W[intended_skill],
                self.current_skill_levels[intended_skill], self.reward_function_type
            )
        else:
            raise ValueError("Invalid action")

        self.chosen_skill = intended_skill
        self.cultivation_outcome = cultivation_outcome
        self.current_step += 1
        self.finish = (self.current_step >= self.T)

        # data collection
        self.logging['new_skill_levels'] = self.current_skill_levels.copy()
        self.logging['cultivation_outcome'] = cultivation_outcome
        self.logging['reward'] = reward
        self.logging['finish'] = self.finish