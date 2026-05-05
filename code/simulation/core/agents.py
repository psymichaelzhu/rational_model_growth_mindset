import math
import numpy as np
from .utils import reward_function, skill_selection, beta_mean


class Agent_full_knowledge:
    """
    Agent with complete knowledge of environment parameters (M, W, R, T).

    Policy: 
    1. For each skill, estimate the optimal switch point (the step at which
       to switch from cultivating to harvesting) and corresponding expected return 
    2. Greedily select the skill with the highest expected return 
    3. Cultivate/harvest that skill according to the estimated switch point

    params keys:
      n_skills: [int]: number of skills
      M: [list [float]]: malleability for each skill, each element is a float between 0 and 1
      W: [list [float]]: reward growth rate for each skill
      R: [list [float]]: reward baseline for each skill
      T: [int]: time horizon
    """

    def __init__(self, params):
        self.n_skills = params['n_skills']
        self.M = params['M'].copy()
        self.W = params['W'].copy()
        self.R = params['R'].copy()
        self.T = params['T']
        self.reward_function_type = 'linear'
        self.skill_selection_type = 'greedy'

        self.initialize()

    def initialize(self):
        self.current_step = None  # current time step index
        self.current_skill_levels = None  # current skill levels for all skills

        self.intended_skill = None  # which skill the agent intends to engage with in the current step
        self.action = None  # whether the agent intends to cultivate or harvest in the current step

        # agent's policy-based estimations
        self.switch_points = [None for i in range(self.n_skills)]  # optimal switch points for each skill, relative to current time step
        self.returns = [None for i in range(self.n_skills)]  # maximum expected returns for each skill, attained at the optimal switch point

        self.logging = {}  # data collector

    def _calculate_return(self, switch_point, skill):
        """
        Expected return for the specified skill and switch point,
        following the optimal policy of cultivating until switch_point and harvesting.

        switch_point is relative to the current time step (i.e., how many steps to cultivate from now on before harvesting).
        D = T - current_step is the number of remaining steps.

        params keys:
          switch_point: [int]: switch point, relative, in range [0, D]
          skill: [int]: skill index, 0-based
        """
        D = self.T - self.current_step  # remaining steps from current time step
        if not (0 <= switch_point <= D):  # [0, D]
            raise ValueError("Switch point must be within remaining steps")

        m = self.M[skill]
        w = self.W[skill]
        r = self.R[skill]
        I_t = self.current_skill_levels[skill]  # current skill level

        # R'_t = R + w * I_t: current harvest payoff given current skill level
        r_prime = r + w * I_t

        n = switch_point  # number of cultivation attempts before harvesting
        expected_harvest_reward = 0
        for k in range(n + 1):  # k: each possible number of cultivation successes
            prob = math.comb(n, k) * (m ** k) * ((1 - m) ** (n - k))
            reward = reward_function(r_prime, w, k, self.reward_function_type)  # harvest reward given k additional skill levels gained
            expected_harvest_reward += prob * reward

        expected_return = expected_harvest_reward * (D - switch_point)  # all harvest steps yield the same reward
        return expected_return

    def _policy_estimate_brute(self, skill=None):
        """
        Brute-force: enumerate all switch points in [0, D] and select the one with the highest expected return.

        params keys:
          skill: [int]: skill index, 0-based, or None to compute for all skills
        """
        if skill is None:
            for skill in range(self.n_skills):
                self._policy_estimate_brute(skill=skill)
        else:
            D = self.T - self.current_step
            returns = [self._calculate_return(d, skill) for d in range(D + 1)]
            self.switch_points[skill] = np.argmax(returns)
            self.returns[skill] = returns[self.switch_points[skill]]

    def _policy_estimate_analytic(self, skill=None):
        """
        Closed-form: uses the analytic solution derived for the linear reward function (see paper).

        The expected return G(d) = (D - d)(R'_t + w * m * d) is a concave quadratic in d,
        maximized at d_tilde = D/2 - R'_t / (2 * w * m).
        The optimal integer switch point clips to [0, D] and rounds to the nearest integer.

        params keys:
          skill: [int]: skill index, 0-based, or None to compute for all skills
        """
        if skill is None:
            for skill in range(self.n_skills):
                self._policy_estimate_analytic(skill=skill)
        else:
            D = self.T - self.current_step
            m = self.M[skill]
            w = self.W[skill]
            r = self.R[skill]
            I_t = self.current_skill_levels[skill]
            r_prime = r + w * I_t  # R'_t = R + w * I_t

            # analytic solution requires w * m > 0; fall back to brute-force
            if w * m <= 0:
                self._policy_estimate_brute(skill=skill)
                return

            # continuous maximizer of G(d) = (D - d)(R'_t + w * m * d)
            d_tilde = D / 2 - r_prime / (2 * w * m)
            # clip to [0, D] and round to nearest integer to get optimal switch point
            self.switch_points[skill] = int(np.clip(round(d_tilde), 0, D))
            d_star = self.switch_points[skill]
            self.returns[skill] = (D - d_star) * (r_prime + w * m * d_star)

    def act(self, env, policy_mode='analytic'):
        """
        policy_mode: [str]: 'analytic' (default) or 'brute_force'
        """
        # environment state
        self.current_step = env.current_step
        self.current_skill_levels = env.current_skill_levels.copy()

        # agent estimates optimal switch point and corresponding expected return
        if policy_mode == 'brute_force':
            self._policy_estimate_brute()
        elif policy_mode == 'analytic':
            self._policy_estimate_analytic()
        else:
            raise ValueError(f"Invalid policy_mode: {policy_mode}. Choose 'brute_force' or 'analytic'.")

        # agent selects the skill to engage with
        self.intended_skill = skill_selection(
            self.returns,
            self.skill_selection_type
        )

        # cultivate if the relative switch point is positive, otherwise harvest (switch point = 0 means harvest right away)
        if self.switch_points[self.intended_skill] > 0:
            self.action = 'cultivate'
        else:
            self.action = 'harvest'

        # data collection
        self.logging['switch_points'] = self.switch_points.copy()
        self.logging['returns'] = self.returns.copy()


class Agent_unknown_malleability(Agent_full_knowledge):
    """
    Agent with unknown malleability, extending Agent_full_knowledge with Bayesian belief updates.

    Maintains a Beta(alpha, beta) belief over M for each skill.
    After each cultivation attempt, updates the belief and recomputes M as the posterior mean.

    params keys:
      M_prior (list of (alpha, beta) tuples), plus all keys from Agent_full_knowledge
    """

    def __init__(self, params):
        super().__init__(params)
        # initialize prior malleability belief
        self.M_alpha = [params['M_prior'][i][0] for i in range(self.n_skills)] # alpha parameter of the Beta distribution, number of successes
        self.M_beta = [params['M_prior'][i][1] for i in range(self.n_skills)] # beta parameter of the Beta distribution, number of failures
        # initialize malleability estimate based on the prior belief
        self.M = [beta_mean(self.M_alpha[i], self.M_beta[i]) for i in range(self.n_skills)]
        # note, although the parent class Agent_full_knowledge stores the M parameter,
        # class Agent_unknown_malleability will not use the true M values for decision-making, but use the estimated M values instead
        # the true M values are overridden by the estimated M values here

    def update_belief(self, env):
        """Update Beta belief for the chosen skill based on last cultivation outcome."""
        cultivation_outcome = env.cultivation_outcome # outcome of the cultivation attempt: 1 for success, 0 for failure, or None if harvest
        chosen_skill = env.chosen_skill # which skill was chosen in the most recent step

        if cultivation_outcome is None:  # harvest action or first step, no update of the belief
            return None
        elif cultivation_outcome == 1:
            self.M_alpha[chosen_skill] += 1 # increment the alpha parameter by 1 if the cultivation is successful
        elif cultivation_outcome == 0:
            self.M_beta[chosen_skill] += 1 # increment the beta parameter by 1 if the cultivation is unsuccessful
        else:
            raise ValueError("Invalid outcome")

        self.M[chosen_skill] = beta_mean(
            self.M_alpha[chosen_skill],
            self.M_beta[chosen_skill]
        )

    def act(self, env, policy_mode='analytic'):
        self.update_belief(env)
        super().act(env, policy_mode=policy_mode)

        # data collection
        self.logging['M_alpha'] = self.M_alpha.copy()
        self.logging['M_beta'] = self.M_beta.copy()
