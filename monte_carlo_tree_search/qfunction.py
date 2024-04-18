from monte_carlo_tree_search.tabular_policy import TabularPolicy
from abc import ABC, abstractmethod

class QFunction:

    """ Update the Q-value of (state, action) by delta """
    @abstractmethod
    def update(self, state, action, delta, visits, reward):
        pass

    """ Get a Q value for a given state-action pair """
    @abstractmethod
    def get_q_value(self, state, action):
        pass

    """ Return a pair containing the action and Q-value, where the
        action has the maximum Q-value in state
    """

    def get_max_q(self, state, actions):
        arg_max_q = None
        max_q = float("-inf")
        for action in actions:
            value = self.get_q_value(state, action)
            if max_q < value:
                arg_max_q = action
                max_q = value
        return (arg_max_q, max_q)

    """ Extract a policy for this Q-function  """

    def extract_policy(self, mdp):
        policy = TabularPolicy()
        for state in mdp.get_states():
            # Find the action with maximum Q-value and make this the
            (action, _) = self.get_max_q(state, mdp.get_actions(state))
            policy.update(state, action)

        return policy
