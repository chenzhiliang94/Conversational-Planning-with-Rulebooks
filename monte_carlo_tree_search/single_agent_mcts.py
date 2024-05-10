import random

from monte_carlo_tree_search.mcts import Node, MCTS

class SingleAgentNode(Node):
    def __init__(
        self,
        mdp,
        parent,
        state,
        qfunction,
        bandit,
        #depth=0,
        reward=0.0,
        action=None,
    ):
        super().__init__(mdp, parent, state, qfunction, bandit, reward, action)

        # A dictionary from actions to a set of node-probability pairs
        self.children = {}
        #self.depth = depth

    
    """ Return true if and only if all child actions have been expanded """

    def is_fully_expanded(self):
        valid_actions = self.mdp.get_actions(self.state)
        print("valid actions:", len(valid_actions))
        print("number of children: ", len(self.children))
        if len(valid_actions) == len(self.children):
            return True
        else:
            return False

    """ Select a node that is not fully expanded """

    def select(self):
        print("selecting...")
        if not self.is_fully_expanded() or self.mdp.is_terminal(self.state):
            return self
        else:
            actions = list(self.children.keys())
            action = self.bandit.select(self.state, actions, self.qfunction)
            return self.get_outcome_child_select(action).select()

    """ Expand a node if it is not a terminal node """

    def expand(self):
        if not self.mdp.is_terminal(self.state):
            # Randomly select an unexpanded action to expand
            print("expanding...")
            print("number of children: ", len(set(self.children.keys())))
            print("number of actions: ", len(set(self.mdp.get_actions(self.state))))
            actions = set(self.mdp.get_actions(self.state)) - set(self.children.keys())

            if len(actions) == 0:
                print("here")
                return Exception("ERROR. action is empty. Why?")
            action = random.choice(list(actions))

            self.children[action] = []
            return self.get_outcome_child_expand(action)
        return self

    """ Backpropogate the reward back to the parent node """

    def back_propagate(self, reward, child):
        action = child.action

        Node.visits[self.state] = Node.visits[self.state] + 1
        Node.visits[(self.state, action)] = Node.visits[(self.state, action)] + 1

        q_value = self.qfunction.get_q_value(self.state, action)
        delta = (1 / (Node.visits[(self.state, action)])) * (
            reward - self.qfunction.get_q_value(self.state, action)
        )
        
        self.qfunction.update(self.state, action, delta, (1 / (Node.visits[(self.state, action)])), reward)

        if self.parent != None:
            self.parent.back_propagate(self.reward + reward, self)

    """ Simulate the outcome of an action, and return the child node. Note this has distinct result mapping because we use this during select and expand stage """

    def get_outcome_child_select(self, action):
        # Choose one outcome based on transition probabilities
        (next_state, reward) = self.mdp.execute_in_selection(self.state, action)

        # Find the corresponding state and return if this already exists
        for (child) in self.children[action]:
            if next_state.response == child.state.response:
                return child

        # This outcome has not occured from this state-action pair previously. Note each action can map to N human responses which are alreayd generated in 
        # execute_in_selection function (already generated, but we actually see it for first time here)
        new_child = SingleAgentNode(
            self.mdp, self, next_state, self.qfunction, self.bandit, reward, action
        )
        #Find the probability of this outcome (only possible for model-based) for visualising tree
        self.children[action].append(new_child)
        return new_child
    
    def get_outcome_child_expand(self, action):
         # Choose one outcome based on transition probabilities
        (next_state, reward) = self.mdp.execute_in_expansion(self.state, action)

        # Find the corresponding state and return if this already exists
        for (child) in self.children[action]:
            if next_state.response == child.state.response:
                print("child is found")
                return child
            
        # This outcome has not occured from this state-action pair previously. Note each action can map to N human responses which are alreayd generated in 
        # execute_in_selection function (already generated, but we actually see it for first time here)
        new_child = SingleAgentNode(
            self.mdp, self, next_state, self.qfunction, self.bandit, reward, action
        )
    
        # Find the probability of this outcome (only possible for model-based) for visualising tree
        self.children[action].append(new_child)
        return new_child

class SingleAgentMCTS(MCTS):
    def create_root_node(self):
        return SingleAgentNode(
            self.mdp, None, self.mdp.get_initial_state(), self.qfunction, self.bandit
        )
