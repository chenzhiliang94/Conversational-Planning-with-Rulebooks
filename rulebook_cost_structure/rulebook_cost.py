from typing import List, Set, Dict
import numpy as np

class RulebookCost:
    def __init__(self, cost):
        self.is_fitted = False
        self.cost_structure = []
        self.cost = cost
    
    def __lt__(self, other):
        assert len(other.cost) == len(self.cost), "cost sizes are different for comparison"
        for c1, c2 in zip(self.cost, other.cost):
            if self.cost < other.cost:
                return True
            elif self.cost > other.cost:
                return False
        return True # if both equal, tie break arbitrarily
    
    def __eq__(self, other):
        assert len(other.cost) == len(self.cost), "cost sizes are different for comparison"
        return self.cost == other.cost
    
    def get_features(self, traj_features : Dict) -> List:
        rules = traj_features[list(traj_features.keys())[0]]
        rules = rules[list(rules.keys())[0]]
        return rules
    
    def manually_insert_cost_structure(self, cost_structure : List):
        self.is_fitted = True
        self.cost_structure = cost_structure
    
    def fit(self, traj_features : Dict, preference_data : np.array):
        assert len(traj_features) > 0
        preference_shape = np.shape(preference_data)
        assert preference_shape[0] > 0 and preference_shape[1] == 2
        
        rules = self.get_features(traj_features)
        
        # rearrange rule priority from data
        # ...
        fitted_rules = rules # write code 
        
        self.cost_structure = fitted_rules
        self.is_fitted = True
    
    def predict(self, traj_a_cost : Dict, traj_b_cost: Dict) -> int:
        if not self.is_fitted:
            raise Exception("Error. Rulebook does not have any fitted rules given.")
        for rule in self.cost_structure:
            if traj_a_cost[rule] < traj_b_cost[rule]:
                return 1 # traj a is better than b
            elif traj_a_cost[rule] > traj_b_cost[rule]:
                return 0 # traj b is better than a
            continue
        
        # a total tie between two trajectories, hence choose 1 as tie breaker by default
        return 1
        