# coding: utf-8
# planner.py

class Planner(object):
    def __init__(self, env):
        self.env = env
        self.log = []
        
    def initialize(self):
        self.env.reset()
        self.log = []
        
    def plan(self, gamma=0.9, threshold=1e-04):
        raise NotImplementedError()
        
    def transition(self, state, action):
        transition_probs = self.env.calc_transition_probs(state, action)
        
        for next_state in transition_probs:
            prob = transition_probs[next_state]
            reward, _ = self.env.reward(next_state)
            yield prob, next_state, reward
            
    def state_value_dict_to_grid(self, state_value_dict):
        grid = []
        
        for i in range(self.env.row_length):
            row = [0] * self.env.column_length
            grid.append(row)
            
        for s in state_value_dict:
            grid[s.row][s.column] = state_value_dict[s]
            
        return grid
    
class ValueIterationPlanner(Planner):
    def __init__(self, env):
        super().__init__(env)
        
    def plan(self, gamma=0.9, threshold=1e-04):
        self.initialize()
        
        V = {}
        
        for s in self.env.states:
            V[s] = 0
            
        while True:
            delta = 0.0
            self.log.append(self.state_value_dict_to_grid(V))
            
            for s in V:
                if not self.env.can_action_at(s):
                    continue
                
                expected_rewards = []
                
                for a in self.env.actions:
                    r = 0.0
                    
                    for prob, next_state, reward in self.transition(s, a):
                        r += prob * (reward + gamma * V[next_state])
                        
                    expected_rewards.append(r)
                    
                expected_reward_max = max(expected_rewards)
                delta = max(delta, abs(expected_reward_max - V[s]))
                V[s] = expected_reward_max
                
            if delta < threshold:
                break
            
        V_grid = self.state_value_dict_to_grid(V)
        
        return V_grid
    
class PolicyIterationPlanner(Planner):
    def __init__(self, env):
        super().__init__(env)
        self.policy = {}
        
    def initialize(self):
        super().initialize()
        self.policy = {}
        
        actions = self.env.actions
        states = self.env.states
        
        for s in states:
            self.policy[s] = {}
            
            for a in actions:
                self.policy[s][a] = 1.0 / len(actions)
                
    def calculate_value(self, gamma, threshold):
        V = {}
        
        for s in self.env.states:
            V[s] = 0
            
        while True:
            delta = 0.0
            
            for s in V:
                value = 0.0
                
                for a in self.policy[s]:
                    action_prob = self.policy[s][a]
                    
                    for prob, next_state, reward in self.transition(s, a):
                        value += action_prob * prob * (reward + gamma * V[next_state])
                
                delta = max(delta, abs(value - V[s]))
                V[s] = value
                
            if delta < threshold:
                break
            
        return V
    
    def calculate_policy(self, gamma, V):
        update_stable = True
        
        for s in self.env.states:
            policy_action = self.take_max_action(self.policy[s])
            
            q_values = {}
            
            for a in self.env.actions:
                r = 0.0
                
                for prob, next_state, reward in self.transition(s, a):
                    r += prob * (reward + gamma * V[next_state])
                    
                q_values[a] = r
                
            best_action = self.take_max_action(q_values)
            
            if policy_action != best_action:
                update_stable = False
                
            for a in self.policy[s]:
                prob = 1.0 if a == best_action else 0.0
                self.policy[s][a] = prob
        
        return update_stable
    
    def take_max_action(self, action_value_dict):
        return max(action_value_dict, key=action_value_dict.get)
        
    def plan(self, gamma=0.9, threshold=1e-04):
        self.initialize()
        
        while True:
            V = self.calculate_value(gamma, threshold)
            self.log.append(self.state_value_dict_to_grid(V))
            
            update_stable = self.calculate_policy(gamma, V)
            
            if update_stable:
                break
            
        V_grid = self.state_value_dict_to_grid(V)
        
        return V_grid
    