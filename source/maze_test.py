# coding: utf-8
# maze_test.py

import numpy as np

from maze_environment import CellType, Environment

class Agent(object):
    def __init__(self, env):
        self.actions = env.actions
        
    def policy(self, state):
        return np.random.choice(self.actions)
    
def main():
    grid = [[CellType.NORMAL, CellType.NORMAL, CellType.NORMAL, CellType.REWARD],
            [CellType.NORMAL, CellType.BLOCK,  CellType.NORMAL, CellType.DAMAGE],
            [CellType.NORMAL, CellType.NORMAL, CellType.NORMAL, CellType.NORMAL]]
    
    env = Environment(grid)
    agent = Agent(env)
    
    for i in range(10):
        state = env.reset()
        total_reward = 0.0
        done = False
        
        while not done:
            action = agent.policy(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state
        
        print("Episode {}: Agent got {} reward".format(i, total_reward))
        
if __name__ == "__main__":
    main()
    