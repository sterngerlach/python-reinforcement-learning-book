# coding: utf-8
# sarsa.py

import gym

from collections import defaultdict
from agent import Agent
from frozen_lake import show_q_value

class SarsaAgent(Agent):
    def __init__(self, epsilon=0.1):
        super().__init__(epsilon)
        
    def learn(self, env, episode_count=1000, gamma=0.9,
              learning_rate=0.1, render=False, report_interval=50):
        self.init_log()
        self.Q = defaultdict(lambda: [0] * len(actions))
        actions = list(range(env.action_space.n))
        
        for e in range(episode_count):
            s = env.reset()
            done = False
            a = self.policy(s, actions)
            
            while not done:
                if render:
                    env.render()
                
                next_state, reward, done, info = env.step(a)
                next_action = self.policy(next_state, actions)
                
                gain = reward + gamma * self.Q[next_state][next_action]
                self.Q[s][a] += learning_rate * (gain - self.Q[s][a])
                
                s = next_state
                a = next_action
            else:
                self.log(reward)
                
            if e != 0 and e % report_interval == 0:
                self.show_reward_log(episode=e)

def main():
    agent = SarsaAgent()
    env = gym.make("FrozenLakeEasy-v0")
    agent.learn(env, episode_count=500)
    show_q_value(agent.Q)
    agent.show_reward_log()
    
if __name__ == "__main__":
    main()
    