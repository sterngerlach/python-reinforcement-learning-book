# coding: utf-8
# monte_carlo.py

import math
import gym

from collections import defaultdict
from agent import Agent
from frozen_lake import show_q_value

class MonteCarloAgent(Agent):
    def __init__(self, epsilon=0.1):
        super().__init__(epsilon)
        
    def learn(self, env, episode_count=1000, gamma=0.9,
              render=False, report_interval=50):
        self.init_log()
        self.Q = defaultdict(lambda: [0] * len(actions))
        
        N = defaultdict(lambda: [0] * len(actions))
        actions = list(range(env.action_space.n))
        
        for e in range(episode_count):
            s = env.reset()
            done = False
            experience = []
            
            while not done:
                if render:
                    env.render()
                
                a = self.policy(s, actions)
                next_state, reward, done, info = env.step(a)
                experience.append({ "state": s, "action": a, "reward": reward })
                s = next_state
            else:
                self.log(reward)
                
            for i, x in enumerate(experience):
                s, a = x["state"], x["action"]
                G, t = 0, 0
                
                for j in range(i, len(experience)):
                    G += math.pow(gamma, t) * experience[j]["reward"]
                    t += 1
                
                N[s][a] += 1
                alpha = 1.0 / N[s][a]
                self.Q[s][a] += alpha * (G - self.Q[s][a])
                
            if e != 0 and e % report_interval == 0:
                self.show_reward_log(episode=e)
                
def main():
    agent = MonteCarloAgent(epsilon=0.1)
    env = gym.make("FrozenLakeEasy-v0")
    agent.learn(env, episode_count=500)
    show_q_value(agent.Q)
    agent.show_reward_log()

if __name__ == "__main__":
    main()
    