# coding: utf-8
# actor_critic.py

import numpy as np
import gym

from agent import Agent
from frozen_lake import show_q_value

class Actor(Agent):
    def __init__(self, env):
        super().__init__(epsilon=-1)
        
        nrow = env.observation_space.n
        ncol = env.action_space.n
        
        self.actions = list(range(env.action_space.n))
        self.Q = np.random.uniform(0.0, 1.0, nrow * ncol).reshape((nrow, ncol))
        
    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    
    def policy(self, s):
        return np.random.choice(self.actions, 1, p=self.softmax(self.Q[s]))[0]
    
class Critic(object):
    def __init__(self, env):
        states = env.observation_space.n
        self.V = np.zeros(states)
        
class ActorCritic(object):
    def __init__(self, actor_cls, critic_cls):
        self.actor_cls = actor_cls
        self.critic_cls = critic_cls
        
    def train(self, env, episode_count=1000, gamma=0.9,
              learning_rate=0.1, render=False, report_interval=50):
        actor = self.actor_cls(env)
        critic = self.critic_cls(env)
        
        actor.init_log()
        
        for e in range(episode_count):
            s = env.reset()
            done = False
            
            while not done:
                if render:
                    env.render()
                    
                a = actor.policy(s)
                next_state, reward, done, info = env.step(a)
                
                gain = reward + gamma * critic.V[next_state]
                td = gain - critic.V[s]
                
                actor.Q[s][a] += learning_rate * td
                critic.V[s] += learning_rate * td
                s = next_state
            else:
                actor.log(reward)
            
            if e != 0 and e % report_interval == 0:
                actor.show_reward_log(episode=e)
                
        return actor, critic
    
def main():
    trainer = ActorCritic(Actor, Critic)
    env = gym.make("FrozenLakeEasy-v0")
    actor, critic = trainer.train(env, episode_count=3000)
    show_q_value(actor.Q)
    actor.show_reward_log()
    
if __name__ == "__main__":
    main()
    