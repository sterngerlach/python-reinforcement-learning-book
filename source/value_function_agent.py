# coding: utf-8
# value_function_agent.py

import random
import argparse
import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

import gym

from framework import FNAgent, Trainer, Observer

class ValueFunctionAgent(FNAgent):
    def save(self, model_path):
        joblib.dump(self.model, model_path)
        
    @classmethod
    def load(cls, env, model_path, epsilon=1e-04):
        actions = list(range(env.action_space.n))
        
        agent = cls(epsilon, actions)
        agent.model = joblib.load(model_path)
        agent.initialized = True
        
        return agent
    
    def initialize(self, experiences):
        scaler = StandardScaler()
        estimator = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1)
        self.model = Pipeline([("scaler", scaler), ("estimator", estimator)])
        
        states = np.vstack([e.s for e in experiences])
        self.model.named_steps["scaler"].fit(states)
        
        self.update([experiences[0]], gamma=0)
        self.initialized = True
        
        print("Initialization done")
        
    def estimate(self, s):
        estimated = self.model.predict(s)[0]
        return estimated
    
    def _predict(self, states):
        if self.initialized:
            predicts = self.model.predict(states)
        else:
            size = len(self.actions) * len(states)
            predicts = np.random.uniform(size=size)
            predicts = predicts.reshape((-1, len(self.actions)))
        
        return predicts
    
    def update(self, experiences, gamma):
        states = np.vstack([e.s for e in experiences])
        next_states = np.vstack([e.n_s for e in experiences])
        
        estimates = self._predict(states)
        future = self._predict(next_states)
        
        for i, e in enumerate(experiences):
            reward = e.r
            
            if not e.d:
                reward += gamma * np.max(future[i])
            
            estimates[i][e.a] = reward
            
        estimates = np.array(estimates)
        states = self.model.named_steps["scaler"].transform(states)
        self.model.named_steps["estimator"].partial_fit(states, estimates)
        
class CartPoleObserver(Observer):
    def transform(self, state):
        return np.array(state).reshape((1, -1))
    
class ValueFunctionTrainer(Trainer):
    def train(self, env, episode_count=420, epsilon=0.1, initial_count=-1,
              render=False):
        actions = list(range(env.action_space.n))
        agent = ValueFunctionAgent(epsilon, actions)
        self.train_loop(env, agent, episode_count, initial_count, render)
        
        return agent
    
    def begin_train(self, episode, agent):
        agent.initialize(self.experiences)
        
    def step(self, episode, step_count, agent, experience):
        if self.training:
            batch = random.sample(self.experiences, self.batch_size)
            agent.update(batch, self.gamma)
            
    def episode_end(self, episode, step_count, agent):
        rewards = [e.r for e in self.get_recent(step_count)]
        self.reward_log.append(sum(rewards))
        
        if self.is_event(episode, self.report_interval):
            recent_rewards = self.reward_log[-self.report_interval:]
            self.logger.describe("reward", recent_rewards, episode=episode)
            
def main(play):
    env = CartPoleObserver(gym.make("CartPole-v0"))
    trainer = ValueFunctionTrainer()
    path = trainer.logger.path_of("value-function-agent.pkl")
    
    if play:
        agent = ValueFunctionAgent.load(env, path)
        agent.play(env)
    else:
        trained = trainer.train(env)
        trainer.logger.plot("Rewards", trainer.reward_log, trainer.report_interval)
        trained.save(path)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Value function agent")
    parser.add_argument("--play", action="store_true", help="play with trained model")
    args = parser.parse_args()
    main(args.play)
    