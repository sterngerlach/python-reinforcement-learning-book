# coding: utf-8
# framework.py

import io
import sys
import os
import re

from collections import deque, namedtuple
from PIL import Image

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.python import keras as K

Experience = namedtuple("Experience", ["s", "a", "r", "n_s", "d"])

class FNAgent(object):
    def __init__(self, epsilon, actions):
        self.epsilon = epsilon
        self.actions = actions
        self.model = None
        self.estimate_probs = False
        self.initialized = False
        
    def save(self, model_path):
        self.model.save(model_path, overwrite=True, include_optimizer=False)
        
    @classmethod
    def load(cls, env, model_path, epsilon=1e-04):
        actions = list(range(env.action_space.n))
        
        agent = cls(epsilon, actions)
        agent.model = K.models.load_model(model_path)
        agent.initialized = True
        
        return agent
    
    def initialize(self, experiences):
        raise NotImplementedError()
        
    def estimate(self, s):
        raise NotImplementedError()
        
    def update(self, experiences, gamma):
        raise NotImplementedError()
        
    def policy(self, s):
        if np.random.random() < self.epsilon or not self.initialized:
            return np.random.randint(len(self.actions))
        else:
            estimates = self.estimate(s)
            
            if self.estimate_probs:
                action = np.random.choice(self.actions, size=1, p=estimates)[0]
                return action
            else:
                return np.argmax(estimates)
            
    def play(self, env, episode_count=5, render=True):
        for e in range(episode_count):
            s = env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                if render:
                    env.render()
                
                a = self.policy(s)
                next_state, reward, done, info = env.step(a)
                episode_reward += reward
                s = next_state
            else:
                print("Got reward {}".format(episode_reward))
                
class Trainer(object):
    def __init__(self, buffer_size=1024, batch_size=32,
                 gamma=0.9, report_interval=10, log_dir=""):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.report_interval = report_interval
        self.logger = Logger(log_dir, self.trainer_name)
        self.experiences = deque(maxlen=buffer_size)
        self.training = False
        self.training_count = 0
        self.reward_log = []
        
    @property
    def trainer_name(self):
        class_name = self.__class__.__name__
        snaked = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", class_name)
        snaked = re.sub("([a-z][0-9])([A-Z])", r"\1_\2", snaked).lower()
        snaked = snaked.replace("_trainer", "")
        return snaked
    
    def train_loop(self, env, agent, episode=200, initial_count=-1,
                   render=False, observe_interval=0):
        self.experiences = deque(maxlen=self.buffer_size)
        self.training = False
        self.training_count = 0
        self.reward_log = []
        
        frames = []
        
        for i in range(episode):
            s = env.reset()
            done = False
            step_count = 0
            self.episode_begin(i, agent)
            
            while not done:
                if render:
                    env.render()
                
                if self.training and observe_interval > 0 and \
                    (self.training_count == 1 or \
                     self.training_count % observe_interval == 0):
                    frames.append(s)
                    
                a = agent.policy(s)
                next_state, reward, done, info = env.step(a)
                e = Experience(s, a, reward, next_state, done)
                self.experiences.append(e)
                
                if not self.training and \
                    len(self.experiences) == self.buffer_size:
                    self.begin_train(i, agent)
                    self.training = True
                
                self.step(i, step_count, agent, e)
                
                s = next_state
                step_count += 1
            else:
                self.episode_end(i, step_count, agent)
                
                if not self.training and \
                    initial_count > 0 and i >= initial_count:
                    self.begin_train(i, agent)
                    self.training = True
                
                if self.training:
                    if len(frames) > 0:
                        self.logger.write_image(self.training_count, frames)
                        frames = []
                    
                    self.training_count += 1
                    
    def episode_begin(self, episode, agent):
        pass
    
    def begin_train(self, episode, agent):
        pass
    
    def step(self, episode, step_count, agent, experience):
        pass
    
    def episode_end(self, episode, step_count, agent):
        pass
    
    def is_event(self, count, interval):
        return count != 0 and count % interval == 0
    
    def get_recent(self, count):
        recent = range(len(self.experiences) - count, len(self.experiences))
        return [self.experiences[i] for i in recent]
    
class Observer(object):
    def __init__(self, env):
        self._env = env
        
    @property
    def action_space(self):
        return self._env.action_space
    
    @property
    def observation_space(self):
        return self._env.observation_space
    
    def reset(self):
        return self.transform(self._env.reset())
    
    def render(self):
        self._env.render()
        
    def step(self, action):
        next_state, reward, done, info = self._env.step(action)
        return self.transform(next_state), reward, done, info
    
    def transform(self, state):
        raise NotImplementedError()
        
class Logger(object):
    def __init__(self, log_dir="", dir_name=""):
        self.log_dir = log_dir
        
        if not log_dir:
            self.log_dir = os.path.join(os.path.dirname(__file__), "logs")
            
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
            
        if dir_name:
            self.log_dir = os.path.join(self.log_dir, dir_name)
            
            if not os.path.exists(self.log_dir):
                os.mkdir(self.log_dir)
            
        self._callback = K.callbacks.TensorBoard(self.log_dir)
        
    @property
    def writer(self):
        return self._callback.writer
    
    def set_model(self, model):
        self._callback.set_model(model)
        
    def path_of(self, file_name):
        return os.path.join(self.log_dir, file_name)
    
    def describe(self, name, values, episode=-1, step=-1):
        mean = np.round(np.mean(values), 3)
        std = np.round(np.std(values), 3)
        desc = "{} is {} (std: {})".format(name, mean, std)
        
        if episode > 0:
            print("At episode {}, {}".format(episode, desc))
        elif step > 0:
            print("At step {}, {}".format(step, desc))
        
    def plot(self, name, values, interval=10):
        indices = list(range(0, len(values), interval))
        means = []
        stds = []
        
        for i in indices:
            _values = values[i:(i + interval)]
            means.append(np.mean(_values))
            stds.append(np.std(_values))
            
        means = np.array(means)
        stds = np.array(stds)
        
        plt.figure()
        plt.title("{} History".format(name))
        plt.grid()
        plt.fill_between(indices, means - stds, means + stds,
                         alpha=0.1, color="g")
        plt.plot(indices, means, "o-", color="g",
                 label="{} per {} episode".format(name.lower(), interval))
        plt.legend(loc="best")
        
        plt.show()
        
    def write(self, index, name, value):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.tag = name
        summary_value.simple_value = value
        
        self.writer.add_summary(summary, index)
        self.writer.flush()
        
    def write_image(self, index, frames):
        last_frames = [f[:, :, -1] for f in frames]
        
        if np.min(last_frames[-1]) < 0:
            scale = 127 / np.abs(last_frames[-1]).max()
            offset = 128
        else:
            scale = 255 / np.max(last_frames[-1])
            offset = 0
        
        channel = 1
        tag = "frames_at_training_{}".format(index)
        values = []
        
        for f in last_frames:
            height, width = f.shape
            array = np.asarray(f * scale + offset, dtype=np.uint8)
            image = Image.fromarray(array)
            output = io.BytesIO()
            image.save(output, format="PNG")
            image_string = output.getvalue()
            output.close()
            image = tf.Summary.Image(height=height, width=width,
                                     colorspace=channel,
                                     encoded_image_string=image_string)
            value = tf.Summary.Value(tag=tag, image=image)
            values.append(value)
            
        summary = tf.Summary(value=values)
        self.writer.add_summary(summary, index)
        self.writer.flush()
        