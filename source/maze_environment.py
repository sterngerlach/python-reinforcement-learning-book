# coding: utf-8
# maze_environment.py

import numpy as np

from enum import Enum

class State(object):
    def __init__(self, row=-1, column=-1):
        self.row = row
        self.column = column
        
    def __repr__(self):
        return "<State: [{}, {}]>".format(self.row, self.column)
    
    def clone(self):
        return State(self.row, self.column)
    
    def __hash__(self):
        return hash((self.row, self.column))
    
    def __eq__(self, other):
        return self.row == other.row and self.column == other.column
    
class Action(Enum):
    UP = 1
    DOWN = -1
    LEFT = 2
    RIGHT = -2
    
class CellType(Enum):
    NORMAL = 0
    DAMAGE = -1
    REWARD = 1
    BLOCK = 9
    
class Environment(object):
    def __init__(self, grid, move_prob=0.8):
        self.grid = grid
        self.agent_state = State()
        self.default_reward = -0.04
        self.move_prob = move_prob
        
        self.reset()
        
    @property
    def row_length(self):
        return len(self.grid)
    
    @property
    def column_length(self):
        return len(self.grid[0])
    
    @property
    def actions(self):
        return [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
    
    @property
    def states(self):
        states = []
        
        for row in range(self.row_length):
            for column in range(self.column_length):
                if self.grid[row][column] != CellType.BLOCK:
                    states.append(State(row, column))
        
        return states
    
    """全ての状態s'についての, 遷移確率P(s' | s, a)の計算"""
    def calc_transition_probs(self, state, action):
        transition_probs = {}
        
        if not self.can_action_at(state):
            return transition_probs
        
        opposite_action = Action(action.value * -1)
        
        for a in self.actions:
            prob = self.move_prob if a == action else \
                   (1.0 - self.move_prob) / 2.0 if a != opposite_action else 0.0
            next_state = self.move(state, a)
            
            if next_state not in transition_probs:
                transition_probs[next_state] = prob
            else:
                transition_probs[next_state] += prob
        
        return transition_probs
        
    def can_action_at(self, state):
        return self.grid[state.row][state.column] == CellType.NORMAL
    
    def move(self, state, action):
        if not self.can_action_at(state):
            raise Exception("Cannot move from current location")
        
        next_state = state.clone()
        
        if action == Action.UP:
            next_state.row -= 1
        elif action == Action.DOWN:
            next_state.row += 1
        elif action == Action.LEFT:
            next_state.column -= 1
        elif action == Action.RIGHT:
            next_state.column += 1
        
        if not (0 <= next_state.row < self.row_length):
            next_state = state
        if not (0 <= next_state.column < self.column_length):
            next_state = state
        if self.grid[next_state.row][next_state.column] == CellType.BLOCK:
            next_state = state
        
        return next_state
    
    def reward(self, next_state):
        reward = self.default_reward
        done = False
        
        cell_type = self.grid[next_state.row][next_state.column]
        
        if cell_type == CellType.REWARD:
            reward = 1.0
            done = True
        elif cell_type == CellType.DAMAGE:
            reward = -1.0
            done = True
        
        return reward, done
    
    def reset(self):
        self.agent_state = State(self.row_length - 1, 0)
        return self.agent_state
    
    def step(self, action):
        next_state, reward, done = self.transition(self.agent_state, action)
        
        if next_state is not None:
            self.agent_state = next_state
        
        return next_state, reward, done
    
    def transition(self, state, action):
        transition_probs = self.calc_transition_probs(state, action)
        
        if len(transition_probs) == 0:
            return None, None, True
        
        next_states = []
        probs = []
        
        for s in transition_probs:
            next_states.append(s)
            probs.append(transition_probs[s])
        
        next_state = np.random.choice(next_states, p=probs)
        reward, done = self.reward(next_state)
        
        return next_state, reward, done
    