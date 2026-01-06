###############################################################################
# File Name: mini_project_02a.py
#
# Description: This program defines a custom inventory management environment 
# for reinforcement learning using OpenAI Gymnasium, where an agent must manage 
# raw materials and product inventory while considering costs, demand, and 
# revenue. It includes an observation normalization wrapper, a replay buffer 
# for experience storage, and a DQN neural network architecture suitable for 
# training a reinforcement learning agent. 
# 
# Note: This script was provided with the assignment and is not my original 
# work. This environment was created by Professor Haygriva Rao at the
# University of Georgia.
###############################################################################

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch.nn as nn
from collections import deque
import random

# -------------------------------
# Environment Class
# -------------------------------
class InventoryManagementEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 initial_cash=1000.0,
                 demand_mean=10.0,
                 demand_volatility=3.0,
                 raw_price_base=5.0,
                 product_price_base=20.0,
                 price_std=0.2,
                 holding_cost=0.1,
                 stockout_penalty=5.0,
                 invalid_action_penalty=2.0,
                 bankruptcy_penalty=100.0,
                 conversion_bonus=10.0,
                 max_steps=500):
        super(InventoryManagementEnv, self).__init__()
        self.action_space = spaces.Discrete(3)
        low = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.initial_cash = initial_cash
        self.demand_mean = demand_mean
        self.demand_volatility = demand_volatility
        self.raw_price_base = raw_price_base
        self.product_price_base = product_price_base
        self.price_std = price_std
        self.holding_cost = holding_cost
        self.stockout_penalty = stockout_penalty
        self.invalid_action_penalty = invalid_action_penalty
        self.bankruptcy_penalty = bankruptcy_penalty
        self.conversion_bonus = conversion_bonus
        self.max_steps = max_steps

        self.current_step = 0
        self.last_product_inventory_before_sale = 0
        self.last_sold_units = 0
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.raw_inventory = 0.0
        self.product_inventory = 0.0
        self.cash = self.initial_cash
        self.raw_price = max(0.1, np.random.normal(self.raw_price_base, self.price_std))
        self.product_price = max(0.1, np.random.normal(self.product_price_base, self.price_std))
        self.demand = max(0, np.random.normal(self.demand_mean, self.demand_volatility))
        self.current_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([self.raw_inventory,
                         self.product_inventory,
                         self.raw_price,
                         self.product_price,
                         self.demand,
                         self.cash], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        reward = 0.0
        invalid_action = False

        if action == 0:
            pass
        elif action == 1:
            if self.cash >= self.raw_price:
                self.cash -= self.raw_price
                self.raw_inventory += 1
            else:
                reward -= self.invalid_action_penalty
                invalid_action = True
        elif action == 2:
            if self.raw_inventory >= 2:
                self.raw_inventory -= 2
                self.product_inventory += 1
                reward += self.conversion_bonus
            else:
                reward -= self.invalid_action_penalty
                invalid_action = True
        else:
            reward -= self.invalid_action_penalty
            invalid_action = True

        self.demand = max(0, np.random.normal(self.demand_mean, self.demand_volatility))
        self.last_product_inventory_before_sale = self.product_inventory
        sold_units = min(self.product_inventory, self.demand)
        self.last_sold_units = sold_units
        revenue = sold_units * self.product_price
        self.product_inventory -= sold_units
        self.cash += revenue
        reward += revenue

        unmet_demand = self.demand - sold_units
        if unmet_demand > 0:
            reward -= self.stockout_penalty * unmet_demand
        
        holding_cost_total = self.holding_cost * (self.raw_inventory + self.product_inventory)
        reward -= holding_cost_total

        self.raw_price = max(0.1, self.raw_price + np.random.normal(0, self.price_std))
        self.product_price = max(0.1, self.product_price + np.random.normal(0, self.price_std))

        terminated = self.cash <= 0
        truncated = self.current_step >= self.max_steps
        if terminated:
            reward -= self.bankruptcy_penalty

        info = {
            'sold_units': sold_units,
            'unmet_demand': self.demand - sold_units,
            'revenue': revenue,
            'holding_cost': holding_cost_total,
            'invalid_action': invalid_action,
            'product_inventory_before_sale': self.last_product_inventory_before_sale,
            'product_inventory_after_sale': self.product_inventory
        }
        return self._get_obs(), reward, terminated, truncated, info

    def render(self, mode='human'):
        print(f"Step: {self.current_step}")
        print(f"Raw Inventory: {self.raw_inventory}")
        print(f"Product Inventory Before Sale: {self.last_product_inventory_before_sale}, After Sale: {self.product_inventory}")
        print(f"Raw Price: {self.raw_price:.2f}, Product Price: {self.product_price:.2f}")
        print(f"Demand: {self.demand:.2f}, Cash: {self.cash:.2f}")

# -------------------------------
# Normalization Wrapper
# -------------------------------
class NormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super(NormalizeObservation, self).__init__(env)
        self.scale_factors = np.array([10, 10, 10, 10, 20, 2000], dtype=np.float32)
        low = self.observation_space.low / self.scale_factors
        high = self.observation_space.high / self.scale_factors
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        return obs / self.scale_factors

# -------------------------------
# Replay Buffer
# -------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states),
                np.array(dones, dtype=np.float32))
    
    def __len__(self):
        return len(self.buffer)

# -------------------------------
# DQN Network
# -------------------------------
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)
