###############################################################################
# File Name: assignment_06a.py
#
# Description: This program implements a Deep Q-Network (DQN) in PyTorch to 
# train an agent on the LunarLander-v3 environment using experience replay, a 
# target network, and an epsilon-greedy exploration strategy.
#
# Record of Revisions (Date | Author | Change):
# 09/29/2025 | Rhys DeLoach | Initial creation
###############################################################################

# Import Libraries
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import random
import copy
import numpy as np
import time

torch.manual_seed(42) # Set seed

device = 'mps' if torch.mps.is_available() else 'cpu' # Set device

env = gym.make('LunarLander-v3')

# Model Architecture
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.lin1 = nn.Linear(8,64)
        self.lin2 = nn.Linear(64,64)
        self.lin3 = nn.Linear(64,4)   
        
    
    def forward(self,x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
    
        return x
        
# Create Q and Target NN
DQNet = Model().to(device)
targetNet = copy.deepcopy(DQNet).to(device)

targetNet.eval() # No back prop

# Network Parameters
lr = 0.001 
optimizer = Adam(params=DQNet.parameters(), lr=lr)

eps_start = 1 # E-Greedy policy
eps_end = 0.0001
eps_decay = 0.9995

loss_fn = nn.MSELoss() # Loss function
episodes = 1500
mini_batch_size = 128

loss_value = 0
loss_history = []
update_freq = 15
gamma = 0.99
scores = []
rewards = 0

def training(Qnet, t_net, replay_memory, optimizer, loss_fn, mini_batch_size=32):
    batch = random.sample(replay_memory, mini_batch_size)
    
    states = torch.tensor([b[0] for b in batch], dtype=torch.float32, device=device)
    actions = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
    rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)
    next_states = torch.tensor([b[3] for b in batch], dtype=torch.float32, device=device)
    dones = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=device)

    # Current Q values
    q_values = Qnet(states)
    expected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Target Q values
    with torch.no_grad():
        next_q_values = t_net(next_states)
        max_next_q = next_q_values.max(1)[0]
        target = rewards + gamma * max_next_q * (1 - dones)

    # Compute loss
    loss = loss_fn(expected, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return Qnet, loss.item()

# Q Explore
eps = eps_start
replay = []

for i in range(episodes):
    start = time.time()
    s = env.reset() # Reset to start state
    s = torch.from_numpy(s[0]) # Convert state to tensor a
    done = False # Episode end flag
    rewards = 0 # Container for rewards accumulation
    t_step = 0

    while not done:
        DQNet.eval()

        s = s.to(device)

        q_values = DQNet(s)

        # Action
        if np.random.random() < eps:
            a = env.action_space.sample()
            new_state, reward, done, _, _ = env.step(a.item())
        else:
            a = torch.argmax(q_values)
            new_state, reward, done, _, _ = env.step(a.item())
        
        new_experience = s.tolist(), a, reward, new_state.tolist(), done # Gather new experience
        replay.append(new_experience) # Append to replay buffer

        if len(replay) > 100000: # Limit replay buffer to 100000
            replay.pop(0)
        
        rewards += reward # Accumulate rewards

        s = torch.from_numpy(new_state)

        if i % update_freq == 0: # Swap weights
            targetNet = copy.deepcopy(DQNet)
            targetNet.eval()

        t_step += 1 # Kill episode after more than 500 time steps
        if t_step > 500:
            break

        if len(replay) > 5000: # Train DQNet
            DQNet,loss = training(Qnet=DQNet,t_net=targetNet, replay_memory=replay, optimizer=optimizer, loss_fn=loss_fn, mini_batch_size=mini_batch_size)
            loss_history.append(loss)   # Save the loss value
            eps = max(eps*eps_decay,eps_end) # Decrease the epsilon

    scores.append(rewards)
    if rewards >= 200 and i > 200:
        torch.save(DQNet,'output/DQN'+str(i)+'pth')
        print(i)

    print(f"Episode {i}/{episodes} Rewards:{rewards} Buffer Length:{len(replay)} eps:{eps} Time:{time.time()-start}")
        

