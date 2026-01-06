###############################################################################
# File Name: assignment_06b.py
#
# Description: This script loads a trained DQN model and runs it in inference 
# mode on the LunarLander-v3 environment to visually evaluate performance. 
# The agent selects actions greedily from the learned Q-values and reports how 
# many steps each episode lasts.
#
# Record of Revisions (Date | Author | Change):
# 09/29/2025 | Rhys DeLoach | Initial creation
###############################################################################

# Import Libraries
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.serialization

device = 'mps' if torch.mps.is_available() else 'cpu'  # Set device

# Define the same model architecture as used during training
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lin1 = nn.Linear(8, 64)
        self.lin2 = nn.Linear(64, 64)
        self.lin3 = nn.Linear(64, 4)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return self.lin3(x)

# Allow this class to be safely unpickled
torch.serialization.add_safe_globals([Model])

# Load model safely
model_500 = torch.load('output/DQN1474.pth', map_location=device, weights_only=False)
model_500.to(device)
model_500.eval()

# Test environment
env = gym.make('LunarLander-v3', render_mode='human')

for episode in range(5):
    state = env.reset()[0]
    state = torch.from_numpy(state).float().to(device)
    done = False
    step = 0

    while not done:
        with torch.no_grad():
            action = torch.argmax(model_500(state))
        next_state, reward, done, info, _ = env.step(action.item())
        state = torch.from_numpy(next_state).float().to(device)
        step += 1

    print(f"Episode {episode} finished in {step} steps")

env.close()
