###############################################################################
# File Name: assignment_05.py
#
# Description: This program implements a simple Q-learning algorithm for a 
# one-dimensional environment with six states, iteratively updating Q-values 
# based on rewards and discounted future returns to learn an optimal policy.
#
# Record of Revisions (Date | Author | Change):
# 09/25/2025 | Rhys DeLoach | Initial creation
###############################################################################

# Import Libraries
import numpy as np
import random

# Initialize Variables
startState = 2 # Start Position (0-5)
episodeCount = 0 # Episode Count
terminalReward = 0 # Terminal Reward
q = np.zeros((6,2)) # Current Episode Q-values
q[0][0] = 100
q[5][1] = 40
qPrev = np.array([]) # Previous Episode Q-values
stateFuture = 0 # Future State 
reward = [100,0,0,0,0,40] # System Map
gamma = 0.9 # Gamma
alpha = 0.1 # Alpha

# while not np.array_equal(q, qPrev):
while episodeCount < 10000:
    episodeCount += 1
    qPrev = q.copy()
    stateCurrent = startState
    # stateCurrent = random.randint(1, 4) # Random start state
    firstStep = True
    while True:
        if firstStep:
            action = random.choice([-1, 1])
            firstStep = False
        else:
            if q[stateCurrent, 0] == q[stateCurrent, 1]:
                action = random.choice([-1, 1])
            else:
                if np.argmax(q[stateCurrent]) == 1:
                    action = 1
                else:
                    action = -1
        stateFuture = stateCurrent + action
        if action == -1:
            action = 0
        if reward[stateCurrent] == 0:
            q[stateCurrent, action] = q[stateCurrent, action] + alpha * (reward[stateCurrent] + gamma * max(q[stateFuture]) - q[stateCurrent, action])
            stateCurrent = stateFuture
        else:
            break

print(f'Final Q:\n {q}')

# Observation: Assignment instructions define no epsilon mechanism and static start state. However, the model never converges 
# because of no epsilon mechanism. Uncomment random start state for psuedo-exploration and convergence.