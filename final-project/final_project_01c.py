###############################################################################
# File Name: final_project_01c.py
#
# Description: This program loads a trained SAC policy for the Furuta pendulum 
# and runs it in real time, rendering the rotary arm and pendulum while logging 
# observations and rewards. It enables visualization and evaluation of the 
# learned control policy over a fixed number of steps.
#
# Record of Revisions (Date | Author | Change):
# 12/04/2025 | Rhys DeLoach | Initial creation
###############################################################################

# Import libraries
import time
from final_project_01a import furutaEnv, furutaParams
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import SAC

params = furutaParams() # Initialize environment params
env = furutaEnv(params, dt=0.01, render_mode="human") # Initialize environment

model = SAC.load("output/checkpoints/furutaFinal.zip") # Load trained SAC policy

obs, _ = env.reset() # Reset the environment

for step in range(500): # Loop for 500 steps
    action, _ = model.predict(obs, deterministic=True) # Predict action for this step
    
    obs, reward, terminated, truncated, info = env.step(action) # Take step

    env.render() # Render Step
    
    print(f"Step {step}: obs={obs}, reward={reward}") # Logging

    if terminated or truncated:
        obs, _ = env.reset()

    time.sleep(env.dt)

env.close()
