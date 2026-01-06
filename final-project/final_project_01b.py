###############################################################################
# File Name: final_project_01b.py
#
# Description: This program trains a Soft Actor-Critic (SAC) agent on the 
# Furuta pendulum environment, using vectorized Stable-Baselines3 wrappers 
# with logging and checkpointing. It enables reinforcement learning on the 
# nonlinear pendulum dynamics while optionally monitoring and saving intermediate 
# models for analysis.
#
# Record of Revisions (Date | Author | Change):
# 12/04/2025 | Rhys DeLoach | Initial creation
###############################################################################

# Import Libraries
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from final_project_01a import furutaEnv, furutaParams

def makeEnv(): # Initialize environment
    params = furutaParams() 
    env = furutaEnv(params)
    env = Monitor(env)
    return env

def main():
    os.makedirs("output/checkpoints", exist_ok=True) # Checkpoint Directory
    os.makedirs("output/logs", exist_ok=True) # Log Directory

    env = DummyVecEnv([makeEnv]) # Vectorize env

    checkpointCallback = CheckpointCallback(
        save_freq=250000,
        save_path="output/checkpoints",
        name_prefix="furuta",
    )

    model = SAC( # Initialize model
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=4,
        gradient_steps=4,
        ent_coef=0.02,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log="output/logs/",
    )

    model.learn( # Train model
        total_timesteps=10000000,
        callback=checkpointCallback,
        log_interval=10,
    )

    model.save("output/checkpoints/furutaFinal") # Save final checkpoint

if __name__ == "__main__":
    main()