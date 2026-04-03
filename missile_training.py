from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from wandb.integration.sb3 import WandbCallback
import wandb
from missile_chase_target_RL import MissileEnv

wandb.init(
    project="missile_guidance",
    name="v4_perturbations",
    config={
        "total_timesteps": 600_000,
        "ent_coef": 0.01,
        "learning_rate": 3e-4,
        "checkpoint_start": "1801472"
    }
)

env = MissileEnv()
model = PPO.load("checkpoints_v4/missile_ppo_v4_4401472_steps.zip", env=env, device='cpu')
model.ent_coef = 0.01

checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./checkpoints_v4/",
    name_prefix="missile_ppo_v4"
)

model.learn(
    total_timesteps=600_000,
    callback=[checkpoint_callback, WandbCallback()],
    reset_num_timesteps=False
)

model.save("missile_ppo_v4")
wandb.finish()
print("Training v4 done")