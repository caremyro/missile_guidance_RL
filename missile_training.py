from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from wandb.integration.sb3 import WandbCallback
import wandb
from missile_chase_target_RL import MissileEnv

wandb.init(
    project="missile_guidance",
    name="v3_from_1.8M",
    config={
        "total_timesteps": 3_000_000,
        "ent_coef": 0.01,
        "learning_rate": 3e-4,
        "checkpoint_start": "1801472"
    }
)

env = MissileEnv()
model = PPO.load("checkpoints/missile_ppo_v2_1801472_steps", env=env, tensorboard_log="./logs/")
model.ent_coef = 0.01

checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./checkpoints_v3/",
    name_prefix="missile_ppo_v3"
)

model.learn(
    total_timesteps=3_000_000,
    callback=[checkpoint_callback, WandbCallback()],
    reset_num_timesteps=False,
    tb_log_name="missile_ppo_v3"
)

model.save("missile_ppo_v3")
wandb.finish()
print("Traning v3 done")