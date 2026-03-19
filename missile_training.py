from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from missile_chase_target_RL import MissileEnv

env = MissileEnv()
check_env(env)

checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./checkpoints/",
    name_prefix="missile_ppo"
)


model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4) #learning_rate=3e-4 -> the default for PPO
model.learn(total_timesteps=1_000_000)
model.save("missile_ppo")
print("Training done")