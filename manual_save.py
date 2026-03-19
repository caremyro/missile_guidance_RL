from stable_baselines3 import PPO
from missile_chase_target_RL import MissileEnv

model = PPO.load("checkpoints_v3/missile_ppo_v3_3001472_steps", env=MissileEnv())
model.save("missile_ppo_v3")
print("Sauvegardé !")