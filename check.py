from stable_baselines3.common.env_checker import check_env
from missile_chase_target_RL import MissileEnv

env = MissileEnv()
check_env(env)
print("OK !")