import gymnasium as gym 
import numpy as np

class MissileEnv(gym.Env):

    def __init__(self):
        self.tmax = 90
        self.dt = 0.1
        self.max_accel = 500.0
        self.miss_vel = 1200
        self.targ_vel = 750.0
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)

    def _get_obs(self):
        los_current = self.aircraft_pos - self.missile_pos
        distance = np.linalg.norm(los_current)
        ulos = los_current / distance
        v_approach = self.missile_vel - self.aircraft_vel
        Vc = np.dot(v_approach, ulos)
        omega_los = np.cross(ulos, v_approach) / distance

        obs = np.concatenate([
            ulos,
            self.missile_vel  / self.miss_vel,
            [Vc / self.miss_vel],
            omega_los,
            [distance / 50000.0]
        ])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        altitude_avion = rng.uniform(8000, 15000)
        cap_initial = rng.uniform(0, 2 * np.pi)

        self.aircraft_start_loc = np.array([
            rng.uniform(-20000, 20000),
            rng.uniform(-20000, 20000),
            altitude_avion
        ])
        self.Straight_time = rng.uniform(15, 30)
        self.curve_time = rng.uniform(15, 25)
        self.Straight_time2 = rng.uniform(15, 30)
        self.turn_angle = rng.choice([-1, 1]) * rng.uniform(np.pi/3, np.pi)
        self.yz_angle = rng.uniform(-np.pi/8, np.pi/8)
        self.climb_rate_curve = rng.uniform(-0.002, 0.002)

        self.missile_start_loc = np.array([
            rng.uniform(-30000, 30000),
            rng.uniform(-30000, 30000),
            0.0
        ])

        self.missile_launch_time = rng.uniform(3, 12)

        self.aircraft_vel = np.array([
            np.cos(cap_initial) * self.targ_vel,
            np.sin(cap_initial) * self.targ_vel,
            0.0
        ])

        self.aircraft_pos = self.aircraft_start_loc.copy()
        self.missile_pos  = self.missile_start_loc.copy()

        los0 = self.aircraft_pos - self.missile_pos
        self.missile_vel = los0 / np.linalg.norm(los0) * self.miss_vel
        self.prev_distance = np.linalg.norm(self.aircraft_pos - self.missile_pos)
        self.t = 0.0

        return self._get_obs(), {}

    def step(self, action):
        accel_cmd = action * self.max_accel
        self.missile_vel += accel_cmd * self.dt
        spd = np.linalg.norm(self.missile_vel)
        self.missile_vel = self.missile_vel / spd * min(spd, self.miss_vel)
        self.missile_pos += self.missile_vel * self.dt # new pos = old pos + vel * dt

        if self.missile_pos[2] < 0:
            self.missile_pos[2] = 0
            self.missile_vel[2] = 0

        self.aircraft_pos += self.aircraft_vel * self.dt
        self.t += self.dt

        done = False 
        distance = np.linalg.norm(self.aircraft_pos - self.missile_pos)
        reward = 0.0
        reward += (self.prev_distance - distance) * 0.1

        if distance < 100.0:
            reward += 1000.0
            done = True
        elif self.t > self.tmax:
            reward -= 100.0
            done = True
        reward -= 0.01 * np.linalg.norm(accel_cmd)
        self.prev_distance = distance

        return self._get_obs(), reward, done, False, {}

    def render(self):
        pass