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

    def _update_aircraft(self):
        dx0 = np.cos(self.cap_initial)
        dy0 = np.sin(self.cap_initial)

        if 0 <= self.t <= self.Straight_time:
            x = self.aircraft_start_loc[0] + self.targ_vel * dx0 * self.t
            y = self.aircraft_start_loc[1] + self.targ_vel * dy0 * self.t
            z = self.aircraft_start_loc[2]

        elif self.Straight_time < self.t <= self.Straight_time + self.curve_time:
            tc = self.t - self.Straight_time
            if not self.curve_initialized:
                self.curve_start_x = self.aircraft_pos[0]
                self.curve_start_y = self.aircraft_pos[1]
                self.curve_start_z = self.aircraft_pos[2]
                self.curve_initialized = True
                sign = np.sign(self.turn_angle)
                self.center_x = self.curve_start_x + abs(self.radius) * (-sign * dy0)
                self.center_y = self.curve_start_y + abs(self.radius) * ( sign * dx0)
                self.center_z = self.curve_start_z
                self.curve_initialized = True
            frac      = tc / self.curve_time
            angle_now = frac * self.turn_angle
            vx0 =  np.sign(self.turn_angle) * dy0
            vy0 = -np.sign(self.turn_angle) * dx0
            cos_a, sin_a = np.cos(angle_now), np.sin(angle_now)
            vx = vx0 * cos_a - vy0 * sin_a
            vy = vx0 * sin_a + vy0 * cos_a

            x = self.center_x + abs(self.radius) * vx
            y = self.center_y + abs(self.radius) * vy
            z = self.curve_start_z \
                + np.sin(self.yz_angle + np.pi/2) * self.targ_vel**2 \
                * (1 - np.cos(np.pi * tc / self.curve_time)) * self.climb_rate_curve

        elif self.Straight_time + self.curve_time < self.t <= self.Straight_time + self.curve_time + self.Straight_time2:
            if not self.straight2_initialized:
                self.straight2_start_x = self.aircraft_pos[0]
                self.straight2_start_y = self.aircraft_pos[1]
                self.straight2_start_z = self.aircraft_pos[2]
                self.straight2_initialized = True

            ts        = self.t - (self.Straight_time + self.curve_time)
            cap_final = self.cap_initial + self.turn_angle
            x = self.straight2_start_x + self.targ_vel * ts * np.cos(cap_final)
            y = self.straight2_start_y + self.targ_vel * ts * np.sin(cap_final)
            z = self.straight2_start_z

        else:
            if self.straight2_initialized:
                cap_final = self.cap_initial + self.turn_angle
                x = self.straight2_start_x + self.targ_vel * self.Straight_time2 * np.cos(cap_final)
                y = self.straight2_start_y + self.targ_vel * self.Straight_time2 * np.sin(cap_final)
                z = self.straight2_start_z
            else:
                x = self.aircraft_start_loc[0] + self.targ_vel * dx0 * self.Straight_time
                y = self.aircraft_start_loc[1] + self.targ_vel * dy0 * self.Straight_time
                z = self.aircraft_start_loc[2]
            
        if self.t > 0:
            self.aircraft_vel = (np.array([x, y, z]) - self.aircraft_pos) / self.dt

        if np.random.random() < 0.02:  # 2% chance of noise
            perturbation = np.random.uniform(-150, 150, 3)
            perturbation[2] *= 0.3 # less vertical noise
            self.aircraft_vel += perturbation
            self.aircraft_vel = self.aircraft_vel / np.linalg.norm(self.aircraft_vel) * self.targ_vel

        return np.array([x, y, z])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        altitude_avion = rng.uniform(8000, 15000)
        cap_initial = rng.uniform(0, 2 * np.pi)
        self.cap_initial = cap_initial
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

        self.curve_initialized    = False
        self.straight2_initialized = False
        self.curve_start          = None
        self.straight2_start      = None
        self.center               = None
        self.radius               = (self.targ_vel * self.curve_time) / self.turn_angle

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

        self.aircraft_pos = self._update_aircraft()
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