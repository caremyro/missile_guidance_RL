from stable_baselines3 import PPO
from missile_chase_target_RL import MissileEnv
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


model = PPO.load("missile_ppo")
env   = MissileEnv()

missile_positions = []
aircraft_positions = []

obs, _ = env.reset()
done   = False

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    missile_positions.append(env.missile_pos.copy())
    aircraft_positions.append(env.aircraft_pos.copy())
missile_positions = np.array(missile_positions)
aircraft_positions = np.array(aircraft_positions)

fig = plt.figure(figsize=(14, 10))
ax  = fig.add_subplot(111, projection='3d')

all_points  = np.vstack([aircraft_positions, missile_positions])
x_center    = (np.max(all_points[:,0]) + np.min(all_points[:,0])) / 2
y_center    = (np.max(all_points[:,1]) + np.min(all_points[:,1])) / 2
z_center    = (np.max(all_points[:,2]) + np.min(all_points[:,2])) / 2
plot_radius = max(np.ptp(all_points[:,0]), np.ptp(all_points[:,1]), np.ptp(all_points[:,2])) / 2 * 1.1

ax.set_xlim(x_center - plot_radius, x_center + plot_radius)
ax.set_ylim(y_center - plot_radius, y_center + plot_radius)
ax.set_zlim(z_center - plot_radius, z_center + plot_radius)
ax.set_box_aspect([1, 1, 1])
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D Missile-Aircraft Simulation — RL Guidance', fontsize=16)
ax.grid(True)
ax.view_init(elev=20, azim=45)

target_point,  = ax.plot([], [], [], 'bo', markersize=10, label='Cible')
target_trail,  = ax.plot([], [], [], 'b-', linewidth=2,   alpha=0.5, label='Trace cible')
missile_point, = ax.plot([], [], [], 'ro', markersize=8,  label='Missile')
missile_trail, = ax.plot([], [], [], 'r-', linewidth=1.5, alpha=0.5, label='Trace missile')
time_text     = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
speed_text    = ax.text2D(0.02, 0.90, '', transform=ax.transAxes, fontsize=10)
distance_text = ax.text2D(0.02, 0.85, '', transform=ax.transAxes, fontsize=10)

ax.scatter(*aircraft_positions[0],  c='green',  s=100, marker='s', label='Départ cible')
ax.scatter(*missile_positions[0], c='orange', s=100, marker='^', label='Départ missile')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx, yy = np.meshgrid(
    np.linspace(xlim[0], xlim[1], 3),
    np.linspace(ylim[0], ylim[1], 3)
)
ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.08, color='green')
ax.plot([missile_positions[0][0], missile_positions[0][0]],
        [missile_positions[0][1], missile_positions[0][1]],
        [0, 0], 'o', color='orange', markersize=8)

ax.set_zlim(0, z_center + plot_radius)

ax.legend()

n_points           = len(missile_positions)
animation_interval = 20
dt                 = env.dt
targ_vel           = env.targ_vel


def init():
    for artist in [target_point, target_trail, missile_point, missile_trail]:
        artist.set_data([], [])
        artist.set_3d_properties([])
    time_text.set_text('')
    speed_text.set_text('')
    distance_text.set_text('')
    return target_point, target_trail, missile_point, missile_trail, time_text, speed_text, distance_text

def update(frame):
    target_point.set_data([aircraft_positions[frame,0]], [aircraft_positions[frame,1]])
    target_point.set_3d_properties([aircraft_positions[frame,2]])
    target_trail.set_data(aircraft_positions[:frame+1,0], aircraft_positions[:frame+1,1])
    target_trail.set_3d_properties(aircraft_positions[:frame+1,2])

    missile_point.set_data([missile_positions[frame,0]], [missile_positions[frame,1]])
    missile_point.set_3d_properties([missile_positions[frame,2]])
    missile_trail.set_data(missile_positions[:frame+1,0], missile_positions[:frame+1,1])
    missile_trail.set_3d_properties(missile_positions[:frame+1,2])

    if frame > 0:
        speed = np.linalg.norm(aircraft_positions[frame] - aircraft_positions[frame-1]) / dt
    else:
        speed = targ_vel

    if frame == len(missile_positions) - 1:
        missile_point.set_data([], [])
        missile_point.set_3d_properties([])
        ax.scatter(*missile_positions[-1], c='yellow', s=500, 
                   marker='*', zorder=10, label='Impact')
        distance_text.set_text('IMPACT !')

    distance = np.linalg.norm(aircraft_positions[frame] - missile_positions[frame])
    time_text.set_text(f'Time     = {frame * env.dt:.2f} s')
    speed_text.set_text(f'Cible    = {speed:.1f} m/s')
    distance_text.set_text(f'Distance = {distance:.1f} m')

    all_current = np.vstack([
        aircraft_positions[:frame+1],
        missile_positions[:frame+1]
    ])
    
    margin = 5000 #m
    ax.set_xlim(np.min(all_current[:,0]) - margin, np.max(all_current[:,0]) + margin)
    ax.set_ylim(np.min(all_current[:,1]) - margin, np.max(all_current[:,1]) + margin)
    ax.set_zlim(0, np.max(all_current[:,2]) + margin)

    return target_point, target_trail, missile_point, missile_trail, time_text, speed_text, distance_text

frame_skip = max(1, n_points // 500)
frames     = range(0, n_points, frame_skip) if n_points > 0 else range(1)
print(f"n_points = {n_points}")
print(f"Durée épisode = {n_points * env.dt:.1f}s")
print(f"Distance finale = {np.linalg.norm(aircraft_positions[-1] - missile_positions[-1]):.1f}m")
print(f"frame_skip = {frame_skip}")
print(f"Animation : {len(list(frames))} frames")

anim = FuncAnimation(fig, update, frames=frames, init_func=init,
                     blit=False, interval=animation_interval, repeat=True)
plt.show()