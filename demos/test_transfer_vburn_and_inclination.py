import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
from yeager_utils import (
    transfer_velocity_continuous,
    transfer_inclination_continuous,
    EARTH_RADIUS, RGEO, VGEO, figpath,
    quickint, get_times
)


if __name__ == "__main__":
    # Initial orbit (roughly circular GEO altitude)
    r0 = np.array([RGEO, 0.0, 0.0])
    v0 = np.array([0.0, VGEO, 0.0])
    a_thrust = 0.1  # Moderate thrust [m/s^2]

    print("=== Velocity Direction Burn ===")
    v_target = -900  # Positive for acceleration along velocity
    r1, v1, t1 = transfer_velocity_continuous(
        r0, v0, v_target=v_target, a_thrust=a_thrust,
        plot=False)
    print(f"After velocity burn: r = {r1[-1]}, v = {v1[-1]}, t = {t1[-1]:.1f} s\n")

    print("=== Inclination Change Burn ===")
    delta_v = -800  # Target inclination change
    r2, v2, t2 = transfer_inclination_continuous(
        r1[-1], v1[-1], delta_v=delta_v, a_thrust=a_thrust,
        plot=False, save_path=figpath("tests/demo_combined_inclination_burn.jpg")
    )
    print(f"After inclination burn: r = {r2[-1]}, v = {v2[-1]}, t = {t2[-1]:.1f} s\n")

    # Total mission time and final state
    total_time = t1[-1] + t2[-1]
    print(f"Total time for combined burns: {total_time:.1f} s")
    print(f"Final position: {r2[-1]}")
    print(f"Final velocity: {v2[-1]}")

    # Coasting orbit after burns (simulate 10,000 seconds)
    r3, v3, t = quickint(r2[-1], v2[-1])

    print(t)

    # Plot combined trajectories
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Earth as a sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    x = EARTH_RADIUS * np.outer(np.cos(u), np.sin(v))
    y = EARTH_RADIUS * np.outer(np.sin(u), np.sin(v))
    z = EARTH_RADIUS * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color='blue', alpha=0.3)

    # Plot trajectory from velocity burn
    ax.plot(r1[:, 0], r1[:, 1], r1[:, 2], 'r-', label='Velocity Burn Trajectory')

    # Plot trajectory from inclination burn
    ax.plot(r2[:, 0], r2[:, 1], r2[:, 2], 'g-', label='Inclination Burn Trajectory')

    # Plot coasting orbit trajectory
    ax.plot(r3[:, 0], r3[:, 1], r3[:, 2], 'b-', label='Coasting Orbit Trajectory')

    # Mark start and end points
    ax.scatter(*r0, color='black', s=50, label='Start Orbit')
    ax.scatter(*r1[-1], color='red', s=50, label='End Velocity Burn')
    ax.scatter(*r2[-1], color='green', s=50, label='End Inclination Burn')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Combined Trajectories of Velocity, Inclination Burns, and Coasting Orbit')
    ax.legend()

    # Equal aspect ratio for 3D plot
    max_range = np.array([r3[:,0].max()-r3[:,0].min(), r3[:,1].max()-r3[:,1].min(), r3[:,2].max()-r3[:,2].min()]).max() / 2.0
    mid_x = (r3[:,0].max() + r3[:,0].min()) * 0.5
    mid_y = (r3[:,1].max() + r3[:,1].min()) * 0.5
    mid_z = (r3[:,2].max() + r3[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    
