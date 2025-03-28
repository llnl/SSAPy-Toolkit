import numpy as np
import matplotlib.pyplot as plt


def calculate_finite_burn_acceleration(delta_v, t_impulsive, a_magnitude):
    """
    Calculate the acceleration to approximate an instantaneous orbital transfer with a finite burn.

    Parameters:
    - delta_v: np.array, shape (3,), impulsive delta-v vector in inertial frame (m/s)
    - t_impulsive: float, time of the impulsive burn (s)
    - a_magnitude: float, constant acceleration magnitude from specific thrust (m/s^2)

    Returns:
    - a_vector: np.array, shape (3,), constant acceleration vector (m/s^2)
    - t_burn: float, duration of the burn (s)
    - t_start: float, start time of the burn (s)
    - t_end: float, end time of the burn (s)
    """
    # Compute the magnitude of delta-v
    dv_norm = np.linalg.norm(delta_v)
    if dv_norm == 0:
        raise ValueError("Delta-v is zero; no burn is required.")

    # Direction of acceleration is the same as delta-v
    direction = delta_v / dv_norm
    a_vector = a_magnitude * direction

    # Burn duration to achieve the required delta-v
    t_burn = dv_norm / a_magnitude

    # Center the burn around the impulsive burn time
    t_start = t_impulsive - t_burn / 2
    t_end = t_impulsive + t_burn / 2

    return a_vector, t_burn, t_start, t_end


def keplerian_acceleration(position):
    """
    Calculate the gravitational acceleration due to a central body (Keplerian acceleration).

    Parameters:
    - position: np.array, shape (3,), spacecraft position (m)

    Returns:
    - acceleration: np.array, shape (3,), gravitational acceleration (m/s^2)
    """
    from yeager_utils import EARTH_MU
    r = np.linalg.norm(position)  # Distance from the center of Earth
    direction = -position / r  # Unit vector pointing towards the center of Earth
    a_gravity = -EARTH_MU / r**2  # Gravitational acceleration magnitude
    return a_gravity * direction


def leapfrog_integrator(position, velocity, a_magnitude, burn_vector, t_start, t_end, dt):
    """
    Propagate the spacecraft's position and velocity using the leapfrog integrator under both burn and Keplerian accelerations.

    Parameters:
    - position: np.array, shape (3,), initial position (m)
    - velocity: np.array, shape (3,), initial velocity (m/s)
    - a_magnitude: float, constant acceleration magnitude from the burn (m/s^2)
    - burn_vector: np.array, shape (3,), direction of the constant burn acceleration (m/s^2)
    - t_start: float, start time of the burn (s)
    - t_end: float, end time of the burn (s)
    - dt: float, time step (s)

    Returns:
    - positions: list of np.array, positions at each time step
    - velocities: list of np.array, velocities at each time step
    """
    positions = [position]
    velocities = [velocity]

    # Half-step initial velocity
    velocity += (a_magnitude * burn_vector) * dt / 2

    t = 0.0
    while t < t_end:
        a_gravity = keplerian_acceleration(position)

        position += velocity * dt
        acceleration = a_gravity + a_magnitude * burn_vector
        velocity += acceleration * dt

        positions.append(position.copy())
        velocities.append(velocity.copy())
        t += dt

    return np.array(positions), np.array(velocities)


# Example usage
if __name__ == "__main__":
    # Example inputs
    delta_v = np.array([-3000.0, -1000.0, 10000.0])  # 100 m/s along x-axis
    t_impulsive = 0.0  # Impulsive burn at t = 0 s
    a_magnitude = 2.0  # Acceleration of 2 m/s^2

    # Calculate the burn parameters
    burn_vector, t_b, t_s, t_e = calculate_finite_burn_acceleration(delta_v, t_impulsive, a_magnitude)
    print(f"Burn vector: {burn_vector} m/s^2")
    print(f"Burn duration: {t_b} s")
    print(f"Burn start time: {t_s} s")
    print(f"Burn end time: {t_e} s")

    # Define initial conditions for position and velocity (assuming a low Earth orbit)
    initial_position = np.array([7000e3, 0.0, 0.0])  # Initial position (m)
    initial_velocity = np.array([0.0, 7.5e3, 0.0])  # Initial velocity (m/s)

    # Simulate the trajectory with the leapfrog integrator
    dt = 1  # Time step (s)
    positions, velocities = leapfrog_integrator(initial_position, initial_velocity, a_magnitude, burn_vector, t_s, t_e, dt)

    # Plot the trajectory
    plt.figure(figsize=(8, 6))
    plt.scatter(positions[:, 0], positions[:, 1], label='Trajectory in XY-plane')
    plt.scatter([0], [0], color='red', label='Earth Center')  # Earth at the origin
    plt.title('Spacecraft Trajectory Under Finite Burn Maneuver and Keplerian Acceleration')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend()
    plt.grid(True)
    plt.show()
