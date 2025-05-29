from yeager_utils import orbit_plot_xy, np, leapfrog, accel_uniform_earth

# Constants (SI units)
G = 6.67430e-11      # Gravitational constant (m^3 kg^-1 s^-2)
M_earth = 5.972e24   # Mass of Earth (kg)
R_earth = 6.371e6    # Radius of Earth (m)
mu = G * M_earth     # Gravitational parameter (m^3/s^2)

# Define orbit parameters
ap = R_earth + 1000e3  # Apoapsis (m), constant
perigees = np.linspace(10e3, R_earth, 10)  # Periapses (m)
rs = []

for peri in perigees:
    # Compute semi-major axis and eccentricity
    a = (peri + ap) / 2
    e = (ap - peri) / (peri + ap)

    # Initial conditions at apogee
    r_a = -ap  # Apoapsis distance (m)
    r0 = np.array([r_a, 0.0, 0.0])  # Position at apogee (m)
    v0_magnitude = np.sqrt(mu * (np.abs(2 / r_a) - 1 / a))  # Velocity at apogee (m/s)
    v0 = np.array([0.0, -v0_magnitude, 0.0])  # Velocity in negative y-direction

    # Compute orbital period
    T = 2 * np.pi * np.sqrt(a**3 / mu)  # Period in seconds
    print(f"Periapsis: {peri/1e3:.1f} km, Period: {T/60:.1f} minutes")

    # Time array for one period
    t = np.linspace(0, T, 2000)
    dt = t[1] - t[0]

    # Integrate using Leapfrog
    r, v = leapfrog(r0=r0, v0=v0, t=t, accel=accel_uniform_earth)
    rs.append(r)

# Plot orbits using orbit_plot_xy
orbit_plot_xy(
    rs,
    save_path="/g/g16/yeager7/workdir/yeager_utils/demos/images/testing_ellipses_leapfrog.jpg",
    pad=500,
    title="Earth as an extended body",
    show=True
)
