import numpy as np
import matplotlib.pyplot as plt
import ssapy

from yeager_utils import equally_spaced_ta, yufig, RGEO


def main():
    # Example elliptical orbit
    a = 12000e3          # meters
    e = 0.7
    i = 0.0              # radians
    pa = 0.0             # radians
    raan = 0.0           # radians
    ta0 = 0.0            # radians
    t0 = 0.0             # GPS seconds

    # Build SSAPy Orbit from Keplerian elements
    orbit = ssapy.Orbit.fromKeplerianElements(a, e, i, pa, raan, ta0, t0)  # [10]

    # Request an even number of equal-arc-length samples
    n_samples = 16
    ta = equally_spaced_ta(a=a, e=e, n_samples=n_samples, degrees=False)

    # Compute radius for each sampled true anomaly
    r = a * (1 - e**2) / (1 + e * np.cos(ta))

    # Focus-centered perifocal coordinates
    x = r * np.cos(ta)
    y = r * np.sin(ta)

    # Dense ellipse for plotting
    ta_dense = np.linspace(0, 2 * np.pi, 1000)
    r_dense = a * (1 - e**2) / (1 + e * np.cos(ta_dense))
    x_dense = r_dense * np.cos(ta_dense)
    y_dense = r_dense * np.sin(ta_dense)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(x_dense, y_dense, label="Orbit")
    ax.scatter(x, y, color="red", zorder=3, label="Equal arc-length samples")
    ax.scatter([0], [0], color="black", marker="*", s=120, label="Focus")

    # Highlight periapsis and apoapsis
    rp = a * (1 - e)
    ra = a * (1 + e)
    ax.scatter([rp], [0], s=150, color="green", label="Periapsis")
    ax.scatter([-ra], [0], s=150, color="purple", label="Apoapsis")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Equal Arc-Length Sampling on an Elliptical Orbit")
    ax.legend(loc="upper left")
    ax.grid(True)

    plt.axis('equal')
    # Save using your utility
    yufig(fig, "tests/equally_spaced_ta.jpg")

    print("Returned ta [deg]:")
    print(np.degrees(ta))


if __name__ == "__main__":
    main()