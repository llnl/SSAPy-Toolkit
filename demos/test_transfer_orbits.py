#!/usr/bin/env python3


from ssapy_toolkit import (
    EARTH_RADIUS,
    transfer_shooter,
    transfer_hohmann,
    transfer_lambertian,
    transfer_coplanar,
    RGEO,
    Time,
    hkoe,
    yufig,
)
from ssapy import Orbit

def main():
    SHOW = True  # Set to False if you want to save without popping windows

    # Epoch
    t0 = Time("2025-01-01").gps

    # Define initial and final orbits via Keplerian elements
    # a in meters; e dimensionless; i, RAAN, argp, ta in degrees
    orbit1 = Orbit.fromKeplerianElements(
        *hkoe([20000e3 + EARTH_RADIUS, 0.1, -60.0, 90.0, 0.0, 180.0]),
        t=t0,
    )
    orbit2 = Orbit.fromKeplerianElements(
        *hkoe([10000e3 + EARTH_RADIUS, 0.1, 0.0, 0.0, 0.0, 0.0]),
        t=t0,
    )

    # Hohmann: Orbit -> Orbit
    print("Running Hohmann (orbit -> orbit)")
    result = transfer_hohmann(orbit1, orbit2, plot=True)
    fig = result["fig"]
    yufig(fig, "tests/transfers_hohmann_orbit_to_orbit")

    # Hohmann: r1, v1, r2
    print("Running Hohmann (r1, v1, r2)")
    result = transfer_hohmann(orbit1.r, orbit1.v, orbit2.r, plot=True)
    fig = result["fig"]
    yufig(fig, "tests/transfers_hohmann_rv")

    # Lambertian: Orbit -> Orbit
    print("Running Lambertian (orbit -> orbit)")
    try:
        result = transfer_lambertian(orbit1, orbit2, plot=True)
        fig = result["fig"]
        yufig(fig, "tests/transfers_lambertian_orbit_to_orbit")
    except Exception as err:
        print("Lambertian (orbit -> orbit) failed:", err)

    # Lambertian: r1, v1, r2
    print("Running Lambertian (r1, v1, r2)")
    try:
        result = transfer_lambertian(orbit1.r, orbit1.v, orbit2.r, plot=True)
        fig = result["fig"]
        yufig(fig, "tests/transfers_lambertian_rv")
    except Exception as err:
        print("Lambertian (r1, v1, r2) failed:", err)

    # Shooter: Orbit -> Orbit
    print("Running shooter (orbit -> orbit)")
    result = transfer_shooter(orbit1, orbit2, plot=True, status=True)
    fig = result["fig"]
    yufig(fig, "tests/transfers_shooter_orbit_to_orbit")

    # Shooter: r1, v1, r2
    print("Running shooter (r1, v1, r2)")
    result = transfer_shooter(orbit1.r, orbit1.v, orbit2.r, plot=True, status=True)
    fig = result["fig"]
    yufig(fig, "tests/transfers_shooter_rv")

    # Coplanar: Orbit -> Orbit
    print("Running coplanar (orbit -> orbit)")
    result = transfer_coplanar(orbit1, orbit2, plot=True, status=True)
    fig = result["fig"]
    yufig(fig, "tests/transfers_coplanar_orbit_to_orbit")


if __name__ == "__main__":
    main()
