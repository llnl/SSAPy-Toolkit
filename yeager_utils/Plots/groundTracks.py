import numpy as np
from typing import Union, List, Optional
import matplotlib.pyplot as plt
import ipyvolume as ipv

from ssapy import groundTrack
from ..Time_Functions import Time
from .plotutils import load_earth_file, save_plot, drawEarth


def groundTrackPlot(r: Union[np.ndarray, List[np.ndarray]],
                    time: Union[np.ndarray, List[np.ndarray]],
                    ground_stations: Optional[np.ndarray] = None,
                    save_path: Optional[str] = None) -> None:
    """
    Parameters
    ----------
    r : (3,) array_like or List of (3,) array_like - Orbit positions in meters or list of such arrays for multiple orbits.
    time : (n,) array_like or List of (n,) array_like - array of Astropy Time objects or time in gps seconds.
             If r is a list of orbits, time should be a list of time vectors (one for each orbit).

    optional - ground_stations: (n,2) array of ground stations (lat, lon) in degrees.

    Author:
    -------
    Travis Yeager (yeager7@llnl.gov)
    """

    # Check if r is a list of orbits
    if isinstance(r, np.ndarray):
        r = [r]
    if isinstance(time, np.ndarray):
        # If multiple orbits, create a corresponding time list
        if isinstance(time, list):
            time_list = time  # If time is already a list of time arrays
        else:
            time_list = []
            for _ in r:
                time_list.append(time)
    else:
        # If only one orbit, time should be a single time array
        time_list = [time]

    # Initialize plot
    fig = plt.figure(figsize=(15, 12))
    plt.imshow(load_earth_file(), extent=[-180, 180, -90, 90])

    # Loop through each orbit and its corresponding time
    standard_colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

    for i, orbit in enumerate(r):
        if len(r) <= len(standard_colors):  # Use standard colors
            print(len(r), len(standard_colors))
            color = standard_colors[i]
        else:  # Use rainbow colors if there are more orbits than standard colors
            color = plt.cm.rainbow(i / len(r))

        lon, lat, height = groundTrack(orbit, time_list[i])  # Assuming output in radians

        # Convert longitude and latitude to degrees for plotting
        lon_deg = np.degrees(lon)
        lat_deg = np.degrees(lat)

        # Identify discontinuities in longitude (crossing the -180/180 boundary)
        discont_indices = np.where(np.abs(np.diff(lon_deg)) > 179)[0]

        # Split the data into segments for clean plotting
        segments = np.split(np.arange(len(lon_deg)), discont_indices + 1)

        # Plot the ground track for each orbit with different label and color
        for j, segment in enumerate(segments):
            plt.plot(lon_deg[segment], lat_deg[segment], color=color, label=f"Orbit {i+1}" if j == 0 else "")

    # Plot ground stations if provided
    if ground_stations is not None:
        for ground_station in ground_stations:
            plt.scatter(ground_station[1], ground_station[0], s=50, color='Red', label="Ground Station")

    # Set plot limits and labels
    plt.ylim(-90, 90)
    plt.xlim(-180, 180)
    plt.xlabel("Longitude [deg]")
    plt.ylabel("Latitude [deg]")

    # Show legend
    plt.legend()

    # Show the plot
    plt.show()

    # Save plot if save_path is provided
    if save_path:
        save_plot(fig, save_path)


def groundTrackVideo(r: np.ndarray, time: Union[np.ndarray, Time]) -> None:
    """
    Visualizes the ground track of an orbiting object using 3D animation.

    This function creates an interactive 3D visualization of the object's ground track,
    displaying the path and the position of the object over time. The Earth is shown,
    and the object's position is marked by a sphere. A line plot represents the object's
    trajectory.

    Parameters
    ----------
    r : np.ndarray
        A (3, N) array representing the position of the orbiting object in meters
        at each time step, where N is the number of time steps.
    time : Union[np.ndarray, astropy.time.Time]
        A 1D array or scalar representing the time steps. If it is an array, it should
        correspond to GPS seconds, i.e., seconds since 1980-01-06 00:00:00 UTC.

    Returns
    -------
    None
        This function does not return any value. It displays an interactive animation
        of the object's ground track.

    Notes
    -----
    - The function uses the `ipv` package for creating interactive 3D visualizations.
    - The object's position is shown as a magenta sphere and the trajectory as a white line.
    - The animation is controlled using the `ipv.animation_control` function.

    Author:
    -------
    Travis Yeager (yeager7@llnl.gov)
    """
    ipvfig = ipv.figure(width=2000 / 2, height=1000 / 2)
    ipv.style.set_style_dark()
    ipv.style.box_off()
    ipv.style.axes_off()
    widgets = []
    widgets.append(drawEarth(time))
    widgets.append(
        ipv.scatter(
            r[:, 0, None],
            r[:, 1, None],
            r[:, 2, None],
            marker='sphere',
            color='magenta',
            size=10  # Increase the dot size (default is 1)
        )
    )
    # Line plot showing the path
    widgets.append(
        ipv.plot(
            r[:, 0],
            r[:, 1],
            r[:, 2],
            color='white',
            linewidth=1
        )
    )
    ipv.animation_control(widgets, sequence_length=len(time), interval=0)
    ipv.xyzlim(-10_000_000, 10_000_000)
    ipvfig.camera.position = (-2, 0, 0.2)
    ipvfig.camera.up = (0, 0, 1)
    ipv.show()
