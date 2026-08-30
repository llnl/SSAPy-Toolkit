import sys

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def test_plotting_fallbacks_work_without_global_land_mask(monkeypatch):
    monkeypatch.setitem(sys.modules, "global_land_mask", None)

    from demos.sensor_coverage import demo_sensor_fov_plot as sensor_demo
    from ssapy_toolkit.plots import eclipse_space_view_plotly as eclipse
    from ssapy_toolkit.plots import globe_orbit_daynight_plotly as globe

    land, lat, lon = globe._land_mask(8, 16)
    assert land.shape == lat.shape == lon.shape == (8, 16)
    assert np.isfinite(land).all()
    assert np.all((0.0 <= land) & (land <= 1.0))

    assert globe._procedural_continents(8, 16).shape == (8, 16, 3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    eclipse._earth_sphere_mpl(ax, np.zeros(3), 1.0, np.array([1.0, 0.0, 0.0]))
    assert ax.collections
    plt.close(fig)

    fig = go.Figure()
    sensor_demo._add_map_background(fig)
    assert all(trace.name != "Land/water background" for trace in fig.data)
