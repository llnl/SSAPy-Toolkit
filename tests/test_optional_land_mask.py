import sys

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def test_plotting_fallbacks_work_without_global_land_mask(monkeypatch):
    monkeypatch.setitem(sys.modules, "global_land_mask", None)

    from demos.sensor_coverage import demo_sensor_fov_plot as sensor_demo
    from ssapy_toolkit.plots import eclipse_space_view_plotly as eclipse
    from ssapy_toolkit.plots import globe_orbit_daynight_plotly as globe

    # _land_mask and _procedural_continents were replaced by one deterministic
    # fallback texture, used only when no real SSAPy Earth image is available.
    # It still prefers global_land_mask for continent shapes and drops to a
    # low-frequency field when the package is missing, which is the path this
    # test exercises. The cache is keyed on resolution, so clear it or an
    # earlier call can return a texture built while the package was importable.
    globe._procedural_earth_texture_cached.cache_clear()
    texture = globe._procedural_earth_texture(8, 16)
    assert texture.shape == (8, 16, 3)
    assert texture.dtype == np.uint8
    assert np.all((0 <= texture) & (texture <= 255))
    # A usable fallback has to vary: a single flat colour would render as a
    # blank sphere and still satisfy the checks above.
    assert texture.reshape(-1, 3).std(axis=0).max() > 1.0

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    eclipse._earth_sphere_mpl(ax, np.zeros(3), 1.0, np.array([1.0, 0.0, 0.0]))
    assert ax.collections
    plt.close(fig)

    fig = go.Figure()
    sensor_demo._add_map_background(fig)
    assert all(trace.name != "Land/water background" for trace in fig.data)
