if __name__ == "__main__":
    from yeager_utils import gifify, figpath, groundtrack_dashboard, RGEO, ssapy_orbit
    import os
    import numpy as np
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt

    # Use a non-interactive backend if available
    try:
        plt.switch_backend("Agg")
    except Exception:
        pass

    # Helper to count frames
    def _count_gif_frames(path):
        with imageio.get_reader(path) as rdr:
            try:
                n = rdr.get_length()
            except Exception:
                n = sum(1 for _ in rdr)
        return n
   # Synthetic data for simple tests
    x = np.linspace(0, 4 * np.pi, 240)
    y = np.sin(x) * np.exp(-0.1 * x)

    # Test 1: pyplot-style function (returns None), chunks mode
    def plot_simple(x, y, label=None):
        plt.figure()
        plt.plot(x, y, label=label)
        if label:
            plt.legend()
        plt.grid(True)

    out1 = gifify(
        plot_simple,
        x, y,
        label="damped wave",
        save_path=figpath("tests/test_chunks.gif"),
        array_arg_indices=(0, 1),
        mode="chunks",
        chunk_size=60,
        step=60,
        fps=12,
        verbose=True,
    )
    assert os.path.exists(out1["path"]) and os.path.getsize(out1["path"]) > 0, "test_chunks.gif not created"
    n1 = _count_gif_frames(out1["path"])
    assert n1 == out1["frames"], f"Frame mismatch (chunks): gif={n1} reported={out1['frames']}"
    print(f"Test 1 OK: {out1['path']} with {n1} frames")

    # Test 2: function that returns an Axes, cumulative mode
    def plot_returns_axes(x, y):
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_title("returns Axes")
        ax.grid(True)
        return ax

    out2 = gifify(
        plot_returns_axes,
        x, y,
        save_path=figpath("tests/test_cumulative.gif"),
        array_arg_indices=(0, 1),
        mode="cumulative",
        step=20,
        fps=10,
        verbose=True,
    )
    assert os.path.exists(out2["path"]) and os.path.getsize(out2["path"]) > 0, "test_cumulative.gif not created"
    n2 = _count_gif_frames(out2["path"])
    assert n2 == out2["frames"], f"Frame mismatch (cumulative): gif={n2} reported={out2['frames']}"
    print(f"Test 2 OK: {out2['path']} with {n2} frames")

    # Test 3: function that expects an injected Axes, sliding mode
    def plot_with_ax(ax, x, y):
        ax.plot(x, y)
        ax.set_title("injected ax")
        ax.grid(True)

    out3 = gifify(
        plot_with_ax,
        x, y,
        save_path=figpath("tests/test_sliding.gif"),
        array_arg_indices=(0, 1),
        mode="sliding",
        chunk_size=50,
        step=15,
        fps=10,
        inject_ax=True,
        ax_arg_index=0,  # inject as first positional arg
        verbose=True,
    )
    assert os.path.exists(out3["path"]) and os.path.getsize(out3["path"]) > 0, "test_sliding.gif not created"
    n3 = _count_gif_frames(out3["path"])
    assert n3 == out3["frames"], f"Frame mismatch (sliding): gif={n3} reported={out3['frames']}"
    print(f"Test 3 OK: {out3['path']} with {n3} frames")

    # Test 4: groundtrack_dashboard (multi-axes), sliding mode
    # - We pass r,t as the two arrays to slice (array_arg_indices=(0,1)).
    
    r, v, t = ssapy_orbit(a=RGEO, e=0.2)
    # Run gifify on the dashboard
    out4 = gifify(
        groundtrack_dashboard,
        r, t.gps,
        save_path=figpath("tests/test_groundtrack.gif"),
        array_arg_indices=(0, 1),
        mode="sliding",
        chunk_size=120,
        step=30,
        fps=8,
        verbose=True,
        fixed_limits=True,   # r,t are not x/y axes; don't auto-limit on them
        show_legend=False,
        pad=1000,
        t0=float(t[0].gps)
    )
    assert os.path.exists(out4["path"]) and os.path.getsize(out4["path"]) > 0, "test_groundtrack.gif not created"
    n4 = _count_gif_frames(out4["path"])
    assert n4 == out4["frames"], f"Frame mismatch (groundtrack): gif={n4} reported={out4['frames']}"
    print(f"Test 4 OK: {out4['path']} with {n4} frames")
    
    print("All tests passed.")
