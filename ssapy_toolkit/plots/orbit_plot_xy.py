from ._orbit_plot_core import _orbit_plot_core


def orbit_plot_xy(r, t=None, title="", figsize=(7, 7), save_path=False, frame="gcrf", show=False, c="black", pad=1):
    return _orbit_plot_core(
        r,
        t=t,
        title=title,
        figsize=figsize,
        save_path=save_path,
        frame=frame,
        show=show,
        c=c,
        pad=pad,
        views=("xy",),
        xy_title_includes_title=True,
        lunar_transform="fixed",
    )
