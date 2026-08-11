from ._cislunar_plot_core import _cislunar_plot_core


def cislunar_plot_xy(r, t=None, figsize=(8, 8), fontsize=12, save_path=False, show=False, title=None, c="black"):
    return _cislunar_plot_core(
        r,
        t=t,
        figsize=figsize,
        fontsize=fontsize,
        save_path=save_path,
        show=show,
        title=title,
        c=c,
        mode="xy",
        legend=True,
    )
