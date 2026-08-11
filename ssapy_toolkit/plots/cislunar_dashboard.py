from ._cislunar_plot_core import _cislunar_plot_core
from .plotutils import _pop_save_path_aliases, _raise_unrecognized_kwargs


def cislunar_dashboard(
    r,
    t=None,
    figsize=(12, 7),
    fontsize=12,
    save_path=False,
    show=False,
    title=None,
    c="white",
    legend=True,
    **save_kwargs,
):
    """Plot the standard two-panel cislunar dashboard layout."""
    save_path, save_kwargs = _pop_save_path_aliases(save_kwargs, save_path=save_path)
    _raise_unrecognized_kwargs(save_kwargs, "cislunar_dashboard")
    return _cislunar_plot_core(
        r,
        t=t,
        figsize=figsize,
        fontsize=fontsize,
        save_path=save_path,
        show=show,
        title=title,
        c=c,
        mode="combined",
        legend=legend,
    )
