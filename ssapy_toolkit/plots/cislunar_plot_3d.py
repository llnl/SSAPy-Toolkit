from ._cislunar_plot_core import _cislunar_plot_core
from .plotutils import _pop_save_path_aliases, _raise_unrecognized_kwargs


def cislunar_plot_3d(r, t=None, figsize=(8, 8), fontsize=12, save_path=False, show=False, legend=True, title="", c="white", **save_kwargs):
    save_path, save_kwargs = _pop_save_path_aliases(save_kwargs, save_path=save_path)
    _raise_unrecognized_kwargs(save_kwargs, "cislunar_plot_3d")
    return _cislunar_plot_core(
        r,
        t=t,
        figsize=figsize,
        fontsize=fontsize,
        save_path=save_path,
        show=show,
        title=title,
        c=c,
        mode="3d",
        legend=legend,
    )
