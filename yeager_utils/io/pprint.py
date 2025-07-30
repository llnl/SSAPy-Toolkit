# my_utils/pretty.py ──────────────────────────────────────────────────────────
import pathlib
import numpy as np
from pprint import pprint as _pprint

try:
    import h5py
except ImportError:  # make h5py optional
    h5py = None


# ────────────────────────── NumPy printing helpers ─────────────────────────
_np_defaults = dict(
    edgeitems  = 3,     # values kept from each end when arrays are long
    threshold  = 20,    # total elements before truncation kicks in
    linewidth  = 120,   # wrap arrays nicely in most terminals
    precision  = 6,     # float digits; set None for NumPy default
    suppress   = True,  # 1e‑10 → 0.000000
)


def _print_numpy(obj):
    """Pretty‑print any Python object, truncating long NumPy arrays."""
    with np.printoptions(**_np_defaults):
        _pprint(obj)


# ─────────────────────────── HDF5 printing helpers ─────────────────────────
def _repr_hdf5(obj, indent=0, max_items=20, depth=None):
    """
    Recursively print HDF5 structure (groups, datasets, attrs).

    Parameters
    ----------
    obj        : h5py.File | h5py.Group | h5py.Dataset
    indent     : int   current indent (spaces)
    max_items  : int   limit per group (show head/tail with … if more)
    depth      : int   max recursion depth; None = unlimited
    """
    pad = " " * indent
    if isinstance(obj, (h5py.File, h5py.Group)):
        typ = "File" if isinstance(obj, h5py.File) else "Group"
        print(f"{pad}{obj.name or '/'} ({typ})")
        # show attributes (if any) before children
        if obj.attrs:
            print(f"{pad}  @attrs:")
            for k, v in obj.attrs.items():
                _print_numpy({k: v})  # handle scalars/arrays
        # show children (groups/datasets)
        keys = list(obj.keys())
        n = len(keys)
        show = keys[:max_items] if n <= max_items else \
               keys[:max_items//2] + ["..."] + keys[-max_items//2:]
        for k in show:
            if k == "...":
                print(f"{pad}  ... ({n - max_items} more)")
                continue
            _repr_hdf5(obj[k], indent + 2, max_items, None if depth is None else depth - 1)
    elif isinstance(obj, h5py.Dataset):
        shape = "x".join(str(s) for s in obj.shape)
        print(f"{pad}{obj.name} (Dataset, {shape}, {obj.dtype})")
        if obj.attrs:
            print(f"{pad}  @attrs:")
            for k, v in obj.attrs.items():
                _print_numpy({k: v})
    else:  # fallback
        print(f"{pad}{obj} (unsupported HDF5 type)")


def _print_hdf5(obj):
    """
    Accept a path or already‑open h5py object and pretty‑print its tree.
    """
    # open file if a path was given
    must_close = False
    if isinstance(obj, (str, pathlib.Path)):
        if h5py is None:
            raise RuntimeError("h5py not installed; cannot open HDF5 files")
        obj = h5py.File(obj, "r")
        must_close = True

    if h5py is None or not isinstance(obj, (h5py.File, h5py.Group, h5py.Dataset)):
        raise TypeError("Object is not an HDF5 file/group/dataset")

    _repr_hdf5(obj)
    if must_close:
        obj.close()


# ───────────────────────────── public pprint ───────────────────────────────
def pprint(obj):
    """
    Pretty‑print *any* object.

    * If `obj` is (or points to) an HDF5 file | group | dataset → show the
      tree structure, dataset shapes/dtypes, and attributes.
    * Otherwise → fall back to NumPy‑aware dict/array pretty‑printer.

    Usage
    -----
    >>> from my_utils.pretty import pprint
    >>> pprint(my_dict)
    >>> pprint("data/sim_results.h5")
    >>> with h5py.File("data.h5") as f:
    ...     pprint(f["/subgroup"])
    """
    if h5py is not None and (
        isinstance(obj, (h5py.File, h5py.Group, h5py.Dataset)) or
        isinstance(obj, (str, pathlib.Path))
    ):
        _print_hdf5(obj)
    else:
        _print_numpy(obj)
