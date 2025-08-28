# my_utils/pretty.py
# -----------------------------------------------------------------------------
# Pretty-print helpers with HDF5 summary support.
#
# - NumPy-aware pretty printer for general Python objects
# - HDF5 "summary of everything" view: lists every group and dataset,
#   showing group child counts and dataset shape/dtype, plus attribute keys,
#   and a small head/tail preview of dataset values.
#
# Usage
# -----
# >>> from my_utils.pretty import pprint
# >>> pprint(my_dict)
# >>> pprint("data/sim_results.h5")            # summarize entire file
# >>> with h5py.File("data.h5") as f:
# ...     pprint(f["/subgroup"])               # summarize subtree
#
# Notes
# -----
# - h5py is optional; if not installed, HDF5 features are disabled.
# - ASCII only; no fancy unicode characters.
# -----------------------------------------------------------------------------

import pathlib
import numpy as np
from pprint import pprint as _pprint

try:
    import h5py
except ImportError:  # make h5py optional
    h5py = None


# --------------------------- NumPy printing helpers --------------------------

_np_defaults = dict(
    edgeitems=3,      # values kept from each end when arrays are long
    threshold=20,     # total elements before truncation kicks in
    linewidth=120,    # wrap arrays nicely in most terminals
    precision=6,      # float digits; set None for NumPy default
    suppress=True,    # 1e-10 -> 0.000000
)


def _print_numpy(obj):
    """Pretty-print any Python object, truncating long NumPy arrays."""
    with np.printoptions(**_np_defaults):
        _pprint(obj)


# ---------------------------- HDF5 summary helpers ---------------------------

def _open_hdf5_like(obj):
    """
    Normalize an HDF5-like input into an open handle and a must_close flag.

    Returns
    -------
    handle : h5py.File | h5py.Group | h5py.Dataset
    must_close : bool
    """
    if h5py is None:
        raise RuntimeError("h5py not installed; cannot work with HDF5")

    if isinstance(obj, (h5py.File, h5py.Group, h5py.Dataset)):
        return obj, False

    if isinstance(obj, (str, pathlib.Path)):
        fh = h5py.File(obj, "r")
        return fh, True

    raise TypeError("Object is not an HDF5 file/group/dataset or path to one")


def _indent_for_name(name: str, indent_per_level: int = 2) -> str:
    """
    Compute indentation based on HDF5 path depth.
    Root '/' -> depth 0, '/grp' -> depth 1, '/grp/sub' -> depth 2, etc.
    """
    if name == "/":
        depth = 0
    else:
        # Count slashes but subtract 1 to make '/grp' depth 1
        depth = max(0, name.count("/") - 1)
    return " " * (indent_per_level * depth)


def _summarize_group(g: "h5py.Group") -> str:
    """
    Summarize a group by counting immediate subgroups and datasets.
    """
    n_groups = 0
    n_dsets = 0
    # Iterate keys without loading data
    for k in g.keys():
        try:
            item = g.get(k, getlink=False)
        except Exception:
            # If access fails, skip counting that child
            continue
        if isinstance(item, h5py.Group):
            n_groups += 1
        elif isinstance(item, h5py.Dataset):
            n_dsets += 1
    return f"{g.name or '/'} (Group: {n_groups} groups, {n_dsets} datasets)"


def _array2string(a: np.ndarray) -> str:
    """Format a small numpy array to a compact one-line string."""
    return np.array2string(
        np.asarray(a),
        max_line_width=_np_defaults["linewidth"],
        precision=_np_defaults["precision"],
        suppress_small=_np_defaults["suppress"],
        separator=", ",
        threshold=1_000_000,  # do not trigger numpy's own ellipsis for our tiny previews
    )


def _preview_dataset_values(d: "h5py.Dataset", head: int = 3, tail: int = 3, small_limit: int = 20) -> str:
    """
    Return a small string preview of dataset values:
    - If scalar: show the singleton value.
    - If total size <= small_limit: show the entire flattened array.
    - Else: show first `head` and last `tail` elements with "..., " in between.
    Notes:
    - For N-D datasets, preview is taken along the first axis with minimal slicing
      to avoid loading large arrays into memory.
    """
    try:
        # Scalar dataset
        if d.shape == ():
            val = d[()]  # scalar
            return _array2string(np.array([val]))[1:-1]  # strip brackets

        # Small datasets: read everything (but keep it small)
        if d.size <= small_limit:
            data = d[...]
            flat = np.ravel(data)
            return _array2string(flat)[1:-1]  # strip brackets

        # Large datasets
        n0 = d.shape[0]

        # Build head and tail slices along the first axis while keeping other dims small
        # Use slice(None) for remaining dims but cap to 1 to avoid big loads.
        rest_slices = tuple(slice(0, 1) for _ in range(d.ndim - 1))

        h = min(head, n0)
        t = min(tail, n0 - h) if n0 > h else 0

        head_slice = (slice(0, h),) + rest_slices
        head_block = d[head_slice]
        head_vals = np.ravel(head_block)

        tail_vals = np.array([], dtype=d.dtype)
        if t > 0:
            tail_slice = (slice(n0 - t, n0),) + rest_slices
            tail_block = d[tail_slice]
            tail_vals = np.ravel(tail_block)

        if head_vals.size + tail_vals.size <= small_limit and t == 0:
            # Just show the head if that covers all elements along axis 0
            return _array2string(head_vals)[1:-1]

        # Format "x0, x1, x2, ..., y1, y2, y3"
        head_str = _array2string(head_vals).strip()
        tail_str = _array2string(tail_vals).strip() if tail_vals.size else ""
        head_str = head_str[1:-1] if head_str.startswith("[") and head_str.endswith("]") else head_str
        tail_str = tail_str[1:-1] if tail_str.startswith("[") and tail_str.endswith("]") else tail_str

        if tail_str:
            return f"{head_str}, ..., {tail_str}"
        else:
            return head_str
    except Exception as e:
        # On any issue reading data, fall back to a generic note
        return f"<preview unavailable: {type(e).__name__}>"


def _summarize_dataset(d: "h5py.Dataset") -> str:
    """
    Summarize a dataset by shape and dtype, plus a small head/tail preview.
    """
    try:
        shape = "x".join(str(s) for s in d.shape)
    except Exception:
        shape = "unknown"
    try:
        dtype = str(d.dtype)
    except Exception:
        dtype = "unknown"

    preview = _preview_dataset_values(d)
    if preview:
        return f"{d.name} (Dataset, shape={shape}, dtype={dtype}) preview=[{preview}]"
    else:
        return f"{d.name} (Dataset, shape={shape}, dtype={dtype})"


def _summarize_attrs(obj) -> str:
    """
    Return a one-line summary of attribute keys, if any.
    """
    try:
        keys = list(obj.attrs.keys())
    except Exception:
        keys = []
    if not keys:
        return ""
    return f"@attrs: {keys}"


def _print_hdf5_summary(obj):
    """
    Print a summary of all levels under the given HDF5 object.

    The summary includes every group and dataset reachable from 'obj',
    one line per object, plus a line of attribute keys when present,
    and a small value preview for datasets.
    Traversal uses h5py visititems to cover the full subtree.
    """
    handle, must_close = _open_hdf5_like(obj)
    try:
        # Print the root (or starting node) first
        if isinstance(handle, (h5py.File, h5py.Group)):
            head_line = _summarize_group(handle if isinstance(handle, h5py.Group) else handle["/"])
            print(head_line)
            attrs_line = _summarize_attrs(handle)
            if attrs_line:
                print(_indent_for_name(handle.name) + "  " + attrs_line)
        elif isinstance(handle, h5py.Dataset):
            print(_summarize_dataset(handle))
            attrs_line = _summarize_attrs(handle)
            if attrs_line:
                print(_indent_for_name(handle.name) + "  " + attrs_line)

        # Visit entire subtree starting at handle
        def _visitor(name, item):
            # Skip duplicating the very first node if visititems calls us for it
            if item is handle:
                return
            pad = _indent_for_name(item.name)
            if isinstance(item, h5py.Group):
                print(pad + _summarize_group(item))
                attrs_line = _summarize_attrs(item)
                if attrs_line:
                    print(pad + "  " + attrs_line)
            elif isinstance(item, h5py.Dataset):
                print(pad + _summarize_dataset(item))
                attrs_line = _summarize_attrs(item)
                if attrs_line:
                    print(pad + "  " + attrs_line)
            else:
                # Other HDF5 object types are uncommon; print a generic line
                try:
                    print(pad + f"{item.name} (unsupported HDF5 type: {type(item).__name__})")
                except Exception:
                    print(pad + f"(unsupported HDF5 type: {type(item).__name__})")

        # For a file or group, walk the subtree; for a dataset there is nothing to walk
        if isinstance(handle, (h5py.File, h5py.Group)):
            handle.visititems(_visitor)

    finally:
        if must_close:
            try:
                handle.close()
            except Exception:
                pass


# ------------------------------ public pprint --------------------------------

def pprint(obj):
    """
    Pretty-print any object.

    If 'obj' is (or points to) an HDF5 file | group | dataset:
        Print a summary of all levels reachable from that node:
        - For each group: the count of immediate subgroups and datasets
        - For each dataset: shape and dtype, plus a head/tail preview
        - For any object with attributes: list of attribute keys
    Otherwise:
        Fall back to NumPy-aware pretty printer for general Python objects.
    """
    if h5py is not None and (
        isinstance(obj, (h5py.File, h5py.Group, h5py.Dataset)) or
        isinstance(obj, (str, pathlib.Path))
    ):
        _print_hdf5_summary(obj)
    else:
        _print_numpy(obj)
