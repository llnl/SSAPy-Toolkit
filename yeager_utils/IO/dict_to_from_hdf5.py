import os
import sys
import pickle
import datetime as _dt
from typing import Any, Mapping, Sequence, Union

import h5py
import numpy as np

try:
    from astropy.time import Time as AstroTime
    _HAS_ASTROPY = True
except ImportError:
    _HAS_ASTROPY = False


# ---------- SAVE ----------

def save_dict_to_hdf5(
    data: Mapping[str, Any],
    filepath: str,
    mode: str = "w",
    *,
    pickle_objects: bool = True,
    compression: Union[str, None] = "gzip",
    compression_opts: Union[int, None] = 4,
) -> None:
    """
    Save a (possibly nested) Python dictionary to an HDF5 file.

    Supported types:
      - dict -> HDF5 group
      - list/tuple -> HDF5 group with numeric keys ("0", "1", ...)
      - numpy arrays -> datasets
      - scalars: int/float/bool, numpy scalar types -> scalar datasets
      - str -> variable-length UTF-8 datasets
      - bytes/bytearray/memoryview -> bytes datasets
      - datetime.datetime, datetime.date, datetime.time -> stored as ISO strings
      - astropy.time.Time -> stored as (mjd, scale, format, meta) so it can be reconstructed
      - other objects -> pickled if pickle_objects=True

    Args:
        data: dict-like with string keys.
        filepath: output HDF5 file path.
        mode: HDF5 file mode ("w", "w-", "a", "r+").
        pickle_objects: if True, unsupported objects are stored via pickle.
        compression: HDF5 compression filter (e.g. "gzip", "lzf", or None).
        compression_opts: compression options (e.g. level 0–9 for gzip).
    """

    def _write_item(
        h5group: h5py.Group,
        key: str,
        value: Any,
    ) -> None:
        # nested dict -> subgroup
        if isinstance(value, Mapping):
            subgroup = h5group.require_group(key)
            for k, v in value.items():
                if not isinstance(k, str):
                    raise TypeError(f"HDF5 requires string keys; got key={k!r}")
                _write_item(subgroup, k, v)

        # list/tuple -> group with numeric keys
        elif isinstance(value, (list, tuple)):
            subgroup = h5group.require_group(key)
            for idx, v in enumerate(value):
                _write_item(subgroup, str(idx), v)

        # astropy Time
        elif _HAS_ASTROPY and isinstance(value, AstroTime):
            # We store as group: mjd (float array), scale, format, and meta as pickled bytes
            if key in h5group:
                del h5group[key]
            t_group = h5group.create_group(key)
            # flatten to MJD in float64
            mjd = value.mjd
            mjd_arr = np.array(mjd, dtype="float64")
            t_group.create_dataset(
                "mjd",
                data=mjd_arr,
                compression=compression,
                compression_opts=compression_opts,
            )
            t_group.attrs["__type__"] = "astropy.time.Time"
            t_group.attrs["scale"] = value.scale
            t_group.attrs["format"] = "mjd"  # we store in this normalized form

            # Store meta as pickled bytes if present and non-empty
            if getattr(value, "meta", None):
                meta_pickled = pickle.dumps(dict(value.meta), protocol=pickle.HIGHEST_PROTOCOL)
                meta_ds = t_group.create_dataset(
                    "meta",
                    data=np.frombuffer(meta_pickled, dtype="uint8"),
                    compression=compression,
                    compression_opts=compression_opts,
                )
                meta_ds.attrs["pickled"] = True

        # datetime types -> ISO strings with type attrs
        elif isinstance(value, _dt.datetime):
            _store_string_with_type(h5group, key, value.isoformat(), "datetime.datetime")
        elif isinstance(value, _dt.date):
            _store_string_with_type(h5group, key, value.isoformat(), "datetime.date")
        elif isinstance(value, _dt.time):
            _store_string_with_type(h5group, key, value.isoformat(), "datetime.time")

        # numpy array
        elif isinstance(value, np.ndarray):
            if key in h5group:
                del h5group[key]
            h5group.create_dataset(
                key,
                data=value,
                compression=compression,
                compression_opts=compression_opts,
            )

        # numeric scalars
        elif isinstance(value, (int, float, bool, np.integer, np.floating, np.bool_)):
            if key in h5group:
                del h5group[key]
            h5group.create_dataset(
                key,
                data=value,
                compression=compression,
                compression_opts=compression_opts,
            )

        # strings -> vlen UTF-8
        elif isinstance(value, str):
            if key in h5group:
                del h5group[key]
            dt = h5py.string_dtype(encoding="utf-8")
            h5group.create_dataset(
                key,
                data=np.array(value, dtype=dt),
                dtype=dt,
                compression=compression,
                compression_opts=compression_opts,
            )

        # bytes-like -> vlen bytes (uint8)
        elif isinstance(value, (bytes, bytearray, memoryview)):
            if key in h5group:
                del h5group[key]
            arr = np.frombuffer(bytes(value), dtype="uint8")
            ds = h5group.create_dataset(
                key,
                data=arr,
                compression=compression,
                compression_opts=compression_opts,
            )
            ds.attrs["__bytes__"] = True

        # fallback: pickle
        else:
            if not pickle_objects:
                raise TypeError(
                    f"Unsupported type for key '{key}': {type(value)!r}. "
                    f"Enable pickle_objects=True to store via pickle."
                )
            if key in h5group:
                del h5group[key]
            pickled = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            ds = h5group.create_dataset(
                key,
                data=np.frombuffer(pickled, dtype="uint8"),
                compression=compression,
                compression_opts=compression_opts,
            )
            ds.attrs["pickled"] = True
            ds.attrs["python_type"] = str(type(value))
            ds.attrs["__type__"] = "pickle"

    def _store_string_with_type(
        h5group: h5py.Group,
        key: str,
        value: str,
        type_name: str,
    ) -> None:
        if key in h5group:
            del h5group[key]
        dt = h5py.string_dtype(encoding="utf-8")
        ds = h5group.create_dataset(
            key,
            data=np.array(value, dtype=dt),
            dtype=dt,
            compression=compression,
            compression_opts=compression_opts,
        )
        ds.attrs["__type__"] = type_name

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    with h5py.File(filepath, mode) as f:
        for k, v in data.items():
            if not isinstance(k, str):
                raise TypeError(f"HDF5 requires string keys; got key={k!r}")
            _write_item(f, k, v)


# ---------- LOAD ----------

def load_dict_from_hdf5(filepath: str) -> dict:
    """
    Load a dictionary previously stored with save_dict_to_hdf5.

    Reconstructs:
      - nested dicts/lists/tuples
      - numpy arrays/scalars
      - strings/bytes
      - datetime.datetime/date/time
      - astropy.time.Time
      - pickled objects (if present)
    """

    def _read_item(obj: Union[h5py.Group, h5py.Dataset]) -> Any:
        # If group, need to inspect children/attrs.
        if isinstance(obj, h5py.Group):
            # astropy Time group?
            if "__type__" in obj.attrs and obj.attrs["__type__"] == "astropy.time.Time":
                if not _HAS_ASTROPY:
                    raise ImportError(
                        "astropy is required to load astropy.time.Time objects."
                    )
                mjd = np.array(obj["mjd"][...], dtype="float64")
                scale = obj.attrs["scale"]
                # we stored normalized as mjd
                t = AstroTime(mjd, format="mjd", scale=scale)

                # restore meta if present
                if "meta" in obj:
                    meta_ds = obj["meta"]
                    if meta_ds.attrs.get("pickled", False):
                        meta_bytes = bytes(meta_ds[...].tolist())
                        meta = pickle.loads(meta_bytes)
                        t.meta.update(meta)
                return t

            # Otherwise treat as dict or list-like.
            # If keys are all integers from 0..n-1, treat as list.
            keys = list(obj.keys())
            if not keys:
                return {}
            try:
                int_keys = sorted(int(k) for k in keys)
                is_seq = int_keys == list(range(len(keys)))
            except ValueError:
                is_seq = False

            if is_seq:
                out = []
                for i in range(len(keys)):
                    out.append(_read_item(obj[str(i)]))
                return out
            else:
                out = {}
                for k in keys:
                    out[k] = _read_item(obj[k])
                return out

        # Dataset
        ds: h5py.Dataset = obj  # type: ignore[assignment]

        # check for special markers
        dtype_str = str(ds.dtype)

        # pickled object?
        if ds.attrs.get("__type__") == "pickle" or ds.attrs.get("pickled", False):
            arr = np.array(ds[...], dtype="uint8")
            return pickle.loads(arr.tobytes())

        # bytes dataset?
        if ds.attrs.get("__bytes__", False):
            arr = np.array(ds[...], dtype="uint8")
            return bytes(arr.tobytes())

        # type-tagged strings (datetime, etc.)
        tname = ds.attrs.get("__type__", None)
        if tname:
            value = ds.asstr()[()]
            if isinstance(value, np.ndarray):
                value = value.tolist()
            if tname == "datetime.datetime":
                return _dt.datetime.fromisoformat(value)
            elif tname == "datetime.date":
                return _dt.date.fromisoformat(value)
            elif tname == "datetime.time":
                return _dt.time.fromisoformat(value)
            # fall through for unknown types -> just return string

        # plain string dataset
        if h5py.check_string_dtype(ds.dtype) is not None:
            return ds.asstr()[()]

        # numeric / array dataset
        arr = ds[...]
        # unwrap scalars
        if arr.shape == ():
            return arr.item()
        return arr

    with h5py.File(filepath, "r") as f:
        result = {}
        for k in f.keys():
            result[k] = _read_item(f[k])
        return result