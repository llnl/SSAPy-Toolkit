import json
from typing import Any
from datetime import datetime

import numpy as np
from astropy.time import Time


def save_json(filename: str, data: Any) -> None:
    """Save data to a JSON file with support for NumPy, datetime, and Astropy Time."""

    def encode(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return {"__type__": "ndarray", "data": obj.tolist()}
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, datetime):
            return {"__type__": "datetime", "data": obj.isoformat()}
        if isinstance(obj, Time):
            return {
                "__type__": "astropy_time",
                "data": obj.isot,
                "scale": obj.scale,
            }
        if isinstance(obj, set):
            return {"__type__": "set", "data": list(obj)}
        if isinstance(obj, tuple):
            return {"__type__": "tuple", "data": list(obj)}
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, default=encode, indent=4)


def load_json(filename: str) -> Any:
    """Load data from a JSON file and decode special types."""

    def decode(obj: Any) -> Any:
        if not isinstance(obj, dict) or "__type__" not in obj:
            return obj
        obj_type = obj["__type__"]
        if obj_type == "ndarray":
            return np.array(obj["data"])
        if obj_type == "datetime":
            return datetime.fromisoformat(obj["data"])
        if obj_type == "astropy_time":
            return Time(obj["data"], scale=obj["scale"])
        if obj_type == "set":
            return set(obj["data"])
        if obj_type == "tuple":
            return tuple(obj["data"])
        return obj

    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f, object_hook=decode)
