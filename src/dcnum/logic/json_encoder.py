import json
import numbers
import pathlib

import numpy as np


class ExtendedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pathlib.Path):
            return str(obj)
        elif isinstance(obj, numbers.Integral):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)
