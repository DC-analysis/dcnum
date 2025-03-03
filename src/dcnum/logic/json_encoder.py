import json
import numbers
import pathlib

import numpy as np


class ExtendedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        """Extended JSON encoder for the **dcnum** logic

        This JSON encoder can handle the following additional objects:

        - ``pathlib.Path``
        - integer numbers
        - ``numpy`` boolean
        - slices (via "PYTHON-SLICE" identifier)
        """
        if isinstance(obj, pathlib.Path):
            return str(obj)
        elif isinstance(obj, numbers.Integral):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, slice):
            return "PYTHON-SLICE", (obj.start, obj.stop, obj.step)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)
