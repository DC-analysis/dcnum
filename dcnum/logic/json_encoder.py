import json
import numbers
import pathlib

import numpy as np


class ExtendedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        """Extended JSON encoder for the **dcnum** logic

        This JSON encoder can handle the following additional objects:

        - ``pathlib.Path``
        - integer numbers
        - ``numpy`` boolean
        - slices (via "PYTHON-SLICE" identifier)
        """
        if isinstance(o, pathlib.Path):
            return str(o)
        elif isinstance(o, numbers.Integral):
            return int(o)
        elif isinstance(o, np.bool_):
            return bool(o)
        elif isinstance(o, slice):
            return "PYTHON-SLICE", (o.start, o.stop, o.step)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, o)
