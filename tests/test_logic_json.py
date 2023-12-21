import json
import pathlib

import pytest

from dcnum import logic
import numpy as np


@pytest.mark.parametrize("input,output", [
    [pathlib.Path("/peter/pan"), str(pathlib.Path("/peter/pan"))],
    [1, int(1)],
    [np.int64(1), int(1)],
    [np.uint8(1), int(1)],
    [True, True],
    [np.bool_(True), True],
    ["hans", "hans"],
])
def test_json_encoder(input, output):
    dumped = json.dumps(input, cls=logic.ExtendedJSONEncoder)
    assert json.loads(dumped) == output
