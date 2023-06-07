import pathlib

import h5py
import numpy as np

from dcnum.feat import brightness

from helper_methods import retrieve_data

data_path = pathlib.Path(__file__).parent / "data"


def test_basic_brightness():
    # This original file was generated with dcevent for reference.
    path = retrieve_data(data_path /
                         "fmt-hdf5_cytoshot_full-features_2023.zip")
    # Make data available
    with h5py.File(path) as h5:
        data = brightness.brightness_features(
            image=h5["events/image"][:],
            image_bg=h5["events/image_bg"][:],
            mask=h5["events/mask"][:],
        )

        assert np.allclose(data["bright_bc_avg"][1],
                           -43.75497215592681,
                           atol=0, rtol=1e-10)
        for feat in brightness.brightness_names:
            assert np.allclose(h5["events"][feat],
                               data[feat])
        # control test
        assert not np.allclose(h5["events"]["bright_perc_10"],
                               data["bright_perc_90"])