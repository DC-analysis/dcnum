import pathlib

import h5py
import numpy as np

from dcnum.feat import moments

from helper_methods import retrieve_data

data_path = pathlib.Path(__file__).parent / "data"


def test_moments_based_features():
    # This original file was generated with dcevent for reference.
    path = retrieve_data(data_path /
                         "fmt-hdf5_cytoshot_full-features_2023.zip")
    feats = [
        "deform",
        "size_x",
        "size_y",
        "pos_x",
        "pos_y",
        "area_msd",
        "area_ratio",
        "area_um",
        "aspect",
        "tilt",
        "inert_ratio_cvx",
        "inert_ratio_raw",
        "inert_ratio_prnc",
    ]

    # Make data available
    with h5py.File(path) as h5:
        data = moments.moments_based_features(
            mask=h5["events/mask"][:],
            pixel_size=0.2645
        )
        for feat in feats:
            assert np.allclose(h5["events"][feat],
                               data[feat])
        # control test
        assert not np.allclose(h5["events"]["inert_ratio_cvx"],
                               data["tilt"])