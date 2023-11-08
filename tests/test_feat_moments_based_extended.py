import pathlib

import h5py
import numpy as np

from dcnum.feat import feat_moments

from helper_methods import retrieve_data

data_path = pathlib.Path(__file__).parent / "data"


def test_moments_based_features():
    # This file has new cell features belonging to
    # fmt-hdf5_cytoshot_full-features_2023.zip
    path = retrieve_data(data_path /
                         "fmt-hdf5_cytoshot_extended-moments-features.zip")

    feats = [
            "area_um_raw",
            "deform_raw",
            "eccentr_prnc",
            "per_ratio",
            "per_um_raw",
    ]

    # Make data available
    with h5py.File(path) as h5:
        data = feat_moments.moments_based_features(
            mask=h5["events/mask"][:],
            pixel_size=0.2645
        )
        for feat in feats:
            rtol = 1e-5
            atol = 1e-8
            assert np.allclose(h5["events"][feat][:],
                               data[feat],
                               rtol=rtol,
                               atol=atol), f"Feature {feat} mismatch!"


def test_mask_2d():
    masks = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=bool)[np.newaxis]
    data = feat_moments.moments_based_features(
                mask=masks,
                pixel_size=0.2645
            )
    assert data["deform_raw"].shape == (1,)
    # This is the deformation of a square (compared to circle)
    assert np.allclose(data["deform_raw"][0], 0.11377307454724206)
    # Without moments-based computation, this would be 4*pxsize=0.066125
    assert np.allclose(data["area_um_raw"][0], 0.06996025)
