import h5py
import numpy as np

from dcnum.feat import feat_brightness

from helper_methods import retrieve_data


def test_basic_brightness():
    # This original file was generated with dcevent for reference.
    path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    # Make data available
    with h5py.File(path) as h5:
        data = feat_brightness.brightness_features(
            image=h5["events/image"][:],
            image_bg=h5["events/image_bg"][:],
            mask=h5["events/mask"][:],
        )

        assert np.allclose(data["bright_bc_avg"][1],
                           -43.75497215592681,
                           atol=0, rtol=1e-10)
        for feat in feat_brightness.brightness_names:
            assert np.allclose(h5["events"][feat][:],
                               data[feat]), f"Feature {feat} mismatch!"
        # control test
        assert not np.allclose(h5["events"]["bright_perc_10"][:],
                               data["bright_perc_90"])


def test_basic_brightness_single_image():
    # This original file was generated with dcevent for reference.
    path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    # Make data available
    with h5py.File(path) as h5:
        data = feat_brightness.brightness_features(
            image=h5["events/image"][1][np.newaxis],
            image_bg=h5["events/image_bg"][1][np.newaxis],
            mask=h5["events/mask"][1][np.newaxis],
        )

        assert np.allclose(data["bright_bc_avg"][0],
                           -43.75497215592681,
                           atol=0, rtol=1e-10)
        for feat in feat_brightness.brightness_names:
            assert np.allclose(h5["events"][feat][1],
                               data[feat][0]), f"Feature {feat} mismatch!"
        # control test
        assert not np.allclose(h5["events"]["bright_perc_10"][1],
                               data["bright_perc_90"][0])


def test_brightness_empty_mask():
    """Brightness features of an empty mask are NaN (dcnum 0.31.2 crashed)"""
    image = np.full((1, 10, 10), 100, dtype=np.uint8)
    image_bg = np.full((1, 10, 10), 110, dtype=np.uint8)
    mask = np.zeros((2, 10, 10), dtype=bool)
    mask[1, 2:6, 2:6] = True
    data = feat_brightness.brightness_features(image=image,
                                               mask=mask,
                                               image_bg=image_bg)
    for key in ["bright_avg", "bright_sd", "bright_bc_avg", "bright_bc_sd",
                "bright_perc_10", "bright_perc_90"]:
        assert np.isnan(data[key][0]), key
        assert not np.isnan(data[key][1]), key
    assert data["bright_avg"][1] == 100
    assert data["bright_bc_avg"][1] == -10
    assert np.all(data["bg_med"] == 110)
