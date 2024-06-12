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
