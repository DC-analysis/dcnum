import h5py
import numpy as np

from dcnum.feat import feat_texture

from helper_methods import retrieve_data


def test_basic_haralick():
    # This original file was generated with dcevent for reference.
    path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    # Make data available
    with h5py.File(path) as h5:
        ret_arr = feat_texture.haralick_texture_features(
            image=h5["events/image"][:],
            image_bg=h5["events/image_bg"][:],
            mask=h5["events/mask"][:],
        )

        assert np.allclose(ret_arr["tex_asm_avg"][1],
                           0.001514295993357114,
                           atol=0, rtol=1e-10)
        for feat in feat_texture.haralick_names:
            assert np.allclose(h5["events"][feat],
                               ret_arr[feat])
        # control test
        assert not np.allclose(h5["events"]["tex_asm_avg"],
                               ret_arr["tex_asm_ptp"])


def test_empty_image():
    masks = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=bool)[np.newaxis]
    image_corr = np.zeros(6*6, dtype=np.int16).reshape(1, 6, 6)
    tex = feat_texture.haralick_texture_features(
        image_corr=image_corr,
        mask=masks,
    )
    assert np.allclose(tex["tex_con_avg"][0], 0)


def test_empty_mask():
    masks = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=bool)[np.newaxis]
    image_corr = np.arange(6*6, dtype=np.int16).reshape(1, 6, 6)
    tex = feat_texture.haralick_texture_features(
        image_corr=image_corr,
        mask=masks,
    )
    assert np.isnan(tex["tex_con_avg"][0])


def test_1d_mask_image():
    masks = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=bool)[np.newaxis]
    image_corr = np.arange(6*6, dtype=np.int16).reshape(1, 6, 6)
    tex = feat_texture.haralick_texture_features(
        image_corr=image_corr,
        mask=masks,
    )
    assert np.isnan(tex["tex_con_avg"][0])


def test_nd_mask_with_1d_image():
    mask = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=bool)
    masks = np.stack([mask, mask, mask, mask])
    image_corr = np.arange(6*6, dtype=np.int16).reshape(1, 6, 6)
    tex = feat_texture.haralick_texture_features(
        image_corr=image_corr,
        mask=masks,
    )
    assert len(tex["tex_con_avg"]) == 4
    assert np.allclose(tex["tex_con_avg"][0], 27.75)


def test_simple_mask_image():
    masks = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=bool)[np.newaxis]
    image_corr = np.arange(6*6, dtype=np.int16).reshape(1, 6, 6)
    tex = feat_texture.haralick_texture_features(
        image_corr=image_corr,
        mask=masks,
    )
    assert np.allclose(tex["tex_con_avg"][0], 27.75)
