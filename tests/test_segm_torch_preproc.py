import numpy as np

from dcnum.segm.segm_torch import torch_preproc

import pytest

pytest.importorskip("torch")


def test_reshape_crop_both():
    image = np.arange(64, dtype=np.float32).reshape(8, 8)
    out = torch_preproc.preprocess_images(images=image,
                                          image_shape=(6, 6),
                                          norm_std=None,
                                          norm_mean=None,
                                          )
    assert out.shape == (1, 1, 6, 6)
    imout = out[0, 0, :, :]
    assert np.allclose(image[1:-1, 1:-1], imout)


def test_reshape_crop_height():
    image = np.arange(64, dtype=np.float32).reshape(8, 8)
    out = torch_preproc.preprocess_images(images=image,
                                          image_shape=(6, 8),
                                          norm_std=None,
                                          norm_mean=None,
                                          )
    assert out.shape == (1, 1, 6, 8)
    imout = out[0, 0, :, :]
    assert np.allclose(image[1:-1, :], imout)


def test_reshape_crop_width():
    image = np.arange(64, dtype=np.float32).reshape(8, 8)
    out = torch_preproc.preprocess_images(images=image,
                                          image_shape=(8, 6),
                                          norm_std=None,
                                          norm_mean=None,
                                          )
    assert out.shape == (1, 1, 8, 6)
    imout = out[0, 0, :, :]
    assert np.allclose(image[:, 1:-1], imout)


def test_reshape_pad_both():
    image = np.arange(64, dtype=np.float32).reshape(8, 8)
    out = torch_preproc.preprocess_images(images=image,
                                          image_shape=(10, 10),
                                          norm_std=None,
                                          norm_mean=None,
                                          )
    assert out.shape == (1, 1, 10, 10)
    imout = out[0, 0, :, :]
    median = np.median(image)
    assert np.allclose(imout[0, :], median)
    assert np.allclose(imout[-1, :], median)
    assert np.allclose(imout[:, 0], median)
    assert np.allclose(imout[:, -1], median)
    assert np.allclose(image, imout[1:-1, 1:-1])


def test_reshape_pad_height():
    image = np.arange(64, dtype=np.float32).reshape(8, 8)
    out = torch_preproc.preprocess_images(images=image,
                                          image_shape=(10, 8),
                                          norm_std=None,
                                          norm_mean=None,
                                          )
    assert out.shape == (1, 1, 10, 8)
    imout = out[0, 0, :, :]
    median = np.median(image)
    assert np.allclose(imout[0, :], median)
    assert np.allclose(imout[-1, :], median)
    assert np.allclose(image, imout[1:-1, :])


def test_reshape_pad_width_crop_height():
    image = np.arange(64, dtype=np.float32).reshape(8, 8)
    out = torch_preproc.preprocess_images(images=image,
                                          image_shape=(6, 10),
                                          norm_std=None,
                                          norm_mean=None,
                                          )
    assert out.shape == (1, 1, 6, 10)
    imout = out[0, 0, :, :]
    median = np.median(image)
    assert np.allclose(imout[:, 0], median)
    assert np.allclose(imout[:, -1], median)
    assert np.allclose(image[1:-1, :], imout[:, 1:-1])
