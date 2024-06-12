import h5py
import numpy as np
import scipy.ndimage as ndi

from dcnum.feat import feat_contour

from helper_methods import retrieve_data


def test_inert_ratio_prnc():
    """Test tilt and equivalence of inert_ratio_raw and inert_ratio_prnc"""
    t = np.linspace(0, 2*np.pi, 3000)

    x1 = 80 * np.cos(t)
    y1 = 90 * np.sin(t)
    offset = 120

    # create a mask from the contour
    mask = np.zeros((250, 250), dtype=bool)
    mask[np.array(np.round(x1 + offset), dtype=int),
         np.array(np.round(y1 + offset), dtype=int)] = True
    mask = ndi.binary_fill_holes(mask)

    # sanity check that fill_holes worked
    assert np.sum(mask) > 22000

    data = feat_contour.moments_based_features(
        mask=mask[np.newaxis],
        pixel_size=0.2645
    )

    raw = data["inert_ratio_raw"][0]
    assert np.allclose(raw, 1.125606191144217)

    phi = np.arctan2(y1, x1)
    rho = np.sqrt(x1**2 + y1**2)

    for theta in np.linspace(0.1, 2*np.pi, 14):  # arbitrary rotation
        for pos_x in np.linspace(-5, 20, 8):  # arbitrary x shift
            for pos_y in np.linspace(-4.6, 17, 4):  # arbitrary y shift
                x2 = rho * np.cos(phi + theta) + pos_x
                y2 = rho * np.sin(phi + theta) + pos_y

                maskij = np.zeros((350, 350), dtype=bool)
                maskij[np.array(np.round(y2) + offset + 20, dtype=int),
                       np.array(np.round(x2) + offset + 20, dtype=int)] = True
                maskij = ndi.binary_fill_holes(maskij)

                # sanity check that fill_holes worked
                assert np.sum(mask) > 22000

                # sanity checks (nothing at boundary)
                assert not np.any(maskij[0, :])
                assert not np.any(maskij[-1, :])
                assert not np.any(maskij[:, 0])
                assert not np.any(maskij[:, -1])

                dataij = feat_contour.moments_based_features(
                    mask=maskij[np.newaxis],
                    pixel_size=0.2645
                )

                prnc = dataij["inert_ratio_prnc"][0]
                tilt = dataij["tilt"][0]

                angle = np.pi/2 - theta
                angle = np.mod(angle, np.pi)
                if angle > np.pi / 2:
                    angle -= np.pi
                angle = np.abs(angle)

                assert np.allclose(raw, prnc, rtol=0, atol=3e-3)
                assert np.allclose(angle, tilt, rtol=0, atol=1e-2)


def test_inert_ratio_prnc_simple_1():
    c = np.array([[0, 0],
                  [0, 1],
                  [0, 2],
                  [1, 2],
                  [2, 2],
                  [3, 2],
                  [3, 1],
                  [3, 0],
                  [2, 0],
                  [1, 0],
                  [0, 0]])

    # create a mask from the contour
    mask = np.zeros((20, 20), dtype=bool)

    mask[c[:, 1] + 10, c[:, 0] + 12] = True
    mask = ndi.binary_fill_holes(mask)

    # sanity check that fill_holes worked
    assert np.sum(mask) == 12

    data = feat_contour.moments_based_features(
        mask=mask[np.newaxis],
        pixel_size=0.2645
    )

    raw = data["inert_ratio_raw"][0]
    prnc = data["inert_ratio_prnc"][0]
    tilt = data["tilt"][0]

    assert np.allclose(raw, 1.5)
    assert np.allclose(prnc, 1.5)
    assert np.allclose(tilt, 0)


def test_inert_ratio_prnc_simple_2():
    c = np.array([[0, 0],
                  [1, 1],
                  [2, 2],
                  [3, 3],
                  [4, 2],
                  [5, 1],
                  [4, 0],
                  [3, -1],
                  [2, -2],
                  [1, -1],
                  [0, 0]])

    # create a mask from the contour
    mask = np.zeros((20, 20), dtype=bool)

    mask[c[:, 1] + 10, c[:, 0] + 12] = True
    mask = ndi.binary_fill_holes(mask)

    # sanity check that fill_holes worked
    assert np.sum(mask) == 18

    data = feat_contour.moments_based_features(
        mask=mask[np.newaxis],
        pixel_size=0.2645
    )

    raw = data["inert_ratio_raw"][0]
    prnc = data["inert_ratio_prnc"][0]
    tilt = data["tilt"][0]

    assert np.allclose(raw, 1)
    assert np.allclose(prnc, 1.5)
    assert np.allclose(tilt, np.pi/4)


def test_moments_based_features():
    # This original file was generated with dcevent for reference.
    path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
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
        data = feat_contour.moments_based_features(
            mask=h5["events/mask"][:],
            pixel_size=0.2645
        )
        for feat in feats:
            if feat.count("inert"):
                rtol = 2e-5
                atol = 1e-8
            else:
                rtol = 1e-5
                atol = 1e-8
            assert np.allclose(h5["events"][feat][:],
                               data[feat],
                               rtol=rtol,
                               atol=atol), f"Feature {feat} mismatch!"
        # control test
        assert not np.allclose(h5["events"]["inert_ratio_cvx"][:],
                               data["tilt"])


def test_mask_0d():
    masks = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=bool)[np.newaxis]
    data = feat_contour.moments_based_features(
        mask=masks,
        pixel_size=0.2645
    )
    assert data["deform"].shape == (1,)
    assert np.isnan(data["deform"][0])
    assert np.isnan(data["area_um"][0])


def test_mask_1d():
    masks = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=bool)[np.newaxis]
    data = feat_contour.moments_based_features(
        mask=masks,
        pixel_size=0.2645
    )
    assert data["deform"].shape == (1,)
    assert np.isnan(data["deform"][0])
    assert np.isnan(data["area_um"][0])


def test_mask_1d_large():
    masks = np.array([
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
    ], dtype=bool)[np.newaxis]
    data = feat_contour.moments_based_features(
        mask=masks,
        pixel_size=0.2645
    )
    assert data["deform"].shape == (1,)
    assert np.isnan(data["deform"][0])
    assert np.isnan(data["area_um"][0])


def test_mask_1d_large_no_border():
    masks = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=bool)[np.newaxis]
    data = feat_contour.moments_based_features(
        mask=masks,
        pixel_size=0.2645
    )
    assert data["deform"].shape == (1,)
    assert np.isnan(data["deform"][0])
    assert np.isnan(data["area_um"][0])


def test_mask_2d():
    masks = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=bool)[np.newaxis]
    data = feat_contour.moments_based_features(
        mask=masks,
        pixel_size=0.2645
    )
    assert data["deform"].shape == (1,)
    # This is the deformation of a square (compared to circle)
    assert np.allclose(data["deform"][0], 0.11377307454724206)
    # Without moments-based computation, this would be 4*pxsize=0.066125
    assert np.allclose(data["area_um"][0], 0.06996025)


def test_mask_mixed():
    mask_valid = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=bool)
    mask_invalid = np.array([
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
    ], dtype=bool)
    mixed_masks = np.append(mask_valid[None, ...],
                            mask_invalid[None, ...], axis=0)
    data = feat_contour.moments_based_features(
        mask=mixed_masks,
        pixel_size=0.2645)
    assert data["deform"].shape == (2,)
    assert np.all(data["valid"][:] == np.array([True, False]))
    assert not np.isnan(data["deform"][0])
    assert np.isnan(data["deform"][1])


def test_tilt():
    t = np.linspace(0, 2*np.pi, 3000)

    x1 = 90 * np.cos(t)
    y1 = 30 * np.sin(t)

    phi = np.arctan2(x1, y1)
    rho = np.sqrt(x1**2 + y1**2)

    for theta in np.linspace(-.3, 2.2*np.pi, 32):  # arbitrary rotation
        x2 = rho * np.cos(phi + theta)
        y2 = rho * np.sin(phi + theta)

        # create a mask from the contour
        mask = np.zeros((320, 320), dtype=bool)
        mask[np.array(np.round(x2 + 120), dtype=int),
             np.array(np.round(y2 + 120), dtype=int)] = True
        mask = ndi.binary_fill_holes(mask)

        # sanity check
        assert np.sum(mask) > 8000
        # sanity checks (nothing at boundary)
        assert not np.any(mask[0, :])
        assert not np.any(mask[-1, :])
        assert not np.any(mask[:, 0])
        assert not np.any(mask[:, -1])

        data = feat_contour.moments_based_features(
            mask=mask[np.newaxis],
            pixel_size=0.2645
        )
        tilt = data["tilt"][0]

        th = np.mod(theta, np.pi)
        if th > np.pi/2:
            th -= np.pi
        th = np.abs(th)
        assert np.allclose(tilt, th, rtol=0, atol=4e-3)
