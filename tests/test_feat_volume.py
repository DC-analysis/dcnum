import itertools

import h5py
import numpy as np
import pytest

from dcnum.feat.feat_contour.volume import volume_from_contours, vol_revolve
from dcnum.feat.feat_contour import moments_based_features

from helper_methods import retrieve_data


def area_of_polygon(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def centroid_of_polygon(points):
    """
    http://stackoverflow.com/a/14115494/190597 (mgamba)
    Centroid of polygon:
    http://en.wikipedia.org/wiki/Centroid#Centroid_of_polygon
    """
    area = area_of_polygon(*zip(*points))
    result_x = 0
    result_y = 0
    N = len(points)
    points = itertools.cycle(points)
    x1, y1 = next(points)
    for _i in range(N):
        x0, y0 = x1, y1
        x1, y1 = next(points)
        cross = (x0 * y1) - (x1 * y0)
        result_x += (x0 + x1) * cross
        result_y += (y0 + y1) * cross
    result_x /= (area * 6.0)
    result_y /= (area * 6.0)
    return abs(result_x), abs(result_y)


def get_ellipse_coords(a, b, x=0.0, y=0.0, angle=0.0, k=2):
    """

    Draws an ellipse using (360*k + 1) discrete points; based on pseudo code
    given at http://en.wikipedia.org/wiki/Ellipse

    k = 1 means 361 points (degree by degree)
    a = major axis distance,
    b = minor axis distance,
    x = offset along the x-axis
    y = offset along the y-axis
    angle = clockwise rotation [in degrees] of the ellipse;
        * angle=0  : the ellipse is aligned with the positive x-axis
        * angle=30 : rotated 30 degrees clockwise from positive x-axis
    """
    pts = np.zeros((360 * k + 1, 2))

    beta = -angle * np.pi / 180
    sin_beta = np.sin(beta)
    cos_beta = np.cos(beta)
    alpha = np.linspace(0, 2*np.pi, 360 * k + 1, endpoint=True)

    sin_alpha = np.sin(alpha)
    cos_alpha = np.cos(alpha)

    pts[:, 0] = x + (a * cos_alpha * cos_beta - b * sin_alpha * sin_beta)
    pts[::-1, 1] = y + (a * cos_alpha * sin_beta + b * sin_alpha * cos_beta)

    return pts


def test_volume_from_ellipse():
    # Helper definitions to get an ellipse
    major = 10
    minor = 5
    ellip = get_ellipse_coords(a=major,
                               b=minor,
                               x=minor,
                               y=5,
                               angle=0,
                               k=100)
    # obtain the centroid (corresponds to pos_x and pos_lat)
    cx, cy = centroid_of_polygon(ellip)
    events = volume_from_contours(contour=[ellip],
                                  pos_x=np.atleast_1d(cx),
                                  pos_y=np.atleast_1d(cy),
                                  pixel_size=1)

    # Analytic solution for volume of ellipsoid:
    v = 4 / 3 * np.pi * major * minor**2
    assert np.allclose(events["volume"], np.atleast_1d(v))


def test_volume_from_file():
    volume = np.array([
        2.56756546e-01, 2.09837048e+02, 1.91567167e+02, 3.86136009e-01,
        2.33808207e-01, 5.91524031e-01, 6.63264261e-01, 2.74712082e+02,
        1.99609918e+02, 2.57740837e+01, 4.33438475e-01, 9.34117316e+01,
        7.65374817e+01, 1.27440926e+01, 8.62865875e+01, 2.14351136e+02,
        4.50435063e+00, 1.84096969e+02, 2.32176623e-01, 2.29960786e+02,
        1.12255249e+02, 1.07491950e+02, 1.02322574e+02, 2.68966571e-01,
        2.45648180e+02, 9.09003162e-01, 1.01967418e+02, 9.74175326e+01,
        1.09564687e+02, 2.39239370e-01, 1.71917437e-01, 8.98323862e+01,
        4.13412223e+00, 2.91659170e+02, 2.00198054e+02, 1.97545320e+00,
        9.15408837e+01, 1.60965362e-01, 3.48553309e-01, 2.04561447e+02])
    path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")

    with h5py.File(path) as h5:
        pixel_size = h5.attrs["imaging:pixel size"]
        events = moments_based_features(
            mask=np.array(h5["events/mask"][:], dtype=bool),
            pixel_size=pixel_size,
            ret_contour=True,
        )

    vevents = volume_from_contours(contour=events["contour"],
                                   pos_x=events["pos_x"],
                                   pos_y=events["pos_y"],
                                   pixel_size=pixel_size,
                                   )
    assert np.allclose(vevents["volume"], volume,
                       atol=1e-6,
                       rtol=0)


def test_volume_from_none():
    vevents = volume_from_contours(contour=[None],
                                   pos_x=np.atleast_1d([10.5]),
                                   pos_y=np.atleast_1d([100.5]),
                                   pixel_size=0.2665,
                                   )
    assert np.all(np.isnan(vevents["volume"]))


@pytest.mark.parametrize("npoints,rtol", [[100, 6.72e-4],
                                          [1000, 6.6e-6]])
def test_vol_revolve_circular_toroid(npoints, rtol):
    """Upstream test function 1

    https://de.mathworks.com/matlabcentral/fileexchange/36525-volrevolve

    % Verification for circular toroid of major radius R0, minor radius a
    % Volume is 2 * pi^2 * R0 * a^2. Run this code:
    % clear all
    % R0 = 5 ;
    % a = 1 ;
    % npoints = 100 ;
    % theta = 2*pi*[0:1:npoints-1]'/double(npoints-1) ;
    % R = R0 + a*cos(theta) ;
    % Z =      a*sin(theta) ;
    % vol_analytic = 2 * pi^2 * R0 * a^2 ;
    %  >> 98.6960
    % vol = volRevolve(R,Z) ;
    %  >> 98.6298 (6.7e-04 relative error)
    % Do it again with npoints = 1000, get:
    %  >> 98.6954 (6.6e-06 relative error)
    % As expected, it's always slightly small because the polygon inscribes the
    % circle.
    """
    r0 = 5
    a = 1
    theta = 2 * np.pi * np.arange(npoints-1) / (npoints-1)
    r = r0 + a*np.cos(theta)
    z = a*np.sin(theta)
    vol_analytic = 2 * np.pi**2 * r0 * a**2
    vol = vol_revolve(r, z)
    assert np.allclose(vol_analytic, 98.6960, rtol=0, atol=0.001)
    assert np.allclose(vol_analytic, vol, rtol=rtol, atol=0)
    assert vol < vol_analytic


def test_vol_revolve_rectangular_toroid():
    """Upstream test function 2

    https://de.mathworks.com/matlabcentral/fileexchange/36525-volrevolve

    % Verification for washer (rectangular toroid), with the radius of the
    % 'hole' in the washer being a, and the outer radius of the washer being b.
    % (Thus the width of the metal cross section is b-a.) The height of the
    % washer is h. Then the volume is pi * (b^2 - a^2) * h. Run this code:
    clear all
    a = 1 ;
    b = 2 ;
    h = 10 ;
    R = [a; b; b; a; a] ;
    Z = [0; 0; h; h; 0] ;
    vol_analytic = pi * (b^2 - a^2) * h ;
    % >> 94.2478
    vol = volRevolve(R,Z) ;
    % >> 94.2478
    """
    a = 1
    b = 2
    h = 10
    r = [a, b, b, a, a]
    z = [0, 0, h, h, 0]
    vol_analytic = np.pi * (b**2 - a**2) * h
    vol = vol_revolve(r, z)
    assert vol_analytic == vol
    assert np.allclose(vol, 94.2478, rtol=0, atol=0.0001)
