import numpy as np

from dcnum.read import concatenated_hdf5_data, detect_flickering

from helper_methods import retrieve_data


def test_detect_flicker_basic():
    image_data = np.full((500, 80, 320), 145)
    flicker_indices = [4, 9, 10, 23, 439]
    for idx in flicker_indices:
        image_data[idx] += 5
    assert detect_flickering(image_data,
                             roi_height=10,
                             brightness_threshold=5,
                             count_threshold=5,
                             max_frames=500,
                             )

    assert not detect_flickering(image_data,
                                 roi_height=10,
                                 brightness_threshold=5,
                                 count_threshold=6,  # threshold too low
                                 max_frames=500,
                                 )

    assert not detect_flickering(image_data,
                                 roi_height=10,
                                 brightness_threshold=6,  # threshold too low
                                 count_threshold=5,
                                 max_frames=500,
                                 )

    assert not detect_flickering(image_data,
                                 roi_height=10,
                                 brightness_threshold=5,
                                 count_threshold=5,
                                 max_frames=400,  # too few frames
                                 )


def test_detect_flicker_hdf5data_instance():
    path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path_out = path.with_name("input.rtdc")
    # create simple concatenated dataset, repeating a file
    with concatenated_hdf5_data([path]*25, path_out=path_out) as hd:
        assert len(hd) == 1000
        assert not detect_flickering(hd.image)
        assert detect_flickering(hd.image, brightness_threshold=1)


def test_detect_flicker_none():
    image_data = np.full((500, 80, 320), 145)
    assert not detect_flickering(image_data,
                                 roi_height=10,
                                 brightness_threshold=5,
                                 count_threshold=5,
                                 max_frames=500,
                                 )


def test_detect_flicker_not_outside_roi():
    image_data = np.full((500, 80, 320), 145)
    flicker_indices = [4, 9, 10, 23, 439]
    for idx in flicker_indices:
        # only modify data outside the ROI
        image_data[idx, 11:, :] += 5
    assert not detect_flickering(image_data,
                                 roi_height=10,
                                 brightness_threshold=5,
                                 count_threshold=5,
                                 max_frames=500,
                                 )


def test_detect_flicker_only_inside_roi():
    image_data = np.full((500, 80, 320), 145)
    flicker_indices = [4, 9, 10, 23, 439]
    for idx in flicker_indices:
        # only modify data inside the ROI
        image_data[idx, :10, :] += 5
    assert detect_flickering(image_data,
                             roi_height=10,
                             brightness_threshold=5,
                             count_threshold=5,
                             max_frames=500,
                             )
