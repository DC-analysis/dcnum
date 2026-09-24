import multiprocessing as mp

from dcnum.read import HDF5Data, concatenated_hdf5_data
from dcnum.logic.slot_register import SlotRegister
from dcnum.logic.job import DCNumPipelineJob

import h5py
import numpy as np

import pytest

from helper_methods import retrieve_data


mp_spawn = mp.get_context("spawn")


def test_slot_register_increment_counter_value():
    path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    hd = HDF5Data(path)
    assert "image" in hd

    job = DCNumPipelineJob(path_in=path)
    sr = SlotRegister(job=job, data=hd, num_slots=1)

    assert sr.get_counter_value("chunks_loaded") == 0
    sr.increment_counter_value("chunks_loaded", 1)
    sr.increment_counter_value("chunks_loaded", 2)
    assert sr.get_counter_value("chunks_loaded") == 3

    lock = sr.get_counter_lock("chunks_loaded")
    with pytest.raises(ValueError, match="A locked `lock` must be passed"):
        sr.increment_counter_value("chunks_loaded", 1, lock=lock)

    with lock:
        sr.increment_counter_value("chunks_loaded", 1, lock=lock)
    assert sr.get_counter_value("chunks_loaded") == 4

    with lock, pytest.raises(TimeoutError, match="Failed to increment"):
        sr.increment_counter_value("chunks_loaded", 1, timeout=0.01)

    with pytest.raises(KeyError, match="No counter defined"):
        sr.increment_counter_value("peterpan", 1)


def test_slot_register_reserve_slot_for_task():
    path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    hd = HDF5Data(path)
    assert "image" in hd

    print("Setting up pipeline job")
    job = DCNumPipelineJob(path_in=path)
    slot_register = SlotRegister(job=job, data=hd, num_slots=1)

    warden = slot_register.reserve_slot_for_task(current_state="i",
                                                 next_state="s")
    assert warden is not None
    with warden as (cs, batch_range):
        assert cs.state == "i"
        assert batch_range == (0, 40)

        # We only have one slot, this means requesting the same thing will
        # not work.
        warden2 = slot_register.reserve_slot_for_task(current_state="i",
                                                      next_state="s")
        assert warden2 is None

    assert cs.state == "s"

    warden3 = slot_register.reserve_slot_for_task(current_state="s",
                                                  next_state="e")
    assert warden3 is not None


def test_slot_register_chunks_with_odd_remainder():
    """Test number of chunks"""
    h5path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path = h5path.with_name("test.hdf5")
    with concatenated_hdf5_data(50 * [h5path], path_out=path):
        pass

    # Make chunked dataset
    with h5py.File(path, "a") as h5:
        images = h5["events/image"][:]
        images_bg = h5["events/image_bg"][:]
        del h5["events/image"]
        del h5["events/image_bg"]
        h5.create_dataset("events/image",
                          data=images,
                          chunks=(49, 80, 320))
        h5.create_dataset("events/image_bg",
                          data=images_bg,
                          chunks=(49, 80, 320))

    hd = HDF5Data(path)

    job = DCNumPipelineJob(path_in=path)
    slot_register = SlotRegister(job=job, data=hd, num_slots=1)

    # With a chunk size of 49, we forced the image data to load in
    # chunks of 980 (20*49) just below the 1000 mark. We will have
    # two chunks, thus 40 is left for the remainder chunk.
    assert len(slot_register.slots) == 2
    assert slot_register.num_chunks == 3
    assert slot_register.chunk_size == 980
    assert slot_register.slots[0].length == 980
    assert slot_register.slots[1].length == 40
    assert not slot_register.slots[0].is_remainder
    assert slot_register.slots[1].is_remainder


def test_slot_register_chunks_with_even_remainder():
    """Test number of chunks"""
    h5path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path = h5path.with_name("test.hdf5")
    with concatenated_hdf5_data(50 * [h5path], path_out=path):
        pass

    # Make chunked dataset
    with h5py.File(path, "a") as h5:
        images = h5["events/image"][:]
        images_bg = h5["events/image_bg"][:]
        del h5["events/image"]
        del h5["events/image_bg"]
        h5.create_dataset("events/image",
                          data=images,
                          chunks=(50, 80, 320))
        h5.create_dataset("events/image_bg",
                          data=images_bg,
                          chunks=(50, 80, 320))

    hd = HDF5Data(path)

    job = DCNumPipelineJob(path_in=path)
    slot_register = SlotRegister(job=job, data=hd, num_slots=1)

    # With an image chunk size of 50, we hit 1000 exactly. There is still
    # a remainder chunk, since the entire dataset is larger than one chunk.
    assert slot_register.num_chunks == 2
    assert slot_register.chunk_size == 1000
    assert slot_register.slots[0].length == 1000
    assert slot_register.slots[1].length == 1000
    assert not slot_register.slots[0].is_remainder
    assert slot_register.slots[1].is_remainder


def test_slot_register_task_load_all_with_odd_remainder():
    """Test number of chunks"""
    h5path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path = h5path.with_name("test.hdf5")
    with concatenated_hdf5_data(50 * [h5path], path_out=path):
        pass

    # Make chunked dataset
    with h5py.File(path, "a") as h5:
        images = h5["events/image"][:]
        images_bg = h5["events/image_bg"][:]
        del h5["events/image"]
        del h5["events/image_bg"]
        h5.create_dataset("events/image",
                          data=images,
                          chunks=(49, 80, 320))
        h5.create_dataset("events/image_bg",
                          data=images_bg,
                          chunks=(49, 80, 320))

    hd = HDF5Data(path)

    job = DCNumPipelineJob(path_in=path)
    slot_register = SlotRegister(job=job, data=hd, num_slots=2)

    slot_register.task_load_all()
    assert slot_register.slots[0].length == 980
    assert slot_register.slots[1].length == 980
    assert slot_register.slots[2].length == 40
    assert slot_register.slots[0].chunk == 0
    assert slot_register.slots[1].chunk == 1
    assert slot_register.slots[2].chunk == 2


def test_slot_register_task_load_all_with_odd_remainder_multi():
    """Test number of chunks"""
    h5path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path = h5path.with_name("test.hdf5")
    with concatenated_hdf5_data(100 * [h5path], path_out=path):
        pass

    # Make chunked dataset
    with h5py.File(path, "a") as h5:
        images = h5["events/image"][:]
        images_bg = h5["events/image_bg"][:]
        del h5["events/image"]
        del h5["events/image_bg"]
        h5.create_dataset("events/image",
                          data=images,
                          chunks=(49, 80, 320))
        h5.create_dataset("events/image_bg",
                          data=images_bg,
                          chunks=(49, 80, 320))

    hd = HDF5Data(path)

    job = DCNumPipelineJob(path_in=path)
    slot_register = SlotRegister(job=job, data=hd, num_slots=2)

    assert slot_register.task_load_all()
    assert slot_register.slots[0].length == 980
    assert slot_register.slots[1].length == 980
    assert slot_register.slots[2].length == 80
    assert slot_register.slots[0].chunk == 0
    assert slot_register.slots[1].chunk == 1
    assert slot_register.slots[2].chunk == -1

    # reset the state to "i" so data are loaded again
    for cs in slot_register.slots:
        cs.state = "i"

    assert slot_register.task_load_all()
    assert slot_register.slots[0].length == 980
    assert slot_register.slots[1].length == 980
    assert slot_register.slots[2].length == 80
    assert slot_register.slots[0].chunk == 2
    assert slot_register.slots[1].chunk == 3

    # reset the state to "i" so the remainder chunk gets loaded
    for cs in slot_register.slots:
        cs.state = "i"

    slot_register.task_load_all()
    assert slot_register.slots[2].chunk == 4


def test_slot_register_task_load_all_with_even_remainder():
    """Test number of chunks"""
    h5path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path = h5path.with_name("test.hdf5")
    with concatenated_hdf5_data(50 * [h5path], path_out=path):
        pass

    # make chunked dataset
    with h5py.File(path, "a") as h5:
        images = h5["events/image"][:]
        images_bg = h5["events/image_bg"][:]
        del h5["events/image"]
        del h5["events/image_bg"]
        h5.create_dataset("events/image",
                          data=images,
                          chunks=(50, 80, 320))
        h5.create_dataset("events/image_bg",
                          data=images_bg,
                          chunks=(50, 80, 320))

    hd = HDF5Data(path)

    job = DCNumPipelineJob(path_in=path)
    slot_register = SlotRegister(job=job, data=hd, num_slots=1)

    slot_register.task_load_all()
    assert slot_register.slots[0].length == 1000
    assert slot_register.slots[1].length == 1000
    assert slot_register.slots[0].chunk == 0
    assert slot_register.slots[1].chunk == 1


def test_slot_register_segment_images():
    """Segment images"""
    h5path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path = h5path.with_name("test.hdf5")
    with concatenated_hdf5_data(50 * [h5path], path_out=path):
        pass

    # Make chunked dataset
    with h5py.File(path, "a") as h5:
        images = h5["events/image"][:]
        images_bg = h5["events/image_bg"][:]
        del h5["events/image"]
        del h5["events/image_bg"]
        h5.create_dataset("events/image",
                          data=images,
                          chunks=(49, 80, 320))
        h5.create_dataset("events/image_bg",
                          data=images_bg,
                          chunks=(49, 80, 320))

    hd = HDF5Data(path)

    job = DCNumPipelineJob(path_in=path)
    slot_register = SlotRegister(job=job, data=hd, num_slots=2)

    # load data into chunks
    slot_register.task_load_all()
    assert slot_register.slots[0].length == 980
    assert slot_register.slots[1].length == 980
    assert slot_register.slots[2].length == 40
    assert slot_register.slots[0].chunk == 0
    assert slot_register.slots[1].chunk == 1
    assert slot_register.slots[2].chunk == 2

    # segment images
    assert slot_register.task_segment_images()
    assert slot_register.task_segment_images()
    assert slot_register.task_segment_images()
    assert not slot_register.task_segment_images()
    assert np.sum(slot_register.slots[0].mask) == 1260920
    assert np.sum(slot_register.slots[1].mask) == 1259101
    assert np.sum(slot_register.slots[2].mask) == 51429
    masks = np.concatenate([
        slot_register.slots[0].mask,
        slot_register.slots[1].mask,
        slot_register.slots[2].mask[:40],
    ])
    assert masks.shape == (2000, 80, 400)
    assert np.sum(masks) == 2571450

    # reset the masks
    slot_register.slots[0].mask[:] = False
    slot_register.slots[1].mask[:] = False
    slot_register.slots[2].mask[:] = False
    assert np.sum(slot_register.slots[0].mask) == 0
    assert np.sum(slot_register.slots[1].mask) == 0
    assert np.sum(slot_register.slots[2].mask) == 0

    # put slot data into "s" state
    slot_register.slots[0].state = "s"
    slot_register.slots[1].state = "s"
    slot_register.slots[2].state = "s"

    # segment images in chunks (done in while loop)
    assert slot_register.task_segment_images_full_chunk()
    assert not slot_register.task_segment_images_full_chunk()
    masks2 = np.concatenate([
        slot_register.slots[0].mask,
        slot_register.slots[1].mask,
        slot_register.slots[2].mask[:40],
    ])
    assert masks2.shape == (2000, 80, 400)
    assert np.sum(masks2) == 2571450

    assert np.all(masks == masks2)
