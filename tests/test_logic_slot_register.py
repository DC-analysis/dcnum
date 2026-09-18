import multiprocessing as mp

from dcnum.read import HDF5Data, concatenated_hdf5_data
from dcnum.logic.slot_register import SlotRegister
from dcnum.logic.job import DCNumPipelineJob

import h5py

from helper_methods import retrieve_data


mp_spawn = mp.get_context("spawn")


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

    with h5py.File(path, "a") as h5:
        # rewrite the image column, making it chunk-less
        images = h5["events/image"][:]
        images_bg = h5["events/image_bg"][:]
        del h5["events/image"]
        del h5["events/image_bg"]
        h5.create_dataset("events/image",
                          data=images,
                          chunks=(49, 80, 320))
        h5.create_dataset("events/image_bg",
                          data=images_bg,
                          # Different chunks!
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

    with h5py.File(path, "a") as h5:
        # rewrite the image column, making it chunk-less
        images = h5["events/image"][:]
        images_bg = h5["events/image_bg"][:]
        del h5["events/image"]
        del h5["events/image_bg"]
        h5.create_dataset("events/image",
                          data=images,
                          chunks=(50, 80, 320))
        h5.create_dataset("events/image_bg",
                          data=images_bg,
                          # Different chunks!
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

    with h5py.File(path, "a") as h5:
        # rewrite the image column, making it chunk-less
        images = h5["events/image"][:]
        images_bg = h5["events/image_bg"][:]
        del h5["events/image"]
        del h5["events/image_bg"]
        h5.create_dataset("events/image",
                          data=images,
                          chunks=(49, 80, 320))
        h5.create_dataset("events/image_bg",
                          data=images_bg,
                          # Different chunks!
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

    with h5py.File(path, "a") as h5:
        # rewrite the image column, making it chunk-less
        images = h5["events/image"][:]
        images_bg = h5["events/image_bg"][:]
        del h5["events/image"]
        del h5["events/image_bg"]
        h5.create_dataset("events/image",
                          data=images,
                          chunks=(49, 80, 320))
        h5.create_dataset("events/image_bg",
                          data=images_bg,
                          # Different chunks!
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

    with h5py.File(path, "a") as h5:
        # rewrite the image column, making it chunk-less
        images = h5["events/image"][:]
        images_bg = h5["events/image_bg"][:]
        del h5["events/image"]
        del h5["events/image_bg"]
        h5.create_dataset("events/image",
                          data=images,
                          chunks=(50, 80, 320))
        h5.create_dataset("events/image_bg",
                          data=images_bg,
                          # Different chunks!
                          chunks=(50, 80, 320))

    hd = HDF5Data(path)

    job = DCNumPipelineJob(path_in=path)
    slot_register = SlotRegister(job=job, data=hd, num_slots=1)

    slot_register.task_load_all()
    assert slot_register.slots[0].length == 1000
    assert slot_register.slots[1].length == 1000
    assert slot_register.slots[0].chunk == 0
    assert slot_register.slots[1].chunk == 1
