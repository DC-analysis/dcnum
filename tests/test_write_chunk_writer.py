import collections
import multiprocessing as mp

import h5py

from dcnum import write

from helper_methods import retrieve_data


mp_spawn = mp.get_context("spawn")


def test_writer_thread_basic():
    path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path_wrt = path.with_name("written.hdf5")
    write_queue_size = mp_spawn.Value("L", 0)

    dq = collections.deque()

    wrthr = write.ChunkWriter(path_out=path_wrt,
                              dq=dq,
                              write_queue_size=write_queue_size)
    wrthr.start()

    with h5py.File(path) as h5:
        deform = h5["events"]["deform"][:]
        image = h5["events"]["image"][:]

        dq.append(("deform", deform))
        dq.append(("deform", deform))
        dq.append(("deform", deform[:10]))

        dq.append(("image", image))
        dq.append(("image", image))
        dq.append(("image", image[:10]))

    wrthr.finished_when_queue_empty()
    wrthr.join()
    # At the end, the write queue should be empty again.
    assert write_queue_size.value == 0

    with h5py.File(path_wrt) as ho:
        events = ho["events"]
        size = deform.shape[0]
        assert events["deform"].shape[0] == 2*size + 10
        assert events["image"].shape[0] == 2 * size + 10
        assert events["image"].shape[1:] == image.shape[1:]


def test_writer_thread_error(monkeypatch):
    path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path_wrt = path.with_name("written.hdf5")
    write_queue_size = mp_spawn.Value("L", 0)

    def store_error(self, feat, data, group_name="events"):
        raise OSError("No space left on device")

    monkeypatch.setattr(write.HDF5Writer, "store_feature_chunk",
                        store_error)

    dq = collections.deque()
    wrthr = write.ChunkWriter(path_out=path_wrt,
                              dq=dq,
                              write_queue_size=write_queue_size)
    assert not wrthr.failed
    wrthr.start()
    with h5py.File(path) as h5:
        dq.append(("deform", h5["events"]["deform"][:]))
    wrthr.finished_when_queue_empty()
    wrthr.join()
    assert wrthr.failed
