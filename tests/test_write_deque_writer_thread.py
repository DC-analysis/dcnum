import collections

import h5py

from dcnum import write

from helper_methods import retrieve_data


def test_writer_thread_basic():
    path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path_wrt = path.with_name("written.hdf5")
    dq = collections.deque()

    wrthr = write.DequeWriterThread(path_out=path_wrt, dq=dq)
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

    with h5py.File(path_wrt) as ho:
        events = ho["events"]
        size = deform.shape[0]
        assert events["deform"].shape[0] == 2*size + 10
        assert events["image"].shape[0] == 2 * size + 10
        assert events["image"].shape[1:] == image.shape[1:]
