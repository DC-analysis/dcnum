import h5py
import numpy as np
import pytest

from dcnum import logic
from dcnum.feat import QueueEventExtractor

from helper_methods import retrieve_data


def test_extractor_default_kwargs():
    """Brightness and Haralick features are computed by default"""
    qee = QueueEventExtractor(slot_register=None,
                              pixel_size=0.2645,
                              gate=None,
                              event_queue=None)
    assert qee.extract_kwargs == {"brightness": True, "haralick": True}
    assert qee.get_ppid() == "legacy:b=1^h=1^v=1"


def test_extractor_get_ppid():
    qee = QueueEventExtractor(slot_register=None,
                              pixel_size=0.2645,
                              gate=None,
                              event_queue=None,
                              extract_kwargs={"haralick": False,
                                              "volume": False})
    assert qee.get_ppid() == "legacy:b=1^h=0^v=0"


def test_extractor_get_ppkw_from_ppid_invalid_code():
    with pytest.raises(ValueError,
                       match="Could not find extraction method 'peter'"):
        QueueEventExtractor.get_ppkw_from_ppid("peter:b=1^h=0^v=1")


def test_extractor_process_label_missing_slot():
    """The chunk of the requested index must be in a chunk slot"""
    class SlotRegister(list):
        chunk_size = 100

    qee = QueueEventExtractor(slot_register=SlotRegister(),
                              pixel_size=0.2645,
                              gate=None,
                              event_queue=None)
    with pytest.raises(ValueError, match="Could not find slot for chunk=2"):
        qee.process_label(index=250)


@pytest.mark.parametrize("online_gates", [True, False])
def test_extractor_box_gates(online_gates):
    """Events outside the online box gates are removed"""
    path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    deform_min = 0.02
    deform_max = 0.05
    with h5py.File(path, "a") as h5:
        h5.attrs["online_filter:deform min"] = deform_min
        h5.attrs["online_filter:deform max"] = deform_max
        h5.attrs["online_filter:deform soft limit"] = False

    job = logic.DCNumPipelineJob(path_in=path,
                                 gate_kwargs={"online_gates": online_gates},
                                 debug=True)
    with logic.DCNumJobRunner(job=job) as runner:
        runner.run()

    with h5py.File(job["path_out"]) as h5:
        deform = h5["events/deform"][:]
        outside = (deform < deform_min) | (deform > deform_max)
        assert len(deform)
        if online_gates:
            assert not np.any(outside)
        else:
            # make sure the gate actually removes events
            assert np.any(outside)
