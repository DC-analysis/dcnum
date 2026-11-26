"""Make sure that the feature extraction stays reproducible

If one of these tests fails because of an intended change of a feature
computation, the results of the pipeline change for unchanged pipeline
identifiers. In this case, increment `DCNUM_PPID_GENERATION` (see the
section "Pipeline generation" in the docs) and regenerate the reference
values with::

    python tests/test_feat_reproducibility.py
"""
import itertools
import json
import pathlib
import types

import h5py
import numpy as np
import pytest

from dcnum import logic, read
from dcnum.feat import Gate, QueueEventExtractor

from helper_methods import retrieve_data


#: reference feature values computed with :func:`compute_features`
PATH_REFERENCE = (pathlib.Path(__file__).parent / "data"
                  / "feat_reference_values.json")

#: scalar features computed with all feature kwargs enabled
FEATURES_ALL = [
    "area_msd", "area_ratio", "area_um", "area_um_raw", "aspect",
    "bg_med", "bright_avg", "bright_bc_avg", "bright_bc_sd",
    "bright_perc_10", "bright_perc_90", "bright_sd",
    "deform", "deform_raw", "eccentr_prnc",
    "inert_ratio_cvx", "inert_ratio_prnc", "inert_ratio_raw",
    "per_ratio", "per_um_raw", "pos_x", "pos_y", "size_x", "size_y",
    "tex_asm_avg", "tex_asm_ptp", "tex_con_avg", "tex_con_ptp",
    "tex_cor_avg", "tex_cor_ptp", "tex_den_avg", "tex_den_ptp",
    "tex_ent_avg", "tex_ent_ptp", "tex_f12_avg", "tex_f12_ptp",
    "tex_f13_avg", "tex_f13_ptp", "tex_idm_avg", "tex_idm_ptp",
    "tex_sen_avg", "tex_sen_ptp", "tex_sva_avg", "tex_sva_ptp",
    "tex_var_avg", "tex_var_ptp",
    "tilt", "volume",
]


def compute_features(**extract_kwargs):
    """Compute the features of all events of the test dataset

    The masks, images, and background images are taken from the test
    dataset "fmt-hdf5_cytoshot_full-features_2023.zip" (40 events),
    so that the result does not depend on background computation or
    segmentation.

    Returns
    -------
    features: dict
        feature name -> array with one value per event
        (non-scalar features such as "mask" are not included)
    """
    path = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    with h5py.File(path) as h5:
        masks = h5["events/mask"][:] > 0
        images = h5["events/image"][:]
        images_bg = h5["events/image_bg"][:]
        pixel_size = h5.attrs["imaging:pixel size"]

    with read.HDF5Data(path) as hd:
        gate = Gate(hd, online_gates=False)
    qee = QueueEventExtractor(slot_register=None,
                              pixel_size=pixel_size,
                              gate=gate,
                              event_queue=None,
                              extract_kwargs=extract_kwargs)

    features = {}
    for ii in range(len(masks)):
        # one image per chunk slot
        chunk_slot = types.SimpleNamespace(
            image=images[ii:ii+1],
            image_bg=images_bg[ii:ii+1],
            image_corr=np.asarray(images[ii:ii+1], dtype=np.int16)
            - images_bg[ii:ii+1],
            bg_off=None,
        )
        events = qee.get_events_from_masks(masks=masks[ii:ii+1],
                                           chunk_slot=chunk_slot,
                                           sub_index=0,
                                           **qee.extract_kwargs)
        for feat, values in events.items():
            if feat == "mask":
                continue
            assert len(values) == 1, "one event per mask"
            features.setdefault(feat, []).append(values[0])
    return {feat: np.array(values) for feat, values in features.items()}


def test_feat_reference_values():
    """Feature values must not change (for unchanged PPIDs)"""
    features = compute_features(brightness=True, haralick=True, volume=True)
    reference = json.loads(PATH_REFERENCE.read_text())
    assert sorted(features) == sorted(reference) == sorted(FEATURES_ALL)
    for feat in sorted(reference):
        ref = reference[feat]
        assert str(features[feat].dtype) == ref["dtype"], feat
        np.testing.assert_allclose(
            features[feat],
            np.array(ref["values"], dtype=float),
            rtol=1e-10,
            atol=0,
            equal_nan=True,
            err_msg=f"Feature '{feat}' changed!")


@pytest.mark.parametrize("brightness,haralick,volume",
                         itertools.product([True, False], repeat=3))
def test_feat_feature_set(brightness, haralick, volume):
    """The extracted features only depend on the feature kwargs"""
    features = compute_features(brightness=brightness,
                                haralick=haralick,
                                volume=volume)
    expected = []
    for feat in FEATURES_ALL:
        if feat.startswith("tex_") and not haralick:
            continue
        if (feat.startswith("bright_") or feat == "bg_med") \
                and not brightness:
            continue
        if feat == "volume" and not volume:
            continue
        expected.append(feat)
    assert sorted(features) == sorted(expected)


def run_pipeline(path, debug, num_procs):
    path_out = path.with_name(f"out_{debug}_{num_procs}.rtdc")
    job = logic.DCNumPipelineJob(path_in=path,
                                 path_out=path_out,
                                 debug=debug,
                                 num_procs=num_procs)
    with logic.DCNumJobRunner(job=job) as runner:
        runner.run()
    return path_out


def test_feat_independent_of_parallelization():
    """Threads, processes, and the number of workers give the same data"""
    path_orig = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path = path_orig.with_name("input.rtdc")
    # five times the test data (several chunks)
    with read.concatenated_hdf5_data(5 * [path_orig], path_out=path):
        pass

    paths_out = [run_pipeline(path, debug=True, num_procs=1),
                 run_pipeline(path, debug=False, num_procs=1),
                 run_pipeline(path, debug=False, num_procs=3),
                 ]

    with h5py.File(paths_out[0]) as h5:
        ref_feats = {f: h5["events"][f][:] for f in h5["events"]}
        ref_hash = h5.attrs["pipeline:dcnum hash"]
    assert len(ref_feats["deform"]) > 100
    for po in paths_out[1:]:
        with h5py.File(po) as h5:
            assert h5.attrs["pipeline:dcnum hash"] == ref_hash
            assert sorted(h5["events"]) == sorted(ref_feats)
            for feat, ref in ref_feats.items():
                assert np.array_equal(h5["events"][feat][:],
                                      ref,
                                      equal_nan=ref.dtype.kind == "f"), \
                    f"{feat} differs in {po}"


if __name__ == "__main__":
    # Regenerate the reference values
    features = compute_features(brightness=True, haralick=True, volume=True)
    data = {feat: {"dtype": str(values.dtype),
                   # JSON does not support NaN; use None instead
                   "values": [None if np.isnan(v) else float(v)
                              for v in np.asarray(values, dtype=float)]}
            for feat, values in sorted(features.items())}
    PATH_REFERENCE.write_text(json.dumps(data, indent=1) + "\n")
    print(f"Wrote {PATH_REFERENCE}")
