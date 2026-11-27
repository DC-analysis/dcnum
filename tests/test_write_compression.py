import json

import h5py
import numpy as np
import pytest

from dcnum import logic, read, write
from dcnum.write import chop

from helper_methods import extract_data, find_data, retrieve_data


def get_dclab_chopped_reference():
    """Return the file chopped by dclab 0.74 (see tests/data/Readme.md)"""
    edest = extract_data("fmt-hdf5_cytoshot_chopped-image-mask_dclab-0.74.zip")
    return find_data(edest)[0]


def get_tap_output(tmp_path):
    """Return a pipeline output file that refers to its input file"""
    path_in = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path_out = tmp_path / "tap.rtdc"
    job = logic.DCNumPipelineJob(path_in=path_in,
                                 path_out=path_out,
                                 basin_strategy="tap",
                                 debug=True)
    with logic.DCNumJobRunner(job=job) as runner:
        runner.run()
    return path_in, path_out


def write_synthetic_file(path, num_events=600, seed=42):
    """Write a file with events of many different sizes

    Events of different sizes result in multiple datasets in the
    chopped data.
    """
    rng = np.random.default_rng(seed)
    roi = (80, 250)
    pixel_size = 0.34
    mask = np.zeros((num_events,) + roi, dtype=bool)
    size_x = np.zeros(num_events)
    size_y = np.zeros(num_events)
    for ii in range(num_events):
        sy = rng.integers(3, 60)
        sx = rng.integers(3, 120)
        # also events at the border of the frame
        oy = rng.integers(0, roi[0] - sy + 1)
        ox = rng.integers(0, roi[1] - sx + 1)
        mask[ii, oy:oy + sy, ox:ox + sx] = True
        size_y[ii] = sy * pixel_size
        size_x[ii] = sx * pixel_size
    # several events per frame
    frame = np.sort(rng.integers(1, num_events // 2, size=num_events))
    # events in the same frame share the image and the background
    frame_bg = np.asarray(
        rng.integers(90, 110, size=(num_events // 2,) + roi), dtype=np.uint8)
    # noise (image data outside of the events)
    frame_image = frame_bg + np.asarray(
        rng.integers(0, 3, size=frame_bg.shape), dtype=np.uint8)
    for ii in range(num_events):
        frame_image[frame[ii]][mask[ii]] = rng.integers(
            0, 255, size=np.sum(mask[ii]), dtype=np.uint8)
    image = frame_image[frame]
    image_bg = frame_bg[frame]
    with h5py.File(path, "w") as h5:
        h5.attrs["experiment:date"] = "2026-11-27"
        h5.attrs["experiment:event count"] = num_events
        h5.attrs["experiment:run index"] = 1
        h5.attrs["experiment:sample"] = "synthetic"
        h5.attrs["experiment:time"] = "12:00:00"
        h5.attrs["imaging:pixel size"] = pixel_size
        h5.attrs["imaging:roi size x"] = roi[1]
        h5.attrs["imaging:roi size y"] = roi[0]
        h5.attrs["setup:channel width"] = 20
        h5.attrs["setup:chip region"] = "channel"
        h5.attrs["setup:flow rate"] = 0.06
        h5.attrs["setup:identifier"] = "synthetic-setup"
        h5.attrs["setup:medium"] = "CellCarrierB"
        for feat, data in [("deform", rng.random(num_events)),
                           ("frame", frame),
                           ("image", image),
                           ("image_bg", image_bg),
                           ("mask", mask),
                           ("size_x", size_x),
                           ("size_y", size_y),
                           ]:
            ds = h5.create_dataset(f"events/{feat}", data=data, chunks=True)
            if len(ds.shape) == 3:
                ds.attrs.create('CLASS', np.bytes_('IMAGE'))
    return path


def assert_chopped_identical(path_a, path_b):
    with h5py.File(path_a) as ha, h5py.File(path_b) as hb:
        ga = ha["basin_events/image"]
        gb = hb["basin_events/image"]
        assert sorted(ga.keys()) == sorted(gb.keys())
        assert dict(ga.attrs) == dict(gb.attrs)
        for key in ga:
            assert np.array_equal(ga[key][:], gb[key][:]), key
            if key != "index":
                assert ga[key].chunks == gb[key].chunks, key


@pytest.mark.parametrize("num_procs", [1, 2])
def test_compress_chopped_same_as_dclab(tmp_path, num_procs):
    """dcnum writes the same chopped data as `dclab-compress`"""
    path_in = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path_ref = get_dclab_chopped_reference()
    path_out = tmp_path / "out.rtdc"
    write.compress(path_in, path_out, crop_event_images=True,
                   num_procs=num_procs)
    assert_chopped_identical(path_out, path_ref)
    with read.HDF5Data(path_out) as hd, read.HDF5Data(path_ref) as hr:
        assert "image" not in hd.h5["events"]
        assert hd.is_chopped("image")
        assert np.array_equal(hd.image[:], hr.image[:])
        # all other features are stored in the "events" group
        assert sorted(hd.h5["events"].keys()) == sorted(
            set(hr.keys()) - {"image"})
        assert np.array_equal(hd.mask[:], hr.mask[:])
        for feat in ["deform", "frame", "image_bg", "time"]:
            assert np.array_equal(hd[feat][:], hr[feat][:])


@pytest.mark.parametrize("crop_method", ["box", "dilation"])
def test_compress_chopped_same_as_dclab_synthetic(tmp_path, crop_method):
    """Several datasets, events sharing a frame, and the frame edge"""
    pytest.importorskip("dclab")
    from dclab.rtdc_dataset.feat_basin.chop_images import (
        write_chopped_images)
    import dclab

    path_in = write_synthetic_file(tmp_path / "synthetic.rtdc")
    path_out = tmp_path / "out.rtdc"
    path_ref = tmp_path / "ref.rtdc"
    write.compress(path_in, path_out, crop_event_images=True,
                   crop_method=crop_method, num_procs=2)
    with dclab.new_dataset(path_in) as ds, h5py.File(path_ref, "w") as h5:
        write_chopped_images(ds=ds, feat="image", h5_dst=h5,
                             crop_method=crop_method)
    with h5py.File(path_out) as h5:
        assert len(h5["basin_events/image"].keys()) > 3
    assert_chopped_identical(path_out, path_ref)

    with read.HDF5Data(path_out) as hd, read.HDF5Data(path_in) as hi:
        image = hd.image[:]
        image_in = hi.image[:]
        mask = hi.mask[:]
        # inside the events, the images are identical
        assert np.array_equal(image[mask], image_in[mask])
        assert not np.array_equal(image, image_in)
    # dclab reads the same frames
    with dclab.new_dataset(path_out) as ds:
        assert np.array_equal(ds["image"][:], image)


def test_compress_chopped_input_without_cropping(tmp_path):
    """Chopped input data are written as full frames"""
    path_ref = get_dclab_chopped_reference()
    path_out = tmp_path / "out.rtdc"
    write.compress(path_ref, path_out, num_procs=1)
    with read.HDF5Data(path_out) as hd, read.HDF5Data(path_ref) as hr:
        assert not hd.is_chopped("image")
        assert not hd.is_chopped("mask")
        assert "image" in hd.h5["events"]
        assert "basin_events" not in hd.h5
        assert len(hd.basins) == 0
        assert np.array_equal(hd.image[:], hr.image[:])
        assert np.array_equal(hd.mask[:], hr.mask[:])


@pytest.mark.parametrize("compression,filter_id", [
    ("none", None),
    ("zstd-1", "32015"),
    ("ZSTD-9", "32015"),
])
def test_compress_compression(tmp_path, compression, filter_id):
    path_in = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path_out = tmp_path / "out.rtdc"
    write.compress(path_in, path_out, compression=compression, num_procs=1)
    with h5py.File(path_out) as h5, h5py.File(path_in) as h5in:
        for feat in ["deform", "image", "mask"]:
            ds = h5["events"][feat]
            assert ds.fletcher32
            filters = ds._filters
            if filter_id is None:
                assert "32015" not in filters
            else:
                assert filters[filter_id][0] == int(compression[-1])
            assert np.array_equal(ds[:], h5in["events"][feat][:])
        assert sorted(h5["events"].keys()) == sorted(h5in["events"].keys())


def test_compress_errors(tmp_path):
    path_in = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path_out = tmp_path / "out.rtdc"
    with pytest.raises(ValueError, match="Unsupported compression"):
        write.compress(path_in, path_out, compression="gzip")
    with pytest.raises(ValueError, match="Invalid crop method"):
        write.compress(path_in, path_out, crop_method="circle")
    path_out.touch()
    with pytest.raises(FileExistsError):
        write.compress(path_in, path_out)


def test_compress_log(tmp_path):
    path_in = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path_1 = tmp_path / "out1.rtdc"
    path_2 = tmp_path / "out2.rtdc"
    write.compress(path_in, path_1, crop_event_images=True, num_procs=1)
    write.compress(path_1, path_2, num_procs=1)
    with read.HDF5Data(path_1) as hd1, read.HDF5Data(path_2) as hd2:
        log1 = json.loads("\n".join(hd1.logs["dcnum-compress"]))
        assert log1["crop_event_images"]
        assert log1["crop_method"] == "dilation"
        assert log1["compression"] == "zstd-5"
        assert log1["input file"] == path_in.name
        log2 = json.loads("\n".join(hd2.logs["dcnum-compress"]))
        assert not log2["crop_event_images"]
        assert log2["input file"] == path_1.name
        # the log of the first compression is kept
        assert hd2.logs[f"dcnum-compress_{hd1.md5_5m}"] == hd1.logs[
            "dcnum-compress"]
        # other logs are copied
        for key in hd1.logs:
            if key != "dcnum-compress":
                assert key in hd2.logs


def test_compress_progress(tmp_path):
    path_in = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    progress = []
    write.compress(path_in, tmp_path / "out.rtdc",
                   crop_event_images=True,
                   num_procs=1,
                   progress_callback=progress.append)
    assert progress == sorted(progress)
    assert progress[-1] == 1


def test_compress_raw_data(tmp_path):
    """Raw data can only be compressed without cropping"""
    path_in = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path_raw = tmp_path / "raw.rtdc"
    with h5py.File(path_in) as h5, h5py.File(path_raw, "w") as h5raw:
        h5raw.attrs.update(h5.attrs)
        for feat in ["image", "frame", "time"]:
            h5.copy(h5[f"events/{feat}"], h5raw.require_group("events"))
    with pytest.raises(ValueError, match="Cropping event images requires"):
        write.compress(path_raw, tmp_path / "out.rtdc",
                       crop_event_images=True)
    write.compress(path_raw, tmp_path / "out.rtdc", num_procs=2)
    with read.HDF5Data(tmp_path / "out.rtdc") as hd, \
            read.HDF5Data(path_raw) as hr:
        assert sorted(hd.keys()) == ["frame", "image", "time"]
        assert np.array_equal(hd.image[:], hr.image[:])


def test_compress_no_image(tmp_path):
    path_in = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    with h5py.File(path_in, "a") as h5:
        del h5["events/image"]
    with pytest.raises(ValueError, match="No image data found"):
        write.compress(path_in, tmp_path / "out.rtdc")


@pytest.mark.parametrize("crop_event_images", [True, False])
def test_compress_tap_output(tmp_path, crop_event_images):
    """Features from file basins are written to the output file"""
    path_in, path_tap = get_tap_output(tmp_path)
    path_out = tmp_path / "out.rtdc"
    write.compress(path_tap, path_out,
                   crop_event_images=crop_event_images,
                   num_procs=2)
    with read.HDF5Data(path_tap) as ht:
        assert "image" not in ht.h5["events"]
        ref = {}
        features = ht.keys()
        for feat in features:
            if feat in ["image", "image_bg", "mask"]:
                ref[feat] = getattr(ht, feat)[:]
            else:
                ref[feat] = ht[feat][:]
    # The output file does not need the input file anymore
    path_in.rename(path_in.with_name("moved.rtdc"))
    with read.HDF5Data(path_out) as hd:
        assert hd.is_chopped("image") == crop_event_images
        assert ("image" in hd.h5["events"]) != crop_event_images
        assert sorted(hd.keys()) == sorted(ref.keys())
        for feat, data_ref in ref.items():
            if feat in ["image", "image_bg", "mask"]:
                data = getattr(hd, feat)[:]
            else:
                data = hd[feat][:]
            if feat == "image" and crop_event_images:
                mask = ref["mask"]
                assert np.array_equal(data[mask], data_ref[mask])
            else:
                assert np.array_equal(data, data_ref,
                                      equal_nan=data.dtype.kind == "f"), feat


def test_compress_tap_output_basin_paths(tmp_path):
    """Moved basin files can be specified with `basin_paths`"""
    path_in, path_tap = get_tap_output(tmp_path)
    with read.HDF5Data(path_tap) as ht:
        image_ref = ht.image[:]
        mask_ref = ht.mask[:]
    moved_dir = tmp_path / "moved"
    moved_dir.mkdir()
    path_moved = moved_dir / "moved.rtdc"
    path_in.rename(path_moved)
    path_out = tmp_path / "out.rtdc"
    with pytest.raises(ValueError, match="file basins are not available"):
        write.compress(path_tap, path_out, num_procs=1)
    assert not path_out.exists()
    # not a basin of the input file
    path_other = write_synthetic_file(tmp_path / "other.rtdc")
    with pytest.raises(ValueError, match="is not a file basin"):
        write.compress(path_tap, path_out, basin_paths=[path_other])

    write.compress(path_tap, path_out, basin_paths=[path_moved],
                   crop_event_images=True, num_procs=1)
    with read.HDF5Data(path_out) as hd:
        assert hd.is_chopped("image")
        assert np.array_equal(hd.mask[:], mask_ref)
        assert np.array_equal(hd.image[:][mask_ref], image_ref[mask_ref])
        log = json.loads("\n".join(hd.logs["dcnum-compress"]))
        assert log["basin_paths"] == [str(path_moved)]


def test_compress_basin_without_identifier(tmp_path):
    """File basins without identifier are matched by their file name"""
    path_in = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path_raw = tmp_path / "raw.rtdc"
    path_derived = tmp_path / "derived.rtdc"
    with h5py.File(path_in) as h5, h5py.File(path_raw, "w") as h5raw:
        h5raw.attrs.update(h5.attrs)
        for feat in ["image", "frame", "time"]:
            h5.copy(h5[f"events/{feat}"], h5raw.require_group("events"))
        # This mapping feature is only meaningful in the basin file.
        h5raw["events/basinmap0"] = np.arange(len(h5["events/frame"]))
        image_ref = h5["events/image"][:]
    with h5py.File(path_in) as h5, \
            write.HDF5Writer(path_derived, mode="w") as hw:
        hw.h5.attrs.update(h5.attrs)
        for feat in ["deform", "mask"]:
            h5.copy(h5[f"events/{feat}"], hw.h5["events"])
        # all features of the basin file, no identifier
        hw.store_basin(name="raw", paths=[path_raw])

    # The basin is the same file
    write.compress(path_derived, tmp_path / "out1.rtdc",
                   basin_paths=[path_raw], num_procs=1)
    # The basin was moved
    moved_dir = tmp_path / "moved"
    moved_dir.mkdir()
    path_raw.rename(moved_dir / "raw.rtdc")
    write.compress(path_derived, tmp_path / "out2.rtdc",
                   basin_paths=[moved_dir / "raw.rtdc"], num_procs=1)
    for name in ["out1.rtdc", "out2.rtdc"]:
        with read.HDF5Data(tmp_path / name) as hd:
            assert sorted(hd.h5["events"].keys()) == [
                "deform", "frame", "image", "mask", "time"]
            assert np.array_equal(hd.image[:], image_ref)
    # A different file name does not work
    path_raw_2 = moved_dir / "raw_2.rtdc"
    (moved_dir / "raw.rtdc").rename(path_raw_2)
    with pytest.raises(ValueError, match="is not a file basin"):
        write.compress(path_derived, tmp_path / "out3.rtdc",
                       basin_paths=[path_raw_2])


def test_compress_cleanup_on_error(tmp_path, monkeypatch):
    """No output file is written if compression fails"""
    from dcnum.write import compression

    def worker_error(**kwargs):
        raise RuntimeError("Compression failed!")

    monkeypatch.setattr(compression, "_worker_compress_feature",
                        worker_error)
    path_in = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    path_out = tmp_path / "out.rtdc"
    with pytest.raises(RuntimeError, match="Compression failed!"):
        write.compress(path_in, path_out, num_procs=1)
    assert not list(tmp_path.iterdir())


def test_compress_default_num_procs_and_trace(tmp_path):
    """Feature groups (e.g. fluorescence traces) are copied"""
    path_in = retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip")
    with h5py.File(path_in, "a") as h5:
        num_events = len(h5["events/frame"])
        trace = np.arange(num_events * 10).reshape(num_events, 10)
        h5["events/trace/fl1_raw"] = trace
    path_out = tmp_path / "out.rtdc"
    write.compress(path_in, path_out)
    with h5py.File(path_out) as h5:
        assert np.array_equal(h5["events/trace/fl1_raw"][:], trace)
        assert "image" in h5["events"]


def test_get_clusters_same_as_dclab():
    pytest.importorskip("dclab")
    from dclab.rtdc_dataset.feat_basin.chop_images import (
        determine_clustering)
    rng = np.random.default_rng(7)
    geometry = np.zeros((5000, 4), dtype=np.uint16)
    geometry[:, 0] = rng.integers(5, 80, size=5000)
    geometry[:, 1] = rng.integers(5, 250, size=5000)
    clusters, num = chop.get_clusters(geometry[:, :2], dtype=np.uint8)
    clusters_ref, num_ref = determine_clustering(geometry, dtype=np.uint8)
    assert num > 5
    assert num == num_ref
    assert np.array_equal(clusters, clusters_ref)


def test_get_sibling_links():
    frame = np.array([1, 2, 2, 3, 5, 5, 5, 7])
    links = chop.get_sibling_links(frame)
    assert np.array_equal(links, [0, 2, 1, 3, 6, 4, 5, 7])


def test_repair_frame():
    frame = np.array([1, 2, 2, 5, 3, 0, 9], dtype=np.uint32)
    with pytest.warns(chop.NegativeFrameCountWarning):
        repaired = chop.repair_frame(frame)
    assert np.array_equal(repaired, [1, 2, 2, 5, 6, 7, 8])
    assert np.array_equal(chop.repair_frame(np.arange(4)), np.arange(4))


def test_chop_events_dilation_requires_mask():
    with pytest.raises(ValueError, match="mask is required"):
        chop.chop_events(data=np.zeros((1, 10, 10), dtype=np.uint8),
                         shapes=np.array([[5, 5]]),
                         offsets=np.array([[0, 0]]),
                         dilation_structure=chop.disk(1))


def test_worker_not_initialized():
    from dcnum.write import compression
    with pytest.raises(RuntimeError, match="not initialized"):
        compression._worker_compress_feature(feat="deform", start=0,
                                             stop=1, chunk_size=1,
                                             ds_kwds={})


def test_disk():
    assert np.array_equal(chop.disk(1), [[0, 1, 0], [1, 1, 1], [0, 1, 0]])


def test_compression_kwargs():
    kw = write.get_compression_kwargs("none")
    assert kw == {"fletcher32": True,
                  "compression": None,
                  "compression_opts": None}
    with pytest.raises(ValueError, match="Unsupported compression"):
        write.get_compression_kwargs("zstd-10")
