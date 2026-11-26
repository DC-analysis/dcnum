"""Segmentation with the "inductor" backend, including compilation

dcnum segments the images chunk by chunk. On a GPU, each chunk is split
into batches of about the same size; on the CPU, each chunk is one batch.
The last chunk of a dataset is usually smaller than the others, so the
model sees at least two different batch sizes. This benchmark segments
2137 frames (1000 + 1000 + 137) with the exported dcnum test model and
the inductor caches disabled, so every repetition includes compiling the
model for every batch size it encounters.

Runs on the GPU if CUDA is available, otherwise on the CPU. Note that the
test model in the dcnum repository was exported on the CPU.
"""
import pathlib
import tempfile
import zipfile

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

from dcnum.segm.segm_torch import SegmentTorchUNI, torch_model


here = pathlib.Path(__file__).parent
data_dir = here.parent / "tests" / "data"


def extract(zip_name, suffix):
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="bm_segm_torch_uni_"))
    with zipfile.ZipFile(data_dir / zip_name) as zf:
        zf.extractall(tmp)
    return sorted(tmp.glob(f"*{suffix}"))[0]


class Benchmark:
    def __init__(self):
        # compile from scratch in every repetition
        torch._inductor.config.force_disable_caches = True
        torch._dynamo.reset()
        torch_model.cache_model.clear()

        self.model_file = extract(
            "segm-torch-model_unet-dcnum-test_g2_0901b.zip", ".dcnm")
        path_data = extract(
            "fmt-hdf5_cytoshot_full-features_2024.zip", ".rtdc")
        with h5py.File(path_data) as h5:
            image = h5["events/image"][:]
        num_frames = 2137
        reps = num_frames // len(image) + 1
        self.images = np.concatenate([image] * reps)[:num_frames]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def benchmark(self):
        for start in range(0, len(self.images), 1000):
            SegmentTorchUNI.segment_algorithm(
                self.images[start:start + 1000],
                model_file=self.model_file,
                backend="inductor",
                device=self.device)
