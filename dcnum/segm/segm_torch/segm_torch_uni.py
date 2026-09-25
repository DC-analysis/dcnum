from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np

from ..segmenter_uni import UNISegmenter

from .segm_torch_base import TorchSegmenterBase
from .torch_model import get_model_meta, load_model
from .torch_setup import torch


if TYPE_CHECKING:
    from ...logic import DCNumPipelineJob


class SegmentTorchUNI(TorchSegmenterBase, UNISegmenter):
    """PyTorch segmentation (Universal worker version)"""
    requires_model_format_version = "2.0"

    def __init__(self,
                 *,
                 kwargs_mask: dict | None = None,
                 backend: str | None = None,
                 device: str | None = None,
                 debug: bool = False,
                 **kwargs
                 ):
        """Segmenter for use with `UniversalWorker`

        Parameters
        ----------
        kwargs_mask: dict
            Keyword arguments for mask post-processing (see `process_labels`)
        backend: str
            Which backend to use for compiling/running the model. This
            parameter is part of the PPID. The default backend is whatever
            torch falls back to, reproducibility implied.
        device: str
            Which device to use (e.g. "cpu", or "cuda"). The device is not
            part of the PPID, because it should not affect reproducibility.
        debug: bool
            Debugging parameters
        kwargs:
            Additional, optional keyword arguments for ``segment_algorithm``
            defined in the subclass.
        """
        super().__init__(kwargs_mask=kwargs_mask,
                         debug=debug,
                         backend=backend,
                         device=device,
                         **kwargs)

        if "model_file" in kwargs:
            model_meta = SegmentTorchUNI.get_model_meta(self.kwargs)
            self.kwargs["backend"] = model_meta["backend"]
            self.kwargs["device"] = model_meta["device"]
            if "batch_size" in model_meta:
                self.required_batch_size = model_meta["batch_size"]

    def log_info(self, logger):
        model_meta = SegmentTorchUNI.get_model_meta_full(self.kwargs)
        backend = self.kwargs.get("backend")
        device = self.kwargs.get("device")
        logger.info(f"Segmenter backend '{backend}' with device '{device}'")
        if self.required_batch_size:
            logger.info(f"Batch size: {self.required_batch_size}")
        else:
            logger.info("No batch size restrictions")

        batch_size = model_meta.get("batch_size_recommended", None)
        logger.info(f"Recommended batch size: {batch_size}")

        if device and device.startswith("cuda"):
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU name: {torch.cuda.get_device_name()}")
            compute_capability = ".".join(
                str(c) for c in torch.cuda.get_device_capability(device))
            logger.info(f"GPU compute capability: {compute_capability}")
            _, total = torch.cuda.mem_get_info(device)
            logger.info(f"Available GPU memory: {total/1024**3:.1f}GB")

    @staticmethod
    def get_model_meta(segm_kwargs) -> dict[str, Any]:
        """Return basic model metadata"""
        model_meta = get_model_meta(
            segm_kwargs["model_file"],
            backend=segm_kwargs.get("backend"),
            device=segm_kwargs.get("device"))
        return model_meta

    @staticmethod
    def get_model_meta_full(segm_kwargs) -> dict[str, Any]:
        """Return model metadata after loading the model

        This will include keys such as "batch_size_recommended".
        """
        _, model_meta = load_model(
            segm_kwargs["model_file"],
            backend=segm_kwargs.get("backend"),
            device=segm_kwargs.get("device"))
        return model_meta

    @classmethod
    def get_ppid_from_ppkw(cls, kwargs, kwargs_mask=None):
        kwargs = copy.copy(kwargs)
        # Make sure the correct backend is given to the PPID generator
        if kwargs.get("backend") is None:
            model_meta = SegmentTorchUNI.get_model_meta(kwargs)
            kwargs["backend"] = model_meta["backend"]
        # Ignore the "device" in the kwargs
        if "device" in kwargs:
            kwargs.pop("device")
        return super().get_ppid_from_ppkw(kwargs, kwargs_mask)

    @staticmethod
    def update_worker_dedications(job: DCNumPipelineJob,
                                  worker_dedications: list[list[str]],
                                  ) -> list[list[str]]:
        """Update worker dedictions based on the segmentation approach"""
        if len(worker_dedications) == 1:
            # Nothing should be changed
            return worker_dedications
        else:
            # This is defined by UniversalWorker
            assert "load_all" in worker_dedications[0]
            assert "load_all" not in worker_dedications[1]

            # Check whether we are running on CPU or GPU/other. In case
            # of GPU, only the first worker should segment_images and the
            # second worker should load_all (segmenter initialization is slow).
            assert job["segmenter_code"] == "torchuni"
            model_meta = SegmentTorchUNI.get_model_meta(
                job["segmenter_kwargs"])
            if model_meta["device"] != "cpu":
                # 1st worker only segments and in full chunks
                worker_dedications[0].clear()
                worker_dedications[0].append("segment_images_full_chunk")
                # All other workers don't segment
                for wds in worker_dedications[1:]:
                    wds.remove("segment_images")
                # 2nd worker loads and joins remaining tasks
                worker_dedications[1].insert(0, "load_all")
            return worker_dedications

    @staticmethod
    def segment_algorithm(images,
                          *,
                          model_file: str | None = None,
                          backend: str | None,
                          device: str | None,
                          ):
        """
        Parameters
        ----------
        images: 3d ndarray
            event image
        model_file: str
            path to or name of a dcnum model file (.dcnm); if only a
            name is provided, then the "torch_model_files" directory
            paths are searched for the file name
        backend: str
            Which backend to use for compiling/running the model. This
            parameter is part of the PPID. The default backend is whatever
            torch falls back to, reproducibility implied.
        device: str
            Which device to use (e.g. "cpu", or "cuda"). The device is not
            part of the PPID, because it should not affect reproducibility.

        Returns
        -------
        mask: 3d boolean or integer ndarray
            mask or labeling image for the given index
        """
        if model_file is None:
            raise ValueError("Please specify a .dcnm model file!")

        with torch.inference_mode():

            # Set number of pytorch threads to 1, because dcnum is doing
            # all the multiprocessing.
            # https://pytorch.org/docs/stable/generated/torch.set_num_threads.html#torch.set_num_threads
            if torch.get_num_threads() != 1:
                torch.set_num_threads(1)
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)

            # Load model and metadata
            model, model_meta = load_model(model_file,
                                           backend=backend,
                                           device=device,
                                           )

            size = len(images)

            if model_meta["device"] == "cpu":
                # we only want one batch, because `images` is already a batch
                batch_size = size
            else:
                # we are using a different computing device (e.g. GPU)
                batch_size = model_meta.get("batch_size_recommended", 300)
                # run at least two batches to make use of async on GPU
                num_batches = int(np.ceil(max(2., size / batch_size)))
                batch_size = int(np.ceil(size / num_batches))

            # output mask array
            mask = np.empty(images.shape, dtype=bool)

            # Move image tensors to device
            batch_dev = torch.tensor(images[:batch_size], device=device)

            mask_func = model_meta["mask_func"]

            for bidx in range(0, size, batch_size):
                # Model inference
                pred_tensor = model(batch_dev)

                if bidx + batch_size < size:
                    # Copy next batch to device (async)
                    batch_dev = torch.tensor(
                        images[bidx + batch_size:bidx + 2 * batch_size],
                        device=device)

                # Convert tensor to numpy and populate mask
                mask[bidx:bidx + batch_size] = mask_func(pred_tensor)

        return mask
