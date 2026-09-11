from ..segmenter_uni import UNISegmenter

from .segm_torch_base import TorchSegmenterBase
from .torch_model import load_model
from .torch_setup import torch


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
        compile_for: str
            For which hardware device to compile the model for
        debug: bool
            Debugging parameters
        kwargs:
            Additional, optional keyword arguments for ``segment_algorithm``
            defined in the subclass.
        """
        super().__init__(kwargs_mask=kwargs_mask,
                         debug=debug,
                         **kwargs)

        if "model_file" in kwargs:
            model_file = kwargs["model_file"]
            _, model_meta = load_model(model_file,
                                       backend=backend,
                                       device=device)
            if "batch_size" in model_meta:
                self.required_batch_size = model_meta["batch_size"]
            self.kwargs_technical["backend"] = model_meta["backend"]
            self.kwargs_technical["device"] = model_meta["device"]

    def log_info(self, logger):
        backend = self.kwargs_technical.get("backend")
        device = self.kwargs_technical.get("device")
        logger.info(f"Segmenter backend: {backend}, device: {device}")

    @staticmethod
    def segment_algorithm(images,
                          backend: str | None,
                          device: str | None,
                          *,
                          model_file: str | None = None):
        """
        Parameters
        ----------
        images: 3d ndarray
            event image
        model_file: str
            path to or name of a dcnum model file (.dcnm); if only a
            name is provided, then the "torch_model_files" directory
            paths are searched for the file name

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

            device = device or "cpu"

            # Load model and metadata
            model, model_meta = load_model(model_file,
                                           backend=backend,
                                           device=device,
                                           )

            # Move image tensors to device
            image_ten_on_device = torch.tensor(images, device=device)
            # Model inference
            pred_tensor = model(image_ten_on_device)

            mask_func = model_meta["mask_func"]
            mask = mask_func(pred_tensor)

        return mask
