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
                 debug: bool = False,
                 **kwargs
                 ):
        """Segmenter for use with `UniversalWorker`

        Parameters
        ----------
        kwargs_mask: dict
            Keyword arguments for mask post-processing (see `process_labels`)
        debug: bool
            Debugging parameters
        kwargs:
            Additional, optional keyword arguments for ``segment_algorithm``
            defined in the subclass.
        """
        if "model_file" in kwargs:
            model_file = kwargs["model_file"]
            _, model_meta = load_model(model_file, "cpu")
            if "batch_size" in model_meta:
                self.required_batch_size = model_meta["batch_size"]
        super(SegmentTorchUNI, self).__init__(kwargs_mask=kwargs_mask,
                                              debug=debug,
                                              **kwargs)

    @staticmethod
    def segment_algorithm(images, *,
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
            mask or labeling image for the give index
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
            device = torch.device("cpu")

            # Load model and metadata
            model, model_meta = load_model(model_file, device)

            image_ten = torch.from_numpy(images)

            # Move image tensors to device
            image_ten_on_device = image_ten.to(device)
            # Model inference
            pred_tensor = model(image_ten_on_device)

            mask = pred_tensor.detach().cpu().numpy()

        return mask
