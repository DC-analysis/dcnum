import numpy as np
import torch

from ..segmenter_mpo import MPOSegmenter

from .segm_torch_base import TorchSegmenterBase
from .torch_model import load_model
from .torch_preproc import preprocess_images
from .torch_postproc import postprocess_masks


class SegmentTorchMPO(TorchSegmenterBase, MPOSegmenter):
    """PyTorch segmentation (multiprocessing version)"""

    @staticmethod
    def segment_algorithm(image, *,
                          model_file: str = None):
        """
        Parameters
        ----------
        image: 2d ndarray
            event image
        model_file: str
            path to or name of a dcnum model file (.dcnm); if only a
            name is provided, then the "torch_model_files" directory
            paths are searched for the file name

        Returns
        -------
        mask: 2d boolean or integer ndarray
            mask or labeling image for the give index
        """
        if model_file is None:
            raise ValueError("Please specify a .dcnm model file!")

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

        image_preproc = preprocess_images(image[np.newaxis, :, :],
                                          **model_meta["preprocessing"])

        image_ten = torch.from_numpy(image_preproc)

        # Move image tensors to device
        image_ten_on_device = image_ten.to(device)
        # Model inference
        pred_tensor = model(image_ten_on_device)

        # Convert cuda-tensor into numpy mask array. The `pred_tensor`
        # array is still of the shape (1, 1, H, W). The `masks`
        # array is of shape (1, H, W). We can optionally label it
        # here (we have to if the shapes don't match) or do it in
        # postprocessing.
        masks = pred_tensor.detach().cpu().numpy()[0] >= 0.5

        # Perform postprocessing in cases where the image shapes don't match
        assert len(masks[0].shape) == len(image.shape), "sanity check"
        if masks[0].shape != image.shape:
            labels = postprocess_masks(
                masks=masks,
                original_image_shape=image.shape,
            )
            return labels[0]
        else:
            return masks[0]
