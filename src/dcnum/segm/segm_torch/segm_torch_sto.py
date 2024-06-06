from dcnum.segm import STOSegmenter
import numpy as np
import torch

from .segm_torch_base import TorchSegmenterBase
from .torch_model import load_model
from .torch_preproc import preprocess_images
from .torch_postproc import postprocess_masks


class SegmentTorchSTO(TorchSegmenterBase, STOSegmenter):
    """PyTorch segmentation (GPU version)"""

    @staticmethod
    def _segment_in_batches(imgs_t, model, batch_size, device):
        """Segment image data in batches"""
        size = len(imgs_t)
        # Create empty array to fill up with segmented batches
        masks = np.empty((len(imgs_t), *imgs_t[0].shape[-2:]),
                         dtype=bool)

        for start_idx in range(0, size, batch_size):
            batch = imgs_t[start_idx:start_idx + batch_size]
            # Move image tensors to cuda
            batch = torch.tensor(batch, device=device)
            # Model inference
            batch_seg = model(batch)
            # Remove extra dim [B, C, H, W] --> [B, H, W]
            batch_seg = batch_seg.squeeze(1)
            # Convert cuda-tensor into numpy arrays
            batch_seg_np = batch_seg.detach().cpu().numpy()
            # Fill empty array with segmented batch
            masks[start_idx:start_idx + batch_size] = batch_seg_np >= 0.5

        return masks

    @staticmethod
    def segment_algorithm(images, gpu_id=None, batch_size=50, *,
                          model_file: str = None):
        """
        Parameters
        ----------
        images: 3d ndarray
            array of N event images of shape (N, H, W)
        gpu_id: str
            optional argument specifying the GPU to use
        batch_size: int
            number of images to process in one batch
        model_file: str
            path to or name of a dcnum model file (.dcnm); if only a
            name is provided, then the "torch_model_files" directory
            paths are searched for the file name

        Returns
        -------
        mask: 2d boolean or integer ndarray
            mask or label images of shape (N, H, W)
        """
        if model_file is None:
            raise ValueError("Please specify a model file!")

        # Determine device to use
        device = torch.device(gpu_id if gpu_id is not None else "cuda")

        # Load model and metadata
        model, model_meta = load_model(model_file, device)

        # Preprocess the images
        image_preproc = preprocess_images(images,
                                          **model_meta["preprocessing"])
        # Model inference
        # The `masks` array has the shape (len(images), H, W), where
        # H and W may be different from the corresponding axes in `images`.
        masks = SegmentTorchSTO._segment_in_batches(image_preproc,
                                                    model,
                                                    batch_size,
                                                    device
                                                    )

        # Perform postprocessing in cases where the image shapes don't match
        assert len(masks.shape[1:]) == len(images.shape[1:]), "sanity check"
        if masks.shape[1:] != images.shape[1:]:
            labels = postprocess_masks(
                masks=masks,
                original_image_shape=images.shape[1:])
            return labels
        else:
            return masks
