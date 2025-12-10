from dcnum.segm import STOSegmenter
import numpy as np
import torch

from .segm_torch_base import TorchSegmenterBase
from .torch_model import load_model
from .torch_preproc import preprocess_images
from .torch_postproc import postprocess_masks


class SegmentTorchSTO(TorchSegmenterBase, STOSegmenter):
    """PyTorch segmentation (GPU version)"""

    def log_info(self, logger, gpu_id=None):
        model_file = self.kwargs["model_file"]
        device = torch.device(gpu_id if gpu_id is not None else "cuda")
        _, total = torch.cuda.mem_get_info(device)
        model, model_meta = load_model(model_file, device)
        batch_size = model_meta["estimated_batch_size_cuda"]
        logger.info(f"Available GPU memory: {total/1024**3:.1f}GB")
        logger.info(f"GPU segmentation batch size: {batch_size}")

    @staticmethod
    def _segment_in_batches(images, model, model_meta, device):
        """Segment image data in batches

        Return mask or label array with same shape as `images`.
        """
        size = len(images)

        # In dcnum <= 0.27.0, we had a fixed batch size of 50 which
        # resulted in a small speed penalty. Here, we use a batch size
        # that is tailored to the GPU memory.
        # Note that a batch size for segmentation larger than the chunk size
        # will result in an effective batch size that is identical to the
        # chunk size. The for-loop below will only have one iteration.
        batch_size = model_meta["estimated_batch_size_cuda"]

        # Preprocess the first image chunk
        batch_next = preprocess_images(images[0:batch_size],
                                       **model_meta["preprocessing"])

        # Create empty array to fill up with segmented batches
        masks = np.empty((size, *batch_next.shape[-2:]), dtype=bool)

        for start_idx in range(0, size, batch_size):
            # Move image tensors to cuda
            batch = torch.tensor(batch_next, device=device)

            # Model inference
            batch_seg = model(batch)
            # perform thresholding on GPU
            batch_seg_bool = batch_seg >= 0.5
            # For debugging and profiling, uncomment the next line.
            # torch.cuda.synchronize()

            # While we are waiting for the GPU, we can load the
            # next batch into memory (model(batch) runs async).
            im_next = images[start_idx + batch_size:start_idx + 2 * batch_size]
            if im_next.size:
                batch_next = preprocess_images(im_next,
                                               **model_meta["preprocessing"])

            # Remove extra dim [B, C, H, W] --> [B, H, W]
            batch_seg_bool = batch_seg_bool.squeeze(1)
            # Convert cuda-tensor to numpy array and fill masks array
            # (This will lock until the GPU computation is complete).
            masks[start_idx:start_idx + batch_size] \
                = batch_seg_bool.detach().cpu().numpy()

        # Perform postprocessing in cases where the image shapes don't match
        if masks.shape[1:] != images.shape[1:]:
            labels = postprocess_masks(
                masks=masks,
                original_image_shape=images.shape[1:])
            return labels
        else:
            return masks

    @staticmethod
    def segment_algorithm(images,
                          gpu_id=None,
                          *,
                          model_file: str = None):
        """
        Parameters
        ----------
        images: 3d ndarray
            array of N event images of shape (N, H, W)
        gpu_id: str
            optional argument specifying the GPU to use
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
            raise ValueError("Please specify a .dcnm model file!")

        # Determine device to use
        device = torch.device(gpu_id if gpu_id is not None else "cuda")

        # Load model and metadata
        model, model_meta = load_model(model_file, device)

        # Model inference
        # The `masks` array has the shape (len(images), H, W), where
        # H and W may be different from the corresponding axes in `images`.
        mol = SegmentTorchSTO._segment_in_batches(
            images=images,
            model=model,
            model_meta=model_meta,
            device=device,
        )

        return mol
