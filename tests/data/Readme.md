## Model files

- ``segm-torch-model_unet-dcnum-test_g1_910c2.zip``: dcnum v1 model file; TorchScript/JIT
- ``segm-torch-model_unet-dcnum-test_g1_6784c.zip``: dcnum v2 model file; ExportedProgram, derived from 910c2
- ``segm-torch-model_unet-dcnum-test_g2_02dcd.zip``: dcnum v2 model file; missing `model_file_type` metadata
- ``segm-torch-model_unet-dcnum-test_g2_17ec6.zip``: dcnum v1 model file, TorchScript/JIT

## Fixed segmentation test data
- ``segm-torch-test-data_unet-dcnum-test_g1_910c2.zip``: images and segmented mask images

## Feature reference values
- ``feat_reference_values.json``: values of all scalar features for the
  40 events of ``fmt-hdf5_cytoshot_full-features_2023.zip`` (computed from
  the masks, images, and background images in that file); used in
  ``tests/test_feat_reproducibility.py``, which also describes how to
  regenerate this file

## Chopped image data
- ``fmt-hdf5_cytoshot_chopped-image-mask_dclab-0.74.zip``: the 40 events of
  ``fmt-hdf5_cytoshot_full-features_2023.zip`` with "image" and "mask"
  stored as chopped internal basins ("h5datasetchop"); created with dclab
  0.74 (``dclab.cli.compress(..., crop_event_images=True)`` and
  ``dclab.rtdc_dataset.feat_basin.chop_images.write_chopped_images`` for
  "mask"); ``reference_dclab.npz`` contains the "image" and "mask" frames
  reconstructed by dclab; used in ``tests/test_read_chopped.py``
