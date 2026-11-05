.. _sec_pipeline:

How the pipeline works
======================

dcnum turns the raw image data of a deformability cytometry measurement
into event features such as area, deformation, brightness or volume.
Every frame recorded by the camera goes through the same sequence of steps,
which is shown in the figure below.

.. plot:: figures/pipeline_overview.py
   :include-source: False
   :show-source-link: False

   Overview of the dcnum pipeline. Background data are computed once for
   the whole dataset. During segmentation, exactly one segmenter is used:
   the thresholding segmenter (orange) works on the background-corrected
   image, while the U-Net segmenters (green) work directly on the raw image.
   Background data are still used for computing the background-corrected
   brightness and texture features (dashed line).


The steps in detail
-------------------

1. Reading the input data
~~~~~~~~~~~~~~~~~~~~~~~~~
The input is an HDF5-based .rtdc file that contains at least the raw
``image`` data of all frames, together with the metadata of the
measurement (e.g. the pixel size and the frame rate). dcnum reads these
data in chunks of frames (:class:`.read.hdf5_data.HDF5Data`). Index
mapping (``data_kwargs={"index_mapping": ...}``) allows you to process
only a subset of the frames.

2. Background computation
~~~~~~~~~~~~~~~~~~~~~~~~~
Most of what the camera sees is static: the walls of the microfluidic
channel, dirt on the chip and the illumination pattern. dcnum estimates
this background by computing the pixel-wise median of many frames, so
that cells, which only pass through, do not contribute. The result is
stored as the ``image_bg`` feature. The available methods
(``background_code``) are:

- ``"sparsemed"`` (default, :class:`.feat.feat_background.bg_sparse_median.BackgroundSparseMed`):
  computes a median image every few seconds of measurement time from
  a large number of frames and assigns to each frame the background image
  closest in time. Background images that still contain cells are removed
  in a cleansing step. Optionally, a per-frame brightness offset
  ``bg_off`` is computed from the channel walls, which corrects for
  flickering of the light source.
- ``"rollmed"`` (:class:`.feat.feat_background.bg_roll_median.BackgroundRollMed`):
  computes a rolling median for every frame.
- ``"copy"`` (:class:`.feat.feat_background.bg_copy.BackgroundCopy`):
  reuses the background data that are already present in the input file.

If the input file already contains background data that were computed with
the same settings (as identified by the pipeline identifier, see
:ref:`sec_pipeline_ppid`), the background computation is skipped.

3. Segmentation
~~~~~~~~~~~~~~~
Segmentation decides, for every pixel of a frame, whether it belongs to an
object (e.g. a cell) or not. The output is a binary mask image. dcnum
provides two kinds of segmenters (``segmenter_code``):

**Thresholding** (``"thresh"``, :class:`.segm.segm_thresh.SegmentThresh`):
In the raw image, cells are hard to separate from the channel walls
by brightness alone. Therefore, the background image (and the offset
``bg_off``) is first subtracted from each frame. In the resulting
background-corrected image, the static structures vanish and only the
cell remains as a region darker than its surroundings. All pixels that
are darker than a threshold (``thresh=-6`` by default) form the mask.
This segmenter *requires* the background correction
(:attr:`.segm.segmenter.Segmenter.requires_background_correction`
is ``True``). The figure below shows the intermediate results.

.. plot:: figures/pipeline_steps_thresh.py
   :include-source: False
   :show-source-link: False

   Segmentation by thresholding of one example frame. The background image
   (2) is subtracted from the raw image (1). In the background-corrected
   image (3), only the cell differs from zero. Thresholding (4) yields the
   cell, but also isolated noise pixels. Mask post-processing and gating
   (5) remove the noise and smooth the mask, whose contour is shown in
   orange. Note that the bright tail of the cell is not part of the mask,
   because only pixels darker than the background are selected.

**U-Net segmentation with PyTorch** (``"torchmpo"``, ``"torchsto"``,
``"torchuni"``, :mod:`.segm.segm_torch`): A pre-trained convolutional
neural network with U-Net architecture predicts, for every pixel, the
probability that it belongs to a cell, and pixels with a probability of
at least 0.5 form the mask. The network has learned during training how
cells look in front of the channel, so it receives **the raw image as
input and does not require background correction**
(:attr:`~.segm.segmenter.Segmenter.requires_background_correction`
is ``False`` for all torch segmenters). The only pre-processing is a
normalization of the image intensities and cropping or padding of the
image to the size the model was trained with. As a consequence,
U-Net segmentation is not affected by an inaccurate background image
(e.g. when a cell sticks to the channel wall for a long time) and it can
also segment objects that are brighter than the background.
The figure below shows example images.

.. plot:: figures/pipeline_steps_unet.py
   :include-source: False
   :show-source-link: False

   Segmentation with a U-Net model. The raw image (1) is the input of the
   model; no background image is involved. The model output (2) is a
   binary mask, which then undergoes the same mask post-processing as
   for the thresholding segmenter (3). Objects touching the image border
   (bottom right) are removed. The masks shown here were produced with
   the small U-Net test model of the dcnum test suite.

To use a U-Net segmenter, install dcnum with the ``torch`` extra
(``pip install dcnum[torch]``) and pass a dcnum model file (.dcnm) to
the job:

.. code:: python

    from dcnum.logic import DCNumPipelineJob, DCNumJobRunner

    job = DCNumPipelineJob(
        path_in="input.rtdc",
        segmenter_code="torchmpo",
        segmenter_kwargs={"model_file": "path/to/model.dcnm"},
    )
    with DCNumJobRunner(job=job) as runner:
        runner.run()

``"torchmpo"`` runs the model on the CPU in several processes,
``"torchsto"`` runs the model in batches in a single process (e.g. on a
CUDA GPU), and ``"torchuni"`` integrates the model into the universal
worker scheme of dcnum and supports different inference backends.
Before the pipeline starts, dcnum checks the metadata stored in the model
file to make sure that the model is applicable to the dataset
(:meth:`.segm.segmenter.Segmenter.validate_applicability`).

.. note::

   The background data are computed for every pipeline run, regardless of
   the segmenter, because they are needed for some of the features
   (see below) and are stored in the output file. If the input file
   already contains background data, you can save time by setting
   ``background_code="copy"``.

4. Labeling and mask post-processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The binary mask is split into connected regions (labeling), so that each
object in a frame becomes one event. The labels are then cleaned up
(:meth:`.segm.segmenter.Segmenter.process_labels`):

- ``clear_border``: objects that touch the image border are removed,
  because they are only partially visible.
- ``fill_holes``: holes inside an object are filled.
- ``closing_disk``: the outline of the mask is smoothed with a disk of
  the given radius in pixels (``0`` disables this step).

You can change these settings via the ``kwargs_mask`` dictionary in
``segmenter_kwargs``.

5. Feature extraction
~~~~~~~~~~~~~~~~~~~~~
For each event, dcnum computes features from the mask and the image data
(:class:`.feat.queue_event_extractor.QueueEventExtractor`):

- contour and moment-based features (:mod:`.feat.feat_contour`), e.g.
  position, area, deformation, aspect ratio, inertia ratio and, from the
  contour, the volume,
- brightness features (:mod:`.feat.feat_brightness`) from the raw image
  (``bright_avg``, ``bright_sd``) and from the background-corrected image
  (``bright_bc_avg``, ``bright_bc_sd``, ``bright_perc_10``,
  ``bright_perc_90``) as well as the background brightness ``bg_med``,
- Haralick texture features (:mod:`.feat.feat_texture`) from the
  background-corrected image.

The background-corrected features are the reason why background data are
also used in pipelines with U-Net segmentation.

6. Gating
~~~~~~~~~
Not every labeled object is a valid event. The gating step
(:class:`.feat.gate.Gate`, ``gate_kwargs``) discards masks that consist of
too few pixels (``size_thresh_mask``, 10 pixels by default) and, if
``online_gates=True``, events outside the feature ranges that were
applied during the measurement (as stored in the metadata of the input
file).

7. Writing the output file
~~~~~~~~~~~~~~~~~~~~~~~~~~
Finally, the events are written in the correct order to the output .rtdc
file (:mod:`.write`). The output file also contains the metadata and logs
of the input file, the log of the dcnum run and the pipeline identifiers.
With ``basin_strategy="tap"``, the output file does not contain a copy of
the raw image data, but refers to the input file via a basin, which keeps
the output file small. With ``basin_strategy="drain"`` (default), all data
are copied to the output file.


.. _sec_pipeline_ppid:

Pipeline identifiers
--------------------
Every step of the pipeline is described by a code and its keyword
arguments. From these, dcnum builds a pipeline identifier (PPID) for each
step and a hash for the entire pipeline, which are stored as attributes
in the output file, for example::

    pipeline:dcnum background = sparsemed:k=200^s=1^t=0^f=0.8^o=1
    pipeline:dcnum segmenter  = thresh:t=-6:cle=1^f=1^clo=2
    pipeline:dcnum feature    = legacy:b=1^h=1^v=1
    pipeline:dcnum gate       = norm:o=0^s=10

Two datasets that were processed with the same PPIDs were analyzed in
exactly the same way. dcnum also uses the PPIDs to skip steps that were
already performed with the same settings. You can obtain the PPID of a
job with :meth:`.logic.job.DCNumPipelineJob.get_ppid`; the module
:mod:`.meta.ppid` contains the underlying logic.


Running the steps in parallel
-----------------------------
For performance, the steps above are not executed one after another for
the whole dataset. Instead, the image data are split into chunks that move
through the steps in parallel. The details are described in
:ref:`sec_design`.
