.. _sec_design:

The design of dcnum
===================


Submodule Structure
-------------------

The general idea of dcnum is to have a toolset for processing raw DC data,
which includes reading images, segmenting events, extracting features for
each event, and writing to an output file.

Each of the individual submodules serves one particular aspect of the
pipeline:

.. list-table:: dcnum submodules
   :header-rows: 1

   * - Submodule
     - Description

   * - :mod:`.feat`
     - Feature extraction from segmented image data.

   * - :mod:`.logic`
     - | Contains the necessary logic (the glue) to combine all
       | the other submodules for processing a dataset.

   * - :mod:`.meta`
     - | Handles metadata, most importantly the pipeline identifiers
       | (PPIDs).

   * - :mod:`.read`
     - For reading raw HDF5 (.rtdc) files.

   * - :mod:`.segm`
     - | Event segmentation finds objects in an image and returns a
       | binary mask for each object.

   * - :mod:`.write`
     - For writing data to HDF5 (.rtdc) files.


Pipeline sequence
-----------------

A pipeline (including its PPID) is defined via the
:class:`.logic.job.DCNumPipelineJob` class which represents the recipe for a
pipeline. The pipeline is executed with the :class:`.logic.ctrl.DCNumJobRunner`.
Here is a simple example that runs the default pipeline for an .rtdc file.

.. code:: python

    from dcnum.logic import DCNumPipelineJob, DCNumJobRunner

    job = DCNumPipelineJob(path_in="input.rtdc")
    with DCNumJobRunner(job=job) as runner:
        runner.run()

Take a look at the keyword arguments that the classes mentioned above
accept. Note that you can specify methods for background correction as
well as segmentation, and that you have full access to the keyword arguments
for every step in the pipeline. Also note that a reproducible PPID is derived
from these keyword arguments (:meth:`.logic.job.DCNumPipelineJob.get_ppid`).
What the individual steps do is explained in :ref:`sec_pipeline`.

The following happens when you run the above code snippet:

1. The file `input.rtdc` is opened using the module :mod:`.read`.
2. If necessary, the background image data are computed with the
   background method selected in the job (a subclass of
   :class:`.feat.feat_background.base.Background`) and written to a
   temporary file.
3. The ``DCNumJobRunner`` creates a :class:`.logic.slot_register.SlotRegister`
   which manages several :class:`.logic.chunk_slot.ChunkSlot` instances.
   A chunk slot holds the data of one chunk of frames (image, background
   image, background-corrected image, mask and labels) in shared memory,
   so that it can be accessed from multiple processes. Each chunk slot
   has a state that defines which task has to be performed next
   (loading image data, segmentation, mask labeling, label processing,
   feature extraction).
4. The ``DCNumJobRunner`` spawns several
   :class:`.logic.universal_worker.UniversalWorkerProcess` instances.
   Each universal worker repeatedly asks the slot register for chunk slots
   that are ready for one of its tasks and performs that task, e.g.

   - loading the image data of the next chunk into a free slot,
   - segmenting the images, if the segmenter is a
     :class:`.segm.segmenter_uni.UNISegmenter` (e.g. ``"thresh"`` or
     ``"torchuni"``),
   - labeling the binary masks and post-processing the labels
     (:meth:`.segm.segmenter.Segmenter.process_labels`),
   - extracting the features of the events
     (:class:`.feat.queue_event_extractor.QueueEventExtractor`), including
     gating (:class:`.feat.gate.Gate`). The events are put into an
     event queue.

   For segmenters that do not run in the universal workers
   (:class:`.segm.segmenter_mpo.MPOSegmenter` and
   :class:`.segm.segmenter_sto.STOSegmenter`, e.g. ``"torchmpo"`` and
   ``"torchsto"``), a :class:`.segm.segmenter_manager_thread.SegmenterManagerThread`
   performs the segmentation of the chunk slots.
5. A :class:`.write.queue_writer_process.QueueWriterProcess` takes the
   events from the event queue, puts them in the right order, and writes
   them in chunks to the output file.
6. Finally, the basin strategy is applied, metadata, logs and pipeline
   identifiers are written, and the temporary output file is renamed to
   the output path.
