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

    job = logic.DCNumPipelineJob(path_in="input.rtdc")
    with logic.DCNumJobRunner(job=job) as runner:
        runner.run()

Take a look at the keyword arguments that the classes mentioned above
accept. Note that you can specify methods for background correction as
well as segmentation, and that you have full access to the keyword arguments
for every step in the pipeline. Also note that a reproducible PPID is derived
from these keyword arguments (:meth:`.logic.job.DCNumPipelineJob.get_ppid`).

The following happens when you run the above code snippet:

1. The file `input.rtdc` is opened using the module :mod:`.read`.
2. The ``DCNumJobRunner`` creates two managers:

   - :class:`.segm.segmenter_manager_thread.SegmenterManagerThread` which spawns
     segmentation workers (subclasses of :class:`.segm.segmenter.Segmenter`)
     in separate subprocesses.
   - :class:`.feat.event_extractor_manager_thread.EventExtractorManagerThread`
     which spawns feature extraction workers
     (:class:`.feat.queue_event_extractor.QueueEventExtractor`) in
     separate subprocesses.
3. The segmentation workers read a chunk of image data and return the label
   image (integer-valued labels, one mask per event in a frame).
4. The label images are fed via a shared array to the feature extraction
   workers.
5. The feature extraction workers put the event information (one event per
   unique integer-labeled mask in the label image) in the event queue.
6. A :class:`write.queue_collector_thread.QueueCollectorThread` puts the
   events in the right order and stages them for writing in chunks.
7. A :class:`write.dequeue_writer_thread.DequeWriterThread` writes the
   chunks to the output file.
