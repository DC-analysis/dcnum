.. _sec_getting_started:

Getting started
===============

Installation
------------
dcnum requires Python 3.11 or newer and can be installed from PyPI::

    pip install dcnum

The default installation contains everything needed for background
computation, segmentation by thresholding, and feature extraction.
Optional dependencies are available as extras:

.. list-table::
   :header-rows: 1

   * - Extra
     - Description

   * - ``torch``
     - | Installs `PyTorch <https://pytorch.org>`_ for segmentation with
       | machine-learning models (segmenters ``"torchmpo"``,
       | ``"torchsto"`` and ``"torchuni"``):
       | ``pip install dcnum[torch]``

   * - ``openvino``
     - | Installs `OpenVINO <https://docs.openvino.ai>`_, an additional
       | inference backend for models that support it (``"torchuni"``
       | segmenter): ``pip install dcnum[openvino]``

For GPU-based segmentation with CUDA, make sure that you install a version
of PyTorch that matches your CUDA installation (see the
`PyTorch installation instructions <https://pytorch.org/get-started/locally/>`_).


Running a pipeline
------------------
The input for dcnum is an .rtdc file that contains the raw image data of
a measurement. A pipeline is described by a
:class:`~dcnum.logic.job.DCNumPipelineJob` and executed by a
:class:`~dcnum.logic.ctrl.DCNumJobRunner`:

.. code:: python

    from dcnum.logic import DCNumPipelineJob, DCNumJobRunner

    job = DCNumPipelineJob(path_in="measurement.rtdc")
    with DCNumJobRunner(job=job) as runner:
        runner.run()

This runs the default pipeline (sparse median background, segmentation by
thresholding, all features) and writes the result to
``measurement_dcn.rtdc`` next to the input file. You can choose a
different output location with the ``path_out`` keyword argument.
The output file contains the extracted features of all events,
the metadata of the input file, and the log of the pipeline run.
You can open it with any software that supports the .rtdc format,
e.g. with `dclab <https://dclab.readthedocs.io>`_:

.. code:: python

    import dclab

    ds = dclab.new_dataset("measurement_dcn.rtdc")
    print(len(ds), ds["deform"].mean())


Configuring the pipeline
------------------------
Each step of the pipeline is selected with a ``*_code`` keyword argument
and configured with the corresponding ``*_kwargs`` dictionary:

.. code:: python

    job = DCNumPipelineJob(
        path_in="measurement.rtdc",
        path_out="analysis/measurement_custom.rtdc",
        # background computation ("sparsemed", "rollmed", or "copy")
        background_code="sparsemed",
        background_kwargs={"kernel_size": 150},
        # segmentation with a lower threshold and less mask smoothing
        segmenter_code="thresh",
        segmenter_kwargs={"thresh": -5,
                          "kwargs_mask": {"closing_disk": 1}},
        # ignore objects with 20 pixels or fewer
        gate_kwargs={"size_thresh_mask": 20},
        # do not copy the raw image data to the output file
        basin_strategy="tap",
    )

The available keyword arguments are documented in the code reference of
the respective classes:

.. list-table::
   :header-rows: 1

   * - Keyword argument
     - Classes

   * - ``background_code``, ``background_kwargs``
     - :class:`~dcnum.feat.feat_background.bg_sparse_median.BackgroundSparseMed`,
       :class:`~dcnum.feat.feat_background.bg_roll_median.BackgroundRollMed`,
       :class:`~dcnum.feat.feat_background.bg_copy.BackgroundCopy`

   * - ``segmenter_code``, ``segmenter_kwargs``
     - :class:`~dcnum.segm.segm_thresh.SegmentThresh`,
       :class:`~dcnum.segm.segm_torch.segm_torch_mpo.SegmentTorchMPO`,
       :class:`~dcnum.segm.segm_torch.segm_torch_sto.SegmentTorchSTO`,
       :class:`~dcnum.segm.segm_torch.segm_torch_uni.SegmentTorchUNI`;
       mask post-processing (``kwargs_mask``):
       :meth:`~dcnum.segm.segmenter.Segmenter.process_labels`

   * - ``feature_kwargs``
     - :meth:`~dcnum.feat.queue_event_extractor.QueueEventExtractor.get_events_from_masks`

   * - ``gate_kwargs``
     - :class:`~dcnum.feat.gate.Gate`

   * - ``data_kwargs``
     - :class:`~dcnum.read.hdf5_data.HDF5Data`

Further keyword arguments of :class:`~dcnum.logic.job.DCNumPipelineJob`
control the compression of the output file (``compression``), the number
of processes (``num_procs``, defaults to the number of physical CPU cores),
and the logging level. With ``debug=True``, all workers run as threads in
the main process, which makes debugging easier.

All keyword arguments that affect the result are part of the pipeline
identifier (PPID) of the job, which you can inspect before running it:

.. code:: python

    print(job.get_ppid())


Segmentation with machine-learning models
-----------------------------------------
The ``torch*`` segmenters require a dcnum model file (.dcnm), which
contains the trained model and metadata describing for which kind of
data the model is suitable. Pass the path to the model file via
``segmenter_kwargs``:

.. code:: python

    job = DCNumPipelineJob(
        path_in="measurement.rtdc",
        segmenter_code="torchmpo",
        segmenter_kwargs={"model_file": "/path/to/model.dcnm"},
    )
    # raises SegmenterNotApplicableError if the model does not fit the data
    job.validate()

Only the file name of the model is stored in the PPID. If you keep your
models in a dedicated directory, you can register that directory and then
pass only the file name:

.. code:: python

    from dcnum.meta import paths

    paths.register_search_path("torch_model_files", "/path/to/models")

Which of the ``torch*`` segmenters to use depends on your hardware:
``"torchmpo"`` runs the model on the CPU in several processes,
``"torchsto"`` processes large batches in a single process (e.g. on a
CUDA GPU), and ``"torchuni"`` additionally supports the inference backends
and devices listed in the model file (``backend`` and ``device`` keyword
arguments).


Monitoring progress
-------------------
:class:`~dcnum.logic.ctrl.DCNumJobRunner` is a
:class:`threading.Thread`. To keep your program responsive (e.g. in a GUI),
start the runner in the background and query its status with
:meth:`~dcnum.logic.ctrl.DCNumJobRunner.get_status`:

.. code:: python

    import time

    runner = DCNumJobRunner(job=job)
    runner.start()
    while runner.is_alive():
        status = runner.get_status()
        print(f"{status['state']}: {status['progress']:.0%}")
        time.sleep(1)
    if runner.state == "error":
        print(runner.error_tb)
    runner.join()

The state of the runner is one of ``"init"``, ``"setup"``,
``"background"``, ``"segmentation"``, ``"plumbing"``, ``"cleanup"``,
``"done"``, or ``"error"``. If an error occurs in the background thread,
the traceback is stored in ``runner.error_tb``.
