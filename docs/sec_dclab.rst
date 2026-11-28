.. _sec_dclab:

dcnum and dclab
===============
dcnum reads and writes the same .rtdc file format as
`dclab <https://dclab.readthedocs.io>`_, and several of its classes have
a counterpart in dclab. The dcnum classes are not replacements for the
dclab classes: they only implement what the processing pipeline needs
and are optimized for it. For analyzing the output files of dcnum, use
dclab (or Shape-Out).


.. _sec_dclab_hdf5data:

HDF5Data and dclab.RTDCBase
---------------------------
:class:`.read.hdf5_data.HDF5Data` is the input reader of the
pipeline. It gives access to the same data as a dclab dataset
(:class:`dclab.rtdc_dataset.RTDCBase`), but differs in what it supports
and how it accesses the data:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Aspect
     - :class:`.read.hdf5_data.HDF5Data`
     - :class:`dclab.rtdc_dataset.RTDCBase`
   * - File formats
     - HDF5-based .rtdc files only (path, :class:`h5py.File`, or
       file-like object).
     - HDF5 (.rtdc), tdms, DCOR, S3, HTTP, dictionaries, and
       hierarchy children (:class:`dclab.rtdc_dataset.RTDC_Hierarchy`).
   * - Multiprocessing
     - Can be pickled: the HDF5 file is opened lazily, and the metadata,
       logs, tables, and basin definitions are passed along, so worker
       processes do not have to read them again.
     - Not designed to be pickled.
   * - Image data
     - ``image``, ``image_bg``, and ``mask`` are accessed via chunked
       caches (:class:`.read.cache.HDF5ImageCache`) whose chunk size is
       aligned to the HDF5 chunks, which makes reading consecutive
       images fast. ``image_corr`` (``image`` minus ``image_bg``) is
       computed on the fly (:class:`.read.cache.ImageCorrCache`).
     - Feature data are accessed via the generic feature interface;
       ``image_corr`` is an ancillary feature.
   * - Scalar features
     - Read completely into memory on first access and cached.
     - Read lazily; ancillary features are cached.
   * - Missing features
     - Not computed. Only features stored in the file or in its basins
       are available (``KeyError`` otherwise).
     - Ancillary features (e.g. volume or Young's modulus) are computed
       on demand if the required data are available.
   * - Basins
     - Only file-based basins (local HDF5 files) and internal basins
       (including chopped image data, "h5datasetchop"); other basin
       types are ignored. Internal basins are searched first.
     - All basin types (file, internal, remote: DCOR, S3, HTTP).
   * - Subsets of events
     - ``index_mapping`` (an integer, slice, or list of indices) selects
       a subset of events transparently. It is part of the data pipeline
       identifier.
     - Box and polygon filters (:attr:`dclab.rtdc_dataset.RTDCBase.filter`)
       and hierarchy children.
   * - Metadata
     - ``meta`` contains the HDF5 attributes as a flat dictionary (e.g.
       ``"imaging:pixel size"``), ``meta_nest`` as a nested dictionary
       similar to dclab's ``config``. The metadata are not validated and
       no default values are added.
     - ``config`` is a validated, nested configuration with default
       values for missing keys.
   * - Pixel size
     - Can be overridden with the ``pixel_size`` argument (e.g. to
       correct a wrong pixel size); the value is part of the data
       pipeline identifier.
     - Can be changed via ``config["imaging"]["pixel size"]``.
   * - Logs and tables
     - Read completely when the instance is created.
     - Read lazily.
   * - Pipeline identifier
     - :meth:`.read.hdf5_data.HDF5Data.get_ppid` returns the data
       pipeline identifier (pixel size and index mapping), which is part
       of the pipeline hash (see :ref:`sec_pipeline_ppid`).
     - No pipeline identifiers.


.. _sec_dclab_hdf5writer:

HDF5Writer and dclab.RTDCWriter
-------------------------------
:class:`.write.writer.HDF5Writer` writes the output file of the
pipeline. In the pipeline, it is used by the queue writer
(:mod:`.write.queue_writer_base`), which collects the events from the
feature extraction workers and writes them in the correct order. It
writes the same file format as dclab's
:class:`dclab.rtdc_dataset.writer.RTDCWriter`, but it is a low-level writer
without the checks and conveniences of dclab:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Aspect
     - :class:`.write.writer.HDF5Writer`
     - :class:`dclab.rtdc_dataset.writer.RTDCWriter`
   * - Feature names
     - Not checked; any feature name can be written
       (:meth:`.write.writer.HDF5Writer.store_feature_chunk`).
     - Must be defined in dclab (``ValueError`` otherwise); the data
       type of special features (e.g. ``index``, contours, traces) is
       handled by dclab.
   * - Writing data
     - Feature data are appended in chunks of events (arrays whose
       first axis is the event axis). Boolean masks are stored as
       ``uint8`` (0 and 255).
     - Feature data are appended per call; ragged data (contours) and
       traces are supported. Masks are stored as ``uint8`` as well.
   * - Existing data
     - New data are always appended to existing features.
     - The ``mode`` argument defines whether data are appended,
       replaced, or the file is reset.
   * - Metadata
     - Written directly to the HDF5 attributes
       (``writer.h5.attrs``), without validation.
       :func:`.write.writer.copy_metadata` copies the metadata, logs,
       and tables of another file.
     - :meth:`~dclab.rtdc_dataset.writer.RTDCWriter.store_metadata` validates
       the metadata. When used as a context manager, the writer
       completes the event count and the ROI size
       (:meth:`~dclab.rtdc_dataset.writer.RTDCWriter.rectify_metadata`) and
       stores the dclab version when the file is closed.
   * - Logs and tables
     - Logs (:meth:`.write.writer.HDF5Writer.store_log`); tables only
       via :func:`.write.writer.copy_metadata`.
     - Logs and tables
       (:meth:`~dclab.rtdc_dataset.writer.RTDCWriter.store_log`,
       :meth:`~dclab.rtdc_dataset.writer.RTDCWriter.store_table`).
   * - Basins
     - File-based basins (HDF5 files) and internal basins
       (:meth:`.write.writer.HDF5Writer.store_basin`).
     - File, internal, and remote basins (DCOR, S3, HTTP).
   * - Compression and checksums
     - Zstandard compression (level 5) and Fletcher-32 checksums by
       default (:func:`.write.writer.set_default_filter_kwargs`); can be
       changed with ``ds_kwds``.
     - Zstandard compression (level 5) and Fletcher-32 checksums by
       default; can be changed with ``compression_kwargs``.
   * - Chunking
     - Chunks of about 1 MiB, but at least 10 and at most 1000 events
       per chunk, so that appending small numbers of events is fast.
     - Chunks of about 1 MiB, at least 10 events per chunk.
   * - HDF5 chunk cache
     - Files opened by the writer use a larger chunk cache (10 MiB per
       dataset) to allow partial writes of chunks.
     - Default h5py chunk cache.


.. _sec_dclab_compress:

compress and dclab-compress
---------------------------
:func:`.write.compression.compress` writes a compressed copy of an .rtdc
file, like ``dclab-compress``. With ``crop_event_images=True``, the
"image" feature is stored as chopped image data (only the pixels around
the events), like ``dclab-compress --crop-event-images``. The chopped
data are identical to those written by dclab, so both packages can read
them.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Aspect
     - :func:`.write.compression.compress`
     - ``dclab-compress``
   * - Parallelization
     - The data are compressed in parallel worker processes
       (``num_procs``); the main process writes the compressed chunks.
     - Single process.
   * - File basins
     - All features of the file basins of the input file are written to
       the output file; ``basin_paths`` only gives the locations of
       basin files (e.g. if they were moved).
     - Only the features of the basin files given with ``-b`` are
       written to the output file.
   * - Cropping
     - Only "image"; methods "dilation" (default) and "box".
     - "image" (via the command line); methods "dilation" (default) and
       "box".
   * - Compression
     - Zstandard (levels 1 to 9) or none, with Fletcher-32 checksums.
       Features that are already compressed with Zstandard at the same or
       a higher level are copied without compressing them again.
     - Zstandard (level 5) with Fletcher-32 checksums.
   * - Log
     - "dcnum-compress" log with the parameters, the wall-clock time,
       the compression time (summed over the worker processes), and the
       time spent writing to disk.
     - "dclab-compress" log with the input files, the date, and the
       system, Python, and library versions (no timing information).
