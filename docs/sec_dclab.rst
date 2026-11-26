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
     - Only file-based basins (local HDF5 files) and internal basins;
       other basin types are ignored. Internal basins are searched
       first.
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
