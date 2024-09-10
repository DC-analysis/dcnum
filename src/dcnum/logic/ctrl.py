import collections
import datetime
import hashlib
import importlib
import json
import logging
from logging.handlers import QueueListener
import multiprocessing as mp
import os
import pathlib
import platform
import socket
import threading
import time
import traceback
import uuid

import h5py
import numpy as np

from ..feat.feat_background.base import get_available_background_methods
from ..feat.queue_event_extractor import QueueEventExtractor
from ..feat import gate
from ..feat import EventExtractorManagerThread
from ..segm import SegmenterManagerThread, get_available_segmenters
from ..meta import ppid
from ..read import HDF5Data, get_mapping_indices
from .._version import version, version_tuple
from ..write import (
    DequeWriterThread, HDF5Writer, QueueCollectorThread, copy_features,
    copy_metadata, create_with_basins, set_default_filter_kwargs
)

from .job import DCNumPipelineJob
from .json_encoder import ExtendedJSONEncoder


# Force using "spawn" method for multiprocessing, because we are using
# queues and threads and would end up with race conditions otherwise.
mp_spawn = mp.get_context("spawn")

#: valid states for a job runnter. The states must be in logical ordern,
#: not in alphabetical order.
valid_states = [
    "created",
    "init",
    "setup",
    "background",
    "segmentation",
    "plumbing",
    "cleanup",
    "done",
    "error",
]


class DCNumJobRunner(threading.Thread):
    def __init__(self,
                 job: DCNumPipelineJob,
                 tmp_suffix: str = None,
                 *args, **kwargs):
        """Run a pipeline as defined by a :class:`DCNumPipelineJob` instance

        Parameters
        ----------
        job: DCNumPipelineJob
            pipeline job to run
        tmp_suffix: str
            optional unique string for creating temporary files
            (defaults to hostname)
        """
        super(DCNumJobRunner, self).__init__(*args, **kwargs)
        self.error_tb = None
        self.job = job
        if tmp_suffix is None:
            tmp_suffix = f"{socket.gethostname()}_{str(uuid.uuid4())[:5]}"
        self.tmp_suffix = tmp_suffix
        self.ppid, self.pphash, self.ppdict = job.get_ppid(ret_hash=True,
                                                           ret_dict=True)
        self.event_count = 0

        self._data_raw = None
        self._data_temp_in = None
        # current job state
        self._state = "init"
        # overall progress [0, 1]
        self._progress_bg = None  # background
        self._progress_ex = None  # segmentation
        self._progress_bn = None  # creating basins
        # segmentation frame rate
        self._segm_rate = 0

        # Set up logging
        # General logger for this job
        self.main_logger = logging.getLogger("dcnum")
        self.main_logger.setLevel(job["log_level"])
        # Log file output in target directory
        self.path_log = job["path_out"].with_suffix(".log")
        self.path_log.parent.mkdir(exist_ok=True, parents=True)
        self.path_log.unlink(missing_ok=True)
        self._log_file_handler = logging.FileHandler(
            filename=self.path_log,
            encoding="utf-8",
            delay=True,
            errors="ignore",
        )
        # Set the log file handler level to DEBUG, so it logs everything
        # presented to it.
        self._log_file_handler.setLevel(logging.DEBUG)
        fmt = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt='%H:%M:%S'
        )
        self._log_file_handler.setFormatter(fmt)
        self.main_logger.addHandler(self._log_file_handler)
        handlers = list(self.main_logger.handlers)

        # Queue for subprocesses to log to
        self.log_queue = mp_spawn.Queue()
        self._qlisten = QueueListener(self.log_queue, *handlers,
                                      respect_handler_level=True)
        self._qlisten.start()

        if job["debug"]:
            self.main_logger.info("Note that in debugging mode, duplicate "
                                  "log entries may appear (logs that are "
                                  "recorded via queues)")

        self.logger = logging.getLogger(f"dcnum.Runner-{self.pphash[:2]}")

        # Sanity checks
        for os_env in ["MKL_NUM_THREADS", "NUMBA_NUM_THREADS",
                       "NUMEXPR_NUM_THREADS", "NUMPY_NUM_THREADS",
                       "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                       "VECLIB_MAXIMUM_THREADS"]:
            # You should disable multithreading for all major tools that
            # use dcnum.logic. We don't want multithreading, because dcnum
            # uses linear code and relies on multiprocessing for
            # parallelization. This has to be done before importing numpy
            # or any other library affected. In your scripts, you can use:
            #
            #    os.environ.setdefault("MKL_NUM_THREADS", "1")
            #    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
            #    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
            #    os.environ.setdefault("NUMPY_NUM_THREADS", "1")
            #    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
            #    os.environ.setdefault("OMP_NUM_THREADS", "1")
            #    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
            #
            val_act = os.environ.get(os_env)
            if val_act != "1":
                self.logger.warning(
                    f"Make sure to set the environment variable {os_env} to "
                    f"'1' (disables multithreading)! Other values will reduce "
                    f"performance and your system may become inresponsive. "
                    f"The current value is '{val_act}'.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # If an error occurred, don't delete the log and basin files.
        delete_temporary_files = exc_type is None
        self.close(delete_temporary_files=delete_temporary_files)

    @property
    def draw(self) -> HDF5Data:
        """Raw input data"""
        if self._data_raw is None:
            # Initialize with the proper kwargs (pixel_size)
            self._data_raw = HDF5Data(self.job["path_in"],
                                      **self.job["data_kwargs"])
        return self._data_raw

    @property
    def dtin(self) -> HDF5Data:
        """Input data with (corrected) background image"""
        if self._data_temp_in is None:
            if not self.path_temp_in.exists():
                # create basin-based input file
                create_with_basins(path_out=self.path_temp_in,
                                   basin_paths=[self.draw.path])
            # Initialize with the proper kwargs (pixel_size)
            self._data_temp_in = HDF5Data(self.path_temp_in,
                                          **self.job["data_kwargs"])
            assert len(self._data_temp_in) > 0
            assert "image_bg" in self._data_temp_in
        return self._data_temp_in

    @property
    def path_temp_in(self):
        po = pathlib.Path(self.job["path_out"])
        return po.with_name(po.stem + f"_input_bb_{self.tmp_suffix}.rtdc~")

    @property
    def path_temp_out(self):
        po = pathlib.Path(self.job["path_out"])
        return po.with_name(po.stem + f"_output_{self.tmp_suffix}.rtdc~")

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        if state not in valid_states:
            raise ValueError(f"Invalid state '{state}' specified!")
        self._state = state

    def close(self, delete_temporary_files=True):
        if self._data_raw is not None:
            self._data_raw.close()
            self._data_raw = None
        if self._data_temp_in is not None:
            self._data_temp_in.close()
            self._data_temp_in = None
        # clean up logging
        if self._log_file_handler in self.main_logger.handlers:
            self.main_logger.removeHandler(self._log_file_handler)
            self._log_file_handler.flush()
            self._log_file_handler.close()
        if self._qlisten is not None:
            self._qlisten.stop()
            self._qlisten = None
        self.log_queue.cancel_join_thread()
        self.log_queue.close()
        if delete_temporary_files:
            # Delete log file on disk
            self.path_log.unlink(missing_ok=True)
            # Delete temporary input file
            self.path_temp_in.unlink(missing_ok=True)
            # We don't have to delete self.path_temp_out, since this one
            # is `rename`d to `self.jon["path_out"]`.

    def join(self, delete_temporary_files=True, *args, **kwargs):
        super(DCNumJobRunner, self).join(*args, **kwargs)
        # Close only after join
        self.close(delete_temporary_files=delete_temporary_files)

    def get_status(self):
        # Compute the total progress. The following weights indicate
        # how much fractional time each processing step takes.
        bgw = 4  # fraction of background
        exw = 27  # fraction of segmentation and feature extraction
        if self.job["basin_strategy"] == "drain":
            drw = 15  # because data need to be copied
        else:
            drw = 1  # just creating the basins in output file
        clw = 1  # fraction of cleanup operations
        tot = bgw + exw + drw + clw
        progress = 0
        st = self.state

        # background
        if valid_states.index(st) > valid_states.index("background"):
            # background already computed
            progress += bgw / tot
        elif self._progress_bg is not None:
            # This is the image count of the input dataset.
            progress += self._progress_bg.value * bgw / tot

        # segmentation
        if valid_states.index(st) > valid_states.index("segmentation"):
            # segmentation already done
            progress += exw / tot
        elif self._progress_ex is not None:
            progress += self._progress_ex * exw / tot

        # draining basins
        if valid_states.index(st) > valid_states.index("plumbing"):
            # plumbing already done
            progress += drw / tot
        if self._progress_bn is not None:
            progress += self._progress_bn * drw / tot

        if self.state == "done":
            progress = 1

        return {
            "progress": progress,
            "segm rate": self._segm_rate,
            "state": self._state,
        }

    def run(self):
        try:
            self.run_pipeline()
        except BaseException:
            self.state = "error"
            self.error_tb = traceback.format_exc()
            if not self.is_alive():
                # Thread has not been started. This means we are not running
                # in a thread but in the main process. Raise the exception.
                raise

    def run_pipeline(self):
        """Execute the pipeline job"""
        time_start = time.monotonic()
        time_string = time.strftime("%Y-%m-%d-%H.%M.%S", time.gmtime())
        self.logger.info(f"Run start: {time_string}")
        if self.job["path_out"].exists():
            raise FileExistsError(
                f"Output file {self.job['path_out']} already exists!")
        # Make sure the output directory exists.
        self.job["path_out"].parent.mkdir(parents=True, exist_ok=True)
        self.state = "setup"
        # First get a list of all pipeline IDs. If the input file has
        # already been processed by dcnum, then we do not have to redo
        # everything.
        # Crucial here is the fact that we also compare the
        # "pipeline:dcnum hash" in case individual steps of the pipeline
        # have been run by a rogue data analyst.
        datdict = {
            "gen_id": self.draw.h5.attrs.get("pipeline:dcnum generation", "0"),
            "dat_id": self.draw.h5.attrs.get("pipeline:dcnum data", "0"),
            "bg_id": self.draw.h5.attrs.get("pipeline:dcnum background", "0"),
            "seg_id": self.draw.h5.attrs.get("pipeline:dcnum segmenter", "0"),
            "feat_id": self.draw.h5.attrs.get("pipeline:dcnum feature", "0"),
            "gate_id": self.draw.h5.attrs.get("pipeline:dcnum gate", "0"),
        }
        # The hash of a potential previous pipeline run.
        dathash = self.draw.h5.attrs.get("pipeline:dcnum hash", "0")
        # The number of events extracted in a potential previous pipeline run.
        evyield = self.draw.h5.attrs.get("pipeline:dcnum yield", -1)
        redo_sanity = (
            # Whether pipeline hash is invalid.
            ppid.compute_pipeline_hash(**datdict) != dathash
            # Whether the input file is the original output of the pipeline.
            or len(self.draw) != evyield
            # If index mapping is defined, then we always redo the pipeline.
            # If the pipeline hashes are identical and index mapping is not
            # None, then both pipelines were done with index mapping.
            # But applying the same pipeline with index mapping in series
            # will lead to a different result in the second run (e.g. 1st
            # pipeline run: take every 2nd event; 2nd pipeline run: take
            # every second event -> results in every 4th event in output of
            # second pipeline run).
            or self.draw.index_mapping is not None
        )
        # Do we have to recompute the background data? In addition to the
        # hash sanity check above, check the generation, input data,
        # and background pipeline identifiers.
        redo_bg = (
            "image_bg" not in self.draw
            or (datdict["gen_id"] != self.ppdict["gen_id"])
            or (datdict["dat_id"] != self.ppdict["dat_id"])
            or (datdict["bg_id"] != self.ppdict["bg_id"]))

        # Do we have to rerun segmentation and feature extraction? Check
        # the segmentation, feature extraction, and gating pipeline
        # identifiers.
        redo_seg = (
            redo_sanity
            or redo_bg
            or (datdict["seg_id"] != self.ppdict["seg_id"])
            or (datdict["feat_id"] != self.ppdict["feat_id"])
            or (datdict["gate_id"] != self.ppdict["gate_id"]))

        self.state = "background"

        if redo_bg:
            # The 'image_bg' feature is written to `self.path_temp_in`.
            # If `job["path_in"]` already has the correct 'image_bg'
            # feature, then we never reach this case here
            # (note that `self.path_temp_in` is basin-based).
            self.task_background()

        self.state = "segmentation"

        # We have the input data covered, and we have to run the
        # long-lasting segmentation and feature extraction step.
        # We are taking into account two scenarios:
        # A) The segmentation step is exactly the one given in the input
        #    file. Here it is sufficient to use a basin-based
        #    output file `self.path_temp_out`.
        # B) Everything else (including background pipeline mismatch or
        #    different segmenters); Here, we simply populate `path_temp_out`
        #    with the data from the segmenter.
        if redo_seg:
            # scenario B (Note this implies `redo_bg`)
            self.task_segment_extract()
        else:
            # scenario A
            # Access the temporary input HDF5Data so that the underlying
            # basin file is created and close it immediately afterward.
            self.dtin.close()
            self._data_temp_in = None
            # Note any new actions that work on `self.path_temp_in` are not
            # reflected in `self.path_temp_out`.
            self.path_temp_in.rename(self.path_temp_out)
            # Since no segmentation was done, the output file now does not
            # contain any events. This is not really what we wanted, but we
            # can still store all features in the output file if required.
            if self.job["basin_strategy"] == "drain":
                orig_feats = []
                for feat in self.draw.h5["events"].keys():
                    if isinstance(self.draw.h5["events"][feat], h5py.Dataset):
                        # copy_features does not support Groups
                        orig_feats.append(feat)
                with h5py.File(self.path_temp_out, "a") as h5_dst:
                    copy_features(h5_src=self.draw.h5,
                                  h5_dst=h5_dst,
                                  features=orig_feats,
                                  mapping=None)

        # Handle basin data according to the user's request
        self.state = "plumbing"
        self.task_enforce_basin_strategy()

        self.state = "cleanup"

        with HDF5Writer(self.path_temp_out) as hw:
            # pipeline metadata
            hw.h5.attrs["pipeline:dcnum generation"] = self.ppdict["gen_id"]
            hw.h5.attrs["pipeline:dcnum data"] = self.ppdict["dat_id"]
            hw.h5.attrs["pipeline:dcnum background"] = self.ppdict["bg_id"]
            hw.h5.attrs["pipeline:dcnum segmenter"] = self.ppdict["seg_id"]
            hw.h5.attrs["pipeline:dcnum feature"] = self.ppdict["feat_id"]
            hw.h5.attrs["pipeline:dcnum gate"] = self.ppdict["gate_id"]
            hw.h5.attrs["pipeline:dcnum hash"] = self.pphash
            hw.h5.attrs["pipeline:dcnum yield"] = self.event_count
            # index mapping information
            im = self.job.kwargs["data_kwargs"].get("index_mapping", None)
            dim = HDF5Data.get_ppid_index_mapping(im)
            hw.h5.attrs["pipeline:dcnum mapping"] = dim
            # regular metadata
            hw.h5.attrs["experiment:event count"] = self.event_count
            hw.h5.attrs["imaging:pixel size"] = self.draw.pixel_size
            # Add job information to resulting .rtdc file
            hw.store_log(f"dcnum-job-{time_string}",
                         json.dumps({
                             "dcnum version": version_tuple,
                             "job": self.job.__getstate__(),
                             "pipeline": {"identifiers": self.ppdict,
                                          "hash": self.pphash,
                                          },
                             "python": {
                                 "build": ", ".join(platform.python_build()),
                                 "implementation":
                                     platform.python_implementation(),
                                 "libraries": get_library_versions_dict([
                                     "cv2",
                                     "h5py",
                                     "mahotas",
                                     "numba",
                                     "numpy",
                                     "scipy",
                                     "skimage",
                                     "torch",
                                 ]),
                                 "version": platform.python_version(),
                                 },
                             "system": {
                                 "info": platform.platform(),
                                 "machine": platform.machine(),
                                 "name": platform.system(),
                                 "release": platform.release(),
                                 "version": platform.version(),
                                 },
                             "tasks": {"background": redo_bg,
                                       "segmentation": redo_seg
                                       },
                             },
                             indent=2,
                             sort_keys=True,
                             cls=ExtendedJSONEncoder,
                         ).split("\n"))

            # copy metadata/logs/tables from original file
            with h5py.File(self.job["path_in"]) as h5_src:
                copy_metadata(h5_src=h5_src, h5_dst=hw.h5)
            if redo_seg:
                # Store the correct measurement identifier. This is used to
                # identify this file as a correct basin in subsequent pipeline
                # steps, and it also makes sure that the original file cannot
                # become a basin by accident (we have different indexing).
                # This is the identifier appendix that we use to identify this
                # dataset. Note that we only override the run identifier when
                # segmentation did actually take place.
                mid_ap = f"dcn-{self.pphash[:7]}"
                # This is the current measurement identifier
                mid_cur = hw.h5.attrs.get("experiment:run identifier")
                if not mid_cur:
                    # Compute a measurement identifier from the metadata
                    m_time = hw.h5.attrs.get("experiment:time", "none")
                    m_date = hw.h5.attrs.get("experiment:date", "none")
                    m_sid = hw.h5.attrs.get("setup:identifier", "none")
                    hasher = hashlib.md5(
                        f"{m_time}_{m_date}_{m_sid}".encode("utf-8"))
                    mid_cur = str(uuid.UUID(hex=hasher.hexdigest()))
                # The new measurement identifier is a combination of both.
                mid_new = f"{mid_cur}_{mid_ap}" if mid_cur else mid_ap
                hw.h5.attrs["experiment:run identifier"] = mid_new

        trun = datetime.timedelta(seconds=round(time.monotonic() - time_start))
        self.logger.info(f"Run duration: {str(trun)}")
        self.logger.info(time.strftime("Run stop: %Y-%m-%d-%H.%M.%S",
                                       time.gmtime()))
        # Add the log file to the resulting .rtdc file
        if self.path_log.exists():
            with HDF5Writer(self.path_temp_out) as hw:
                hw.store_log(
                    f"dcnum-log-{time_string}",
                    self.path_log.read_text().strip().split("\n"))

        # Rename the output file
        self.path_temp_out.rename(self.job["path_out"])
        self.state = "done"

    def task_background(self):
        """Perform background computation task

        This populates the file `self.path_temp_in` with the 'image_bg'
        feature.
        """
        self.logger.info("Starting background computation")
        if self._data_temp_in is not None:
            # Close the temporary input data file, so we can write to it.
            self._data_temp_in.close()
            self._data_temp_in = None
        # Start background computation
        bg_code = self.job["background_code"]
        bg_cls = get_available_background_methods()[bg_code]
        with bg_cls(
                input_data=self.job["path_in"],
                output_path=self.path_temp_in,
                # always compress, the disk is usually the bottleneck
                compress=True,
                num_cpus=self.job["num_procs"],
                # custom kwargs
                **self.job["background_kwargs"]) as bic:
            self._progress_bg = bic.image_proc
            bic.process()
        self.logger.info("Finished background computation")

    def task_enforce_basin_strategy(self):
        """Transfer basin data from input files to output if requested

        The user specified the "basin_strategy" keyword argument in
        `self.job`. If this is set to "drain", then copy all basin
        information from the input file to the output file. If it
        is set to "tap", then only create basins in the output file.
        """
        self._progress_bn = 0
        t0 = time.perf_counter()
        # We have these points to consider:
        # - We must use the `basinmap` feature to map from the original
        #   file to the output file.
        # - We must copy "bg_off" and "image_bg" to the output file.
        # - For the "drain" basin strategy, we also have to copy all the
        #   other features.
        # - If "image_bg" is defined as an internal basin in the input
        #   file, we have to convert the mapping and store a corresponding
        #   internal basin in the output file.

        # Determine the basinmap feature
        with HDF5Writer(self.path_temp_out) as hw:
            hout = hw.h5
            # First, we have to determine the basin mapping from input to
            # output. This information is stored by the QueueCollectorThread
            # in the "basinmap0" feature, ready to be used by us.
            if "index_unmapped" in hout["events"]:
                # The unmapped indices enumerate the events in the output file
                # with indices from the mapped input file. E.g. if for the
                # first image in the input file, two events are found and for
                # the second image in the input file, three events are found,
                # then this would contain [0, 0, 1, 1, 1, ...]. If the index
                # mapping of the input file was set to slice(1, 100), then the
                # first image would not be there, and we would have
                # [1, 1, 1, ...].
                idx_um = hout["events/index_unmapped"][:]

                # If we want to convert this to an actual basinmap feature,
                # then we have to convert those indices to indices that map
                # to the original input HDF5 file.
                raw_im = self.draw.index_mapping
                if raw_im is None:
                    # Create a hard link to save time and space
                    hout["events/basinmap0"] = hout["events/index_unmapped"]
                    basinmap0 = idx_um
                else:
                    self.logger.info("Converting input mapping")
                    basinmap0 = get_mapping_indices(raw_im)[idx_um]
                    # Store the mapped basin data in the output file.
                    hw.store_feature_chunk("basinmap0", basinmap0)
                self.logger.info("Input mapped to output with basinmap0")
                # We don't need them anymore.
                del hout["events/index_unmapped"]

                # Note that `size_raw != (len(self.draw))` [sic!]. The former
                # is the size of the raw dataset and the latter is its mapped
                # size!
                size_raw = self.draw.h5.attrs["experiment:event count"]
                if (len(basinmap0) == size_raw
                        and np.all(basinmap0 == np.arange(size_raw))):
                    # This means that the images in the input overlap perfectly
                    # with the images in the output, i.e. a "copy" segmenter
                    # was used or something is very reproducible.
                    # We set basinmap to None to be more efficient.
                    basinmap0 = None

            else:
                # The input is identical to the output, because we are using
                # the same pipeline identifier.
                basinmap0 = None

            # List of features we have to copy from input to output.
            # We need to make sure that the features are correctly attributed
            # from the input files. E.g. if the input file already has
            # background images, but we recompute the background images, then
            # we have to use the data from the recomputed background file.
            # We achieve this by keeping a specific order and only copying
            # those features that we don't already have in the output file.
            feats_raw = [
                # background data from the temporary input image
                [self.dtin.h5, ["bg_off"], "critical"],
                [self.draw.h5, self.draw.features_scalar_frame, "optional"],
                [self.draw.h5, ["image", "bg_off"], "optional"],
            ]

            # Store image_bg as an internal basin, if defined in input
            for idx in range(len(self.dtin.basins)):
                bn_dict = self.dtin.basins[idx]
                if (bn_dict["type"] == "internal"
                        and "image_bg" in bn_dict["features"]):
                    self.logger.info(
                        "Copying internal basin background images")
                    bn_grp, bn_feats, bn_map = self.dtin.get_basin_data(idx)
                    assert "image_bg" in bn_feats
                    # Load all images into memory (should only be ~600)
                    bg_images1 = self.dtin.h5["basin_events"]["image_bg"][:]
                    # Get the original internal mapping for these images
                    # Note that `basinmap0` always refers to indices in the
                    # original raw input file, and not to indices in an
                    # optional mapped input file (using `index_mapping`).
                    # Therefore, we do `self.dtin.h5["events"]["basinmap0"]`
                    # instead of `self.dtin["basinmap0"]`
                    basinmap_in = self.dtin.h5["events"][bn_dict["mapping"]][:]
                    # Now we have to convert the indices in `basinmap_in`
                    # to indices in the output file.
                    basinmap1 = basinmap_in[basinmap0]
                    # Store the internal mapping in the output file
                    hw.store_basin(name=bn_dict["name"],
                                   description=bn_dict["description"],
                                   mapping=basinmap1,
                                   internal_data={"image_bg": bg_images1}
                                   )
                    break
            else:
                self.logger.info("Background images must be copied")
                # There is no internal image_bg feature, probably because
                # the user did not use the sparsemed background correction.
                # In this case, we simply add "image_bg" to the `feats_raw`.
                feats_raw += [
                    [self.dtin.h5, ["image_bg"], "critical"],
                    [self.draw.h5, ["image_bg"], "optional"],
                ]

            # Copy the features required in the output file.
            for hin, feats, importance in feats_raw:
                # Only consider features that are available in the input
                # and that are not already in the output.
                feats = [f for f in feats
                         if (f in hin["events"] and f not in hout["events"])]
                if not feats:
                    continue
                elif (self.job["basin_strategy"] == "drain"
                      or importance == "critical"):
                    # DRAIN: Copy all features over to the output file.
                    self.logger.debug(f"Transferring {feats} to output file")
                    copy_features(h5_src=hin,
                                  h5_dst=hout,
                                  features=feats,
                                  mapping=basinmap0)
                else:
                    # TAP: Create basins for the "optional" features in the
                    # output file. Note that the "critical" features never
                    # reach this case.
                    self.logger.debug(f"Creating basin for {feats}")
                    # Relative and absolute paths.
                    pin = pathlib.Path(hin.filename).resolve()
                    paths = [pin]
                    pout = pathlib.Path(hout.filename).resolve().parent
                    try:
                        paths.append(os.path.relpath(pin, pout))
                    except ValueError:
                        # This means it is impossible to compute a relative
                        # path (e.g. different drive letter on Windows).
                        pass
                    hw.store_basin(name="dcnum basin",
                                   features=feats,
                                   mapping=basinmap0,
                                   paths=paths,
                                   description=f"Created with dcnum {version}",
                                   )
                self._progress_bn += 1 / len(feats_raw)
        t_tot = time.perf_counter() - t0
        self.logger.info(f"Enforcing basin strategy time: {t_tot:.1f}s")

    def task_segment_extract(self):
        self.logger.info("Starting segmentation and feature extraction")
        # Start writer thread
        writer_dq = collections.deque()
        ds_kwds = set_default_filter_kwargs()
        thr_write = DequeWriterThread(
            path_out=self.path_temp_out,
            dq=writer_dq,
            mode="w",
            ds_kwds=ds_kwds,
        )
        thr_write.start()

        # Start segmentation thread
        seg_cls = get_available_segmenters()[self.job["segmenter_code"]]
        if seg_cls.requires_background_correction:
            imdat = self.dtin.image_corr
        else:
            imdat = self.dtin.image

        if self.job["debug"]:
            num_slots = 1
            num_extractors = 1
            num_segmenters = 1
        elif seg_cls.hardware_processor == "cpu":  # MPO segmenter
            # We could in principle set the number of slots to one and
            # have both number of extractors and number of segmenters set
            # to the total number of CPUs. However, we would need more RAM
            # (for caching the image data) and we also have more overhead.
            # Having two slots shared between all workers is more efficient.
            num_slots = 2
            # Split segmentation and feature extraction workers evenly.
            num_extractors = self.job["num_procs"] // 2
            num_segmenters = self.job["num_procs"] - num_extractors
            # leave one CPU for the writer and the remaining Threads
            num_segmenters -= 1
        else:  # GPU segmenter
            num_slots = 3
            num_extractors = self.job["num_procs"]
            # leave one CPU for the writer and the remaining Threads
            num_extractors -= 1
            num_segmenters = 1
        num_extractors = max(1, num_extractors)
        num_segmenters = max(1, num_segmenters)
        self.job.kwargs["segmenter_kwargs"]["num_workers"] = num_segmenters
        self.job.kwargs["segmenter_kwargs"]["debug"] = self.job["debug"]
        slot_chunks = mp_spawn.Array("i", num_slots, lock=False)
        slot_states = mp_spawn.Array("u", num_slots, lock=False)

        self.logger.debug(f"Number of slots: {num_slots}")
        self.logger.debug(f"Number of segmenters: {num_segmenters}")
        self.logger.debug(f"Number of extractors: {num_extractors}")

        # Initialize segmenter manager thread
        thr_segm = SegmenterManagerThread(
            segmenter=seg_cls(**self.job["segmenter_kwargs"]),
            image_data=imdat,
            bg_off=self.dtin["bg_off"] if "bg_off" in self.dtin else None,
            slot_states=slot_states,
            slot_chunks=slot_chunks,
        )
        thr_segm.start()

        # Start feature extractor thread
        fe_kwargs = QueueEventExtractor.get_init_kwargs(
            data=self.dtin,
            gate=gate.Gate(self.dtin, **self.job["gate_kwargs"]),
            num_extractors=num_extractors,
            log_queue=self.log_queue,
            log_level=self.logger.level,
            )
        fe_kwargs["extract_kwargs"] = self.job["feature_kwargs"]

        thr_feat = EventExtractorManagerThread(
            slot_chunks=slot_chunks,
            slot_states=slot_states,
            fe_kwargs=fe_kwargs,
            num_workers=num_extractors,
            labels_list=thr_segm.labels_list,
            writer_dq=writer_dq,
            debug=self.job["debug"])
        thr_feat.start()

        # Start the data collection thread
        thr_coll = QueueCollectorThread(
            event_queue=fe_kwargs["event_queue"],
            writer_dq=writer_dq,
            feat_nevents=fe_kwargs["feat_nevents"],
            write_threshold=500,
        )
        thr_coll.start()

        data_size = len(self.dtin)
        t0 = time.monotonic()

        # So in principle we are done here. We do not have to do anything
        # besides monitoring the progress.
        while True:
            counted_frames = thr_coll.written_frames
            self.event_count = thr_coll.written_events
            td = time.monotonic() - t0
            # set the current status
            self._progress_ex = counted_frames / data_size
            self._segm_rate = counted_frames / (td or 0.03)
            time.sleep(.5)
            if counted_frames == data_size:
                break

        self.logger.debug("Flushing data to disk")

        # join threads
        join_thread_helper(thr=thr_segm,
                           timeout=30,
                           retries=10,
                           logger=self.logger,
                           name="segmentation")
        # Join the collector thread before the feature extractors. On
        # compute clusters, we had problems with joining the feature
        # extractors, maybe because the event_queue was not depleted.
        join_thread_helper(thr=thr_coll,
                           timeout=600,
                           retries=10,
                           logger=self.logger,
                           name="collector for writer")
        join_thread_helper(thr=thr_feat,
                           timeout=30,
                           retries=10,
                           logger=self.logger,
                           name="feature extraction")
        thr_write.finished_when_queue_empty()
        join_thread_helper(thr=thr_write,
                           timeout=600,
                           retries=10,
                           logger=self.logger,
                           name="writer")

        self.event_count = thr_coll.written_events
        if self.event_count == 0:
            self.logger.error(
                f"No events found in {self.draw.path}! Please check the "
                f"input file or revise your pipeline")

        self.logger.info("Finished segmentation and feature extraction")


def get_library_versions_dict(library_name_list):
    version_dict = {}
    for library_name in library_name_list:
        try:
            lib = importlib.import_module(library_name)
        except BaseException:
            version = None
        else:
            version = lib.__version__
        version_dict[library_name] = version
    return version_dict


def join_thread_helper(thr, timeout, retries, logger, name):
    for _ in range(retries):
        thr.join(timeout=timeout)
        if thr.is_alive():
            logger.info(f"Waiting for '{name}' ({thr}")
        else:
            logger.debug(f"Joined thread '{name}'")
            break
    else:
        logger.error(f"Failed to join thread '{name}'")
        raise ValueError(f"Thread '{name}' ({thr}) did not join"
                         f"within {timeout * retries}s!")
