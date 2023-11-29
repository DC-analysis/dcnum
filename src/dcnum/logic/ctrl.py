import collections
import logging
from logging.handlers import QueueListener
import multiprocessing as mp
import os
import pathlib
import socket
import threading
import time
import uuid

import hdf5plugin
import h5py

from ..feat.feat_background.base import get_available_background_methods
from ..feat.queue_event_extractor import QueueEventExtractor
from ..feat import gate
from ..feat import EventExtractorManagerThread
from ..segm import SegmenterManagerThread, get_available_segmenters
from ..meta import ppid
from ..read import HDF5Data
from ..write import (
    DequeWriterThread, HDF5Writer, QueueCollectorThread,
    copy_metadata, create_with_basins,
)

from .job import DCNumPipelineJob


# Force using "spawn" method for multiprocessing, because we are using
# queues and threads and would end up with race conditions otherwise.
mp_spawn = mp.get_context("spawn")


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
        self._progress = 0
        # segmentation frame rate
        self._segm_rate = 0

        # Set up logging
        # General logger for this job
        self.logger = logging.getLogger(__name__).getChild(
            f"Runner-{self.pphash[:5]}")
        self.logger.setLevel(
            logging.DEBUG if job["debug"] else logging.WARNING)
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
        fmt = logging.Formatter(
            "%(asctime)s %(levelname)s %(processName)s/%(threadName)s "
            + "in %(name)s: %(message)s")
        self._log_file_handler.setFormatter(fmt)
        self.logger.addHandler(self._log_file_handler)
        handlers = list(self.logger.handlers)
        # Queue for subprocesses to log to
        self.log_queue = mp_spawn.Queue()
        self._qlisten = QueueListener(self.log_queue, *handlers)
        self._qlisten.start()

        # Sanity checks
        for os_env in [
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
                "NUMBA_NUM_THREADS"]:
            # You should disable multithreading for all major tools that
            # use dcnum.logic. We don't want multithreading, because dcnum
            # uses linear code and relies on multiprocessing for
            # parallelization. This has to be done before importing numpy
            # or any other library affected. In your scripts, you can use:
            #
            #    os.environ.setdefault("OMP_NUM_THREADS", "1")
            #    os.environ.setdefault("MKL_NUM_THREADS", "1")
            #    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
            #    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
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

    def close(self, delete_temporary_files=True):
        if self._data_raw is not None:
            self._data_raw.close()
            self._data_raw = None
        if self._data_temp_in is not None:
            self._data_temp_in.close()
            self._data_temp_in = None
        # clean up logging
        if self._log_file_handler in self.logger.handlers:
            self.logger.removeHandler(self._log_file_handler)
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

    def join(self, *args, **kwargs):
        super(DCNumJobRunner, self).join(*args, **kwargs)
        # Close only after join
        self.close()

    def get_status(self):
        return {
            "progress": self._progress,
            "segm rate": self._segm_rate,
            "state": self._state,
        }

    def run(self):
        """Execute the pipeline job"""
        if self.job["path_out"].exists():
            raise FileExistsError(
                f"Output file {self.job['path_out']} already exists!")
        self._state = "setup"
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
        )
        # Do we have to recompute the background data? In addition to the
        # hash sanity check above, check the generation, input data,
        # and background pipeline identifiers.
        redo_bg = (
            (datdict["gen_id"] != self.ppdict["gen_id"])
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

        self._state = "background"

        if redo_bg:
            # The 'image_bg' feature is written to `self.path_temp_in`.
            # If `job["path_in"]` already has the correct 'image_bg'
            # feature, then we never reach this case here
            # (note that `self.path_temp_in` is basin-based).
            self.task_background()

        self._progress = 0.1
        self._state = "segmentation"

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

        self._progress = 0.95
        self._state = "cleanup"

        # The user would normally expect the output file to be something
        # that is self-contained (copying the file wildly across file
        # systems and network shares should not impair feature availability).
        # Therefore, we copy any remaining basin-based features to the
        # temporary output file.
        if self.job["no_basins_in_output"]:
            self.task_transfer_basin_data()

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
            # regular metadata
            hw.h5.attrs["experiment:event count"] = self.event_count
            hw.h5.attrs["imaging:pixel size"] = self.draw.pixel_size
            if self.path_log.exists():
                # Add the log file to the resulting .rtdc file
                hw.store_log(
                    time.strftime("dcnum-process-%Y-%m-%d-%H.%M.%S"),
                    self.path_log.read_text().split("\n"))
            # copy metadata/logs/tables from original file
            with h5py.File(self.job["path_in"]) as h5_src:
                copy_metadata(h5_src=h5_src,
                              h5_dst=hw.h5,
                              # don't copy basins
                              copy_basins=False)

        # Rename the output file
        self.path_temp_out.rename(self.job["path_out"])
        self._progress = 1.0
        self._state = "done"

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

            bic.process()
        self.logger.info("Finished background computation")

    def task_segment_extract(self):
        self.logger.info("Starting segmentation and feature extraction")
        # Start writer thread
        writer_dq = collections.deque()
        ds_kwds = dict(hdf5plugin.Zstd(clevel=5))
        ds_kwds["fletcher32"] = True
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
        elif seg_cls.hardware_processor == "cpu":  # CPU segmenter
            num_slots = 2
            num_extractors = self.job["num_procs"] // 2
        else:  # GPU segmenter
            num_slots = 3
            num_extractors = self.job["num_procs"]
        num_extractors = max(1, num_extractors)

        slot_chunks = mp_spawn.Array("i", num_slots)
        slot_states = mp_spawn.Array("u", num_slots)

        # Initialize thread
        thr_segm = SegmenterManagerThread(
            segmenter=seg_cls(**self.job["segmenter_kwargs"]),
            image_data=imdat,
            slot_states=slot_states,
            slot_chunks=slot_chunks,
        )
        thr_segm.start()

        # Start feature extractor thread
        fe_kwargs = QueueEventExtractor.get_init_kwargs(
            data=self.dtin,
            gate=gate.Gate(self.dtin, **self.job["gate_kwargs"]),
            log_queue=self.log_queue)
        fe_kwargs["extract_kwargs"] = self.job["feature_kwargs"]

        thr_feat = EventExtractorManagerThread(
            slot_chunks=slot_chunks,
            slot_states=slot_states,
            fe_kwargs=fe_kwargs,
            num_workers=num_extractors,
            labels_list=thr_segm.labels_list,
            debug=self.job["debug"])
        thr_feat.start()

        # Start the data collection thread
        thr_coll = QueueCollectorThread(
            data=self.dtin,
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
        pmin = 0.1  # from background computation
        pmax = 0.95  # 5% reserved for cleanup
        while True:
            counted_frames = thr_coll.written_frames
            self.event_count = thr_coll.written_events
            td = time.monotonic() - t0
            # set the current status
            self._progress = round(
                pmin + counted_frames / data_size * (pmax - pmin),
                3)
            self._segm_rate = counted_frames / (td or 0.03)
            time.sleep(.5)
            if counted_frames == data_size:
                break

        self.logger.debug("Flushing data to disk...")

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
                f"input file or revise your pipeline.")

        self.logger.info("Finished segmentation and feature extraction")

    def task_transfer_basin_data(self):
        with h5py.File(self.path_temp_out, "a") as hout:
            hd = HDF5Data(hout)
            for ii, _ in enumerate(hd.basins):
                hindat, features = hd.get_basin_data(ii)
                for feat in features:
                    if feat not in hout["events"]:
                        self.logger.debug(
                            f"Transferring {feat} to output file.")
                        h5py.h5o.copy(src_loc=hindat.h5["events"].id,
                                      src_name=feat.encode(),
                                      dst_loc=hout["events"].id,
                                      dst_name=feat.encode(),
                                      )


def join_thread_helper(thr, timeout, retries, logger, name):
    for _ in range(retries):
        thr.join(timeout=timeout)
        if thr.is_alive():
            logger.info(f"Waiting for '{name}' ({thr}")
        else:
            logger.info(f"Joined thread '{name}'")
            break
    else:
        logger.error(f"Failed to join thread '{name}'")
        raise ValueError(
            f"Thread '{name}' ({thr}) did not join within {timeout*retries}s!")
