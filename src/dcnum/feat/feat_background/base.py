import abc
import functools
import inspect
import logging
import multiprocessing as mp
import pathlib
import time

import h5py

from ...meta import ppid
from ...read import HDF5Data
from ...write import HDF5Writer, create_with_basins, set_default_filter_kwargs


# All subprocesses should use 'spawn' to avoid issues with threads
# and 'fork' on POSIX systems.
mp_spawn = mp.get_context('spawn')


class Background(abc.ABC):
    def __init__(self, input_data, output_path, compress=True, num_cpus=None,
                 **kwargs):
        """

        Parameters
        ----------
        input_data: array-like or pathlib.Path
            The input data can be either a path to an HDF5 file with
            the "evtens/image" dataset or an array-like object that
            behaves like an image stack (first axis enumerates events)
        output_path: pathlib.Path
            Path to the output file. If `input_data` is a path, you can
            set `output_path` to the same path to write directly to the
            input file. The data are written in the "events/image_bg"
            dataset in the output file.
        compress: bool
            Whether to compress background data. Set this to False
            for faster processing.
        num_cpus: int
            Number of CPUs to use for median computation. Defaults to
            `multiprocessing.cpu_count()`.
        kwargs:
            Additional keyword arguments passed to the subclass.
        """
        self.logger = logging.getLogger(
            f"dcnum.feat.feat_background.{self.__class__.__name__}")
        # proper conversion to Path objects
        output_path = pathlib.Path(output_path)
        self.output_path = output_path
        if isinstance(input_data, str):
            input_data = pathlib.Path(input_data)
        # kwargs checks
        self.check_user_kwargs(**kwargs)

        # Using spec is not really necessary here, because kwargs are
        # fully populated for background computation, but this might change.
        spec = inspect.getfullargspec(self.check_user_kwargs)
        #: background keyword arguments
        self.kwargs = spec.kwonlydefaults or {}
        self.kwargs.update(kwargs)

        if num_cpus is None:
            num_cpus = mp_spawn.cpu_count()
        #: number of CPUs used
        self.num_cpus = num_cpus

        #: number of images in the input data
        self.image_count = None
        #: fraction images that have been processed
        self.image_proc = mp_spawn.Value("d", 0)

        #: HDF5Data instance for input data
        self.hdin = None
        #: input h5py.File
        self.h5in = None
        #: output h5py.File
        self.h5out = None
        #: reference paths for logging to the output .rtdc file
        self.paths_ref = []
        # Check whether user passed an array or a path
        if isinstance(input_data, pathlib.Path):
            if str(input_data.resolve()) == str(output_path.resolve()):
                self.h5in = h5py.File(input_data, "a", libver="latest")
                self.h5out = self.h5in
            else:
                self.paths_ref.append(input_data)
                self.h5in = h5py.File(input_data, "r", libver="latest")
            # TODO: Properly setup HDF5 caching.
            #       Right now, we are accessing the raw h5ds property of
            #       the ImageCache. We have to go via the ImageCache route,
            #       because HDF5Data properly resolves basins and the image
            #       feature might be in a basin.
            self.hdin = HDF5Data(self.h5in)
            self.input_data = self.hdin.image.h5ds
        else:
            self.input_data = input_data

        #: shape of event images
        self.image_shape = self.input_data[0].shape
        #: total number of events
        self.image_count = len(self.input_data)

        if self.h5out is None:
            if not output_path.exists():
                # If the output path does not exist, then we create
                # an output file with basins (for user convenience).
                create_with_basins(path_out=output_path,
                                   basin_paths=self.paths_ref)
            # "a", because output file already exists
            self.h5out = h5py.File(output_path, "a", libver="latest")

        # Initialize writer
        self.writer = HDF5Writer(
            obj=self.h5out,
            ds_kwds=set_default_filter_kwargs(compression=compress),
        )

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.writer.close()
        # Close h5in and h5out
        if self.hdin is not None:  # we have an input file
            self.hdin.close()  # this closes self.h5in
        if self.h5in is not self.h5out and self.h5out is not None:
            self.h5out.close()

    @abc.abstractmethod
    def check_user_kwargs(self, **kwargs):
        """Implement this to check the kwargs during init"""

    def get_ppid(self):
        """Return a unique background pipeline identifier

        The pipeline identifier is universally applicable and must
        be backwards-compatible (future versions of dcnum will
        correctly acknowledge the ID).

        The segmenter pipeline ID is defined as::

            KEY:KW_BACKGROUND

        Where KEY is e.g. "sparsemed" or "rollmed", and KW_BACKGROUND is a
        list of keyword arguments for `check_user_kwargs`, e.g.::

            kernel_size=100^batch_size=10000

        which may be abbreviated to::

            k=100^b=10000
        """
        return self.get_ppid_from_ppkw(self.kwargs)

    @classmethod
    def get_ppid_code(cls):
        if cls is Background:
            raise ValueError("Cannot get `key` for `Background` base class!")
        key = cls.__name__.lower()
        if key.startswith("background"):
            key = key[10:]
        return key

    @classmethod
    def get_ppid_from_ppkw(cls, kwargs):
        """Return the PPID based on given keyword arguments for a subclass"""
        code = cls.get_ppid_code()
        cback = ppid.kwargs_to_ppid(cls, "check_user_kwargs", kwargs)
        return ":".join([code, cback])

    @staticmethod
    def get_ppkw_from_ppid(bg_ppid):
        """Return keyword arguments for any subclass from a PPID string"""
        code, pp_check_user_kwargs = bg_ppid.split(":")
        for bg_code in get_available_background_methods():
            if bg_code == code:
                cls = get_available_background_methods()[bg_code]
                break
        else:
            raise ValueError(
                f"Could not find background computation method '{code}'!")
        kwargs = ppid.ppid_to_kwargs(cls=cls,
                                     method="check_user_kwargs",
                                     ppid=pp_check_user_kwargs)
        return kwargs

    def get_progress(self):
        """Return progress of background computation, float in [0,1]"""
        if self.image_count == 0:
            return 0.
        else:
            return self.image_proc.value

    def process(self):
        """Perform the background computation

        This irreversibly removes/overrides any "image_bg" and
        "bg_off" features defined in the output file `self.h5out`.
        """
        t0 = time.perf_counter()
        # Delete any old background data
        for ds_key in ["image_bg", "bg_off"]:
            for grp_key in ["events", "basin_events"]:
                if grp_key in self.h5out and ds_key in self.h5out[grp_key]:
                    del self.h5out[grp_key][ds_key]
        # Perform the actual background computation
        self.process_approach()
        bg_ppid = self.get_ppid()
        # Store pipeline information in the image_bg/bg_off feature
        for ds_key in ["image_bg", "bg_off"]:
            for grp_key in ["events", "basin_events"]:
                if grp_key in self.h5out and ds_key in self.h5out[grp_key]:
                    self.h5out[f"{grp_key}/{ds_key}"].attrs[
                        "dcnum ppid background"] = bg_ppid
                    self.h5out[F"{grp_key}/{ds_key}"].attrs[
                        "dcnum ppid generation"] = ppid.DCNUM_PPID_GENERATION
        self.logger.info(
            f"Background computation time: {time.perf_counter()-t0:.1f}s")

    @abc.abstractmethod
    def process_approach(self):
        """The actual background computation approach"""


@functools.cache
def get_available_background_methods():
    """Return dictionary of background computation methods"""
    methods = {}
    for cls in Background.__subclasses__():
        methods[cls.get_ppid_code()] = cls
    return methods
