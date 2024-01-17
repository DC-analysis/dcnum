import abc
import functools
import inspect
import multiprocessing as mp
import pathlib
import uuid

import h5py
import numpy as np

from ...meta import ppid
from ...read import HDF5Data
from ...write import create_with_basins, set_default_filter_kwargs


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
        # proper conversion to Path objects
        output_path = pathlib.Path(output_path)
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
        #: number of images that have been processed
        self.image_proc = mp_spawn.Value("L", 0)

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

        #: unique identifier
        self.name = str(uuid.uuid4())
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

        # Initialize background data
        ds_kwargs = set_default_filter_kwargs(compression=compress)
        h5bg = self.h5out.require_dataset(
            "events/image_bg",
            shape=self.input_data.shape,
            dtype=np.uint8,
            chunks=(min(100, self.image_count),
                    self.image_shape[0],
                    self.image_shape[1]),
            **ds_kwargs,
        )
        h5bg.attrs.create('CLASS', np.string_('IMAGE'))
        h5bg.attrs.create('IMAGE_VERSION', np.string_('1.2'))
        h5bg.attrs.create('IMAGE_SUBCLASS', np.string_('IMAGE_GRAYSCALE'))

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
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
        be backwards-compatible (future versions of dcevent will
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
            return self.image_proc.value / self.image_count

    def process(self):
        self.process_approach()
        bg_ppid = self.get_ppid()
        # Store pipeline information in the image_bg feature
        self.h5out["events/image_bg"].attrs["dcnum ppid background"] = bg_ppid
        self.h5out["events/image_bg"].attrs["dcnum ppid generation"] = \
            ppid.DCNUM_PPID_GENERATION

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
