import abc
import inspect
import multiprocessing as mp
import pathlib

import h5py

from ...meta import ppid


class Background(abc.ABC):
    def __init__(self, input_data, output_path, num_cpus=None, **kwargs):
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
        num_cpus: int
            Number of CPUs to use for median computation. Defaults to
            `multiprocessing.cpu_count()`.
        kwargs:
            Additional keyword arguments passed to the subclass.
        """
        self.check_user_kwargs(**kwargs)

        # Using spec is not really necessary here, because kwargs are
        # fully populated for background computation, but this might change.
        spec = inspect.getfullargspec(self.check_user_kwargs)
        #: background keyword arguments
        self.kwargs = spec.kwonlydefaults or {}
        self.kwargs.update(kwargs)

        if num_cpus is None:
            num_cpus = mp.cpu_count()
        #: number of CPUs used
        self.num_cpus = num_cpus

        #: number of frames
        self.event_count = None

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
            self.input_data = self.h5in["events/image"]
        else:
            self.input_data = input_data

        if self.h5out is None:
            # "a", because output file is already an .rtdc file
            # TODO:
            #  - properly setup HDF5 caching
            #  - create image_bg here instead of in subclasses
            self.h5out = h5py.File(output_path, "a", libver="latest")

    @staticmethod
    def get_kwargs_from_ppid(bg_ppid):
        """Return keyword arguments for any subclass from a PPID string"""
        name, pp_check_user_kwargs = bg_ppid.split(":")
        for cls in Background.__subclasses__():
            if cls.key() == name:
                break
        else:
            raise ValueError(
                f"Could not find background computation method '{name}'!")
        kwargs = ppid.ppid_to_kwargs(cls=cls,
                                     method="check_user_kwargs",
                                     ppid=pp_check_user_kwargs)
        return kwargs

    @classmethod
    def get_ppid_from_kwargs(cls, kwargs):
        """Return the PPID based on given keyword arguments for a subclass"""
        key = cls.key()
        cback = ppid.kwargs_to_ppid(cls, "check_user_kwargs", kwargs)
        return ":".join([key, cback])

    @classmethod
    def key(cls):
        if cls is Background:
            raise ValueError("Cannot get `key` for `Background` base class!")
        key = cls.__name__.lower()
        if key.startswith("background"):
            key = key[10:]
        return key

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
        return self.get_ppid_from_kwargs(self.kwargs)

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
