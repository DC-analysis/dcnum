import queue
import time

import numpy as np
from scipy import ndimage

from .base import mp_spawn, Background


class BackgroundRollMed(Background):
    def __init__(self, input_data, output_path, kernel_size=100,
                 batch_size=10000, compress=True, num_cpus=None):
        """Rolling median RT-DC background image computation

        1. There is one big shared array `shared_input` that contains
           the image data for each batch.
        2. User specifies batch size (10000) and kernel size (default
           is 100)
        3. There is a second shared array `shared_output` that contains
           the median values corresponding to the data in `shared_input`.
        4. Background computation is done by copying the input images
           from a file into the shared array.
        5. The input array is split into and workers compute the
           rolling median for each point in `shared_input`.

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
        kernel_size: int
            Kernel size for median computation. This is the number of
            events that are used to compute the median for each pixel.
        batch_size: int
            Number of events to process at the same time. Increasing this
            number much more than two orders of magnitude larger than
            `kernel_size` will not increase computation speed. Larger
            values lead to a higher memory consumption.
        compress: bool
            Whether to compress background data. Set this to False
            for faster processing.
        num_cpus: int
            Number of CPUs to use for median computation. Defaults to
            `multiprocessing.cpu_count()`.
        """
        super(BackgroundRollMed, self).__init__(
            input_data=input_data,
            output_path=output_path,
            compress=compress,
            num_cpus=num_cpus,
            kernel_size=kernel_size,
            batch_size=batch_size)

        if kernel_size > len(self.input_data):
            raise ValueError(f"Cannot compute background when the input data "
                             f"size {len(self.input_data)} is larger than the "
                             f"kernel size {kernel_size}!")

        #: kernel size used for median filtering
        self.kernel_size = kernel_size
        #: number of events processed at once
        self.batch_size = batch_size

        #: mp.RawArray for temporary batch input data
        self.shared_input_raw = mp_spawn.RawArray(
            np.ctypeslib.ctypes.c_uint8,
            int(np.prod(self.image_shape)) * (batch_size + kernel_size))
        #: mp.RawArray for temporary batch output data
        self.shared_output_raw = mp_spawn.RawArray(
            np.ctypeslib.ctypes.c_uint8,
            int(np.prod(self.image_shape)) * batch_size)
        # Convert the RawArray to something we can write to fast
        # (similar to memoryview, but without having to cast) using
        # np.ctypeslib.as_array. See discussion in
        # https://stackoverflow.com/questions/37705974
        #: numpy array reshaped view on `self.shared_input_raw` with
        #: first axis enumerating the events
        self.shared_input = np.ctypeslib.as_array(
            self.shared_input_raw).reshape(batch_size + kernel_size, -1)
        #: numpy array reshaped view on `self.shared_output_raw` with
        #: first axis enumerating the events
        self.shared_output = np.ctypeslib.as_array(
            self.shared_output_raw).reshape(batch_size, -1)
        #: current batch index (see `self.process` and `process_next_batch`)
        self.current_batch = 0

        #: counter tracking process of workers
        self.worker_counter = mp_spawn.Value("l", 0)
        #: queue for median computation jobs
        self.queue = mp_spawn.Queue()
        #: list of workers (processes)
        self.workers = [WorkerRollMed(self.queue,
                                      self.worker_counter,
                                      self.shared_input_raw,
                                      self.shared_output_raw,
                                      self.batch_size,
                                      self.kernel_size)
                        for _ in range(self.num_cpus)]
        [w.start() for w in self.workers]

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.worker_counter.value = -1000
        [w.join() for w in self.workers]
        super(BackgroundRollMed, self).__exit__(type, value, tb)

    @staticmethod
    def check_user_kwargs(*,
                          kernel_size: int = 100,
                          batch_size: int = 10000):
        """Check user-defined properties of this class

        This method primarily exists so that the CLI knows which
        keyword arguements can be passed to this class.

        Parameters
        ----------
        kernel_size: int
            Kernel size for median computation. This is the number of
            events that are used to compute the median for each pixel.
        batch_size: int
            Number of events to process at the same time. Increasing this
            number much more than two orders of magnitude larger than
            `kernel_size` will not increase computation speed. Larger
            values lead to a higher memory consumption.
        """
        assert kernel_size > 0
        assert batch_size > kernel_size

    def get_slices_for_batch(self, batch_index=0):
        """Returns slices for getting the input and writing to output

        The input slice is `self.kernel_size` longer.
        """
        # We always take `kernel_size` more events from the input data,
        # but we increment with `batch_size`.
        start = batch_index * self.batch_size
        stop_in = (batch_index + 1) * self.batch_size + self.kernel_size
        stop_out = (batch_index + 1) * self.batch_size

        if stop_in > self.image_count:
            stop_in = self.image_count
            stop_out = self.image_count - self.kernel_size

        slice_in = slice(start, stop_in)
        slice_out = slice(start, stop_out)
        output_size = max(0, stop_out - start)
        return slice_in, slice_out, output_size

    def map_iterator(self):
        """Iterates over arguments for `compute_median_for_slice`"""
        pixels_per_job = self.image_shape[0]
        ii = 0
        while True:  # TODO: turn into for-loop using data size
            job_slice = slice(pixels_per_job * ii,
                              pixels_per_job * (ii + 1))
            yield job_slice
            if (ii + 1) * pixels_per_job == np.prod(self.image_shape):
                break
            ii += 1

    def process_approach(self):
        """Perform median computation on entire input data"""
        num_steps = int(np.ceil(self.image_count / self.batch_size))
        for ii in range(num_steps):
            self.process_next_batch()
        # Set the remaining kernel_size median values to the last one
        last_image = self.h5out["events/image_bg"][-self.kernel_size-1]
        for ii in range(self.kernel_size):
            self.h5out["events/image_bg"][self.image_count-ii-1] = last_image
        self.image_proc.value = self.image_count

    def process_next_batch(self):
        """Process one batch of input data"""
        cur_slice_in, cur_slice_out, output_size = \
            self.get_slices_for_batch(self.current_batch)

        if output_size:
            input_size = output_size + self.kernel_size
            self.shared_input[:input_size] = \
                self.input_data[cur_slice_in].reshape(input_size, -1)

            # reset worker counter
            self.worker_counter.value = 0

            num_jobs = 0
            for job_slice in self.map_iterator():
                # prepend output size to arguments
                args = (output_size, job_slice)
                self.queue.put(args)
                num_jobs += 1

            # block until workers are done
            while True:
                time.sleep(.03)
                if self.worker_counter.value == num_jobs:
                    break

            # Write output data to HDF5 file
            # TODO:
            #  Do this in a different thread so workers can keep going
            #  and use a lock somewhere in case the disk is too slow.
            self.h5out["events/image_bg"][cur_slice_out] = \
                self.shared_output[:output_size].reshape(output_size,
                                                         *self.image_shape)

        self.current_batch += 1
        self.image_proc.value += self.batch_size


class WorkerRollMed(mp_spawn.Process):
    def __init__(self, job_queue, counter, shared_input, shared_output,
                 batch_size, kernel_size, *args, **kwargs):
        """Worker process for median computation"""
        super(WorkerRollMed, self).__init__(*args, **kwargs)
        self.queue = job_queue
        self.queue.cancel_join_thread()
        self.counter = counter
        self.shared_input_raw = shared_input
        self.shared_output_raw = shared_output
        self.batch_size = batch_size
        self.kernel_size = kernel_size

    def run(self):
        """Main loop of worker process (breaks when `self.counter` <0)"""
        # Create the ctypes arrays here instead of during __init__, because
        # for some reason they are copied in __init__ and not mapped.
        shared_input = np.ctypeslib.as_array(
            self.shared_input_raw).reshape(
            self.batch_size + self.kernel_size, -1)
        shared_output = np.ctypeslib.as_array(
            self.shared_output_raw).reshape(self.batch_size, -1)
        while True:
            if self.counter.value < 0:
                break
            try:
                args = self.queue.get(timeout=.1)
            except queue.Empty:
                pass
            else:
                compute_median_for_slice(shared_input, shared_output,
                                         self.kernel_size, *args)
                with self.counter.get_lock():
                    self.counter.value += 1


def compute_median_for_slice(shared_input, shared_output, kernel_size,
                             output_size, job_slice):
    """Compute the rolling median for a slice of a shared array

    Parameters
    ----------
    shared_input: multiprocessing.RawArray
        Input data for which to compute the median. For each pixel
        in the original image, batch_size + kernel_size events are
        stored in this array one after another in a row.
        The total size of this array is
        `batch_size` * `kernel_size` * `number_of_pixels_in_the_image`.
    shared_output: multiprocessing.RawArray
        Used for storing the result. Note that the last `kernel_size`
        elements for each pixel in this output array are junk data
        (because it is a rolling median).
    kernel_size: int
        Kernel size for median computation. This is the number of
        events that are used to compute the median for each pixel.
    output_size: int
        The partial batch size, i.e. the number of events for which to
        compute the rolling median. Note that output_size + kernel_size
        events are taken from shared_input
    job_slice: slice
        Now this is the important part. We can write to `shared_input`
        and shared_output from multiple processes. This slice tells
        us which part of the data we are working on. Only this slice
        will be edited in `shared_output`. This slice defines how
        many pixels we are looking at.
    """
    input_size = output_size + kernel_size
    # perform median filter on input data
    # write median data into output array. Nnote that the values at
    # filtered[-kernel_size:] are just junk data, but we keep it to
    # make the code simpler.
    shared_output[:output_size, job_slice] = ndimage.median_filter(
        input=shared_input[:input_size, job_slice],
        size=(kernel_size, 1),
        mode="constant",
        cval=0,
        # This means that the median at output[i] corresponds
        # to the values in input[i:i+kernel_size] (the default is
        # centered).
        origin=(-kernel_size//2, 0),
    )[:output_size, :]
