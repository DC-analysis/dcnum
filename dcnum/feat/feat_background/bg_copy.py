from ...common import h5py
from ...write import HDF5Writer
from .base import Background


class BackgroundCopy(Background):
    def __init__(self, *args, **kwargs):
        """Copy the input background data to the output file"""
        super().__init__(*args, **kwargs)

    @staticmethod
    def check_user_kwargs():
        pass

    def process(self):
        """Copy input data to output dataset"""
        assert self.h5in is not None
        assert self.h5out is not None
        assert self.hdin is not None

        if self.h5in != self.h5out:
            # check input basins for background image
            for ii, bn_dict in enumerate(self.hdin.basins):
                if "image_bg" in bn_dict["features"]:
                    # load the basin data
                    h5group, features, mapping = self.hdin.get_basin_data(ii)
                    if isinstance(mapping, (int, slice, list)):
                        raise NotImplementedError(
                            f"Image data mapping for {type(mapping)} "
                            f"not supported yet")
                    with HDF5Writer(self.h5out) as hw:
                        hw.store_basin(
                            name=bn_dict["name"],
                            description=bn_dict.get("description"),
                            mapping=mapping,
                            internal_data={"image_bg": h5group["image_bg"][:]},
                        )
                    break

            # check regular events for "image_bg" and "bg_off"
            hin = self.hdin.h5
            for feat in ["image_bg", "bg_off"]:
                if feat in hin["events"]:
                    h5py.h5o.copy(src_loc=hin["events"].id,
                                  src_name=feat.encode("utf-8"),
                                  dst_loc=self.h5out["events"].id,
                                  dst_name=feat.encode("utf-8"),
                                  )

        # set progress to 100%
        self.image_proc.value = 1

    def process_approach(self):
        # We do the copying in `process`, because we do not want to modify
        # any metadata or delete datasets as is done in the base class.
        # But we still have to implement this method, because it is an
        # abstractmethod in the base class.
        pass
