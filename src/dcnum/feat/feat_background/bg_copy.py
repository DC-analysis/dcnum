import h5py

from .base import Background


class BackgroundCopy(Background):
    @staticmethod
    def check_user_kwargs():
        pass

    def process(self):
        """Copy input data to output dataset"""
        if self.h5in != self.h5out:
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
