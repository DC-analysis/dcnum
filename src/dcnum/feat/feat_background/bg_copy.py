import h5py

from .base import Background


class BackgroundCopy(Background):
    @staticmethod
    def check_user_kwargs():
        pass

    def process(self):
        """Perform median computation on entire input data"""
        if self.h5in != self.h5out:
            hin = self.hdin.image_bg.h5ds
            h5py.h5o.copy(src_loc=hin.parent.id,
                          src_name=b"image_bg",
                          dst_loc=self.h5out["events"].id,
                          dst_name=b"image_bg",
                          )

        # set progress to 100%
        self.image_proc.value = self.image_count

    def process_approach(self):
        # We do the copying in `process`, because we do not want to modify
        # any metadata or delete datasets as is done in the base class.
        # But we still have to implement this method, because it is an
        # abstractmethod in the base class.
        pass
