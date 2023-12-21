import h5py

from .base import Background


class BackgroundCopy(Background):

    @staticmethod
    def check_user_kwargs():
        pass

    def process_approach(self):
        """Perform median computation on entire input data"""
        if self.h5in != self.h5out:
            hin = self.hdin.image_bg.h5ds
            if "image_bg" in self.h5out["events"]:
                del self.h5out["events/image_bg"]
            h5py.h5o.copy(src_loc=hin.parent.id,
                          src_name=b"image_bg",
                          dst_loc=self.h5out["events"].id,
                          dst_name=b"image_bg",
                          )

        # set progress to 100%
        self.image_proc.value = self.image_count
