#: Scalar features that apply to all events in a frame and which are
#: not computed from image or image_bg data.
PROTECTED_FEATURES = [
    "flow_rate",
    "frame",
    "g_force",
    "pressure",
    "temp",
    "temp_amb",
    "time"
]

for ii in range(10):
    PROTECTED_FEATURES.append(f"userdef{ii}")
