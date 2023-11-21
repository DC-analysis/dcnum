#: Scalar features that apply to all events in a frame
PROTECTED_FEATURES = [
    "bg_med",
    "flow_rate",
    "frame",
    "g_force",
    "index_online",
    "pressure",
    "temp",
    "temp_amb",
    "time"
]

for ii in range(10):
    PROTECTED_FEATURES.append(f"userdef{ii}")
