#: Scalar features that apply to all events in a frame and which are
#: not computed for individual events.
PROTECTED_FEATURES = [
    "bg_off",
    "flow_rate",
    "frame",
    "g_force",
    "pressure",
    "temp",
    "temp_amb",
    "time",
]

# User-defined features may be anything, but if the user needs something
# very specific for the pipeline, having them protected is a nice feature.
for ii in range(10):
    PROTECTED_FEATURES.append(f"userdef{ii}")
