import dcnum


def test_init():
    # Checks if the object `dcnum` has an attribute named `__version__`.
    # If not, an AssertionError is raised.
    assert hasattr(dcnum, "__version__")
