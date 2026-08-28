from dcnum.common import LazyLoader


def test_lazy_loader_simple():
    np = LazyLoader("numpy")
    assert np.module_available()
    assert len(np.zeros(10)) == 10
