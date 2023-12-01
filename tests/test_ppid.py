import inspect

import pytest

from dcnum.meta import ppid


class ExampleClass:
    def __init__(self,
                 *,
                 kitchen_size: str = "large",
                 **kwargs):
        """A cooking class"""
        self.kitchen_size = kitchen_size
        self.kwargs = kwargs

    def cook(self, *,
             temperature: float = 90.0,
             te: str = "a",
             outside: bool = False,
             with_water: bool = True,
             amount: int = 1000,
             wine_type: str = "red",
             test_oven: bool = True):
        return id(self)

    @classmethod
    def get_ppid_code(cls):
        return cls.__name__


def test_compute_pipeline_hash():
    pp_hash = ppid.compute_pipeline_hash(
        gen_id="7",
        dat_id="hdf:p=0.34",
        bg_id="sparsemed:k=200^s=1^t=0^f=0.8",
        seg_id="thresh:t=-3:cle=1^f=1^clo=2",
        feat_id="legacy:b=1^h=0",
        gate_id="norm:o=0^s=11",
    )
    assert pp_hash == "ec11977fc233e133c29642736161f201"


@pytest.mark.parametrize("in_list,out_list", [
    (["camera", "campus"],
     ["came", "camp"]),
    (["cole", "coleman"],
     ["cole", "colem"]),
    (["cole", "coleman", "colemine"],
     ["cole", "colema", "colemi"]),
    (["cole", "coleman", "cundis"],
     ["cole", "colem", "cu"]),
    (["an", "and", "anderson", "andersrum", "ant"],
     ["an", "and", "anderso", "andersr", "ant"]),
])
def test_unique_prefix_ordered(in_list, out_list):
    t_list = ppid.get_unique_prefix(in_list)
    assert t_list == out_list


@pytest.mark.parametrize("in_list,out_list", [
    (["campus", "camera"],
     ["camp", "came"]),
    (["coleman", "cole"],
     ["colem", "cole"]),
    (["coleman", "cole", "colemine"],
     ["colema", "cole", "colemi"]),
    (["cole", "cundis", "coleman"],
     ["cole", "cu", "colem"]),
    (["an", "andersrum", "and", "anderson", "ant"],
     ["an", "andersr", "and", "anderso", "ant"]),
])
def test_unique_prefix_unordered(in_list, out_list):
    t_list = ppid.get_unique_prefix(in_list)
    assert t_list == out_list


@pytest.mark.parametrize("kwargs, pid", [
    ({},
     "tem=90^te=a^o=0^wit=1^a=1000^win=red^tes=1"),
    ({"temperature": 10.1},
     "tem=10.1^te=a^o=0^wit=1^a=1000^win=red^tes=1"),
    ({"with_water": False, "wine_type": "blue"},
     "tem=90^te=a^o=0^wit=0^a=1000^win=blue^tes=1"),
])
def test_kwargs_to_ppid(kwargs, pid):
    ptest = ppid.kwargs_to_ppid(ExampleClass, "cook", kwargs)
    assert pid == ptest


def test_kwargs_to_ppid_init():
    """kwargs defined in __init__ must be allowed!"""
    ppid.kwargs_to_ppid(ExampleClass,
                        method="cook",
                        kwargs={"kitchen_size": "small"},
                        allow_invalid_keys=False)


def test_kwargs_to_ppid_init_invalid():
    """kwargs defined in __init__ must be allowed!"""
    with pytest.raises(KeyError, match="cook_size"):
        ppid.kwargs_to_ppid(ExampleClass,
                            method="cook",
                            kwargs={"cook_size": "small"},
                            allow_invalid_keys=False)


def test_kwargs_to_ppid_invalid():
    with pytest.raises(KeyError, match="hansel"):
        ppid.kwargs_to_ppid(ExampleClass,
                            method="cook",
                            kwargs={"hansel": 4},
                            allow_invalid_keys=False)


@pytest.mark.parametrize("kwargs, pid", [
    ({},
     "tem=90^te=a^o=0^wit=1^a=1000^win=red^tes=1"),
    ({"temperature": 10.1},
     "tem=10.1^te=a^o=0^wit=1^a=1000^win=red^tes=1"),
    ({"with_water": False, "wine_type": "blue"},
     "tem=90^te=a^o=0^wit=0^a=1000^win=blue^tes=1"),
    ({},
     "te=a^tem=90^o=0^wit=1^a=1000^win=red^tes=1"),  # switch order (evil!)
    ({},
     "tem=90^te=a^o=0^wit=1^a=1000^win=red"),  # remove things
    ({},
     "tem=90^te=a^o=0^w=1^a=1000"),  # remove more things!
])
def test_ppid_to_kwargs(kwargs, pid):
    # get the default keyword arguments
    meth = getattr(ExampleClass, "cook")
    spec = inspect.getfullargspec(meth)
    kwargs_full = spec.kwonlydefaults
    # update with given kwargs
    kwargs_full.update(kwargs)

    kwargs_test = ppid.ppid_to_kwargs(ExampleClass, "cook", pid)
    assert kwargs_test == kwargs_full
