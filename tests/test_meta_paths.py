import pytest

from dcnum.meta import paths


@pytest.fixture(autouse=True)
def clear_search_paths():
    paths.search_path_registry.clear()


def test_find_file_in_path(tmp_path):
    file = tmp_path / "myfile.txt"
    file.touch()
    # register a search path
    paths.register_search_path("hans", tmp_path)
    act = paths.find_file("hans", file.name)
    assert str(act.resolve()) == str(file.resolve())


def test_find_file_in_path_multiple(tmp_path):
    dir1 = tmp_path / "foo"
    dir2 = tmp_path / "bar"
    file = dir2 / "myfile.txt"
    dir1.mkdir()
    dir2.mkdir()
    file.touch()
    # register a search path
    paths.register_search_path("mantra", dir1)
    paths.register_search_path("mantra", dir2)
    act = paths.find_file("mantra", file.name)
    assert str(act.resolve()) == str(file.resolve())


def test_find_file_in_path_not_found(tmp_path):
    # find non-existent file
    with pytest.raises(KeyError, match="Could not find"):
        paths.find_file("peter", "myfile.txt")

    # register a search path
    paths.register_search_path("hans", tmp_path / "does_not_exist")
    with pytest.raises(KeyError, match="Could not find"):
        paths.find_file("hans", "myfile.txt")
