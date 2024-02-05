import pathlib

search_path_registry = {}


def register_search_path(topic: str,
                         search_path: str | pathlib.Path):
    """Register a search path for a given topic

    Search paths are a global solution for organizing the locations
    of resources that are part of an analysis pipeline. For instance,
    if the location of such a file that depends on where your pipeline is
    running, you can register multiple search paths and the file will
    be found using :func:`find_file`.
    """
    topic_list = search_path_registry.setdefault(topic, [])
    topic_list.append(pathlib.Path(search_path))


def find_file(topic: str,
              file_name: str):
    """Find a file in the search path for the given topic"""
    search_paths = search_path_registry.get(topic, [])
    for pp in search_paths:
        pf = pp / file_name
        if pf.is_file():
            return pf
    else:
        raise KeyError(f"Could not find {file_name} for {topic} in the "
                       f"registered search paths {search_paths}")
