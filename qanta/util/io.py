import os
import pickle
from functools import wraps
import subprocess


def safe_open(path, mode):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, mode=mode)


def call(args):
    return subprocess.run(args, check=True)


def shell(command):
    return subprocess.run(command, check=True, shell=True)


def safe_path(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def make_dirs(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def file_backed_cache_decorator(cache_location):
    def decorator(func):
        lookup_table = None
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                args_t = tuple(args)
                kwargs_t = tuple(kwargs.items())
                cache_key = (args_t, kwargs_t)
                nonlocal lookup_table
                if lookup_table is None:
                    func_name = func.__name__
                    if os.path.exists(cache_location):
                        with open(cache_location, 'rb') as f:
                            lookup_table = pickle.load(f)
                            if '_func_name' not in lookup_table:
                                raise ValueError(
                                    'Could not verify matching function names for function "{}" in cache file "{}"'.format(func_name, cache_location))
                            elif lookup_table['_func_name'] != func_name:
                                raise ValueError(
                                    'The cached file {} containing the cache for function {} does not match the input function {}'.format(cache_location, lookup_table['_func_name'], func_name))
                    else:
                        lookup_table = {'_func_name': func_name}

                if cache_key in lookup_table:
                    return lookup_table[cache_key]
                else:
                    cache_value = func(*args, **kwargs)
                    lookup_table[cache_key] = cache_value
                    with open(cache_location, 'wb') as f:
                        pickle.dump(lookup_table, f)
                    return cache_value

            except TypeError:
                raise ValueError('Function arguments must be hashable')
        return wrapper
    return decorator
