from functools import wraps
import os
import pickle
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


def pickle_cache(f_name_creator):
    def cache(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            f_name = f_name_creator(*args, **kwargs)
            if os.path.exists(f_name):
                with safe_open(f_name, 'rb') as f:
                    return pickle.load(f)
            else:
                result = func(*args, **kwargs)
                with safe_open(f_name, 'wb') as f:
                    pickle.dump(result, f)
            return result
        return wrapper
    return cache
