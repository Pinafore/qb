import os


def safe_open(path, mode):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, mode=mode)
