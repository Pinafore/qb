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
