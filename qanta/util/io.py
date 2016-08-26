import os
import subprocess


def safe_open(path, mode):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, mode=mode)


def call(args):
    return subprocess.run(args, check=True)


def shell(command):
    return subprocess.run(command, check=True, shell=True)
