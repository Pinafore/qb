import subprocess


def call(args):
    return subprocess.run(args, check=True)


def shell(command):
    return subprocess.run(command, check=True, shell=True)
