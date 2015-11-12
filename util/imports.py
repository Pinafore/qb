"""
This module exists to fix some imports that six has issues with on Mac OSX and difficult to fix
due to El Capitan file system changes
"""
import six

if six.PY2:
    import cPickle as pickle
else:
    import pickle
