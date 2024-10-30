r"""
    From MYOW.utils https://github.com/nerdslab/myow-neuro
"""
import scipy
from scipy import io as spio
import numpy as np
import mat73

def get_struct_or_dict(data, key):
    if isinstance(data, dict):
        if key in data:
            return data[key]
        else:
            return None
    else:
        return getattr(data, key, None)

def _check_keys(d):
    r"""Checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries.
    """
    for key in d:
        if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
            d[key] = _todict(d[key])
    return d

def _todict(matobj):
    r"""A recursive function which constructs from matobjects nested dictionaries."""
    d = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            d[strg] = _todict(elem)
        elif isinstance(elem, np.ndarray):
            d[strg] = _tolist(elem)
        else:
            d[strg] = elem
    return d

def _tolist(ndarray):
    r"""A recursive function which constructs lists from cellarrays
    (which are loaded as numpy ndarrays), recursing into the elements
    if they contain matobjects.
    """
    elem_list = []
    for sub_elem in ndarray:
        if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
            elem_list.append(_todict(sub_elem))
        elif isinstance(sub_elem, np.ndarray):
            elem_list.append(_tolist(sub_elem))
        else:
            elem_list.append(sub_elem)
    return elem_list

def cast_struct(data):
    return _check_keys(data)    

def loadmat(filename, do_check=True):
    r"""This function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects.
    
    - Do-check can be set to false if you know the struct you need; do_check is recursive and exhaustive and can be very slow. Bad for preproc.
    """

    
    try:
        data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
        if do_check:
            return _check_keys(data)
        else:
            return data
    except Exception as e:
        # print(f"Error loading {filename}: {e}")
        return mat73.loadmat(filename)
