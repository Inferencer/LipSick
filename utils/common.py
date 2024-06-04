# utils/common.py

import os

def get_versioned_filename(filepath):
    """ Append a version number to the filepath if it already exists. """
    base, ext = os.path.splitext(filepath)
    counter = 1
    while os.path.exists(filepath):
        filepath = f"{base}({counter}){ext}"
        counter += 1
    return filepath
