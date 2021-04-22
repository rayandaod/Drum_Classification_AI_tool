import os
from os import path
import warnings


# Check that the given file path does not already exists.
# If it does, add "_new" at the end (before the extension)
# If it doesn't, simply return the originally given path
def can_write(file_path):
    if path.exists(file_path):
        warnings.warn("The given path already exists, adding \"_new\" at the end.")
        path_split = os.path.splitext(file_path)
        return can_write(path_split[0] + "_new" + path_split[1])
    else:
        return file_path
