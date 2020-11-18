from pathlib import Path
import os
import numpy as np


def file_exist_query(filename):
    path = Path(filename)
    if path.is_file():
        res = None
        while res not in ['y', 'Y', 'n', 'N']:
            res = input("\nThe file in '{}' already exists, do you really wish to re-write its contents? [y/n]".format(filename))
            if res not in ['y', 'Y', 'n', 'N']:
                print("Please reply with 'y' or 'n'")
        if res in ['n', 'N']:
            return False
    return True


def file_exists(filename):
    path = Path(filename)
    if path.is_file():
        return True
    return False


def folder_exists(folder_name):
    return os.path.isdir(folder_name)


def folder_create(folder_name, exist_ok=False, parents=True):
    path = Path(folder_name)
    try:
        path.mkdir(parents=parents, exist_ok=exist_ok)
    except:
        raise OSError("Trying to create an already existing folder!!")
    return True

def get_filenames(folder="./", file_ending='', file_beginning='', contains=''):
    return list(np.sort([file for file in os.listdir(folder)
                         if file.endswith(file_ending)
                         and file.startswith(file_beginning)
                         and contains in file]))
