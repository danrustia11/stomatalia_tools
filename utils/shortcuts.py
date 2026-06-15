import os
import math
from natsort import natsorted
import numpy as np 

def roundup(x):
    return int(math.ceil(x / 100.0)) * 100

def rounddown(x):
    return int(math.floor(x / 100.0)) * 100



def list_files(folder, level=1, ext_filter=[]):

    if level == 1:
        if len(ext_filter) > 0:
            files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(ext_filter)]
        else:
            files = [os.path.join(folder, f) for f in os.listdir(folder)]
        files = natsorted(files)
        return files
    if level == 2:
        files = []
        for sub_folder in os.listdir(folder):
            sub_folder = os.path.join(folder, sub_folder)
            sub_files = [os.path.join(sub_folder, f) for f in os.listdir(sub_folder)]

            for f in sub_files:
                filename = os.path.join(sub_folder, f)
                files.append(filename)
        return files 

def get_ext(f):
    f = os.path.splitext(f)[1]
    return f

def remove_ext(f):
    f = os.path.splitext(f)[0]
    return f


def get_basename(f):
    f = os.path.splitext(os.path.basename(f))[0]
    return f


def get_sub_folder_name(f, level=1):
    
    if level == 1:
        folder_name = os.path.basename(os.path.split(f)[0])
    if level == 2:
        folder_name = os.path.basename(os.path.split(os.path.splitext(f)[0])[0])

    return folder_name

def create_dir(f):
    if os.path.exists(f) == 0:
        os.makedirs(f, exist_ok=True)


def compute_statistics(data, decimals=2):

    if len(data) > 0:
        data_min = round(np.min(data), decimals)
        data_max = round(np.max(data), decimals)
        data_ave = round(np.average(data), decimals)
        data_std = round(np.std(data), decimals)
    else:
        data_min = 0
        data_max = 0
        data_ave = 0
        data_std = 0

    return [data_min, data_ave, data_max, data_std]