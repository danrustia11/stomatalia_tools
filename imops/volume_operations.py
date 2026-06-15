    
import numpy as np


def measure_volume(m, depth_img):
    
    volume = depth_img[m == 255].astype(np.int16)
    volume = volume[volume.nonzero()]
    volume = volume[volume != 255]
    max_dist = volume.max()
    # print(volume[volume.nonzero()].min(), volume[volume.nonzero()].max())
    volume -= max_dist
    # # TODO change the substraction of the background to belt level instaed of max dist, correct for non zeros
    volume = abs(volume.sum())

    return volume
