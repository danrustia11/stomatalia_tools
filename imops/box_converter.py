import numpy as np
from itertools import groupby
import pycocotools.mask


def x1y1x2y2_2_x1x2y1y2(this_list):
	new_list = [0]*4
	new_list[0] = this_list[0]
	new_list[1] = this_list[2]
	new_list[2] = this_list[1]
	new_list[3] = this_list[3]
	return new_list


def x1x2y1y2_2_x1y1x2y2(this_list):
	new_list = [0]*4
	new_list[0] = this_list[0]
	new_list[1] = this_list[2]
	new_list[2] = this_list[1]
	new_list[3] = this_list[3]
	return new_list

def xywh2xyxy_yolo(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y



def xywh2xxyy(list_input):
    this_list = list_input.copy()
    this_list[0] = list_input[0]					# x1
    this_list[1] = list_input[0] + list_input[2]	# x2
    this_list[2] = list_input[1]					# y1
    this_list[3] = list_input[1] + list_input[3]	# y2
    return this_list

def xywh2xyxy(list_input):
    this_list = list_input.copy()
    this_list[2] = list_input[0] + list_input[2]
    this_list[3] = list_input[1] + list_input[3]
    this_list[0] = list_input[0]
    this_list[1] = list_input[1]
    return this_list

def xyxy2xywh(list_input):
	x1 = int(list_input[0])
	y1 = int(list_input[1])
	w = abs(int(list_input[2]-list_input[0]))
	h = abs(int(list_input[3]-list_input[1]))
	return [x1,y1,w,h]

def x1y1x2y2_2_cxcywh(list_input,height_img, width_img):
	# conversin of [x1,y1,x2,y2] to normalized yolo! [center_x,center_y,width,height]
	center_x = (list_input[0]+list_input[2])/2
	center_y = (list_input[1]+list_input[3])/2
	width = (list_input[2]-list_input[0])
	height = (list_input[3]-list_input[1])
	return [center_x/width_img,center_y/height_img,width/width_img,height/height_img]

def x1y1wh_2_cxcywh(list_input,height_img, width_img):
	# conversin of [x1,y1,x2,y2] to normalized yolo! [center_x,center_y,width,height]
	center_x = (list_input[0]+list_input[2]*0.5)
	center_y = (list_input[1]+list_input[3]*0.5)
	width = list_input[2]
	height = list_input[3]
	return [center_x/width_img,center_y/height_img,width/width_img,height/height_img]

def binary_mask_to_rle(binary_mask):
	## input is numpy array with either 0 or 1 : np.array([0,0,1,1,1,0,1])
	## output: rle={'counts': [2, 3, 1, 1], 'size': [7]}
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle

def rle_to_binary_mask(rle):
	## converts rle={'counts': [2, 3, 1, 1], 'size': [7]} to np.array([0,0,1,1,1,0,1])
	compressed_rle = pycocotools.mask.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
	mask = pycocotools.mask.decode(compressed_rle)
	return mask

if __name__=='__main__':
	print(binary_mask_to_rle(np.array([[0,0,1,1,1,0,1]])))
	rle = {'counts': [2, 3, 1, 1], 'size': [1,7]}
	print(rle_to_binary_mask(rle))
