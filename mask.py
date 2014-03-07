import numpy as np
import os
import sys
import gflags
import pandas as pd
import time
import skimage.io
import skimage.transform
import selective_search_ijcv_with_python as selective_search
import caffe

NET = None

IMAGE_DIM = None
CROPPED_DIM = None
IMAGE_CENTER = None

IMAGE_MEAN = None
CROPPED_IMAGE_MEAN = None

BATCH_SIZE = None
NUM_OUTPUT = None

MASK_SIZE = None;
MASK_TYPES = ['full', 'bottom', 'top', 'left', 'right']

def load_image(filename):
	"""
	Input:
    filename: string

    Output:
    image: an image of size (H x W x 3) of type uint8.
    """
    img = skimage.io.imread(filename)
    if img.ndim == 2:
    img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
  	elif img.shape[2] == 4:
    img = img[:, :, :3]
  	return img

def transfrom_box(box, idx):
	"""
	Input:
	box: a list [u,l,b,r];
	ul: the upper left index
	br: the bottom right index

	idx: an integer indicating the mask type
	"""
	[u,l,b,r] = box
	if MASK_TYPES[idx] == 'full':
		return box
	elif MASK_TYPES[idx] == 'bottom':
		return [(u+b)/2, l, b, r]
	elif MASK_TYPES[idx] == 'top':
		return [u, l, (u+b)/2, r]
	elif MASK_TYPES[idx] == 'left':
		return [u, l, b, (l+r)/2]
	elif MASK_TYPES[idx] == 'right':
		return [u, (l+r)/2, b, r]
	else 
		# throw error: idx not in range

def intersection_area(box, T):
	if (box[0]>T[2] and box[1]<T[3]) and (T[0]>box[2] and T[1]<box[3]):
		S1 = (box[0] - T[2]) * (box[1] - T[3])
		S2 = (T[0] - box[2]) * (T[1] - box[3])
		return min(S1, S2);
	else
		return 0;

def generate_mask(image, box):
	"""

	output: mask, an 2 dimensional array of size MASK_SIZE
	"""

	height = image.shape[0]/MASK_SIZE
	width = image.shape[1]/MASK_SIZE
	mask = zeros((MASK_SIZE, MASK_SIZE))

	for r in xrange(MASK_SIZE):
		for c in xrange(MASK_SIZE):
			T = [height*(r - 1), width*(c - 1), height*r, width*r]
			mask[r, c] = intersection_area(box, T) / (height*width)

	return mask;




