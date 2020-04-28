from __future__ import print_function

import time

# from utils import list_images
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from train import train_mask_net #, train_boundary_net
from generate import generate
import scipy.ndimage

BATCH_SIZE = 16
EPOCHES = 7
LOGGING = 50
IS_TRAINING = True

def main():
	if IS_TRAINING:
		f = h5py.File('multi_focus_dataset106.h5', 'r')
		sources = f['data'][:]
		sources = np.transpose(sources, (0, 3, 2, 1))
		print("sources shape:", sources.shape)
		print(('\nBegin to train the network ...\n'))
		train_mask_net(sources, './models_mn/', EPOCHES, BATCH_SIZE, logging_period = LOGGING)
		# train_boundary_net(sources, './models_bn/', './models_mn/', mask_net_model_num, 4, BATCH_SIZE)

	else:
		print('\nBegin to generate pictures ...\n')
		path = 'test_imgs/'
		T = []

		for i in range(10):
			index = i + 1
			path1 = path + 'far/' + str(index) + '.jpg'
			path2 = path + 'near/' + str(index) + '.jpg'
			begin = time.time()
			generate(path1, path2, './model/model.ckpt', index, output_path='./results/')
			end = time.time()
			T.append(end - begin)
			print(T)


if __name__ == '__main__':
	main()
