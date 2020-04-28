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

		test_imgs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
		for k in range(1):
			model_num = 1300 # 1540
			print("model num:", model_num)
			for i in range(len(test_imgs)):
				index = test_imgs[i]
				path1 = path + 'far/' + str(index) + '.jpg'
				path2 = path + 'near/' + str(index) + '.jpg'
				begin = time.time()
				generate(path1, path2, './models_mn/', index, output_path='./results/', model_num=model_num)
				end = time.time()
				T.append(end - begin)
				print(T)


if __name__ == '__main__':
	main()
