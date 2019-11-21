import os
import sys
import cv2

import utils as utils
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from IPython import embed
import mongodb_api as mongo

class WAV_dataset_task1(Dataset):
	def __init__(self, paths, mode='train', images=False):
		self.tags = {"airport":0,
					"bus":1,
					"shopping_mall":2,
					"street_pedestrian":3,
					"street_traffic":4,
					"metro_station":5,
					"park":6,
					"metro":7,
					"public_square":8,
					"tram":9}
		self.paths = paths
		self.size = [250, 250] #patilla maxim, ja es veura
		self.list_names = [] #cada array de dins correspon al path i al tag
		self.images = images
		self.read_from_database(split=mode)

		print("Total of {} images.".format(len(self.list_names)))

	def __len__(self):
		return len(self.list_names)

	def __getitem__(self, index):

		img, tag = self.list_names[index]
		if self.images:
			img = self.load_image(file_name=img)

		return img, self.tags[tag]

	def read_from_database(self, split="train"):
		items = mongo.get_from(filt={"split": split})
		for it in items:
			self.list_names.append([it["file_name"], it["tag"]])

	def load_image(self, file_name):
		img = utils.load_image(os.path.join(
			self.paths['spectra'],
			'spect_task1',
			file_name.split('.')[0]
		))
		img = torch.from_numpy(img).float()

		return img

class WAV_dataset_task5(Dataset):
	def __init__(self, paths, mode='train', images=False):
		self.paths = paths
		self.list_names = [] #cada array de dins correspon al path i al tag
		self.images = images
		self.read_from_database(split=mode)

		print("Total of {} images.".format(len(self.list_names)))

	def __len__(self):
		return len(self.list_names)

	def __getitem__(self, index):

		img, tag = self.list_names[index]
		if self.images:
			img = self.load_image(file_name=img)

		return img, np.array(tag).astype(int)

	def read_from_database(self, split="train"):

		items = mongo.get_from(filt={"split": split}, collection="task5")
		for it in items:
			self.list_names.append([it["file_name"], it["high_labels"]])


	def load_image(self, file_name):
		img = utils.load_image(os.path.join(
			self.paths['spectra'],
			'spect_task5',
			file_name.split('.')[0]
		))
		img = torch.from_numpy(img).float()

		return img

if __name__ == '__main__':
	ds = WAV_dataset_task5(mode='train')
	for x in ds:
		img, tag = x
		print(img)
		print(tag)
		break
