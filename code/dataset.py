import os
import sys
import cv2

import utils as utils
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from IPython import embed
import mongodb_api as mongo

import random

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

		#print("Total of {} images.".format(len(self.list_names)))

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
	def __init__(self, paths, mode='train', images=False, mixup={"apply": False, "alfa":0.5, "rate":10}):
		self.paths = paths
		self.list_names = [] #cada array de dins correspon al path i al tag
		self.images = images
		self.read_from_database(split=mode)
		self.count = [0,1] #First: For real images, Second: Mixup rate count
		self.mixup = mixup

		print("Total of {} images.".format(self.__len__()))

	def __len__(self):
		if self.mixup["apply"]:
			return len(self.list_names)*(self.mixup["rate"]+1)
		else:
			return len(self.list_names)


	def __getitem__(self, index):

		if self.mixup["apply"]:
			img, tag = self.apply_mixup(index = index) #With mixup, img and tag is always shuffled
			tag = np.array(tag).astype(float)
			img = torch.from_numpy(img).float()

		else:
			img, tag = self.list_names[index]
			if self.images:
				img = self.load_image(file_name=img)
				img = torch.from_numpy(img).float()
			tag = np.array(tag).astype(int)
		
		return img, tag

	def apply_mixup(self, index):
		if self.count[1]%(self.mixup["rate"]+1) == 0:
			self.count[1] = 1
			img, tag = self.list_names[self.count[0]]
			tag = np.array(tag).astype(int)
			if self.images:
				img = self.load_image(file_name=img) 
			#print("------------------" + str(self.count[0]))
			self.count[0] += 1

		else:
			#print("+")
			#We took two random index from the real list of images:
			index1 = random.randint(0, len(self.list_names)-1)
			index2 = index1
			while index1==index2:		#Make shure we don't take the same image
				index2 = random.randint(0, len(self.list_names)-1)

			#We took those names and tag images
			img1, tag1 = self.list_names[index1]
			img2, tag2 = self.list_names[index2]

			#We load both images
			img1 = self.load_image(file_name=img1)
			img2 = self.load_image(file_name=img2)

			img1 = np.array(img1)
			img2 = np.array(img2)
			tag1 = np.array(tag1)
			tag2 = np.array(tag2)

			#We apply mixup
			img = (self.mixup["alfa"]*img1+(1-self.mixup["alfa"])*img2)
			tag = (self.mixup["alfa"]*tag1+(1-self.mixup["alfa"])*tag2)

			self.count[1] +=1
		
		return img, np.array(tag)

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
		#img = torch.from_numpy(img).float()

		return img

if __name__ == '__main__':
	paths = {"audio": "/home/data/audio/",
  			"spectra": "/home/data/spectrogram/",
  			"weights": "/home/weights/",
  			"tensorboard": "/home/tensorboard/"}

	ds = WAV_dataset_task5(paths, mode='validate', images=True, mixup={"apply": False, "alfa":0.5, "rate":10})
	dl = DataLoader(dataset=ds, batch_size=1)
	for i, x in enumerate(dl):
		img, tag = x
		print(i+1, end="\r")
		print(img)
		#print(tag)
		if i==10:
			break
	print("\n")
		
