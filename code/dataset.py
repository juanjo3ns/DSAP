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

TRAIN = 'train'
VAL = 'validate'

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

class WAV_task5(Dataset):
	def __init__(self, paths, mode='train', images=False, mixup={"apply": False, "alfa":0.5, "rate":1}, features="mfcc", classes=8):
		self.paths = paths
		self.images = images
		self.classes = classes
		self.mixup = mixup

		if features == "mfcc":
			self.final_path = os.path.join(self.paths['spectra'],'spect_task5')
		elif features == "nmf":
			self.final_path = os.path.join(self.paths['nmf'],'activ_task5')
		elif features == "deltas":
			self.final_path = os.path.join(self.paths['deltas'],'deltas_task5')
		elif features == "all":
			self.final_path = os.path.join(self.paths['all'],'all_task5')
		self.images_data = []
		self.read_from_database(split=mode)


	def __len__(self):
		return len(self.images_data)

	def __getitem__(self, index):
		img, tag = self.images_data[index]
		return  torch.from_numpy(img).float(), np.array(tag).astype(float)


	def read_from_database(self, split="train"):

		items = mongo.get_from(filt={"split": split}, collection="task5")
		names = []
		for it in items:
			if self.classes == 8:
				names.append([it["file_name"], it["high_labels"]])
			else:
				names.append([it["file_name"], it["low_labels"]])

			img = self.load_image(file_name=it["file_name"])
			#img = torch.from_numpy(img).float()
			if self.classes == 8:
				tag = np.array(it["high_labels"]).astype(int)
			else:
				tag = np.array(it["low_labels"]).astype(int)


			self.images_data.append([img, tag])

		if self.mixup["apply"] and split==TRAIN:
			for it in range(round(len(names)*self.mixup["rate"])):
				img, tag = self.apply_mixup(names)
				self.images_data.append([img, tag])


	def apply_mixup(self, names):
		#We took two random index from the real list of images:
		index1 = random.randint(0, len(names)-1)
		index2 = index1
		while index1==index2:		#Make shure we don't take the same image
			index2 = random.randint(0, len(names)-1)

		#We took those names and tag images
		img1, tag1 = names[index1]
		img2, tag2 = names[index2]

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

		return img, np.array(tag)

	def load_image(self, file_name):
		img = utils.load_image(os.path.join(self.final_path, file_name.split('.')[0]))
		#img = torch.from_numpy(img).float()
		return img


class WAV_dataset_task5(Dataset):
	def __init__(self, paths, mode='train', images=False, mixup={"apply": False, "alfa":0.5, "rate":10}, features="mfcc"):
		self.paths = paths
		self.list_names = [] #cada array de dins correspon al path i al tag
		self.images = images
		self.read_from_database(split=mode)
		self.count = [0,1] #First: For real images, Second: Mixup rate count
		self.mixup = mixup
		if features == "mfcc":
			self.final_path = os.path.join(self.paths['spectra'],'spect_task5')
		elif features == "nmf":
			self.final_path = os.path.join(self.paths['nmf'],'activ_task5')
		elif features == "deltas":
			self.final_path = os.path.join(self.paths['deltas'],'deltas_task5')
		elif features == "all":
			self.final_path = os.path.join(self.paths['all'],'all_task5')
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
		if self.count[0]+1==len(self.list_names):
			self.count[0] = 0
		return img, np.array(tag)

	def read_from_database(self, split="train"):

		items = mongo.get_from(filt={"split": split}, collection="task5")
		for it in items:
			self.list_names.append([it["file_name"], it["high_labels"]])



	def load_image(self, file_name):
		img = utils.load_image(os.path.join(self.final_path, file_name.split('.')[0]))
		#img = torch.from_numpy(img).float()

		return img

if __name__ == '__main__':
	paths = {"audio": "/home/data/audio/",
  			"spectra": "/home/data/spectrogram/",
  			"weights": "/home/weights/",
  			"tensorboard": "/home/tensorboard/"}

	ds = WAV_task5_8(paths, mode=TRAIN, images=True, mixup={"apply": True, "alfa":0.5, "rate":0.3})
	dl = DataLoader(dataset=ds, batch_size=5, shuffle=True)
	print(dl.__len__())
	"""
	for i, (img, tag) in enumerate(dl):

		print(i+1, end="\r")
		print(tag)
		if i==10:
			break
	print("\n")
	"""
