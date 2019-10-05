import os
import sys
import cv2

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import mongodb_api as mongo

PATH_IMAGES = "/home/data/audio/"

class WAV_dataset(Dataset):
	def __init__(self, mode='train'):
		
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

		self.size = [250, 250] #patilla maxim, ja es veura
		self.list_names = [] #cada array de dins correspon al path i al tag
		#self.read_from_database(split=mode)
		self.read_from_database_test(split=mode)

		print("Total of {} images.".format(len(self.list_names)))

	def __len__(self):
		return len(self.list_names)

	def __getitem__(self, index):

		img, tag = self.list_names[index]
		#img = self.load_image(file_name=img)
		
		return img, self.tags[tag]

	def read_from_database(self, split="train"): #de moment nomes afegim de airport i bus per fer probes aixi el dataset es mes petit
		
		items = mongo.get_from(filt={"tag": "airport", "split": split})	
		
		for it in items:
			self.list_names.append([it["file_name"], it["tag"]])
		
		
		items = mongo.get_from(filt={"tag": "bus", "split": split})
		for it in items:
			self.list_names.append([it["file_name"], it["tag"]])
		
	def read_from_database_test(self, split):
		path_to_read_spectrograms_for_testing = "/home/data/spect"
		path = path_to_read_spectrograms_for_testing
		names = os.listdir(path)
		for n in names:
			on = n.split(".")[0] + ".wav"
			item = mongo.get_from(filt={"file_name": on})
			self.list_names.append([on.split(".")[0], item[0]["tag"]])
			

	def load_image(self, file_name):
		pass
		# reshape
		#img = cv2.resize(img, (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)
		#image = image.astype(np.float32)
		#image = torch.FloatTensor(image)
			

if __name__ == '__main__':
	ds = WAV_dataset(mode='train')
	for x in ds:
		img, tag = x
		print(img)
		print(tag)
		break

