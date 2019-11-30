import os
import numpy as np
import scipy as sc
from scipy import io
from scipy.io import wavfile
import yaml
import pylab
from IPython import embed
import matplotlib.pyplot as plt
import cv2 as cv2
import wavio
from read_config import getConfig
from IPython import embed


import dataset as dataset
import preprocessing_functions as pf
from processing import Processing

import torch
from torch.utils.data import TensorDataset, DataLoader

class Generator():
	def __init__(self):
		self.config = getConfig('config_yes') #reads configuration file
		self.paths = self.config['paths']
		self.configuration()

		self.proc = Processing(self.config['processing'])
		self.data = DataLoader(dataset=self.dataset(self.config['paths'], mode=self.config['processing']['mode'], images=False), batch_size=1, shuffle=False)

		for i, d in enumerate(self.data):
			features = self.proc.process(os.path.join(self.path_audio,str(d[0][0])))
			self.save(features, d[0][0].split('.')[0])

	def configuration(self):
		if self.config['processing']['task'] == 1:
			self.task = 'task1'
			self.dataset = dataset.WAV_dataset_task1
			self.path_spectra = os.path.join(
				self.config['paths']['spectra'],
				'spect_' + self.task
			)
			self.path_audio = os.path.join(
				self.config['paths']['audio'],
				'audio_' + self.task
			)
		else:
			self.task = 'task5'
			self.dataset = dataset.WAV_dataset_task5
			if self.config['processing']['features'] == 'nmf':
				features = 'nmf'
				folder = 'activ_'
			else:
				features = 'spectra'
				folder = 'spect_'
			self.setPath(features, folder)
			self.path_audio = os.path.join(
				self.paths['audio'],
				'audio_' + self.task,
				self.config['processing']['mode']
			)
	def setPath(self, features, folder):
		if not os.path.exists(self.paths[features]):
			os.mkdir(self.paths[features])
		if not os.path.exists(os.path.join(self.paths[features], folder + self.task)):
			os.mkdir(os.path.join(self.paths[features], folder + self.task))
		self.path_features = os.path.join(
			self.paths[features],
			folder + self.task
		)

	def show(self, mfccs, filter_banks, periodogram):
		plt.imshow(np.transpose(mfccs), cmap='jet', origin='lowest', aspect='auto')
		plt.colorbar()
		plt.show()
		plt.imshow(np.transpose(filter_banks), cmap='jet', origin='lowest', aspect='auto')
		plt.colorbar()
		plt.show()
		plt.imshow(np.transpose(periodogram), cmap='jet', origin='lowest', aspect='auto')
		plt.colorbar()
		plt.show()


	def save(self, img, name):
		final_path = os.path.join(self.path_features, name + '.png')
		print(final_path)
		cv2.imwrite(final_path, img)

if __name__ == '__main__':
	Generator()
