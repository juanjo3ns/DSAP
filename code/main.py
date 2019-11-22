import time as time
import os
import numpy as np
import cv2 as cv2
from collections import defaultdict
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from read_config import getConfig

import utils as utils
import model as model
import dataset as dataset
from torch import nn

from IPython import embed

from metrics import recall, accuracy

TRAIN = 'train'
VAL = 'validate'

class main():
	def __init__(self, prints=True):
		cfg = getConfig() #reads configuration file
		self.paths = cfg['paths']
		if cfg['task'] == 1:
			self.task = 1
			self.config = cfg['task1']
		else:
			self.task = 5
			self.config = cfg['task5']

		#e.g: task5/test1
		self.exp_path = 'task' + str(self.task) + '/' + self.config['exp_name']

		self.model = None
		self.LastTime = time.time()
		self.prints = prints
		self.writer = SummaryWriter(
			log_dir=os.path.join(
				self.paths['tensorboard'],
				self.exp_path)
		)
		self.start()

	def start(self):

		print(self.config)
		mode = self.config['mode']
		loader = self.get_loader(mode=mode)

		self.model = self.get_model()

		lossFunction, optimizer = self.get_LossOptimizer()

		total_step = len(loader)
		
		

		for epoch in range(self.config['init_epoch'], self.config['epochs']):
			loss_list = []
			total_outputs = []
			total_solutions = []
			for i, (img, tag) in enumerate(loader):

				img = img.unsqueeze(1)

				output, loss = self.run(img=img, criterion=lossFunction, solution=tag)

				loss_list.append(loss.item())

				if mode == TRAIN:
					# Backprop and perform Optimizer
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()

				outs_argmax = output.argmax(dim=1)
				total_outputs.extend(outs_argmax.cpu().numpy())
				total_solutions.extend(tag.numpy())

				# Print de result for this step (sha de canviar el typ? estava aixi tant a val com a train)
				self.print_info(typ="trainn", epoch=epoch, i=i, total_step=total_step, loss=loss.item(), num_epoch=self.config['epochs'])


			self.print_info(typ="epoch_loss", epoch=epoch, loss_list=loss_list)
			# acc = accuracy(total_outputs, total_solutions)
			# self.print_info(typ="epoch_acc", epoch=epoch, accuracy=100*len(acc[0])/len(total_outputs))
			# recall = recall(total_outputs, total_solutions, self.config["NUM_CLASSES"])
			# self.print_info(typ="epoch_recall", epoch=epoch, recall=recall)

			self.writer.add_scalar('Loss/'+ mode, sum(loss_list)/len(loss_list), epoch)

			if self.config['save_weights'] and epoch%self.config['save_weights_freq']==0 and mode == TRAIN:
				if not os.path.exists(os.path.join(self.paths['weights'], self.exp_path)):
					os.mkdir(os.path.join(self.paths['weights'], self.exp_path))
				torch.save(self.model.state_dict(),os.path.join(self.paths['weights'], self.exp_path, 'epoch_{}.pt'.format(epoch)))


	def run(self, img, criterion, solution):

		if self.config['gpu']:
			output = self.model(img.cuda())
		else:
			output = self.model(img)

		loss = self.compute_loss(criterion=criterion, output=output, solution=solution)

		return output, loss

	def compute_loss(self, criterion, output, solution, GPU=True):
		if self.config['gpu']:
			if self.task == 1:
				loss = criterion(output, solution.cuda())
			else:
				loss = criterion(output, solution.type(torch.DoubleTensor).cuda())
		else:
			loss = criterion(output, solution)

		return loss

	def get_loader(self, mode="train", shuffle=True):
		if self.task == 1:
			loader = DataLoader(dataset=dataset.WAV_dataset_task1(self.paths, mode=mode, images=True), batch_size=self.config['batch_size'], shuffle=shuffle)
		elif self.task == 5:
			loader = DataLoader(dataset=dataset.WAV_dataset_task5(self.paths, mode=mode, images=True), batch_size=self.config['batch_size'], shuffle=shuffle)
		return loader

	def get_model(self, GPU=True):

		if self.config['transfer_learning']['load']:
			self.print_info(typ="LoadModel", Weights = "From experiment: " + self.config['transfer_learning']['exp_name'] + '/' + self.config['transfer_learning']['exp_epoch'])
		else:
			self.print_info(typ="LoadModel", Weights = "From Scratch")

		mod, num = model.resnet18(num_classes=8)
		#mod = model.BaselineModel(num_classes=self.config['num_classes'])
		#mod = model.WAV_model_test()

		if self.config['gpu']:
			mod.cuda()

		if self.config['transfer_learning']['load']:
			mod.load_state_dict(torch.load(
				os.path.join(
					self.paths['weights'],
					self.config['transfer_learning']['exp_task'],
					self.config['transfer_learning']['exp_name'],
					self.config['transfer_learning']['exp_epoch'],
				)
			))
		self.model = mod
		self.print_info(typ="LoadModel", Status="Done")
		return mod

	def get_LossOptimizer(self):
		if self.task == 1:
			criterion = nn.CrossEntropyLoss()
		else:
			criterion = nn.BCEWithLogitsLoss()
		optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])

		self.print_info(typ="LossOptimizer", LossFunction="CrossEntropyLoss", optimizer="Adam")

		return criterion, optimizer

	def print_info(self, typ="default", **param):
		if not self.prints:
			return

		# Loading Model ---------------------------------------------------
		if typ == "LoadModel":

			if param.get("Status") == "Done":
				tim = time.time() - self.LastTime
				print("Took {:.2f} ms".format(tim*1000))
				print("Pramaters: {:.2f} M".format(sum(p.numel() for p in self.model.parameters())/1000000))
				print("-"*55)
				return

			print("-"*55 + "\n" + "-"*24 + " MODEL " + "-"*24)
			for itm in param:
				print(itm + ": " + str(param.get(itm)) )
			print("Loading Model ...")
			self.LastTime = time.time()

		# Config parameters  ----------------------------------------------
		if typ == "Init":
			print("-"*55 + "\n" + "-"*21 + " INIT CONFIG " + "-"*21 + "\n" + "-"*55)

			for itm in self.config:
				print(str(itm) + ": " + str(self.config[itm]))
			print("{} GPU's Available with cuda {} version.".format(torch.cuda.device_count()+1, torch.version.cuda))
			print("-"*55 + "\n" + "-"*55)

		# LOSS and Optimizer ----------------------------------------------
		if typ == "LossOptimizer":
			for itm in param:
				print(itm + ": " + str(param.get(itm)) )
			print("-"*55)
			print("-"*23 + " TRAINING " + "-"*22 )

		# Training --------------------------------------------------------
		if typ == "train":
			print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(param.get("epoch") + 1,
																		param.get("num_epoch"), param.get("i") + 1,
																		param.get("total_step"),
																		param.get("loss") ))
		# Training 2 -----------------------------------------------------
		if typ == "trainn":
			index = round( (param.get("i") + 1)/(param.get("total_step"))*20 )
			#maxim = 20 - index
			print("Epoch [{}/{}]".format(param.get("epoch") + 1, param.get("num_epoch")) +
					"[" + "#"*index + " "*(20-index) + "] " + "[{}/{}]".format(param.get("i") + 1, param.get("total_step")) +
					", Loss: {:.4f}".format(param.get("loss"))

					, end="\r" )


			if (param.get("i")+1) == param.get("total_step"):
				print("")

		# Training 3 -----------------------------------------------------
		if typ == "epoch_loss":

			loss_list= param.get("loss_list")
			epoch = param.get("epoch")

			avg_loss = sum(loss_list)/len(loss_list)

			print("Epoch {} , loss: {}".format(epoch, avg_loss))
			self.writer.add_scalar('Loss/train', avg_loss, epoch)

		# Training 4 -----------------------------------------------------
		if typ == "epoch_acc":
			accuracy = param.get("accuracy")
			epoch = param.get("epoch")
			print("Epoch {} , acc: {:.4f} %".format(epoch, accuracy))
			self.writer.add_scalar('Accuracy/train', accuracy, epoch)


		# Training 5 -----------------------------------------------------
		if typ == "epoch_recall":

			recall = param.get("recall")
			epoch = param.get("epoch")

			for k in recall:
				print("\tClass:{} -> Recall: {:.4f} %".format(k, recall[k]))
			print(recall)
			#self.writer.add_scalars("Recall", recall, epoch)


if __name__ == "__main__":
	main()
