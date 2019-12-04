import sys
import time as time
import os
import numpy as np
import cv2 as cv2
from collections import defaultdict
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from read_config import getConfig
from telegram import send

import utils as utils
import model as model
import dataset as dataset
from torch import nn

from IPython import embed

from metrics import *

TRAIN = 'train'
VAL = 'validate'

class main():
	def __init__(self, prints=True):

		config_file = sys.argv[1]
		cfg = getConfig(config_file) #reads configuration file
		self.paths = cfg['paths']
		if cfg['task'] == 1:
			self.task = 1
			self.config = cfg['task1']
		else:
			self.task = 5
			self.config = cfg['task5']

		self.best_accuracy = 0
		self.best_epoch = -1
		#e.g: task5/test1
		self.exp_path = 'task' + str(self.task) + '/' + self.config['exp_name']

		self.model = None
		self.LastTime = time.time()
		self.prints = prints

		if self.config['telegram']:
			send("Running " + self.config['exp_name'] + "...")

		if self.config['save_tensorboard']:
			self.writer = SummaryWriter(
				log_dir=os.path.join(
					self.paths['tensorboard'],
					self.exp_path)
			)
		self.start()

	def start(self):

		#print(self.config)
		mode = self.config['mode']

		if mode == TRAIN:
			loader_eval = self.get_loader(mode=VAL)
			lossFunction_eval, _ = self.get_LossOptimizer(mode=VAL, show=False)

		loader = self.get_loader(mode=mode)

		self.model = self.get_model()
		lossFunction, optimizer = self.get_LossOptimizer()

		total_step = len(loader)

		self.accuracy_eval = 0

		if mode == TRAIN:
			self.model.train()
		else:
			self.model.eval()

		for epoch in range(self.config['init_epoch'], self.config['epochs']):
			time_epoch = time.time()

			loss_list = []
			total_outputs = []
			total_solutions = []
			for i, (img, tag) in enumerate(loader):
				time_batch = time.time()
				img = img.unsqueeze(1)

				output, loss = self.run(img=img, criterion=lossFunction, solution=tag)

				loss_list.append(loss.item())

				if mode == TRAIN:
					# Backprop and perform Optimizer
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()

				if self.task == 1:
					output = output.argmax(dim=1)

				total_outputs.extend(output.cpu().detach().numpy())
				total_solutions.extend(tag.numpy())

				self.timm_batch = (time.time() - time_batch)
				self.print_info(typ="trainn", epoch=epoch, i=i, total_step=total_step, loss=loss.item(), num_epoch=self.config['epochs'], tim=self.timm_batch)
			self.timm_epoch = (time.time() - time_epoch)
			self.print_info(typ="epoch_loss", epoch=epoch, loss_list=loss_list, tim=self.timm_epoch)
			if epoch%1 == 0:
				show = True
			else:
				show = False
			self.compute_metrics(total_outputs=total_outputs, total_solutions=total_solutions, mode=mode, epoch=epoch, acc_eval=self.accuracy_eval, show=show)
			if mode==TRAIN:
				self.evaluate(criterion=lossFunction_eval, loader=loader_eval, epoch=epoch, show=show)

	def compute_metrics(self, total_outputs, total_solutions, mode, show=True, epoch=0, acc_eval=0):
		if show:
			print("---------------- " + str(mode) + " ----------------")
		if self.task == 1:
			acc = accuracy(total_outputs, total_solutions)
			acc = 100*len(acc[0])/len(total_outputs)
			recall = recall(total_outputs, total_solutions, self.config['num_classes'])
		elif self.task == 5:
			acc, recall, micro_auprc, macro_auprc = multilabel_metrics(total_outputs, total_solutions, self.config['threshold'], self.config['mixup']['apply'])
		if show:
			self.print_info(typ="epoch_acc", epoch=epoch, accuracy=acc)
			self.print_info(typ="epoch_recall", epoch=epoch, recall=recall)
		self.log(acc, micro_auprc, macro_auprc, mode, epoch)
		if acc > self.best_accuracy:
			if self.config['save_weights'] and mode == TRAIN:
				if not os.path.exists(os.path.join(self.paths['weights'], self.exp_path)):
					os.mkdir(os.path.join(self.paths['weights'], self.exp_path))
				torch.save(self.model.state_dict(),os.path.join(self.paths['weights'], self.exp_path, 'epoch_{}.pt'.format(epoch)))
				try:
					os.remove(os.path.join(self.paths['weights'], self.exp_path, 'epoch_{}.pt'.format(self.best_epoch)))
				except:
					pass
			self.best_accuracy = acc
			self.best_epoch = epoch

	def log(self, acc, micro_auprc, macro_auprc, mode, epoch):
		if self.config['save_tensorboard']:
			self.writer.add_scalar('Accuracy/'+mode, acc, epoch)
			self.writer.add_scalar('Micro-AUPRC/'+mode, micro_auprc, epoch)
			self.writer.add_scalar('Macro-AUPRC/'+mode, macro_auprc, epoch)
		if self.config['telegram'] and epoch%int(self.config['epochs']/5)==0:
			send("Epoch " + str(epoch) + \
			"\nMode " + mode + \
			"\n\tMicro-AUPRC: " + str(round(micro_auprc,2)) + \
			"\n\tMacro-AUPRC: " + str(round(macro_auprc,2)) + \
			"\n\tAccuracy: " + str(round(acc,2)) + \
			"\n\tEpoch time: " + str(round(self.timm_epoch,2)) + \
			"\n\tBatch time: " + str(round(self.timm_batch,2)))

	def evaluate(self, criterion, loader, epoch=0, show=True):
		self.model.eval()

		loss_list_eval = []
		total_outputs_eval = []
		total_solutions_eval = []

		for i, (img, tag) in enumerate(loader):

			img = img.unsqueeze(1)
			output_eval, loss_eval = self.run(img=img, criterion=criterion, solution=tag)

			loss_list_eval.append(loss_eval.item())

			if self.task == 1:
				output_eval = output_eval.argmax(dim=1)

			total_outputs_eval.extend(output_eval.cpu().detach().numpy())
			total_solutions_eval.extend(tag.numpy())

		self.print_info(typ="epoch_loss_eval", epoch=epoch, loss_list=loss_list_eval)

		#print("---------------- Evaluation ----------------")
		self.compute_metrics(total_outputs=total_outputs_eval, total_solutions=total_solutions_eval, mode=VAL, show=show, epoch=epoch)



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
				loss = criterion(output, solution.type(torch.FloatTensor).cuda())
		else:
			loss = criterion(output, solution)

		return loss

	def get_loader(self, mode="train", shuffle=True):
		self.print_info(typ="dataset", mode=mode)

		if self.task == 1:
			datasett = dataset.WAV_dataset_task1(self.paths, mode=mode, images=True)
			loader = DataLoader(dataset=datasett, batch_size=self.config['batch_size'], shuffle=shuffle)

		elif self.task == 5:
			if mode==TRAIN:
				datasett = dataset.WAV_task5(self.paths, mode=mode, images=True, mixup=self.config["mixup"], features=self.config['features'], classes=self.config['num_classes'])
				loader = DataLoader(dataset=datasett, batch_size=self.config['batch_size'], shuffle=shuffle, drop_last=True)
			else:
				datasett = dataset.WAV_task5(self.paths, mode=mode, images=True, features=self.config['features'], classes=self.config['num_classes'])
				loader = DataLoader(dataset=datasett, batch_size=1, shuffle=shuffle)



		#print("Total of {} images.".format(datasett.__len__()))

		if mode==TRAIN:
			if self.config["mixup"]["apply"]:
				self.print_info(typ="data_aug", aug=["mixup"], val1=self.config["mixup"])
		else:
			self.print_info(typ="data_aug", aug=["mixup"], val1={'apply': False, 'alfa': 0, 'rate': 0})
		return loader

	def get_model(self, GPU=True):

		if self.config['transfer_learning']['load']:
			self.print_info(typ="LoadModel", Weights = "From experiment: " + self.config['transfer_learning']['exp_name'] + '/' + self.config['transfer_learning']['exp_epoch'])
		else:
			self.print_info(typ="LoadModel", Weights = "From Scratch")

		if self.config['model'] == 'baseline':
			mod = model.BaselineModel(num_classes=self.config['num_classes'], p_dropout=self.config['dropout'], features=self.config['features'])
		elif self.config['model'] == 'resnet':
			# mod, num = model.resnet18(num_classes=self.config["num_classes"], p_dropout=self.config['dropout'], features=self.config['features'])
			mod = model.SOTANet(num_classes=self.config["num_classes"], p_dropout=self.config['dropout'])
		elif self.config['model'] == 'rnn':
			mod = model.WAV_model_test()
		else:
			print("Choose correct model!")
			sys.exit()

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

	def get_LossOptimizer(self, mode=TRAIN, show=True):
		if self.task == 1:
			criterion = nn.CrossEntropyLoss()
			if show:
				self.print_info(typ="LossOptimizer", LossFunction="CrossEntropyLoss", optimizer="Adam")
		else:
			if self.config['pondweights']:
				f8, f23 = utils.frequency(filterr={"split":mode})
				#criterion = nn.BCELoss(torch.Tensor(f8).cuda())
				if self.config['num_classes'] == 8:
					criterion = nn.BCELoss(torch.Tensor(f8).cuda())
				else:
					criterion = nn.BCELoss(torch.Tensor(f23).cuda())

			else:
				criterion = nn.BCELoss()
			if show:
				self.print_info(typ="LossOptimizer", LossFunction="BCELoss", optimizer="Adam")
		if mode==TRAIN:
			optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])
		else:
			optimizer = None

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
				print("Trainable parameters: {:.2f} M".format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)/1000000) )
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
					", Loss: {:.4f} took {}s".format(param.get("loss"), round(param.get("tim"), 2)), end="\r" )


			if (param.get("i")+1) == param.get("total_step"):
				print("")

		# Epcoh loss -----------------------------------------------------
		if typ == "epoch_loss":

			loss_list= param.get("loss_list")
			epoch = param.get("epoch")

			avg_loss = sum(loss_list)/len(loss_list)

			print("Epoch {} , loss: {}  took {}s".format(epoch+1, avg_loss, round(param.get("tim"), 2)))
			if self.config['save_tensorboard']:
				self.writer.add_scalar('Loss/train', avg_loss, epoch)

		# Accuracy -----------------------------------------------------
		if typ == "epoch_acc":
			accuracy = param.get("accuracy")
			epoch = param.get("epoch")
			print("Epoch {} , acc: {:.4f} %".format(epoch+1, accuracy))


		# Epoch recall -----------------------------------------------------
		if typ == "epoch_recall":

			recall = param.get("recall")
			epoch = param.get("epoch")
			if type(recall) is dict:
				for k in recall:
					print("\tClass:{} -> Recall: {:.4f} %".format(k, recall[k]))
			else:
				for i,j in enumerate(recall):
					print("\tClass:{} -> Recall: {:.4f} %".format(i, j))

			#self.writer.add_scalars("Recall", recall, epoch)
		#Data Augmentation -------------------------------------------------
		if typ == "data_aug":
			augments = param.get("aug")
			print("Data augmentation: ", end="\n")
			for au in augments:
				if au=="mixup":
					print(" - mixup: " + str(param.get("val1")))

		#Dataset ------------------------------------------------------
		if typ == "dataset":
			print("-"*55 + "\n" + "-"*23 + " DATASET " + "-"*23)
			print(str(param.get("mode")) + ": ", end="")

		# Epcoh loss Eval -----------------------------------------------------
		if typ == "epoch_loss_eval":

			loss_list= param.get("loss_list")
			epoch = param.get("epoch")

			avg_loss = sum(loss_list)/len(loss_list)

			print("Epoch {} , loss: {}".format(epoch+1, avg_loss))
			if self.config['save_tensorboard']:
				self.writer.add_scalar('Loss/validate', avg_loss, epoch)

if __name__ == "__main__":
	main()
