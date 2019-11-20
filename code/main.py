import time as time
import numpy as np
import cv2 as cv2
from collections import defaultdict
import torch
import yaml
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import utils as utils
import model as model
import dataset as dataset
from torch import nn

from IPython import embed

from metrics import recall, accuracy

TRAIN = 'train'
VAL = 'validate'

class main():
    def __init__(self, cfg, prints=True):
        if cfg['task'] == '1':
            self.task = 1
            self.config = cfg['task1']
        else:
            self.task = 5
            self.config = cfg['task5']
        self.model = None
        self.LastTime = time.time()
        self.prints = prints
        self.writer = SummaryWriter(log_dir=self.config['paths']['path_tensorboard'])
        self.start()

    def start(self):
        nn_config = self.config['nn']

        print(nn_config)
        mode = nn_config['mode']
        loader = self.get_loader(mode=mode)

        self.model = self.get_model()

        lossFunction, optimizer = self.get_LossOptimizer()

        total_step = len(loader)


        for epoch in range(nn_config['init_epoch'], nn_config['epochs']):
            loss_list = []
            total_outputs = []
            total_solutions = []
            for i, (img, tag) in enumerate(loader):

                img = img.unsqueeze(1)

                output, loss = self.run(img=img, criterion=lossFunction, solution=tag)

                loss_list.append(loss.item())

                if nn_config['mode'] == TRAIN:
                    # Backprop and perform Optimizer
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                outs_argmax = output.argmax(dim=1)
                total_outputs.extend(outs_argmax.cpu().numpy())
                total_solutions.extend(tag.numpy())

                # Print de result for this step (sha de canviar el typ? estava aixi tant a val com a train)
                self.print_info(typ="trainn", epoch=epoch, i=i, total_step=total_step, loss=loss.item(), num_epoch=self.config['nn']['epochs'])


            self.print_info(typ="epoch_loss", epoch=epoch, loss_list=loss_list)
            acc = accuracy(total_outputs, total_solutions)
            self.print_info(typ="epoch_acc", epoch=epoch, accuracy=100*len(acc[0])/len(total_outputs))
            recall = recall(total_outputs, total_solutions, self.config["NUM_CLASSES"])
            self.print_info(typ="epoch_recall", epoch=epoch, recall=recall)

            self.writer.add_scalar('Loss/'+ nn_config['mode'], sum(loss_list)/len(loss_list), epoch)

            if nn_config['save_weights'] and epoch%nn_config['save_weights_freq']==0:
                torch.save(self.model.state_dict(), "/home/data/models/test3/epoch_{}_.pt".format(epoch))


    def run(self, img, criterion, solution):

        if self.config['nn']['gpu']:
            output = self.model(img.cuda())
        else:
            output = self.model(img)

        loss = self.compute_loss(criterion=criterion, output=output, solution=solution)

        return output, loss

    def compute_loss(self, criterion, output, solution, GPU=True):
        #solution = torch.from_numpy(solution.cpu()).long()
        #print(solution)
        if self.config['nn']['gpu']:
            loss = criterion(output.cpu(), solution.type(torch.DoubleTensor))
        else:
            loss = criterion(output, solution)

        return loss

    def get_loader(self, mode="train", shuffle=True):
        if self.task == 1:
            loader = DataLoader(dataset=dataset.WAV_dataset(mode=mode, images=True, self.config['paths']), batch_size=self.config['nn']['batch_size'], shuffle=shuffle)
        elif self.task == 5:
            loader = DataLoader(dataset=dataset.WAV_dataset_task5(mode=mode, images=True, self.config['paths']), batch_size=self.config['nn']['batch_size'], shuffle=shuffle)
        return loader

    def get_model(self, GPU=True):

        if self.config['nn']['load_weights']:
            self.print_info(typ="LoadModel", Weights = "From file: " + str(self.config['paths']['path_weights'].split("/")[-1]))
        else:
            self.print_info(typ="LoadModel", Weights = "From Scratch")


        mod = model.BaselineModel()
        #mod = model.WAV_model_test()

        if self.config['nn']['gpu']:
            mod.cuda()

        if self.config['nn']['load_weights']:
            mod.load_state_dict(torch.load(self.config['paths']['path_weights']))

        self.print_info(typ="LoadModel", Status="Done")
        return mod

    def get_LossOptimizer(self):
        #criterion = nn.CrossEntropyLoss()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['nn']['lr'])

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

            for itm in self.config['nn']:
                print(str(itm) + ": " + str(self.config['nn'][itm]))
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
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    main(cfg)
