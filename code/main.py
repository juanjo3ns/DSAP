import time as time
import numpy as np
import cv2 as cv2
from collections import defaultdict
import torch
from torch.utils.data import TensorDataset, DataLoader

import utils as utils
import model as model
import dataset as dataset
from torch import nn

from IPython import embed

PATH_SPECTROGRAM = "/home/data/spect/"


class main():
    def __init__(self, mode="train", prints=True):
        self.model = None
        self.LastTime = time.time()
        self.prints = prints
        self.config = {}
        self.run()


    def run(self):
        self.set_config(NUM_EPOCHS=5,
                        NUM_CLASSES=10,
                        BATCH_SIZE=32,
                        LEARNING_RATE=0.00001,
                        GPU=True)

        train_loader = self.get_loader(mode="train")

        self.model = self.get_model()


        lossFunction, optimizer = self.get_LossOptimizer()

        total_step = len(train_loader)


        for epoch in range(self.config["NUM_EPOCHS"]):
            loss_list = []
            total_outputs = []
            total_solutions = []
            for i, (img, tag) in enumerate(train_loader):
                img = img.unsqueeze(1)
                output = self.model(img.cuda())
                loss = self.compute_loss(criterion=lossFunction, output=output, solution=tag)
                loss_list.append(loss.item())


                # Backprop and perform Optimizer
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                outs_argmax = output.argmax(dim=1)
                total_outputs.extend(outs_argmax.cpu().numpy())
                total_solutions.extend(tag.numpy())
                # Print de result for this step
                self.print_info(typ="trainn", epoch=epoch, i=i, total_step=total_step, loss=loss.item(), num_epoch=self.config["NUM_EPOCHS"])
            self.print_info(typ="epoch_loss", epoch=epoch, loss_list=loss_list)
            self.accuracy(total_outputs, total_solutions, epoch)
            self.recall(total_outputs, total_solutions, epoch)

    def recall(self, output, solutions, epoch):
        recall = defaultdict(int)
        for i in range(self.config["NUM_CLASSES"]):
            positions = np.where(np.array(solutions)==i)
            for p in positions[0]:
                if output[p] == i:
                    recall[i] += 1
            recall[i] /= len(positions[0])
            recall[i] *= 100

        self.print_info(typ="epoch_recall", epoch=epoch, recall=recall)

    def accuracy(self, output, solutions, epoch):
        a = np.where(np.array(output)==solutions)
        self.print_info(typ="epoch_acc", epoch=epoch, accuracy=100*len(a[0])/len(output))

    def load_image(self, img):
        img = utils.load_image(PATH_SPECTROGRAM + img[0].split('.')[0])
        img = torch.from_numpy(img).float()
        img = img.permute(2,0,1)

        return img

    def compute_loss(self, criterion, output, solution, GPU=True):
        # solution = torch.from_numpy(solution).long()
        if self.config['GPU']:
            loss = criterion(output.cuda(), solution.cuda())
        else:
            loss = criterion(output, solution)

        return loss

    def get_loader(self, mode="train", shuffle=True):

        loader = DataLoader(dataset=dataset.WAV_dataset(mode=mode, images=True), batch_size=self.config["BATCH_SIZE"], shuffle=shuffle)
        return loader

    def set_config(self, **param):
        for par in param:
            self.config[par] = param.get(par)
        self.print_info(typ="Init")

    def get_model(self, GPU=True):
        self.print_info(typ="LoadModel", Weights= "From Scratch")

        #mod = model.WAV_model()
        mod = model.BaselineModel()
        if self.config['GPU']:
            mod.cuda()

        self.print_info(typ="LoadModel", Status="Done")
        return mod

    def get_LossOptimizer(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["LEARNING_RATE"])

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
            avg_loss = sum(loss_list)/len(loss_list)
            print("Epoch {} , loss: {}".format(param.get("epoch"), avg_loss))

        # Training 4 -----------------------------------------------------
        if typ == "epoch_acc":
            accuracy = param.get("accuracy")
            print("Epoch {} , acc: {:.4f} %".format(param.get("epoch"), accuracy))

        # Training 5 -----------------------------------------------------
        if typ == "epoch_recall":
            recall = param.get("recall")
            for k in recall:
                print("\tClass:{} -> Accuracy: {:.4f} %".format(k, recall[k]))


if __name__ == "__main__":
    main()
