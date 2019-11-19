import time as time
import numpy as np
import cv2 as cv2
from collections import defaultdict
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import utils as utils
import model as model
import dataset as dataset
from torch import nn

from IPython import embed

PATH_SPECTROGRAM = "/home/data/spect/"
PATH_INIT_WEIGHTS = "/home/data/models/test2/epoch_396_.pt"


class main():
    def __init__(self, mode="train", prints=True):
        self.model = None
        self.LastTime = time.time()
        self.prints = prints
        self.config = {}
        self.writer = SummaryWriter(log_dir="/home/code/tensorboard/test_train")
        if mode=="train":
            self.train()
        elif mode=="val":
            self.val()


    def val(self):
        self.set_config(NUM_EPOCHS=1,
                        INIT_EPOCH=0,
                        NUM_CLASSES=11,
                        BATCH_SIZE=128,
                        LEARNING_RATE=0.00001,
                        GPU=True,
                        WEIGHTS=True)

        val_loader = self.get_loader(mode="val")

        self.model = self.get_model()
        #self.model.load_state_dict(torch.load(PATH_INIT_WEIGHTS))

        lossFunction, optimizer = self.get_LossOptimizer()

        total_step = len(val_loader)

        for epoch in range(self.config["INIT_EPOCH"], self.config["NUM_EPOCHS"]):
            loss_list = []
            total_outputs = []
            total_solutions = []
            for i, (img, tag) in enumerate(val_loader):
                img = img.unsqueeze(1)

                output, loss = self.run(img=img, criterion=lossFunction, solution=tag)

                #loss = self.compute_loss(criterion=lossFunction, output=output, solution=tag)
                loss_list.append(loss.item())

                outs_argmax = output.argmax(dim=1)
                total_outputs.extend(outs_argmax.cpu().numpy())
                total_solutions.extend(tag.numpy())

                # Print de result for this step
                self.print_info(typ="trainn", epoch=epoch, i=i, total_step=total_step, loss=loss.item(), num_epoch=self.config["NUM_EPOCHS"])


            self.print_info(typ="epoch_loss", epoch=epoch, loss_list=loss_list)
            self.accuracy(total_outputs, total_solutions, epoch)
            self.recall(total_outputs, total_solutions, epoch)


    def train(self):
        self.set_config(NUM_EPOCHS=1000,
                        INIT_EPOCH=0,
                        NUM_CLASSES=10,
                        BATCH_SIZE=64,
                        LEARNING_RATE=0.00001,
                        GPU=True,
                        WEIGHTS=False)

        train_loader = self.get_loader(mode="train")

        self.model = self.get_model()

        lossFunction, optimizer = self.get_LossOptimizer()

        total_step = len(train_loader)


        for epoch in range(self.config["INIT_EPOCH"], self.config["NUM_EPOCHS"]):
            loss_list = []
            total_outputs = []
            total_solutions = []
            for i, (img, tag) in enumerate(train_loader):

                img = img.unsqueeze(1)

                output, loss = self.run(img=img, criterion=lossFunction, solution=tag)

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
            self.writer.add_scalar('Loss/train', sum(loss_list)/len(loss_list), epoch)
            # if epoch%10==0:
            #     torch.save(self.model.state_dict(), "/home/data/models/test3/epoch_{}_.pt".format(epoch))



    def run(self, img, criterion, solution):

        if self.config["GPU"]:
            output = self.model(img.cuda())
        else:
            output = self.model(img)

        loss = self.compute_loss(criterion=criterion, output=output, solution=solution)

        return output, loss

    def recall(self, output, solutions, epoch, show=True):
        recall = defaultdict(int)
        #recall = {}
        for i in range(self.config["NUM_CLASSES"]):
            positions = np.where(np.array(solutions)==i)
            for p in positions[0]:
                if output[p] == i:
                    recall[str(i)] += 1
            recall[str(i)] /= len(positions[0])
            recall[str(i)] *= 100

        if show:
            self.print_info(typ="epoch_recall", epoch=epoch, recall=recall)

    def accuracy(self, output, solutions, epoch, show=True):
        a = np.where(np.array(output)==solutions)
        if show:
            self.print_info(typ="epoch_acc", epoch=epoch, accuracy=100*len(a[0])/len(output))

    def load_image(self, img):
        img = utils.load_image(PATH_SPECTROGRAM + img[0].split('.')[0])
        img = torch.from_numpy(img).float()
        img = img.permute(2,0,1)

        return img

    def compute_loss(self, criterion, output, solution, GPU=True):
        #solution = torch.from_numpy(solution.cpu()).long()
        #print(solution)
        if self.config['GPU']:
            loss = criterion(output.cpu(), solution.type(torch.DoubleTensor))
        else:
            loss = criterion(output, solution)

        return loss

    def get_loader(self, mode="train", shuffle=True):

        #loader = DataLoader(dataset=dataset.WAV_dataset(mode=mode, images=True), batch_size=self.config["BATCH_SIZE"], shuffle=shuffle)
        loader = DataLoader(dataset=dataset.WAV_dataset_task5(mode=mode, images=True), batch_size=self.config["BATCH_SIZE"], shuffle=shuffle)
        return loader

    def set_config(self, **param):
        for par in param:
            self.config[par] = param.get(par)
        self.print_info(typ="Init")

    def get_model(self, GPU=True):

        if self.config['WEIGHTS']:
            self.print_info(typ="LoadModel", Weights = "From file: " + str(PATH_INIT_WEIGHTS.split("/")[-1]))
        else:
            self.print_info(typ="LoadModel", Weights = "From Scratch")

        
        mod = model.BaselineModel()
        #mod = model.WAV_model_test()

        if self.config['GPU']:
            mod.cuda()

        if self.config['WEIGHTS']:
            mod.load_state_dict(torch.load(PATH_INIT_WEIGHTS))

        self.print_info(typ="LoadModel", Status="Done")
        return mod

    def get_LossOptimizer(self):
        #criterion = nn.CrossEntropyLoss()
        criterion = nn.BCEWithLogitsLoss()
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
    a = main(mode="train")
