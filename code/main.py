import time as time
import numpy as np 
import cv2 as cv2

import torch
from torch.utils.data import TensorDataset, DataLoader

import utils as utils
import model as model
import dataset as dataset
from torch import nn

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
                        NUM_CLASSES=2, 
                        BATCH_SIZE=1, 
                        LEARNING_RATE=0.001, 
                        GPU=True)

        train_loader = self.get_loader(mode="train")

        self.model = self.get_model()

        lossFunction, optimizer = self.get_LossOptimizer()

        total_step = len(train_loader)
        loss_list = []
        acc_list = []



        for epoch in range(self.config["NUM_EPOCHS"]):
            for i, (img, tag) in enumerate(train_loader):

                img = self.load_image(img=img)
                output = self.model(img.cuda())             
                
                sol = np.array(tag, dtype=np.float64)

                loss = self.compute_loss(criterion=lossFunction, output=output, solution=sol)
                loss_list.append(loss.item())


                # Backprop and perform Optimizer
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Print de result for this step
                self.print_info(typ="trainn", epoch=epoch, i=i, total_step=total_step, loss=loss.item(), num_epoch=self.config["NUM_EPOCHS"])

    def load_image(self, img):

        img = utils.load_image(PATH_SPECTROGRAM + img[0])
        img = img.reshape(1, 3, 1090, 1480)
        #np.reshape(img, (1,3,1090, 1480))
        img = torch.from_numpy(img).float()

        return img

    def compute_loss(self, criterion, output, solution, GPU=True):

        solution = torch.from_numpy(solution).long()

        if GPU:
            loss = criterion(output.cuda(), solution.cuda())
        else:
            loss = criterion(output, solution)

        return loss

    def get_loader(self, mode="train", shuffle=True):

        loader = DataLoader(dataset=dataset.WAV_dataset(mode=mode), batch_size=self.config["BATCH_SIZE"], shuffle=shuffle)
        return loader

    def set_config(self, **param):
        for par in param:
            self.config[par] = param.get(par)
        self.print_info(typ="Init")

    def get_model(self, GPU=True):
        self.print_info(typ="LoadModel", Weights= "From Scratch")

        mod = model.WAV_model()
        if GPU:
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
            maxim = 20 - index
            print("Epoch [{}/{}]".format(param.get("epoch") + 1, param.get("num_epoch")) + 
                    "[" + "#"*index + " "*maxim + "] " + "[{}/{}]".format(param.get("i") + 1, param.get("total_step")) +
                    ", Loss: {:.4f}".format(param.get("loss"))
                    
                    , end="\r" )


            if (param.get("i")+1) == param.get("total_step"):
                print("")

if __name__ == "__main__":
    main()
