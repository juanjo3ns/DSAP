import numpy as np 
import cv2 as cv2

import model as model
import dataset as dataset

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn

import utils as utils
import torch.backends.cudnn as cudnn



# Hyperparameters
num_epochs = 5
num_classes = 2
batch_size = 1
learning_rate = 0.001


train_loader = DataLoader(dataset=dataset.WAV_dataset(mode="train"), batch_size=batch_size, shuffle=True)

model = model.WAV_model()

print(torch.cuda.device_count())   # --> 0
print(torch.cuda.is_available())   # --> False
print(torch.version.cuda) 
model.cuda()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Train the model
total_step = len(train_loader)
loss_list = []
acc_list = []

for epoch in range(num_epochs):
    for i, (img, tag) in enumerate(train_loader):
        # Run the forward pass
        img = utils.load_image("/home/data/spect/" + img[0])

        img = img.reshape(1, 3, 1090, 1480)
        img = torch.from_numpy(img).float()
        output = model(img.cuda())
        
        #print("output: " + str(output))
        
        
        sol = np.array(tag, dtype=np.float64)
        sol = torch.from_numpy(sol).long()

        print("------------------------------------------------------------------")              
        print(output)
        print(sol)
        loss = criterion(output.cuda(), sol.cuda())
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, 
                                                                                num_epochs, i + 1, 
                                                                                total_step, 
                                                                                loss.item() ))
        print("------------------------------------------------------------------")                                  

        # Track the accuracy
        """
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)
        """
        """
        
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, 
                                                                                num_epochs, i + 1, 
                                                                                total_step, 
                                                                                loss.item(), 
                                                                                (correct / total) * 100))
        """

