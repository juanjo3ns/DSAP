import numpy as np 
import cv2 as cv2

import model as model
import dataset as dataset

import torch
from torch.utils.data import TensorDataset, DataLoader


# Hyperparameters
num_epochs = 1
num_classes = 2
batch_size = 100
learning_rate = 0.001


train_loader = DataLoader(dataset=dataset.WAV_dataset(mode="train"), batch_size=batch_size, shuffle=True)

model = model.WAV_dataset()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Train the model
total_step = len(train_loader)
loss_list = []
acc_list = []

for epoch in range(num_epochs):
    for i, (imag, tag) in enumerate(train_loader):
        # Run the forward pass
        output = model(img)
        loss = criterion(output, tag)
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

        
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, 
                                                                                num_epochs, i + 1, 
                                                                                total_step, 
                                                                                loss.item(), 
                                                                                (correct / total) * 100))

