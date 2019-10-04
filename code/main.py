import numpy as np 
import model
import cv2 as cv2


x_train = cv2.imread("/home/data/foto1.png", 1)
x_shape = x_train.shape


lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for
bs=64
model = model.GRUNet()

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        #         set_trace()
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()

        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()
