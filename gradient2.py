import numpy as np
import pandas as pd
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
# extract data
file_path = r'python-oriented.xlsx'  # train set
raw_data = pd.read_excel(file_path, header=0)
data = raw_data.values
y = np.array(data[:, 5:12], dtype=float)
features = np.array(data[:, 13:16], dtype=float)
# weight and bias is initialized by the output of multiple linear regression
weight = np.array([[0.8593377, 6.783135, 10.975, 0, 0, -6.61993, 0],
                  [75.64179, 479.6341, 822.8969, 0, -706.2732, 0, 0],
                  [-0.3259305, -2.798011, -6.579005, 0, 4.696396, 4.247723, 1.848496]], dtype=float)
bias = np.array([0.5606372, 7.304416, 27.73993, 34.93901, 20.33828, 7.979418, 0], dtype=float)
y_hut = np.dot(features, weight) + bias

# gradient descent
x = torch.tensor(features, requires_grad=False, dtype=torch.float32)
w = torch.tensor(weight, requires_grad=True, dtype=torch.float32)
b = torch.tensor(bias, requires_grad=True, dtype=torch.float32)
y2 = torch.tensor(y, requires_grad=False, dtype=torch.float32)

# use simple linear network to prevent overfitting
class Net(torch.nn.Module):
    def __init__(self, weight, bias):
        super(Net, self).__init__()
        self.weight = Parameter(weight)
        self.bias = Parameter(bias)
        self.activate = nn.Sigmoid()
        self.Linear = nn.Linear(7, 7)

    def forward(self, z):
        out = torch.mm(x, self.weight) + self.bias
        out = self.activate(out)
        out = self.Linear(out)
        return out


net = Net(w, b)

# loss
loss_func = torch.nn.MSELoss()

# optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

# training
for i in range(50000):
    predict = net.forward(x)
    loss = loss_func(predict, y2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("ite:{}, loss:{}".format(i, loss))

loss = predict - y2
loss = loss * loss
loss = torch.sum(loss)
print(loss)
print(w)
print(b)

# test
file_path = r'python-oriented2.xlsx'  # train set
raw_data = pd.read_excel(file_path, header=0)
data = raw_data.values
y_test = torch.tensor(np.array(data[:, 5:12], dtype=float), dtype=torch.float32)
features_test = torch.tensor(np.array(data[:, 13:16], dtype=float), dtype=torch.float32)

y_hut_test = torch.mm(features_test, w) + b
loss = y_hut_test - y_test
loss = loss * loss
loss = torch.sum(loss)
print(loss)