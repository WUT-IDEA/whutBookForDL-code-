# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
import time

# hyper-parameter
training_epoch = 100
num_classes = 10
learning_rate = 1e-3
batch_size = 10000

# 查看Pytorch是否支持GPU
GPU_FLAG = torch.cuda.is_available()
print('CUDA available?', GPU_FLAG)

# Mnist digits dataset
train_data = torchvision.datasets.MNIST(
    root='data/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)

test_data = torchvision.datasets.MNIST(
    root='data/',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)

train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

'''
data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6,
                                  pin_memory=True, drop_last=True, collate_fn=patch_data) #num_workers就是使用多进程
'''


# 创建网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out


# GPU info
GPU_list = ['/GPU:%s' % (i) for i in xrange(torch.cuda.device_count())]
print(GPU_list)
device_ids = list(xrange(torch.cuda.device_count()))

net = nn.DataParallel(Net(), device_ids=device_ids)
# net = Net()

# 将模型的参数送到GPU中
if GPU_FLAG == True:
    net = net.cuda()
    # 定义loss函数
    criterion = nn.CrossEntropyLoss().cuda()
print(net)  # 输出模型结构

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


def data_parallel(module, input, device_ids, output_device=None):
    if not device_ids:
        return module(input)

    if output_device is None:
        output_device = device_ids[0]

    replicas = nn.parallel.replicate(module, device_ids)
    inputs = nn.parallel.scatter(input, device_ids)
    replicas = replicas[:len(inputs)]
    print(len(replicas), len(inputs))
    outputs = nn.parallel.parallel_apply(replicas, inputs)
    return nn.parallel.gather(outputs, output_device)


# training
for i in xrange(1, 1 + training_epoch):
    cost, accuracy = 0.0, 0.0
    start = time.time()
    for j, (images, labels) in enumerate(train_loader):
        x = Variable(images).view(-1, 28 ** 2).cuda()
        y = Variable(labels).cuda()

        optimizer.zero_grad()
        y_ = net(x)
        # y_ = data_parallel(net, input=x, device_ids=device_ids)
        loss = criterion(y_, y)
        loss.backward()
        optimizer.step()

        cost += loss
        accuracy += (torch.max(y_, dim=1)[1] == y).float().mean()
    cost /= len(train_loader)
    accuracy /= len(train_loader)
    print('Epoch %s / %s, time: %-.5f, training loss: %-.5f, accuracy: %-.5f' %
          (i, training_epoch, time.time() - start, float(cost), float(accuracy)))
