# -*- coding: utf-8 -*-

from __future__ import print_function
from six.moves import xrange

import os

"""
|---------------------------------------------------------------------------------------------------|
| vgg16 | pretrained | freeze |                         annotation                       | accuracy |
|---------------------------------------------------------------------------------------------------|
| False | False      | False  | use some ConVs to extract images features,               |   0.831  |
|       |            |        | the ConVs is trained from scratch.                       |          |
|---------------------------------------------------------------------------------------------------|
| True  | False      | False  | use the ConVs of VGG16 model to extract images features, |   0.101  |
|       |            |        | the ConVs is trained from scratch.                       |          |
|---------------------------------------------------------------------------------------------------|
| True  | True       | True   | use the ConVs of VGG16 model to extract images features, |   0.658  |
|       |            |        | the ConVs is trained on the base of VGG16 model.         |          |
|---------------------------------------------------------------------------------------------------|
| True  | True       |  False | use the ConVs of VGG16 model to extract images features, |   0.890  |
|       |            |        | the ConVs is trained on the base of VGG16 model,         |          |
|       |            |        | and the ConVs are freezed, namely untrainable or         |          |
|       |            |        | parameters cannot be changed.                            |          |
|---------------------------------------------------------------------------------------------------|
"""

model = (True, True, True)

# 根据环境选择python版本
cmd = "python3 cifar10_train.py --vgg16=%s --pretrained=%s --freeze=%s" % model
print(cmd)
# os.system(cmd)
cmd = "python3 cifar10_eval.py --vgg16=%s --pretrained=%s --freeze=%s" % model
print(cmd)
# os.system(cmd)
