import numpy as np
from PIL import Image
'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import sys
import os
import argparse
from sklearn.metrics import classification_report

from datasets import Medmnist
#modelsの中にあるプログラムを全部importしている　resnet.pyなど
from models import *

import numpy as np

# データの取得
val_dataset, class_names = Medmnist.load_test()
j=0
#train , val , test = 13940, 2452, 8829
for i, data in enumerate(val_dataset, 0):
    inputs,label = data
    inputs += 1.0
    inputs *= 127.5
    # inputs /= 2.0
    # inputs *= 255.0
    # print(torch.max(inputs), torch.min(inputs))
    # print(torch.min(inputs))
    if label ==11:
        img = Image.fromarray(np.uint8(inputs.squeeze()))
        j+=1
        # print(inputs)
        img.save("./img_save_t/10/11_{}.png".format(j))
