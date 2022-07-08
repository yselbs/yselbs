from sklearn.cluster import k_means
import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import argparse
import os

import numpy as np
import pandas as pd
# from tsne import bh_sne
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import numpy
import torch
import torchvision
from resnet import ResNet18
import numpy
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch.nn.functional as F
from sklearn.datasets import load_digits
import load_test
from sklearn.metrics import confusion_matrix

test_total_acc = 0
model = ResNet18() #torchvision.models.resnet18(pretrained=False)

#LOADの因数を作成したモデルに変更
# model.load_state_dict(torch.load("./model_sample/best_model.pth"))
model.load_state_dict(torch.load("./model_sample/best_model.pth"))
model.eval()
#load_testのバッチを1にする必要あり
val_loader, class_names = load_test.load_test()
pred_list = []
true_list = []
# a=b=c=d=e=f=g=h=i=j=k=0
data_train=13940
data_val=2452
data_test=8829

def cal_acc(t,p):
    p_arg = torch.argmax(p,dim=1)
    return torch.sum(t == p_arg) 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with torch.no_grad():
    for n,input in enumerate(val_loader):
        data,label = input
        output = model(data)
        test_total_acc += cal_acc(label,output)
        pred = torch.argmax(output , dim =1)
        pred_list += pred.detach().cpu().numpy().tolist()
        true_list += label.detach().cpu().numpy().tolist() 
        # print(n)
        # if label == 0:
        #     a+=1
        # if label == 1:
        #     b+=1
        # if label == 2:
        #     c+=1                                
        # if label == 3:
        #     d+=1
        # if label == 4:
        #     e+=1
        # if label == 5:
        #     f+=1
        # if label == 6:
        #     g+=1        
        # if label == 7:
        #     h+=1
        # if label == 8:
        #     i+=1
        # if label == 9:
        #     j+=1
        # if label == 10:
        #     k+=1




    # print(a)
    # print(b)
    # print(c)
    # print(d)
    # print(e)
    # print(f)
    # print(g)
    # print(h)
    # print(i)
    # print(j)
    # print(k)


print(confusion_matrix(true_list,pred_list))
print(test_total_acc)
print(len(val_loader))
print(f"test acc:{100*test_total_acc/len(val_loader)}")


