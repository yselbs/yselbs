from pickletools import optimize
import torch
import torchvision
from resnet import ResNet18
import load_test
import numpy
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch.optim as optim
from sklearn.datasets import load_digits
import torch
from torch import nn
import os
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import pickle
from tqdm import tqdm
import argparse
from sklearn.metrics import classification_report


def input_create(loader,x_name,y_name,mode):
    x=[]
    y=[]
    for i, data in enumerate(loader, 0):
        inputs,label = data
        # inputs, label = inputs.to(device), label.to(device)
        # inputs, labels = inputs.to(device), labels.to(device)
        #print(inputs[0])
        #print(inputs[0].shape)
        #print(label[0])
        # input = load_val()
        # input, _ = load_val()
        # print(input[0][0: ].unsqueeze(0).shape)
        #今プログラムで指定されているrenetを指定
        model = ResNet18() #torchvision.models.resnet18(pretrained=False)

        #LOADの因数を作成したモデルに変更"./model_save/best_model_24.pth"
        model.load_state_dict(torch.load("./model_save/best_model_24.pth"))
        model.eval()
        #outputはモデルに対して入力を行い、判別を行っている
        #out  = model(input[0].unsqueeze(0))
        #out = model(inputs)
        #  print(out.shape)

        #linearは全結合層、Identityで全結合層を除去（中間層をそのまま出力） 
        model.linear = torch.nn.Identity()
        #中間層のみの場合の出力結果＝重み
        # out  = model(inputs[0].unsqueeze(0))
        out  = model(inputs)
        out = out.unsqueeze(0)
        #xにappendする前にcatをしようするかも(拡張を増やす場合)
        #順番入れ替え
        out = torch.transpose(out, 0, 1)
        # print(out.shape)
        if mode=="train":
            label = torch.eye(11)[label] 
        # print(label)
        # x.append(out.detach().numpy())
        # y.append(label.detach().numpy())
        x.append(out)
        y.append(label)
        print(i)
    f = open(x_name,'wb')
    pickle.dump(x,f)
    f.close()
    g = open(y_name,'wb')
    pickle.dump(y,g)
    g.close()



device = 'cuda' if torch.cuda.is_available() else 'cpu'

numpy.set_printoptions(threshold=numpy.inf)
#適当な画像を入力、ノイズ
#input = torch.rand(1, 1, 28, 28)
train_loader, class_names = load_test.load_train()
val_loader,class_names=load_test.load_val()
test_loader,class_names=load_test.load_test()


# input_create(train_loader,"./conv_save/train_inputs.txtfile", "./conv_save/train_labels.txtfile")
# input_create(val_loader, "./conv_save/val_inputs.txtfile", "./conv_save/val_labels.txtfile")
# input_create(test_loader, "./conv_save/test_inputs.txtfile","./conv_save/test_labels.txtfile","test")
input_create(train_loader,"./conv_save/train_inputs_nasi.txtfile", "./conv_save/train_labels_tsne.txtfile","test")
input_create(val_loader, "./conv_save/val_inputs_nasi.txtfile", "./conv_save/val_labels_tsne.txtfile","test")
