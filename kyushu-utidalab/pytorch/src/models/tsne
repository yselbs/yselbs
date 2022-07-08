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
import load_test
import numpy
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from sklearn.datasets import load_digits

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
numpy.set_printoptions(threshold=numpy.inf)
#適当な画像を入力、ノイズ
#input = torch.rand(1, 1, 28, 28)
val_loader, class_names = load_test.load_test()
x=[]
y=[]

for i, data in enumerate(val_loader, 0):
    inputs,label = data

    # inputs, labels = inputs.to(device), labels.to(device)
    #print(inputs[0])
    #print(inputs[0].shape)
    #print(label[0])
    # input = load_val()
    # input, _ = load_val()
    # print(input[0][0: ].unsqueeze(0).shape)
    #今プログラムで指定されているrenetを指定
    model = ResNet18() #torchvision.models.resnet18(pretrained=False)

    #LOADの因数を作成したモデルに変更
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
    
    print(out.shape)
    x.append(out.detach().numpy())
    y.append(label.detach().numpy())

def tsne_plot( targets, outputs):
    print('generating t-SNE plot...')
    # tsne_output = bh_sne(outputs)
    outputs = np.concatenate(outputs, 0)
    targets = np.concatenate(targets, 0)
    tsne = TSNE(random_state=0)
    tsne_output = tsne.fit_transform(outputs)

    df = pd.DataFrame(tsne_output, columns=['x', 'y'])
    df['targets'] = targets

    plt.rcParams['figure.figsize'] = 10, 10
    sns.scatterplot(
        x='x', y='y',
        hue='targets',
        palette=sns.color_palette("hls", 11),
        data=df,
        marker='o',
        legend="full",
        alpha=0.5
    )

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')

    plt.savefig(os.path.join('tsne.png'), bbox_inches='tight')
    print('done!')

tsne_plot(y, x)
