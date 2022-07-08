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





def parse_args():
	# Set arguments.
	arg_parser = argparse.ArgumentParser(description="Image Classification")
	
	arg_parser.add_argument("--dataset_name", type=str, default='Medmnist')
	arg_parser.add_argument("--data_dir", type=str, default='D:/workspace/datasets/')
	#arg_parser.add_argument("--data_dir", type=str, default='../data/')
	arg_parser.add_argument("--model_name", type=str, default='Tramsformer')
	arg_parser.add_argument("--model_ckpt_dir", type=str, default='../experiments/models/checkpoints/')
	arg_parser.add_argument("--model_ckpt_path_temp", type=str, default='./models/model_save_trans/{}_{}_epoch={}.pth')
	arg_parser.add_argument('--n_epoch', default=1000, type=int, help='The number of epoch')
	arg_parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')

	args = arg_parser.parse_args()

	# Make directory.
	os.makedirs("data", exist_ok=True)
	os.makedirs("model_ckpt_dir", exist_ok=True)

	# Validate paths.
	assert os.path.exists("data")
	assert os.path.exists("model_ckpt_dir")

	return args

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    #imageはcls以外のデータ拡張数(おそらく).IMGいらない説 patchはいらない　num_classesはimagと同じ　dim,depth=1(layerの数),heads=マルチヘッド(), mlp_dim=入力×２ぐらい(原田さんに),poolはok,channels=512(),dimhead=attentionの式のd,
    def __init__(self, *, image_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        # patch_height, patch_width = pair(patch_size)

        # assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # num_patches = (image_height // patch_height) * (image_width // patch_width)
        # patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
        #     nn.Linear(patch_dim, dim),
        # )

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # x = self.to_patch_embedding(img)
        x=img
        b, _ ,_= x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)



# device = 'cuda' if torch.cuda.is_available() else 'cpu'
numpy.set_printoptions(threshold=numpy.inf)
#適当な画像を入力、ノイズ
#input = torch.rand(1, 1, 28, 28)
m=open("./conv_save/val_inputs_nasi.txtfile","rb")
m=pickle.load(m)
n=open("./conv_save/val_labels_tsne.txtfile","rb")
n=pickle.load(n)


x=[]
y=[]

for inputs, label in tqdm(zip(m,n),leave = False):

    # inputs, labels = inputs.to(device), labels.to(device)
    #print(inputs[0])
    #print(inputs[0].shape)
    #print(label[0])
    # input = load_val()
    # input, _ = load_val()
    # print(input[0][0: ].unsqueeze(0).shape)
    #今プログラムで指定されているrenetを指定
    model=ViT(image_size=1, num_classes=11, dim=512, depth=1, heads=8, mlp_dim=1024, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.)

    #LOADの因数を作成したモデルに変更
    model.load_state_dict(torch.load("./model_save_trans/best_model_trans.pth"))
    model.eval()
    #outputはモデルに対して入力を行い、判別を行っている
    #out  = model(input[0].unsqueeze(0))
    #out = model(inputs)
    #  print(out.shape)

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

    plt.savefig(os.path.join('tsne_trans.png'), bbox_inches='tight')
    print('done!')

tsne_plot(y, x)
