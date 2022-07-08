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

from tqdm import tqdm
import argparse
from sklearn.metrics import classification_report

def parse_args():
	# Set arguments.
	arg_parser = argparse.ArgumentParser(description="Image Classification")
	
	arg_parser.add_argument("--dataset_name", type=str, default='CIFAR10')
	arg_parser.add_argument("--data_dir", type=str, default='D:/workspace/datasets/')
	#arg_parser.add_argument("--data_dir", type=str, default='../data/')
	arg_parser.add_argument("--model_name", type=str, default='ResNet18')
	arg_parser.add_argument("--model_ckpt_dir", type=str, default='../experiments/models/checkpoints/')
	arg_parser.add_argument("--model_ckpt_path_temp", type=str, default='../experiments/models/checkpoints/{}_{}_epoch={}.pth')
	arg_parser.add_argument('--n_epoch', default=1, type=int, help='The number of epoch')
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


device = 'cuda' if torch.cuda.is_available() else 'cpu'

numpy.set_printoptions(threshold=numpy.inf)
#適当な画像を入力、ノイズ
#input = torch.rand(1, 1, 28, 28)
train_loader, class_names = load_test.load_train()
x=[]
y=[]
for i, data in enumerate(train_loader, 0):
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
    out = out.unsqueeze(0)
    #xにappendする前にcatをしようするかも(拡張を増やす場合)
    #順番入れ替え
    out = torch.transpose(out, 0, 1)
    # print(out.shape)
    label = torch.eye(11)[label] 
    # print(label)
    # x.append(out.detach().numpy())
    # y.append(label.detach().numpy())
    x.append(out)
    y.append(label)
    print(i)
print("train")
m=[]
n=[]

val_loader, class_names = load_test.load_val()
for i, data in enumerate(val_loader, 0):
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
    out = out.unsqueeze(0)
    #xにappendする前にcatをしようするかも(拡張を増やす場合)
    #順番入れ替え
    out = torch.transpose(out, 0, 1)
    #print(out.shape)
    label = torch.eye(11)[label] 
    # print(label)
    # x.append(out.detach().numpy())
    # y.append(label.detach().numpy())
    m.append(out)
    n.append(label)
    print(i)
print("val")



#image_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.)
#imageはcls以外のデータ拡張数(おそらく) patchはいらない　num_classesはimagと同じ　dim,depth=1(layerの数),heads=マルチヘッド(), mlp_dim=入力×２ぐらい(原田さんに),poolはok,channels=512(),dimhead=attentionの式のd,
# model=ViT(1,1,512,1,8,1024, pool = 'cls', dim_head = 64, dropout = 0, emb_dropout = 0)

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #GPU有効化

# out_T=model(out)


# loss_history=nn.CrossEntropyLoss()
#tqdmモジュール
def train(data,label):
    args = parse_args()
    model=ViT(image_size=1, num_classes=11, dim=512, depth=1, heads=8, mlp_dim=1024, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.).to(device)
    model.train()
    total_loss = 0.0
    total_samples = len(x)
    optimizer=optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    for data, label in tqdm(zip(x,y),leave = False):
        optimizer.zero_grad()
        data = data.to(device)  #GT*** #to(device)でデータをGPUに載せる
        label = label.to(device)
        # ***data1 = torch.clone(data)   #input***
        batch_size = len(data)
        # ***for batch in range(batch_size):
        #     index1 = np.random.randint(0, image_size - (missing_pixels_height-1))
        #     index2 = np.random.randint(0, image_size - (missing_pixels_height-1))
        #     data1[batch][0][index1:index1+missing_pixels_height, index2:index2+missing_pixels_height] = missing_token   #hole***      
        output = model(data) #ViTに入力
        output = output.to(device)
        loss_function = nn.CrossEntropyLoss() #損失関数定義
        label = label.squeeze_()
        loss = loss_function(output, label) #loss計算
        total_loss += loss.item()
        loss.backward()
        del loss
        optimizer.step()

        avg_loss = total_loss / total_samples
        #    loss_history.append(avg_loss)
        print('Average train loss: ' + '{:.10f}'.format(avg_loss))
    return avg_loss


def val(data,label):
    args = parse_args()
    model=ViT(image_size=1, num_classes=11, dim=512, depth=1, heads=8, mlp_dim=1024, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.).to(device)
    model.eval()
    total_loss = 0.0
    total_samples = len(x)
    optimizer=optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    for data, label in tqdm(zip(x,y),leave = False):
        optimizer.zero_grad()
        data = data.to(device)  #GT*** #to(device)でデータをGPUに載せる
        label = label.to(device)
        # ***data1 = torch.clone(data)   #input***
        batch_size = len(data)
        # ***for batch in range(batch_size):
        #     index1 = np.random.randint(0, image_size - (missing_pixels_height-1))
        #     index2 = np.random.randint(0, image_size - (missing_pixels_height-1))
        #     data1[batch][0][index1:index1+missing_pixels_height, index2:index2+missing_pixels_height] = missing_token   #hole***      
        output = model(data) #ViTに入力
        output = output.to(device)
        loss_function = nn.CrossEntropyLoss() #損失関数定義
        label = label.squeeze_()
        loss = loss_function(output, label) #loss計算
        total_loss += loss.item()
        loss.backward()
        del loss
        optimizer.step()

        val_avg_loss = total_loss / total_samples
        #    loss_history.append(avg_loss)
        print('Average train loss: ' + '{:.10f}'.format(val_avg_loss))
    return val_avg_loss


class EarlyStopping:
    """earlystoppingクラス"""

    def __init__(self, patience=5, verbose=False, path='./model_save/best_model.pth'):
        """引数：最小値の非更新数カウンタ、表示設定、モデル格納path"""

        self.patience = patience  # 設定ストップカウンタ
        self.verbose = verbose  # 表示の有無
        self.counter = 0  # 現在のカウンタ値
        self.best_score = None  # ベストスコア
        self.early_stop = False  # ストップフラグ
        self.val_loss_min = np.Inf  # 前回のベストスコア記憶用
        self.path = path  # ベストモデル格納path

    def __call__(self, val_loss, model):
        """
        特殊(call)メソッド
        実際に学習ループ内で最小lossを更新したか否かを計算させる部分
        """
        score = -val_loss

        if self.best_score is None:  # 1Epoch目の処理
            self.best_score = score  # 1Epoch目はそのままベストスコアとして記録する
            self.checkpoint(val_loss, model)  # 記録後にモデルを保存してスコア表示する
        elif score < self.best_score:  # ベストスコアを更新できなかった場合
            self.counter += 1  # ストップカウンタを+1
            if self.verbose:  # 表示を有効にした場合は経過を表示
                # 現在のカウンタを表示する
                print(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:  # 設定カウントを上回ったらストップフラグをTrueに変更
                self.early_stop = True
        else:  # ベストスコアを更新した場合
            self.best_score = score  # ベストスコアを上書き
            self.checkpoint(val_loss, model)  # モデルを保存してスコア表示
            self.counter = 0  # ストップカウンタリセット

    def checkpoint(self, val_loss, model):
        '''ベストスコア更新時に実行されるチェックポイント関数'''
        if self.verbose:  # 表示を有効にした場合は、前回のベストスコアからどれだけ更新したか？を表示
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)  # ベストモデルを指定したpathに保存
        print("best")
        self.val_loss_min = val_loss  # その時のlossを記録する


args = parse_args()
# patienceはlossが最小値を「何回=patience」更新されなかったらepochを止める
earlystopping = EarlyStopping(patience=10, verbose=True)

# Train and val.
for epoch in range(args.n_epoch):
    # 学習　trainではmodelの中身のパラメータが知らず知らずのうちに更新されている
    train_loss = train(x,y)
	#modelはtrainで更新されたmodelを使用し、評価
	#valの今回の実質的な役割はearly stopping
	#valは本来、手元にあるデータからtrain,valに分けている
	#仮想の未知データ
	#しかし、earlyでvalが最もいいときに止めている
	#つまり、valにとって都合がいい
	#だから、全く関係ないtestでもう一度試す
	#train時のmodelとvalのmodelは違う、trainで更新されたmodelを使用
    val_loss = val(m,n)

    # スコア　表示
    stdout_temp = 'epoch: {:>3}, train acc: {:<8}, train loss: {:<8}, val acc: {:<8}, val loss: {:<8}'
    print(stdout_temp.format(epoch+1, val_loss))
	# callメソッド呼び出し
    earlystopping((val_loss), model)  
	# ストップフラグがTrueの場合、breakでforループを抜ける
	#要するにpatienceの数だけ連続でlossが更新できなければ学習を終了する
    if earlystopping.early_stop: 
        print("Early Stopping!")
        break

    # Save a model checkpoint.
    model_ckpt_path = args.model_ckpt_path_temp.format(args.dataset_name, args.model_name, epoch+1)
    torch.save(model.state_dict(),"./model_save_trans/trans_epoch{}".format(epoch+1))
    print('Saved a model checkpoint at {}'.format(model_ckpt_path))
    print('')

    
