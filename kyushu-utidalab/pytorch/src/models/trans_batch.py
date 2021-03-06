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
    #image???cls???????????????????????????(????????????).IMG??????????????? patch??????????????????num_classes???imag????????????dim,depth=1(layer??????),heads=??????????????????(), mlp_dim=????????????????????(???????????????),pool???ok,channels=512(),dimhead=attention?????????d,
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


#image_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.)
#image???cls???????????????????????????(????????????) patch??????????????????num_classes???imag????????????dim,depth=1(layer??????),heads=??????????????????(), mlp_dim=????????????????????(???????????????),pool???ok,channels=512(),dimhead=attention?????????d,
# model=ViT(1,1,512,1,8,1024, pool = 'cls', dim_head = 64, dropout = 0, emb_dropout = 0)

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #GPU?????????

# out_T=model(out)


# loss_history=nn.CrossEntropyLoss()
#tqdm???????????????
def train(x,y):
    args = parse_args()
    model=ViT(image_size=1, num_classes=11, dim=512, depth=1, heads=8, mlp_dim=1024, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.).to(device)
    model.train()
    total_loss = 0.0
    total_samples = len(x)

    optimizer=optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    for data, label in tqdm(zip(x,y),leave = False):
        correct = 0 
        optimizer.zero_grad()
        data = data.to(device)  #GT*** #to(device)???????????????GPU????????????
        label = label.to(device)
        # ***data1 = torch.clone(data)   #input***
        batch_size = len(data)
        # ***for batch in range(batch_size):
        #     index1 = np.random.randint(0, image_size - (missing_pixels_height-1))
        #     index2 = np.random.randint(0, image_size - (missing_pixels_height-1))
        #     data1[batch][0][index1:index1+missing_pixels_height, index2:index2+missing_pixels_height] = missing_token   #hole***      
        output = model(data) #ViT?????????
        output = output.to(device)
        loss_function = nn.CrossEntropyLoss() #??????????????????
        label = label.squeeze_()
        loss = loss_function(output, label) #loss??????
        total_loss += loss.item()
        loss.backward()
        del loss
        optimizer.step()

        ############################################################
        # # n_data???????????????????????????
        # n_data += data.size(0)
        # output?????????????????????????????????????????????(_, )???_????????????????????????????????????axis=1???????????????
        #.data??????????????????????????????
        _, predicted = torch.max(output.data, 1)
        _, labels = torch.max(label.data, 1)
        # predicted=label??????????????????
        correct += (predicted == labels).sum().item()
        # 0???????????????10?????????????????????
        # if i % 10 == 9 and i != 0:    # print every 2000 mini-batches
        print(
            f'[{epoch + 1}] train_loss: {total_loss / batch_size:.3f}, train_acc:{100 * correct / batch_size:3f}')
    avg_loss = total_loss / total_samples
        # #    loss_history.append(avg_loss)
        # print('Average train loss: ' + '{:.10f}'.format(avg_loss))
    return avg_loss


def val(x,y):
    args = parse_args()
    model=ViT(image_size=1, num_classes=11, dim=512, depth=1, heads=8, mlp_dim=1024, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.).to(device)
    model.eval()
    total_loss = 0.0
    total_samples = len(x)
    optimizer=optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    for data, label in tqdm(zip(x,y),leave = False):
        correct = 0 
        optimizer.zero_grad()
        data = data.to(device)  #GT*** #to(device)???????????????GPU????????????
        label = label.to(device)
        # ***data1 = torch.clone(data)   #input***
        batch_size = len(data)
        # ***for batch in range(batch_size):
        #     index1 = np.random.randint(0, image_size - (missing_pixels_height-1))
        #     index2 = np.random.randint(0, image_size - (missing_pixels_height-1))
        #     data1[batch][0][index1:index1+missing_pixels_height, index2:index2+missing_pixels_height] = missing_token   #hole***      
        output = model(data) #ViT?????????
        output = output.to(device)
        loss_function = nn.CrossEntropyLoss() #??????????????????
        label = label.squeeze_()
        loss = loss_function(output, label) #loss??????
        total_loss += loss.item()
        loss.backward()
        del loss
        optimizer.step()
    _, predicted = torch.max(output.data, 1)
    _, labels = torch.max(label.data, 1)
    # predicted=label??????????????????
    correct += (predicted == labels).sum().item()
    # 0???????????????10?????????????????????
    # if i % 10 == 9 and i != 0:    # print every 2000 mini-batches
    print("")
    print(
        f'[{epoch + 1}] val_loss: {total_loss / batch_size:.3f}, val_acc:{100 * correct / batch_size:3f}')
    # avg_loss = total_loss / total_samples
    # # #    loss_history.append(avg_loss)
    #  # print('Average train loss: ' + '{:.10f}'.format(avg_loss))
    val_avg_loss = total_loss / total_samples
    #    loss_history.append(avg_loss)
    # print('Average train loss: ' + '{:.10f}'.format(val_avg_loss))
    return val_avg_loss


class EarlyStopping:
    """earlystopping?????????"""

    def __init__(self, patience=5, verbose=False, path='./model_save_trans/best_model_trans.pth'):
        """??????????????????????????????????????????????????????????????????????????????path"""

        self.patience = patience  # ??????????????????????????????
        self.verbose = verbose  # ???????????????
        self.counter = 0  # ????????????????????????
        self.best_score = None  # ??????????????????
        self.early_stop = False  # ?????????????????????
        self.val_loss_min = np.Inf  # ????????????????????????????????????
        self.path = path  # ????????????????????????path

    def __call__(self, val_loss, model):
        """
        ??????(call)????????????
        ????????????????????????????????????loss????????????????????????????????????????????????
        """
        score = -val_loss

        if self.best_score is None:  # 1Epoch????????????
            self.best_score = score  # 1Epoch?????????????????????????????????????????????????????????
            self.checkpoint(val_loss, model)  # ?????????????????????????????????????????????????????????
        elif score < self.best_score:  # ???????????????????????????????????????????????????
            self.counter += 1  # ???????????????????????????+1
            if self.verbose:  # ????????????????????????????????????????????????
                # ????????????????????????????????????
                print(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:  # ????????????????????????????????????????????????????????????True?????????
                self.early_stop = True
        else:  # ???????????????????????????????????????
            self.best_score = score  # ??????????????????????????????
            self.checkpoint(val_loss, model)  # ???????????????????????????????????????
            self.counter = 0  # ????????????????????????????????????

    def checkpoint(self, val_loss, model):
        '''???????????????????????????????????????????????????????????????????????????'''
        if self.verbose:  # ????????????????????????????????????????????????????????????????????????????????????????????????????????????
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)  # ?????????????????????????????????path?????????
        print("best")
        self.val_loss_min = val_loss  # ????????????loss???????????????



device = 'cuda' if torch.cuda.is_available() else 'cpu'

numpy.set_printoptions(threshold=numpy.inf)

x=open("./conv_save/train_inputs.txtfile","rb")
x=pickle.load(x)
y=open("./conv_save/train_labels.txtfile","rb")
y=pickle.load(y)
m=open("./conv_save/val_inputs.txtfile","rb")
m=pickle.load(m)
n=open("./conv_save/val_labels.txtfile","rb")
n=pickle.load(n)
print("ok")

args = parse_args()
# patience???loss????????????????????????=patience??????????????????????????????epoch????????????
earlystopping = EarlyStopping(patience=10, verbose=True)
model=ViT(image_size=1, num_classes=11, dim=512, depth=1, heads=8, mlp_dim=1024, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.).to(device)
# Train and val.
for epoch in range(args.n_epoch):
    # ?????????train??????model?????????????????????????????????????????????????????????????????????????????????
    train_loss = train(x,y)
	#model???train??????????????????model?????????????????????
	#val?????????????????????????????????early stopping
	#val??????????????????????????????????????????train,val??????????????????
	#????????????????????????
	#????????????early???val???????????????????????????????????????
	#????????????val???????????????????????????
	#??????????????????????????????test?????????????????????
	#train??????model???val???model????????????train??????????????????model?????????
    val_loss = val(m,n)

    # ??????????????????
    # stdout_temp = 'epoch: {:>3}, train acc: {:<8}, train loss: {:<8}, val acc: {:<8}, val loss: {:<8}'
    # stdout_temp = 'epoch: {:>3}, train loss: {:<8},  val loss: {:<8}'
    # print(stdout_temp.format(epoch+1, val_loss))
    print(epoch)
	# call????????????????????????
    earlystopping((val_loss), model)  
	# ????????????????????????True????????????break???for?????????????????????
	#????????????patience?????????????????????loss????????????????????????????????????????????????
    if earlystopping.early_stop: 
        print("Early Stopping!")
        break

    # Save a model checkpoint.
    model_ckpt_path = args.model_ckpt_path_temp.format(args.dataset_name, args.model_name, epoch+1)
    torch.save(model.state_dict(),"./model_save_trans/trans_epoch{}".format(epoch+1))
    print('Saved a model checkpoint at {}'.format(epoch+1))
    print('')

    
