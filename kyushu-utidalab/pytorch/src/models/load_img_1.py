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


device = 'cuda' if torch.cuda.is_available() else 'cpu'
numpy.set_printoptions(threshold=numpy.inf)
#適当な画像を入力、ノイズ
#input = torch.rand(1, 1, 28, 28)
val_loader, class_names = load_test.load_test()
x=[]
y=[]
for i, data in enumerate(val_loader, 0):
    inputs,label = data

#inputs, labels = inputs.to(device), labels.to(device)
print("input[0]")
print(inputs[0])
print("input[0].shape")
print(inputs[0].shape)
print("label")
print(label[0])
#input = load_val()
#input, _ = load_val()
#print(input[0][0: ].unsqueeze(0).shape)
#今プログラムで指定されているrenetを指定
model = ResNet18() #torchvision.models.resnet18(pretrained=False)

#LOADの因数を作成したモデルに変更
model.load_state_dict(torch.load("./model_save/best_model_24.pth"))
model.eval()
#outputはモデルに対して入力を行い、判別を行っている
#out  = model(input[0].unsqueeze(0))
# out = model(inputs)
# print(out.shape)

#linearは全結合層、Identityで全結合層を除去（中間層をそのまま出力） 
model.linear = torch.nn.Identity()
#中間層のみの場合の出力結果＝重み
out  = model(inputs[0].unsqueeze(0))
#out  = model(inputs)
print("out.shape")

print(out)
print(out.shape)

# x = out.to('cpu').detach().numpy().copy()
# print("x")
# print(x)
# print(x.shape)
# y=[[1,2,3]]
# print(y.shape)

#x.append(out.detach().numpy())
#y.append(label.detach().numpy())

# print(len(x))
# # 可視化
# x = np.concatenate(x, 0)
# y = np.concatenate(y, 0)
# print(x.shape)
# print(x[1].shape)

# tsne = TSNE(n_components=2, random_state=1)
# tsne_reduced = tsne.fit_transform(x)
# #
# print(tsne_reduced.shape)

# plt.subplot(122)
# plt.scatter(tsne_reduced[:,0],tsne_reduced[:,1],  c = y, 
#             cmap = "coolwarm", edgecolor = "None", alpha=0.35)
# plt.colorbar()
# plt.title('TSNE Scatter Plot')
# plt.savefig("tsne.png")


# print(out.shape)
# #print(out)
# print(out)

