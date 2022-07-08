import torch
import torchvision
from resnet import ResNet18


#適当な画像を入力、ノイズ
input = torch.rand(1, 1, 28, 28)
#今プログラムで指定されているrenetを指定
model = ResNet18() #torchvision.models.resnet18(pretrained=False)

#LOADの因数を作成したモデルに変更
model.load_state_dict(torch.load("model_re_med1"))
model.eval()
#outputはモデルに対して入力を行い、判別を行っている
out  = model(input)
# print(out.shape)

#linearは全結合層、Identityで全結合層を除去（中間層をそのまま出力）
model.linear = torch.nn.Identity()
#中間層のみの場合の出力結果＝重み
out  = model(input)
print(out.shape)
print(out)
