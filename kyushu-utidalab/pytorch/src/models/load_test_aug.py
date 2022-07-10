import torch
import torchvision
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO, Evaluator
import random
import numpy as np
import torch

def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


torch_fix_seed()


#test_dで使用されている
def load_val():
    data_flag = 'organsmnist'
    # data_flag = 'breastmnist'
    download = True

    NUM_EPOCHS = 3
    BATCH_SIZE = 64
    lr = 0.001

    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5]),
        transforms.RandomRotation(degrees=[15,15])
    ])

    #load the data
    val_dataset = DataClass(split='val', transform=data_transform, download=download)

    # print(len(test_dataset))
    # encapsulate data into dataloader form
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    #shuffleは入力する際にデータをシャッフルするかしないか、基本的にtrainデータにしか適応させない
    val_at_eval = data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    class_names = ('bladder','femur-left','femur-right','heart','kindeny-left','kindeny-right','liver','lung-left','lung-right','pancreas','spleen')
    
    # print(train_dataset)
    # print("===================")
    # print(test_dataset)

    return val_at_eval,class_names

def load_train():
    data_flag = 'organsmnist'
    # data_flag = 'breastmnist'
    download = True

    NUM_EPOCHS = 3
    BATCH_SIZE = 64
    lr = 0.001

    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5]),
        transforms.RandomRotation(degrees=[15,15])
    ])

    #load the data
    train_dataset = DataClass(split='train', transform=data_transform, download=download)

    # print(len(test_dataset))
    # encapsulate data into dataloader form
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    #shuffleは入力する際にデータをシャッフルするかしないか、基本的にtrainデータにしか適応させない
    train_at_eval = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    class_names = ('bladder','femur-left','femur-right','heart','kindeny-left','kindeny-right','liver','lung-left','lung-right','pancreas','spleen')
    
    # print(train_dataset)
    # print("===================")
    # print(test_dataset)

    return train_loader,class_names



def load_test():
    data_flag = 'organsmnist'
    # data_flag = 'breastmnist'
    download = True

    NUM_EPOCHS = 3
    BATCH_SIZE = 1
    lr = 0.001

    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5]),
        transforms.RandomRotation(degrees=[15,15])
    ])

    #load the data
    test_dataset = DataClass(split='test', transform=data_transform, download=download)

    # print(len(test_dataset))
    # encapsulate data into dataloader form
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    #shuffleは入力する際にデータをシャッフルするかしないか、基本的にtrainデータにしか適応させない
    test_at_eval = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    class_names = ('bladder','femur-left','femur-right','heart','kindeny-left','kindeny-right','liver','lung-left','lung-right','pancreas','spleen')
    
    # print(train_dataset)
    # print("===================")
    # print(test_dataset)

    return test_at_eval,class_names


#適当な画像を入力、ノイズ
#input = torch.rand(1, 1, 28, 28)
#load_val()
#今プログラムで指定されているrenetを指定
# model = ResNet18() #torchvision.models.resnet18(pretrained=False)

# #LOADの因数を作成したモデルに変更
# model.load_state_dict(torch.load("model_re_med1"))
# model.eval()
# #outputはモデルに対して入力を行い、判別を行っている
# out  = model(input)
# # print(out.shape)

# #linearは全結合層、Identityで全結合層を除去（中間層をそのまま出力）
# model.linear = torch.nn.Identity()
# #中間層のみの場合の出力結果＝重み
# out  = model(input)
# print(out.shape)
# print(out)
