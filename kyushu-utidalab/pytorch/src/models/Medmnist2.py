from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO, Evaluator

# print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")


def load_data():
    data_flag = 'organsmnist'
    # data_flag = 'breastmnist'
    download = True

    NUM_EPOCHS = 3
    BATCH_SIZE = 128
    lr = 0.001

    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])

    data_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[.5], std=[.5])])

    #load the data
    train_dataset = DataClass(split='train', transform=data_transform, download=download)
    # train_dataset.data = train_dataset[train_dataset.targets == 0]
    val_dataset = DataClass(split='val', transform=data_transform, download=download)

    pil_dataset = DataClass(split='train', download=download)

    # encapsulate data into dataloader form
    #data_loaderはdatasetをバッチサイズごとに分けること、今回なら128ごとに分けられている
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

    class_names = ('bladder','femur-left','femur-right','heart','kindeny-left','kindeny-right','liver','lung-left','lung-right','pancreas','spleen')
    
    # print(train_dataset)
    # print("===================")
    # print(val_dataset)

    return train_loader,val_loader,class_names

# train_loader, val_loader, class_names = load_data()
# for i, data in enumerate(train_loader, 0):
#     # get the inputs; data is a list of [inputs, labels]
#     inputs, labels = data
#     print(i)

load_data()

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO, Evaluator

# print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")


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
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    #load the data
    train_dataset = DataClass(split='train', transform=data_transform, download=download)
    # train_dataset.data = train_dataset[train_dataset.targets == 0]
    val_dataset = DataClass(split='val', transform=data_transform, download=download)

    pil_dataset = DataClass(split='train', download=download)

    # encapsulate data into dataloader form
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    class_names = ('bladder','femur-left','femur-right','heart','kindeny-left','kindeny-right','liver','lung-left','lung-right','pancreas','spleen')
    
    # print(train_dataset)
    # print("===================")
    # print(val_dataset)

    return val_loader,class_names

# train_loader, val_loader, class_names = load_data()
# for i, data in enumerate(train_loader, 0):
#     # get the inputs; data is a list of [inputs, labels]
#     inputs, labels = data
#     print(i)







