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
import Vit
import earlystop
from torchvision.models import resnet18

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


#tqdmモジュール
def train(x,y,model):
    args = parse_args()
    model.train()
    total_loss = 0.0
    total_samples = len(x)
    train_total=13940
    correct = 0 
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

        ############################################################
        # # n_dataは入力データの総数
        # n_data += data.size(0)
        # outputの最大値のラベルを抽出している(_, )　_には値が入っている　１はaxis=1ということ
        #.dataは勾配情報を取り除く
        _, predicted = torch.max(output.data, 1)
        _, labels = torch.max(label.data, 1)
        # predicted=labelを数えている
        correct += (predicted == labels).sum().item()
        # 0スタートで10刻みにするため
        # if i % 10 == 9 and i != 0:    # print every 2000 mini-batches
    print(
        f'[{epoch + 1}] loss: {total_loss / train_total:.3f}, acc:{100 * correct / train_total:3f}')
    avg_loss = total_loss / train_total
        # #    loss_history.append(avg_loss)
        # print('Average train loss: ' + '{:.10f}'.format(avg_loss))
    return avg_loss


def val(x,y,model):
    args = parse_args()
    model.eval()
    total_loss = 0.0
    correct = 0 
    total_samples = len(x)
    val_total=2452
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

        # # n_dataは入力データの総数
        # n_data += data.size(0)
        # outputの最大値のラベルを抽出している(_, )　_には値が入っている　１はaxis=1ということ
        #.dataは勾配情報を取り除く
        _, predicted = torch.max(output.data, 1)
        _, labels = torch.max(label.data, 1)
        # predicted=labelを数えている
        correct += (predicted == labels).sum().item()
        # 0スタートで10刻みにするため
        # if i % 10 == 9 and i != 0:    # print every 2000 mini-batches
    print(
        f'[{epoch + 1}] loss: {total_loss / val_total:.3f}, acc:{100 * correct / val_total:3f}')
    val_avg_loss = total_loss / val_total
    #    loss_history.append(avg_loss)
    # print('Average train loss: ' + '{:.10f}'.format(val_avg_loss))

    return val_avg_loss


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = parse_args()
    #resnet
    ########################################################################################################
    model1 = resnet18(pretrained=True)
    model1.conv1 = torch.nn.Conv2d(1,64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model1.fc = torch.nn.Linear(512,11)
    model1.load_state_dict(torch.load("./model_resnet/best_model.pth"))
    model1.fc = torch.nn.Identity()
    model1.to(device)
    model1.eval()   
    # unfreeze all layers
    for param in model1.parameters():
        param.requires_grad = True
    ###########################################################################################################

    model2=Vit.ViT(image_size=1, num_classes=11, dim=512, depth=1, heads=8, mlp_dim=1024, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.).to(device)

    # patienceはlossが最小値を「何回=patience」更新されなかったらepochを止める
    earlystopping = earlystop.EarlyStopping(patience=10, verbose=True,path='./model_save_trans2/best_model_trans.pth')

    # Train and val.
    for epoch in range(args.n_epoch):
        # 学習　trainではmodelの中身のパラメータが知らず知らずのうちに更新されている
        x=[]
        y=[]
        train_loader, class_names = load_test.load_train()
        for i, data in enumerate(train_loader, 0):
            t_inputs,t_label = data
            t_inputs, t_label = t_inputs.to(device), t_label.to(device)

            #linearは全結合層、Identityで全結合層を除去（中間層をそのまま出力） 

            #中間層のみの場合の出力結果＝重み
            # out  = model(inputs[0].unsqueeze(0))
            tout  = model1(t_inputs)
            tout = tout.unsqueeze(0)
            #xにappendする前にcatをしようするかも(拡張を増やす場合)
            #順番入れ替え
            tout = torch.transpose(tout, 0, 1)
            # print(out.shape)
            t_label = torch.eye(11)[t_label] 
            # print(label)
            # x.append(out.detach().numpy())
            # y.append(label.detach().numpy())
            x.append(tout)
            y.append(t_label)
            #この後ろにつなげる

        m=[]
        n=[]
        val_loader, class_names = load_test.load_val()
        for i, data in enumerate(val_loader, 0):
            vinputs,vlabel = data
            vinputs, vlabel = vinputs.to(device), vlabel.to(device)

            #linearは全結合層、Identityで全結合層を除去（中間層をそのまま出力） 

            #中間層のみの場合の出力結果＝重み
            # out  = model(inputs[0].unsqueeze(0))
            vout  = model1(vinputs)
            vout = vout.unsqueeze(0)
            #xにappendする前にcatをしようするかも(拡張を増やす場合)
            #順番入れ替え
            vout = torch.transpose(vout, 0, 1)
            # print(out.shape)
            vlabel = torch.eye(11)[vlabel] 
            # print(label)
            # x.append(out.detach().numpy())
            # y.append(label.detach().numpy())
            m.append(vout)
            n.append(vlabel)
            #この後ろにつなげる
    
        train_loss = train(x,y,model2)
	    #modelはtrainで更新されたmodelを使用し、評価
	    #valの今回の実質的な役割はearly stopping
	    #valは本来、手元にあるデータからtrain,valに分けている
	    #仮想の未知データ
	    #しかし、earlyでvalが最もいいときに止めている
	    #つまり、valにとって都合がいい
	    #だから、全く関係ないtestでもう一度試す
	    #train時のmodelとvalのmodelは違う、trainで更新されたmodelを使用
        val_loss = val(m,n,model2)

        # スコア　表示
        # stdout_temp = 'epoch: {:>3}, train acc: {:<8}, train loss: {:<8}, val acc: {:<8}, val loss: {:<8}'
        # stdout_temp = 'epoch: {:>3}, train loss: {:<8},  val loss: {:<8}'
        # print(stdout_temp.format(epoch+1, val_loss))
        print(epoch+1)
	    # callメソッド呼び出し
        earlystopping((val_loss), model2)  
	    # ストップフラグがTrueの場合、breakでforループを抜ける
	    #要するにpatienceの数だけ連続でlossが更新できなければ学習を終了する
        if earlystopping.early_stop: 
            print("Early Stopping!")
            break

        #  Save a model checkpoint.
        model_ckpt_path = args.model_ckpt_path_temp.format(args.dataset_name, args.model_name, epoch+1)
        torch.save(model2.state_dict(),"./model_save_trans2/trans_epoch{}".format(epoch+1))
        print('Saved a model checkpoint at {}'.format(epoch+1))
        print('')

    
