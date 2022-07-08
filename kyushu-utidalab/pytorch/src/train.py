import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import sys
import os
import argparse
from sklearn.metrics import classification_report

from datasets import Medmnist
#modelsの中にあるプログラムを全部importしている　resnet.pyなど
from models import *

import numpy as np

#medmnistのorgansを使用
#trainで使用するデータセット=train,val
#流れの確認
#①データの取得(train_loader,val_loader,class_names)をMedmnist.pyのload_data()から
#②モデルの定義　models内のresnet.pyから　今回はResNet18
#③損失関数、最適化関数の決定
#④epochの止め方(earlystopping)を決める
#⑤trainで学習を開始、model内のパラメータ更新　model→model'
#⑥valをmodel'で学習し、lossを算出、earlystoppingの基準に　
#⑥あくまで、基準で実質学習には関係ない



def train(model, device, train_loader, criterion, optimizer):
	#batchnormalization層やdropout層が含まれていた場合
	#これらは学習時では使うが推論時では使わないと言う指定をしてあげなければいけません．
	#学習時は使用
    model.train()

    output_list = []
    target_list = []
    n_data = 0
    running_loss = 0.0
    correct = 0
    # 全画像データとラベルを入手している
    #train_loaderはデータがバッチサイズごとに別れている,[]
    #enumerateはオブジェクト要素と同時に番号（カウント）を取得できる
    #https://note.nkmk.me/python-enumerate-start/
    for i, data in enumerate(train_loader, 0):
        # dataから画像データとラベル情報を入手
        inputs, labels = data
        # 計算をgpuで行うため、gpuにデータを移動させている
        inputs, labels = inputs.to(device), labels.to(device)
        # forward + backward + optimize
        outputs = model(inputs)
        # 次元を１つ()内の数字削除している
        # https://torimakujoukyou.com/numpy-squeeze/
        labels = labels.squeeze(1)

     ######################################################
        loss = criterion(outputs, labels)
		# Backward processing.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
     ######################################################
        # Set data to calculate score.
        output_list += [int(o.argmax()) for o in outputs]
        target_list += [int(t) for t in labels]
        running_loss += loss.item()
     ########################################################
        

        # n_dataは入力データの総数
        n_data += inputs.size(0)
        # outputの最大値のラベルを抽出している(_, )　_には値が入っている　１はaxis=1ということ
        _, predicted = torch.max(outputs.data, 1)
        # predicted=labelを数えている
        correct += (predicted == labels).sum().item()
        # 0スタートで10刻みにするため
        if i % 10 == 9 and i != 0:    # print every 2000 mini-batches
            print(
                f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / n_data:.3f}, acc:{100 * correct / n_data:3f}')


    test_acc = 100 * correct / n_data
    test_loss = running_loss / n_data
    print(n_data)
    return test_acc,test_loss


def val(model, device, val_loader, criterion):
	#train参照
	#推定時は使用しない
    model.eval()

    output_list = []
    target_list = []
    n_data = 0
    running_loss = 0.0
    correct = 0
    for i, data in enumerate(val_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # forward + backward + optimize
        outputs = model(inputs)
        # 次元を１つ()内の数字削除している
        # https://torimakujoukyou.com/numpy-squeeze/
        labels = labels.squeeze(1)
        # 損失関数の値を入力
        loss = criterion(outputs, labels)
        #データの総数を数える
        n_data += inputs.size(0)
        # print statistics
        running_loss += loss.item()
        # outputの最大値のラベルを抽出している(_, )　_には値が入っている　１はaxis=1ということ
        _, predicted = torch.max(outputs.data, 1)
		# predicted=labelを数えている
        correct += (predicted == labels).sum().item()

        # 10刻みで表示させている
        if i % 10 == 9 and i != 0:    # print every 2000 mini-batches
            print(
                f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / n_data:.3f}, acc:{100 * correct / n_data:3f}')

    val_loss = running_loss / n_data
    val_acc = 100 * correct / n_data
    print(n_data)
    return val_acc, val_loss

#from models import *で呼び出している
def get_model(model_name):

    if model_name == 'ResNet18':
        model = ResNet18()
    else:
        print('{} does NOT exist in repertory.'.format(model_name))
        sys.exit(1)

    return model

# 混合行列の値を算出してくれている？　今回は使用してない おそらくtest時に使用する
def calc_score(output_list, target_list, running_loss, data_loader):
    # Calculate accuracy.
    result = classification_report(output_list, target_list, output_dict=True)
    acc = round(result['weighted avg']['f1-score'], 6)
    loss = round(running_loss / len(data_loader.dataset), 6)

    return acc, loss

# 細かい定義
def parse_args():
    # Set arguments.
    arg_parser = argparse.ArgumentParser(description="Image Classification")

    arg_parser.add_argument("--dataset_name", type=str, default='medmnist')
    arg_parser.add_argument("--data_dir", type=str,
                            default='D:/workspace/datasets/')
    #arg_parser.add_argument("--data_dir", type=str, default='../data/')
    arg_parser.add_argument("--model_name", type=str, default='ResNet18')
    arg_parser.add_argument(
        "--model_ckpt_dir", type=str, default='/model_save/')
    arg_parser.add_argument("--model_ckpt_path_temp",
                            type=str, default='/model_save/{}_{}_epoch={}.pth')
    arg_parser.add_argument('--n_epoch', default=100000,
                            type=int, help='The number of epoch')
    arg_parser.add_argument('--lr', default=0.1,
                            type=float, help='Learning rate')

    args = arg_parser.parse_args()

    # # Make directory.
    # os.makedirs("data", exist_ok=True)
    # os.makedirs("model_ckpt_dir", exist_ok=True)

    # # Validate paths.
    # assert os.path.exists("data")
    # assert os.path.exists("model_ckpt_dir")

    return args

#epochの止まるタイミングを決定
class EarlyStopping:
    """earlystoppingクラス"""

    def __init__(self, patience=5, verbose=False, path='./model_sample/best_model.pth'):
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


# Parse arguments.
args = parse_args()

# gpuに使用する
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# データの取得
train_loader, val_loader, class_names = Medmnist.load_data()

# 使用するモデルの設定
model = get_model("ResNet18")
model = model.to(device)
print(model)

# 損失関数と最適化関数の決定
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr,momentum=0.9, weight_decay=5e-4)

# patienceはlossが最小値を「何回=patience」更新されなかったらepochを止める
earlystopping = EarlyStopping(patience=10, verbose=True)

# Train and val.
for epoch in range(args.n_epoch):
    # 学習　trainではmodelの中身のパラメータが知らず知らずのうちに更新されている
    train_acc, train_loss = train(model, device, train_loader, criterion, optimizer)
	#modelはtrainで更新されたmodelを使用し、評価
	#valの今回の実質的な役割はearly stopping
	#valは本来、手元にあるデータからtrain,valに分けている
	#仮想の未知データ
	#しかし、earlyでvalが最もいいときに止めている
	#つまり、valにとって都合がいい
	#だから、全く関係ないtestでもう一度試す
	#train時のmodelとvalのmodelは違う、trainで更新されたmodelを使用
    val_acc, val_loss = val(model, device, val_loader, criterion)

    # スコア　表示
    stdout_temp = 'epoch: {:>3}, train acc: {:<8}, train loss: {:<8}, val acc: {:<8}, val loss: {:<8}'
    print(stdout_temp.format(epoch+1, train_acc, train_loss, val_acc, val_loss))
	# callメソッド呼び出し
    earlystopping((val_loss), model)  
	# ストップフラグがTrueの場合、breakでforループを抜ける
	#要するにpatienceの数だけ連続でlossが更新できなければ学習を終了する
    if earlystopping.early_stop: 
        print("Early Stopping!")
        break

    # Save a model checkpoint.
    model_ckpt_path = args.model_ckpt_path_temp.format(args.dataset_name, args.model_name, epoch+1)
    torch.save(model.state_dict(),"./model_sample/medmnist_ResNet18_epoch{}".format(epoch+1))
    print('Saved a model checkpoint at {}'.format(model_ckpt_path))
    print('')

# カーネルの値，重み
# print(model.state_dict())
# print(model.put)
