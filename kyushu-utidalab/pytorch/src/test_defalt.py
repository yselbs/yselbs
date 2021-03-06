'''Train CIFAR10 with PyTorch.'''
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
from models import load_test
from datasets import Medmnist
from models import *

import numpy as np


def test(model, device, test_loader, criterion):
	model.eval()

	output_list = []
	labels_list = []
	running_loss = 0.0
	for  i,data in enumerate(test_loader):
		# Forward processing.
		inputs, labels = data
		inputs, labels = inputs.to(device), labels.to(device)
		outputs = model(inputs)
		labels = labels.squeeze(1)
		loss = criterion(outputs, labels)
		# 配列要素の最大要素のインデックスを返す
		output_list += [int(o.argmax()) for o in outputs]
		labels_list += [int(t) for t in labels]
		running_loss += loss.item()

		

	# Calculate score.
	test_acc, test_loss = calc_score(output_list, labels_list, running_loss, test_loader)
    
	return test_acc, test_loss


def get_model(model_name):
	if model_name == 'ResNet18':
		model = ResNet18()
	else:
		print('{} does NOT exist in repertory.'.format(model_name))
		sys.exit(1)
	
	return model


def calc_score(output_list, target_list, running_loss, data_loader):
	# Calculate accuracy.
	result = classification_report(output_list, target_list, output_dict=True)
	acc = round(result['weighted avg']['f1-score'], 6)
	loss = round(running_loss / len(data_loader.dataset), 6)

	return acc, loss


def parse_args():
	# Set arguments.
	arg_parser = argparse.ArgumentParser(description="Image Classification")
	
	#arg_parser.add_argument("--data_dir", type=str, default='D:/workspace/datasets/')
	arg_parser.add_argument("--data_dir", type=str, default='/data/')
	arg_parser.add_argument("--model_name", type=str, default='ResNet18')
	arg_parser.add_argument("--model_path", type=str, default='./model_sample/best_model.pth')

	args = arg_parser.parse_args()

	# Validate paths.
	assert os.path.exists(args.data_dir)
	assert os.path.exists(args.model_path)

	return args


# Parse arguments.
args = parse_args()

# Set device.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load dataset.
test_loader, class_names = load_test.load_test()

# Load a model.
model = get_model(args.model_name)
model.load_state_dict(torch.load(args.model_path))
model = model.to(device)
print('Loaded a model from {}'.format(args.model_path))
# Define loss function.
criterion = nn.CrossEntropyLoss()

# Test a model.
test_acc, test_loss = test(model, device, test_loader, criterion)


	

# Output score.
stdout_temp = 'test acc: {:<8}, test loss: {:<8}'
print(stdout_temp.format(test_acc, test_loss))

