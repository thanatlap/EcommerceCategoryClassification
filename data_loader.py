from __future__ import print_function, division
import os
import torch
from pandas import read_csv
from glob import glob
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from sklearn.model_selection import train_test_split

def load_datapath_from_csv(datapath, datafile):

	return read_csv(os.path.join(datapath,datafile)).values


class ImageLoader(torch.utils.data.Dataset):

	def __init__(self, datadir, datafile, input_size, is_training=False):
		
		self.datapath = load_datapath_from_csv(datadir,datafile)
		self.datadir = datadir
		self.input_size = input_size

		if is_training:
			self.transforms = transforms.Compose([
								transforms.RandomResizedCrop(input_size, scale=(0.6, 1.0)),
								# transforms.Resize([input_size,input_size]),
								transforms.ColorJitter(),
								transforms.RandomGrayscale(0.25),
								transforms.RandomHorizontalFlip(p=0.25),
								transforms.RandomVerticalFlip(p=0.25),
								transforms.ToTensor(),
								transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
								transforms.RandomErasing(p=0.5, scale=(0.1, 0.4), ratio=(0.3, 3.3))
								])
		else:
			self.transforms = transforms.Compose([
								transforms.Resize([input_size,input_size]),
								transforms.ToTensor(),
								transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
								])

	def __getitem__(self, index):

		if torch.is_tensor(index):
			index = index.tolist()

		img_file, img_class = self.datapath[index]
		img_file = os.path.join(self.datadir, 'train','train',str(img_class), img_file)

		image = Image.open(img_file).convert("RGB")

		image = self.transforms(image)
		

		return (image, img_class)

	def __len__(self):
		return self.datapath.shape[0]

def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)


def batch_to_gpu(batch):

	features, targets = batch
	features = to_gpu(features).float()
	targets = to_gpu(targets).long()

	return (features, targets)   


if __name__ == '__main__':

	SUBSET = True
	DATA_DIR = 'D:\\ShoppeeChallenge_1_data'
	TRAIN_SET = 0.85

	np.random.seed(0)

	if not SUBSET:
		data = load_datapath_from_csv(DATA_DIR,'train.csv')
		train_file = 'train_train.csv'
		val_file = 'train_val.csv'
	else:
		data = read_csv(os.path.join(DATA_DIR,'train.csv'))
		data = data.loc[(data['category'] == 0) | (data['category'] == 1) | (data['category'] == 2) | (data['category'] == 3)| (data['category'] == 4)].values
		train_file = 'train_subtrain.csv'
		val_file = 'train_subval.csv'

	indices = np.random.permutation(data.shape[0])
	training_idx, val_idx = indices[:int(data.shape[0]*TRAIN_SET)], indices[int(data.shape[0]*TRAIN_SET):]
	training_data, validating_data = data[training_idx,:], data[val_idx,:]

	# training_idx = np.random.choice(np.arange(data.shape[0]), size=int(data.shape[0]*TRAIN_SET), replace=False)
	# training_data = data[training_idx]
	# validating_data = np.array([d for i, d in enumerate(data) if i not in training_idx])
	print(sum([1 if i in training_data else 0 for i in validating_data]))
	pd.DataFrame(training_data, columns=['image','class']).to_csv(os.path.join(DATA_DIR,train_file), index=None)
	pd.DataFrame(validating_data, columns=['image','class']).to_csv(os.path.join(DATA_DIR,val_file), index=None)

	print(data[0,0])
