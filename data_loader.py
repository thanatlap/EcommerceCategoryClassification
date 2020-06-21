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

def load_datapath_from_csv(datapath, datafile):

	return read_csv(os.path.join(datapath,datafile)).values


class ImageLoader(torch.utils.data.Dataset):

	def __init__(self, datadir, datafile, input_size=None):
		
		self.datapath = load_datapath_from_csv(datadir,datafile)
		self.datadir = datadir
		self.input_size = input_size
		self.transforms = transforms.Compose([
							transforms.RandomResizedCrop(input_size),
							transforms.ToTensor(),
							# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
							])

	def __getitem__(self, index):

		if torch.is_tensor(index):
			index = index.tolist()

		img_file, img_class = self.datapath[index]
		img_file = os.path.join(self.datadir, 'train','train',str(img_class), img_file)

		image = Image.open(img_file).convert("RGB")

		image = self.transforms(image)
		image = image/255 # scale

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

	np.random.seed(0)
	data = load_datapath_from_csv('D:\\ShoppeeChallenge_1_data','train.csv')
	training_idx = np.random.choice(np.arange(data.shape[0]), size=int(data.shape[0]*0.95), replace=False)
	training_data = data[training_idx]
	validating_data = np.array([d for i, d in enumerate(data) if i not in training_idx])
	print(sum([1 if i in training_data else 0 for i in validating_data]))
	print(data.shape[0])
	print(training_data.shape)
	print(validating_data.shape)
	pd.DataFrame(training_data, columns=['image','class']).to_csv(os.path.join('D:\\ShoppeeChallenge_1_data','train_train.csv'), index=None)
	pd.DataFrame(validating_data, columns=['image','class']).to_csv(os.path.join('D:\\ShoppeeChallenge_1_data','train_val.csv'), index=None)

	print(data[0,0])
