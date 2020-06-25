from __future__ import print_function
from __future__ import division
import torch
import json
import os, sys
from datetime import datetime
import argparse
from apex import amp
import time
from sklearn.metrics import confusion_matrix
from contextlib import contextmanager
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import pandas as pd
from skimage import io, transform
from torchvision import transforms, utils
from PIL import Image

from model import Clf_model

np.set_printoptions(threshold=sys.maxsize)


MODEL_WEIGHT = 'weights/resnet152_pretrain_tune4/model.pt'
DATA_DIR = "D:\\ShoppeeChallenge_1_data"
TEST_FILE = "test.csv"

def zero_leading_pad(value):
    try:
        return '{0:0>2}'.format(int(value))
    except:
        return value


class ImageLoader(torch.utils.data.Dataset):

	def __init__(self, datadir, datafile, input_size):
		
		self.datapath = pd.read_csv(os.path.join(datadir,datafile)).values
		self.datadir = datadir
		self.input_size = input_size
		self.transforms = transforms.Compose([
							transforms.Resize([input_size,input_size]),
							transforms.ToTensor(),
							transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
							])

	def __getitem__(self, index):

		if torch.is_tensor(index):
			index = index.tolist()

		img_file, img_class = self.datapath[index]
		img_file = os.path.join(self.datadir, 'test','test', img_file)

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


def inference(model, data_loader):

	model.load_state_dict(torch.load(MODEL_WEIGHT, map_location='cpu')['state_dict'])
	model.cuda()
	model.eval()
	total_pred = []
	print('[INFO] Predicting')
	with torch.no_grad():

		for i, batch in enumerate(data_loader):
			x, _ = batch_to_gpu(batch)
			y_pred = model(x)

			_, predicted = torch.max(y_pred.data, 1)
			total_pred.extend(predicted.cpu().numpy())


	data = pd.read_csv(os.path.join(DATA_DIR, TEST_FILE))
	data = data.drop(['category'], axis=1)

	try:
		vfunc = np.vectorize(zero_leading_pad)
		total_pred = vfunc(total_pred)
		data['category'] = np.array(total_pred)
		data.to_csv('submission_{}.csv'.format(datetime.now().strftime("%Y%m%d_%H%M")), index=None)
		
	except:
		data['category'] = np.array(total_pred)
		data.to_csv('submission_temp.csv', index=None)

	


def inference_clf(data_dir, test_file, input_size, model_config):

	# load model
	model = Clf_model(**model_config)
	
	# dataloader    
	pred_set = ImageLoader(data_dir, test_file, input_size)
	data_loader = DataLoader(pred_set, num_workers=4, shuffle=False, sampler=None, 
		batch_size=128, pin_memory=True, drop_last=False, collate_fn=torch.utils.data.dataloader.default_collate) 
	 
	return {'model':model, 
			'data_loader':data_loader}

def main():

	model_config = {
	    "n_class":42,
	    "use_pretrained":True, 
	    "feature_extract":False
	    }
	input_size = 224
	

	prep_predict = inference_clf(DATA_DIR, TEST_FILE, input_size, model_config)
	inference(**prep_predict)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--device", default=0, type=int)
	
	args = parser.parse_args()
	
	torch.cuda.set_device("cuda:{}".format(args.device))
	print('GPU Enable: {}'.format(torch.cuda.is_available()))
	if not torch.cuda.is_available():
		raise ValueError('This code is implement to run on GPU only')
	torch.backends.cudnn.benchmark = True
	
	main()

