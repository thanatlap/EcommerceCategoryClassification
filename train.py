from __future__ import print_function
from __future__ import division
import torch
import json
import os
from datetime import datetime
import argparse
from apex import amp
import time
import sklearn
from contextlib import contextmanager
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import Clf_model
from data_loader import ImageLoader, batch_to_gpu

def load_checkpoint(checkpoint_path, model, optimizer):
	""" Load model checkpoint
	------------------------------------------
	INPUTS:
	1. checkpoint_path (str): path to load the model
	2. model (obj ref): init model
	3. optimizer (obj ref): init optimizer
	------------------------------------------
	OUTPUTS:
	1. epoch (int): last epoch from checkpoint
	2. iteration (int): last iteration from the checkpoint

	- The model state and optimizer is passed as reference. Thus,
	  no need to return

	"""
	if os.path.isfile(checkpoint_path):
		print("[INFO] Loading checkpoint '{}'".format(checkpoint_path))
		checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
		model.load_state_dict(checkpoint_dict['state_dict'])
		optimizer.load_state_dict(checkpoint_dict['optimizer'])
		epochs = checkpoint_dict['epochs']
		iterations = checkpoint_dict['iterations']
		print("Loaded checkpoint '{}' from iterations {}" .format(checkpoint_path, iterations))
	else:
		print('[Warning] Weights not found, start training from scratch')
		epochs = 0
		iterations = 0
	return epochs, iterations


def save_checkpoint(model, optimizer, epochs, iterations, checkpoint_path):
	""" Save model checkpoint
	-----------------------------------------
	INPUTS:
	1. model (obj ref): model state
	2. optimizer (obj ref): optimizer state
	3. epoch (int): recent epoch 
	4. iteration (int): recent iteration 
	5. checkpoint_path (str): path to save the model

	"""
	print("[INFO] Saving model and optimizer state at epochs/iterations {}/{} to {}".format(epochs, iterations, checkpoint_path))
	os.makedirs('weights',exist_ok=True)
	torch.save({'iterations': iterations,
				'epochs': epochs,
				'state_dict': model.state_dict(),
				'optimizer': optimizer.state_dict()}, checkpoint_path)


# adapted from: https://discuss.pytorch.org/t/opinion-eval-should-be-a-context-manager/18998/3
# Following snippet is licensed under MIT license
@contextmanager
def evaluating(model):
	'''Temporarily switch to evaluation mode.'''
	istrain = model.training
	try:
		model.eval()
		yield model
	finally:
		if istrain:
			model.train()
			
			
def validate(model, criterion, valset, epochs, checkpoint_path, batch_to_gpu):
	"""Model validation function
	----------------------------------------------------
	INPUTS:
	1. model (obj ref): model state
	2. criterion (obj): loss function
	3. valset (obj): validation data
	4. epoch (int): recent epoch for logging
	5. checkpoint_path (str): path to save the model
	6. batch_to_gpu (fn): helper function for moving data to gpu

	"""
	correct = 0
	total = 0
	all_pred = []
	all_true = []
	model.eval()
	with torch.no_grad():
		val_loader = DataLoader(valset, sampler=None, num_workers=4,
								shuffle=False, batch_size=1,
								pin_memory=True, collate_fn=torch.utils.data.dataloader.default_collate)

		val_loss = 0.0
		for i, batch in enumerate(val_loader):
			x, y = batch_to_gpu(batch)
			y_pred = model(x)
			loss = criterion(y_pred, y)
			val_loss += loss.item()

			_, predicted = torch.max(y_pred.data, 1)
			total += y.size(0)
			correct += (predicted == y).sum().item()

			all_pred.extend([predicted])
			all_true.extend([y])

		val_loss = val_loss / (i + 1)
	
	val_log_path = os.path.join(checkpoint_path,'log_validate.txt')
	with open(val_log_path, 'a') as f:
		f.write("[INFO] {} Log Validation Result Epoch {} Validation loss {:9f} Top-1 Acc {}\n".format(datetime.now(), epochs, val_loss, 100*correct/total))
		f.write(confusion_matrix(all_true, all_pred))
	print("[INFO] {} Log Validation Result Epoch {} Validation loss {:9f} Top-1 Acc {}\n".format(datetime.now(), epochs, val_loss, 100*correct/total))
	
	model.train()


def get_optimizer(model, train_config):
	""" Get optimizer function
	select optimizer for trianing from config json file
	----------------------------------------------------
	INPUTS:
	1. model (obj): init model
	2. config (dict): training configuration 
	"""
	
	if train_config['opts'] == 'adam':
		optimizer = torch.optim.Adam(model.parameters(), 
									 lr=train_config['learning_rate'],
									 weight_decay=train_config['weight_decay'])
	elif train_config['opts'] == 'sgd':
		optimizer = torch.optim.SGD(model.parameters(), 
									lr=train_config['learning_rate'],
									momentum=train_config['sgd_momentum'],
									nesterov=train_config['sgd_nesterov'],
									weight_decay=train_config['weight_decay'])
	else:
		return NotImplementedError('optimizer [%s] is not implemented', train_config['opts'])
		
	return optimizer


def get_scheduler(optimizer, train_config, iterations=-1):
	""" Get learning rate scheduler
	select optimizer for trianing from config json file
	----------------------------------------------------
	INPUTS:
	1. optimizer (obj): init optimizer
	2. config (dict): training configuration 
	3. iteration (int): recent training iteration
	"""

	if iterations == 0 or  iterations == 1:
		  iterations=-1
	if 'lr_policy' not in train_config or train_config['lr_policy'] == 'constant':
		scheduler = None  # constant scheduler
	elif train_config['lr_policy'] == 'step':
		scheduler = lr_scheduler.StepLR(optimizer, step_size=train_config['epoch_step_size'], gamma=train_config['gamma'], last_epoch=iterations)
	elif train_config['lr_policy'] == 'expo':
		scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=train_config['gamma'], last_epoch=iterations)
	else:
		return NotImplementedError('learning rate policy [%s] is not implemented', train_config['lr_policy'])
	return scheduler

			
def train(model, criterion, train_loader, batch_to_gpu, train_config, valset):
	""" Main function for training the model
	----------------------------------------------------
	INPUTS:
	1. model (obj): init optimizer
	2. criterion (obj): loss function 
	3. train_loader (obj): data loader for model training
	4. batch_to_gpu (fn): helper function to move data to GPU
	5. train_config (dict): training configuration
	6. valset (obj): validation dataset
	"""
	
	# init variable
	run_start_time = time.perf_counter()
	iterations = 0
	val_loss = 0.0
	num_iters = 0
	model.train()
	start_epochs = 0
	summary_writer = SummaryWriter(os.path.join(train_config['checkpoint_path'], train_config["tensorboard_dir"]))
	checkpoint_filepath = os.path.join(train_config['checkpoint_path'], train_config['checkpoint_filename'])
	
	# init optimizer
	optimizer = get_optimizer(model, train_config)
	
	# to cuda
	model = model.cuda()
	criterion.cuda()
	
	if train_config['amp_run']:
		model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
	
	# resume training
	start_epochs, iterations = load_checkpoint(checkpoint_filepath, model, optimizer)
	
	# get scheduler
	scheduler = get_scheduler(optimizer, train_config, iterations)
	lr = scheduler.get_last_lr()

	# train
	for epoch in range(start_epochs, train_config['epochs']):

		# used to calculate avg loss over epoch
		train_epoch_avg_loss = 0.0
		
		for i, batch in enumerate(train_loader, iterations):
			
			# for resume training
			if iterations*train_config['batch_size'] > len(train_loader):
				break
			elif iterations != 0 and iterations*train_config['batch_size'] != i:
				continue 
			
			iter_start_time = time.perf_counter()
			model.zero_grad()
			x, y = batch_to_gpu(batch)

			y_pred = model(x)
			loss = criterion(y_pred, y)
			
			# store loss
			reduced_loss = loss.item()
			train_epoch_avg_loss += reduced_loss
			
			if train_config['amp_run']:
				with amp.scale_loss(loss, optimizer) as scaled_loss:
					scaled_loss.backward()
			else:
				loss.backward()
				
			grad_norm = torch.nn.utils.clip_grad_norm_(
				amp.master_params(optimizer), train_config['grad_clip_thresh'])
			
			optimizer.step()
			
			iter_stop_time = time.perf_counter()
			iter_time = iter_stop_time - iter_start_time
			
			print('[INFO] {:1f}s, epoch: {}, train step: {}, loss: {:5f}, lr: {:5f}, datetime: {}'.format(
					iter_time, epoch, i, loss.item(), optimizer.param_groups[-1]['lr'], datetime.now().strftime("%H:%M:%S")))
			
			if iterations%train_config['iterations_per_checkpoint'] == train_config['iterations_per_checkpoint']-1:
				save_checkpoint(model, optimizer, epoch, iterations, checkpoint_filepath)
			
			iterations += 1
			
		
		# save at the end of epoch
		save_checkpoint(model, optimizer, epoch, iterations, checkpoint_filepath)
			
		# validating
		validate(model, criterion, valset, epoch, train_config['checkpoint_path'], batch_to_gpu)
		
		if scheduler is not None:
			lr = scheduler.get_last_lr()
			scheduler.step()
				
		summary_writer.add_scalar('Loss/train_epoch', train_epoch_avg_loss/iterations, epoch * len(train_loader) + i) 
		summary_writer.add_scalar('Learning Rate', optimizer.param_groups[-1]['lr'], epoch * len(train_loader) + i) 
		iterations = 0
		
	run_stop_time = time.perf_counter()
	run_time = run_stop_time - run_start_time
	save_checkpoint(model, optimizer, epoch, iterations, checkpoint_filepath)
	
	
def train_clf(model_config, train_config, data_config, image_config):
	"""
	Train classification model function
	init variable for training model
	---------------------------------------
	INOUT:
	- model_config: model configuration
	- train_config: training configuration
	- data_config: dataloader configuration
	- image_config: image data configuration
	---------------------------------------
	RETURN:
	- a dictionary of model, loss function, training dataloader, load to gpu function, train_config,
	  valset loader, and collate_fn for validation data loader
	"""
	
	# load model
	model = Clf_model(**model_config)
	criterion = torch.nn.CrossEntropyLoss()
	
	# dataloader    
	train_set = ImageLoader(data_config['data_path'], data_config['training_files'], image_config['input_size'])
	valset = ImageLoader(data_config['data_path'], data_config['validation_files'], image_config['input_size'])
	
	train_loader = DataLoader(train_set, num_workers=4, shuffle=True, sampler=None, 
		batch_size=train_config['batch_size'], pin_memory=True, drop_last=True, collate_fn=torch.utils.data.dataloader.default_collate) 
	 
	return {'model':model, 
			'criterion':criterion, 
			'train_loader':train_loader, 
			'batch_to_gpu': batch_to_gpu, 
			'train_config': train_config,
			'valset': valset}


def main():

	with open('config.json', 'r') as f:
		config = json.load(f)

	prep_train = train_clf(**config)
	train(**prep_train)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--device", default=0, type=int)
	
	args = parser.parse_args()
	
	torch.cuda.set_device("cuda:{}".format(args.device))
	print(torch.cuda.is_available())
	torch.backends.cudnn.benchmark = True
	
	main()