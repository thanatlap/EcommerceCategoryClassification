from __future__ import print_function
from __future__ import division
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, models, transforms
import sys
from torchvision.models import resnet152, resnext101_32x8d, wide_resnet50_2, densenet161

def set_parameter_requires_grad(model, feature_extracting):
	if feature_extracting:
		for param in model.parameters():
			param.requires_grad = False

class Clf_model_dense(nn.Module):

	def __init__(self, n_class, use_pretrained, feature_extract):
		super(Clf_model_dense, self).__init__()
		self.backbone = densenet161(pretrained=use_pretrained)
		set_parameter_requires_grad(self.backbone, feature_extract)
		num_ftrs = self.backbone.classifier.in_features
		# self.backbone.fc = nn.Linear(num_ftrs, 1024)
		self.backbone.classifier  = nn.Linear(num_ftrs, 512)
		self.bn1 = nn.BatchNorm1d(512)
		self.fc2 = nn.Linear(512, n_class)
		nn.init.kaiming_normal_(self.backbone.classifier.weight, mode='fan_out', nonlinearity='relu')
		nn.init.xavier_uniform_(self.fc2.weight)
		nn.init.constant_(self.bn1.weight, 1)
		nn.init.constant_(self.bn1.bias, 0)

	def forward(self, x):
		x = self.backbone(x)
		x = self.bn1(x)
		# x = nn.functional.relu(x)
		x = x* torch.tanh(F.softplus(x))
		x = self.fc2(x)
		return x

class Clf_model_wrn(nn.Module):

	def __init__(self, n_class, use_pretrained, feature_extract):
		super(Clf_model_wrn, self).__init__()
		self.backbone = wide_resnet50_2(pretrained=use_pretrained)
		set_parameter_requires_grad(self.backbone, feature_extract)
		num_ftrs = self.backbone.fc.in_features
		self.backbone.fc = nn.Linear(num_ftrs, 1024)
		self.bn1 = nn.BatchNorm1d(1024)
		self.fc2 = nn.Linear(1024, n_class)
		nn.init.kaiming_normal_(self.backbone.fc.weight, mode='fan_out', nonlinearity='relu')
		nn.init.xavier_uniform_(self.fc2.weight)
		nn.init.constant_(self.bn1.weight, 1)
		nn.init.constant_(self.bn1.bias, 0)

	def forward(self, x):
		x = self.backbone(x)
		x = self.bn1(x)
		# x = nn.functional.relu(x)
		x = x* torch.tanh(F.softplus(x))
		x = self.fc2(x)
		return x

class Clf_model_resnext(nn.Module):

	def __init__(self, n_class, use_pretrained, feature_extract):
		super(Clf_model_resnext, self).__init__()
		self.backbone = resnext101_32x8d(pretrained=use_pretrained)
		set_parameter_requires_grad(self.backbone, feature_extract)
		num_ftrs = self.backbone.fc.in_features
		self.backbone.fc = nn.Linear(num_ftrs, 1024)
		self.bn1 = nn.BatchNorm1d(1024)
		self.fc2 = nn.Linear(1024, n_class)
		nn.init.kaiming_normal_(self.backbone.fc.weight, mode='fan_out', nonlinearity='relu')
		nn.init.xavier_uniform_(self.fc2.weight)
		nn.init.constant_(self.bn1.weight, 1)
		nn.init.constant_(self.bn1.bias, 0)

	def forward(self, x):
		x = self.backbone(x)
		x = self.bn1(x)
		# x = nn.functional.relu(x)
		x = x* torch.tanh(F.softplus(x))
		x = self.fc2(x)
		return x

class Clf_model_resnet152(nn.Module):

	def __init__(self, n_class, use_pretrained, feature_extract):
		super(Clf_model_resnet152, self).__init__()
		self.backbone = resnet152(pretrained=use_pretrained)
		set_parameter_requires_grad(self.backbone, feature_extract)
		num_ftrs = self.backbone.fc.in_features
		self.backbone.fc = nn.Linear(num_ftrs, 512)
		self.bn1 = nn.BatchNorm1d(512)
		self.fc2 = nn.Linear(512, n_class)
		nn.init.kaiming_normal_(self.backbone.fc.weight, mode='fan_out', nonlinearity='relu')
		nn.init.xavier_uniform_(self.fc2.weight)
		nn.init.constant_(self.bn1.weight, 1)
		nn.init.constant_(self.bn1.bias, 0)

	def forward(self, x):
		x = self.backbone(x)
		x = self.bn1(x)
		# x = nn.functional.relu(x)
		x = x* torch.tanh(F.softplus(x))
		x = self.fc2(x)
		return x


class ClfStackModel(nn.Module):

	def __init__(self, n_class):
		super(ClfStackModel, self).__init__()

		dense_path = 'weights/dense161_pretrain_tune1/model.pt'
		wrn_path = 'weights/wrn50_pretrain_tune5/model.pt'
		resnext_path = 'weights/resnet152_pretrain_tune4/model.pt'
		res_path = 'weights/resnet152_pretrain_tune3/model.pt'


		self.dense = Clf_model_dense(n_class, True, True) 
		self.dense.load_state_dict(torch.load(dense_path, map_location='cpu')['state_dict'])
		self.wrn = Clf_model_wrn(n_class, True, True) 
		self.wrn.load_state_dict(torch.load(wrn_path, map_location='cpu')['state_dict'])
		self.resnex = Clf_model_resnext(n_class, True, True) 
		self.resnex.load_state_dict(torch.load(resnext_path, map_location='cpu')['state_dict'])
		self.res = Clf_model_resnet152(n_class, True, True) 
		self.res.load_state_dict(torch.load(res_path, map_location='cpu')['state_dict'])

		set_parameter_requires_grad(self.dense, True)
		set_parameter_requires_grad(self.wrn, True)
		set_parameter_requires_grad(self.resnex, True)
		set_parameter_requires_grad(self.res, True)

		self.fc1 = nn.Linear(int(n_class*4), 64)
		self.bn1 = nn.BatchNorm1d(64)
		self.fc2 = nn.Linear(64, n_class)

		nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
		nn.init.xavier_uniform_(self.fc2.weight)
		nn.init.constant_(self.bn1.weight, 1)
		nn.init.constant_(self.bn1.bias, 0)

	def forward(self, x):
		x1 = self.dense(x)
		x1 = nn.functional.softmax(x1, dim=1).unsqueeze(1)
		x2 = self.wrn(x)
		x2 = nn.functional.softmax(x2, dim=1).unsqueeze(1)
		x3 = self.resnex(x)
		x3 = nn.functional.softmax(x3, dim=1).unsqueeze(1)
		x4 = self.res(x)
		x4 = nn.functional.softmax(x4, dim=1).unsqueeze(1)

		x = torch.cat([x1, x2, x3, x4], dim=1)
		x = torch.reshape(x, (x.size()[0], -1))
		x = self.fc1(x)
		x = self.bn1(x)
		# x = nn.functional.relu(x)
		x = x* torch.tanh(F.softplus(x))
		x = self.fc2(x)
		return x


if __name__ == '__main__':

	model = ClfStackModel(42)