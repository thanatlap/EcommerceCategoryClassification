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

class Clf_model(nn.Module):

	def __init__(self, n_class, use_pretrained, feature_extract):
		super(Clf_model, self).__init__()
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