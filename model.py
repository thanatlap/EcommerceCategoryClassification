from __future__ import print_function
from __future__ import division
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, models, transforms
import sys

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class Clf_model(nn.Module):

	def __init__(self, n_class, use_pretrained, feature_extract):
		super(Clf_model, self).__init__()
		self.backbone = models.resnet152(pretrained=use_pretrained)
		set_parameter_requires_grad(self.backbone, feature_extract)
		num_ftrs = self.backbone.fc.in_features
		self.backbone.fc = nn.Linear(num_ftrs, n_class)
		torch.nn.init.xavier_uniform_(self.backbone.fc.weight)

	def forward(self, x):
		x = self.backbone(x)
		return x