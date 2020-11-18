# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
# @Time    : 21/10/2018 10:14 PM
# @Author  : Boyu Zhang
# @Site    : 
# @File    : stacksr.py
# @Software: PyCharm
"""

from models.base_module import ResBlock, ConvBlock
import torch.nn as nn
from bases.model_base import ModelBase
from utils.utils import resize_tensor, weights_init_kaming
import torch.nn.functional as F
from models import lapsrn, vdsr, drrn
import numpy as np
import torch
import math


class Net(ModelBase):
	def __init__(self, config):
		super(Net, self).__init__(config)

		num = config['num_pretrained']

		self.branch = nn.ModuleList([
			self.make_layers(config['num_residuals']) for _ in range(num)
		])
		# self.weights = torch.randn(6, requires_grad=True).cuda()

		self.weight_init()

		# self.w_output = nn.Parameter(torch.Tensor(config['init_wo']))
		self.w_output = nn.Parameter(torch.ones(num)/num)
		self.w_inter = torch.ones([num, num])*0.2 / (num-1)
		self.w_inter[torch.eye(num).byte()] = 0.8
		self.w_inter = nn.Parameter(self.w_inter)

		# self.em_generator = Em_Generator(config)

	def make_layers(self, num_blocks):
		blocks = [ConvBlock(self.config['in_channels'], self.config['num_filter'], 3, 1, 1, activation='lrelu',
						   norm=None, bias=False)]
		for i in range(num_blocks):
			blocks.append(ResBlock(self.config['num_filter'], self.config['num_filter'], 3, 1, 1, activation='lrelu'))

		out_conv = nn.Sequential(*[ConvBlock(self.config['num_filter'], self.config['num_filter'], 3, 1, 1,
											 activation=None,
											 norm=None, bias=False),
								   ConvBlock(self.config['num_filter'], self.config['in_channels'], 3, 1, 1,
											 activation=None,
											 norm=None, bias=False)
								   ])

		blocks.append(out_conv)

		return nn.Sequential(*blocks)

	def forward(self, input_list, just_w=False):
		input_list = [input_list[:, i, :, :].unsqueeze(1) for i in range(input_list.shape[1])]

		residual = input_list

		for idy in range(len(self.branch[0])):
			outs = []
			for i, inputs in enumerate(input_list):
				# print("idy and i are {} and {}".format(idy, i))
				outs.append(self.branch[i][idy](inputs))

			input_list = outs
			if self.config['is_net'] and idy != len(self.branch[0])-1:
				# sum_inputs = sum(input_list)
				# input_list = [(sum_inputs-input_list[i])+input_list[i] for i in range(len(input_list))]
				input_list = [sum(list(map(lambda x, y: x*y, input_list, self.w_inter[i]))) for i in range(len(input_list))]

		input_list = list(map(lambda x, y: x+y, input_list, residual))
		if just_w:
			input_list = [p.detach() for p in input_list]

		out_sum = sum([torch.mul(input_list[i], self.w_output[i]) for i in range(len(input_list))])

		return out_sum, input_list

	def weight_init(self):
		for m in self.modules():
			weights_init_kaming(m)


class Em_Generator(ModelBase):
	def __init__(self, config):
		super(Em_Generator, self).__init__(config)
		self.upscale = config['upscale']

		self.is_cuda = torch.cuda.is_available()

		if self.config['pop'] != 0:
			self.vdsr = vdsr.Net()
			self.load_vdsr(self.config['vdsr']['path'])
			if self.is_cuda:
				self.vdsr = self.vdsr.cuda()

		if self.config['pop'] != 1:
			self.drrn = drrn.Net()
			self.load_drrn(self.config['drrn']['path'])
			if self.is_cuda:
				self.drrn = self.drrn.cuda()

		if self.config['pop'] != 2:
			self.lapsrn = lapsrn.Net(config=self.config['lapsrn'])
			self.load_lapsrn(self.config['lapsrn']['path'])
			if self.is_cuda:
				self.lapsrn = self.lapsrn.cuda()

	def forward(self, x):
		bases = []

		bic = resize_tensor(x, scale=self.upscale)
		if self.is_cuda:
			bic = bic.cuda()

		if self.config['use_inter']:
			nea = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
			bases.append(nea)

			bil = F.interpolate(x, scale_factor=self.upscale, mode='bilinear')
			bases.append(bil)

			bases.append(bic)

		if self.config['pop'] != 0:
			vds = self.vdsr(bic)
			bases.append(vds)
			del vds
			torch.cuda.empty_cache()

		if self.config['pop'] != 1:
			drr = self.drrn(bic)
			bases.append(drr)
			del drr
			torch.cuda.empty_cache()

		if self.config['pop'] != 2:
			lap = self.lapsrn(x, step=int(np.ceil(math.log(self.upscale, 2))))[-1]
			if self.upscale == 3:
				hr_w = x.shape[2]*self.upscale
				hr_h = x.shape[3]*self.upscale
				lap = resize_tensor(lap, size=(hr_w, hr_h))
				if self.is_cuda:
					lap = lap.cuda()
			bases.append(lap)
			del lap
			torch.cuda.empty_cache()

		if self.is_cuda:
			bases = torch.cat(tuple(bases), dim=1).cuda().detach()
		else:
			bases = torch.cat(tuple(bases), dim=1).detach()

		return bases

	def load_lapsrn(self, path):
		import scipy.io as sio

		mat_model = sio.loadmat(path)
		mat_params = mat_model['net']['params']

		for i, (name, param) in enumerate(self.lapsrn.named_parameters()):
			param_array = mat_params[0, 0]['value'][0, i]

			if 'conv' in mat_params[0, 0]['name'][0, i][0]:
				param_array = np.transpose(param_array, axes=[-1, *range(len(param_array.shape) - 1)])
				if len(param_array.shape) == 4:
					param_array = np.transpose(param_array, axes=[0, 3, 1, 2])

			if 'b' in mat_params[0, 0]['name'][0, i][0]:
				param_array = param_array.squeeze(-1)
			if mat_params[0, 0]['name'][0, i][0] == 'img_up_f':
				param_array = np.expand_dims(np.expand_dims(param_array, 0), 0)
			if mat_params[0, 0]['name'][0, i][0] == 'residual_conv_f':
				param_array = np.expand_dims(param_array, 0)

			if param.data.shape == param_array.shape:
				param.data = torch.from_numpy(param_array)
			else:
				print('layer {} assigned wrong!, torch shape {}, mat shape {}'.format(name,
																					  param.data.shape,
																					  param_array.shape))

	def load_vdsr(self, path):
		weights = torch.load(path)
		self.vdsr.load_state_dict(weights)

	def load_edsr(self, path):
		weights = torch.load(path)
		self.edsr.load_state_dict(weights)

	def load_srdensenet(self, path):
		weights = torch.load(path)
		self.srdensenet.load_state_dict(weights)

	def load_drrn(self, path):
		weights = torch.load(path)
		self.drrn.load_state_dict(weights)
