# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by B. Y. Zhang on 2018/8/20
"""
import os

from bases.infer_base import InferBase
from skimage import io, measure
import time

import torch
import cv2
import numpy as np
from scipy import misc
import math
import pywt

from utils import np_utils, imresize
from utils.utils import mkdir_if_not_exist, colorize, shave, modcrop


class SRInfer(InferBase):
	def __init__(self, model, config=None):
		super(SRInfer, self).__init__(config)
		if self.config['cuda'] and not torch.cuda.is_available():
			print("GPU is not available on this device! Running in CPU!")
			self.config['cuda'] = False
		# self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		# model_path = os.path.join(self.config['cp_dir'], self.config['checkpoint'])
		model_path = self.config['checkpoint']

		# model = weights_init_normal(model, std=1)
		# # for name, param in model.named_parameters():
		# # 	print(name, param.cpu().data.numpy().mean(), param.cpu().data.numpy().mean())
		model = torch.nn.DataParallel(model)

		# noise_model = copy.deepcopy(model)
		# model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
		model.load_state_dict(torch.load(model_path))

		self.model = model
		# self.add_noise(noise_model, 0.008)
		# self.model.load_from_mat([os.path.join(self.config['cp_dir'], path) for path in model_path])
		# self.load_model_from_matlab(model_path)
		if self.config['cuda']:
			self.model = self.model.cuda()

		self.load_generator()

	def add_noise(self, model, alpha=0.01):
		noise_dict = model.state_dict()
		model_dict = self.model.state_dict()
		for k, t0_param in noise_dict.items():
			t1_param = model_dict[k]
			if 'w_output' not in k:
				try:
					if t0_param.shape[0] == t1_param.shape[0]:
						model_dict[k] = t1_param + alpha * t0_param
				except IndexError:
					continue
			else:
				print('Got ya')
		self.model.load_state_dict(model_dict)

	def load_generator(self):
		if 'pop' in self.config.keys():
			from models.stacksr import Em_Generator
		else:
			from models.stacksr_uni import Em_Generator
		self.em_generator = Em_Generator(self.config)
		print(type(self.em_generator))
		if self.config['cuda']:
			# if torch.cuda.device_count() > 1:
			self.em_generator = torch.nn.DataParallel(self.em_generator)
			self.em_generator = self.em_generator.cuda()

	def load_model_from_matlab(self, path):
		import scipy.io as sio

		mat_model = sio.loadmat(path)
		mat_params = mat_model['net']['params']

		for i, (name, param) in enumerate(self.model.named_parameters()):
			# param_array = np.transpose(mat_params[0, 0]['value'][0, i])
			param_array = mat_params[0, 0]['value'][0, i]

			# print(name, param.data.shape, mat_params[0, 0]['name'][0, i][0], param_array.shape)

			if 'conv' in mat_params[0, 0]['name'][0, i][0]:
				param_array = np.transpose(param_array, axes=[-1, *range(len(param_array.shape)-1)])
				# print(mat_params[0, 0]['name'][0, i][0], param_array.shape)
				if len(param_array.shape) == 4:
					param_array = np.transpose(param_array, axes=[0, 3, 1, 2])
					# print(mat_params[0, 0]['name'][0, i][0], param_array.shape)

			if 'b' in mat_params[0, 0]['name'][0, i][0]:
				param_array = param_array.squeeze(-1)
				# print(mat_params[0, 0]['name'][0, i][0], param_array.shape)
			if mat_params[0, 0]['name'][0, i][0] == 'img_up_f':
				param_array = np.expand_dims(np.expand_dims(param_array, 0), 0)
				# print(mat_params[0, 0]['name'][0, i][0], param_array.shape)
			if mat_params[0, 0]['name'][0, i][0] == 'residual_conv_f':
				param_array = np.expand_dims(param_array, 0)
				# print(mat_params[0, 0]['name'][0, i][0], param_array.shape)

			if param.data.shape == param_array.shape:
				param.data = torch.from_numpy(param_array)
			else:
				print('layer {} assigned wrong!, torch shape {}, mat shape {}'.format(name,
				                                                                      param.data.shape,
				                                                                      param_array.shape))

	def chisquare(self, parts):
		num = parts.shape[0]
		dis = 0
		for i in range(num):
			for j in range(i+1, num):
				dis += torch.mean((parts[i]-parts[j])**2/parts[j])
		dis /= num*(num-1)/2
		return dis

	def predict(self, data, **kwargs):
		# eng = meng.start_matlab()
		# for name, param in self.model.named_parameters():
		#     a = param.clone().cpu().data.numpy()
		#     print(name, a.max(), a.min())
		# print('\n')

		cost_time = 0
		save_dir = os.path.join(self.config['preds_dir'], kwargs['testset'])
		mkdir_if_not_exist(save_dir)
		self.model.eval()
		psnr_list = []
		ssim_list = []
		b_psnr_list = []
		b_ssim_list = []
		gap_list = {}
		diversity = 0
		with torch.no_grad():
			for img_bundle in data:
				# print(img_bundle['name'])
				if "color" in self.config.keys() and self.config["color"]:
					x = img_bundle['origin']
					y = img_bundle['y']
					multichannel = True
				else:
					x = img_bundle['x']
					y = img_bundle['y']
					if len(y.shape) == 3:
						(rows, cols, channel) = y.shape
						y, _, _ = np.split(y, indices_or_sections=channel, axis=2)
					else:
						(rows, cols) = y.shape

					multichannel = False

				x = torch.from_numpy(x).float().view(1, -1, x.shape[0], x.shape[1])
				if self.config['cuda']:
					x = x.cuda()
				# print(x[:5])
				lr_size = (x.shape[2], x.shape[3])
				hr_size = img_bundle['size']
				if self.config['progressive']:
					inter_sizes = np_utils.interval_size(lr_size, hr_size, self.config['max_gradual_scale'])
				else:
					inter_sizes = []
				inter_sizes.append(hr_size)

				start_time = time.time()
				if self.config['net'] == 'rrgun':
					preds = self.model(x, y_sizes=inter_sizes)
				elif self.config['net'] == 'lapsrn':
					# step = len(inter_sizes)
					# if kwargs['upscale'] % 2 != 0:
					#     step = step + 1
					step = int(np.ceil(math.log(kwargs['upscale'], 2)))
					preds = self.model(x, step=step)[-1]

					# y_numpy = preds[-1].data.cpu().numpy().squeeze()
					# x = misc.imresize(y_numpy, size=hr_size,
					#                    interp='bicubic', mode='F')
					# x = np.array(x, dtype=np.float64)
					# preds = torch.from_numpy(x)

					# resize = tfs.Compose([tfs.ToPILImage(), tfs.Resize(hr_size, interpolation=Image.BICUBIC),
					#                       tfs.ToTensor()])
					# preds = resize(preds[-1].squeeze(0))
					# preds = F.upsample(preds[-1], size=hr_size, mode='bilinear')
					# preds = preds[-1]
				elif self.config['net'] == 'lapgun':
					preds = self.model(x, y_sizes=inter_sizes)
				elif self.config['net'] in ['lapinternet', 'lapmtnet']:
					# print(img.shape)
					preds = self.model(x, size=inter_sizes[-1], step=self.config['step'])
				elif self.config['net'] in ['ensemsr', 'stacksr', 'stacksr_back', 'stacksr_uni']:
					input_list = self.em_generator(x)
					preds, parts = self.model(input_list)
					parts = torch.cat(tuple(parts), 0)
					diversity += self.chisquare(parts)
					# preds = com[-1]
				elif self.config['net'] == 'wmcnn':
					preds = self.model(x)
					preds = [p.data.cpu().numpy() for p in preds]
					# preds = [matlab.double(p.data.cpu().numpy().squeeze().tolist()) for p in preds]
					# preds = eng.idwt2(*preds, 'bior1.1')
					preds = pywt.idwt2((preds[0], (preds[1:])), 'bior1.1')
				else:
					preds = self.model(x)

				# for c in com:
				#     c = c.data.cpu().numpy()
				#     continue
				cost_time += time.time() - start_time
				if isinstance(preds, list):
					preds = np.clip(preds[-1].data.cpu().numpy(), 16/255, 235/255).astype(np.float64)
					# preds = np.clip(preds[-1].data.cpu().numpy(), 0, 1).astype(np.float64)
				else:
					try:
						preds = preds.data.cpu().numpy()
					except AttributeError:
						preds = preds
					# preds = preds.mul(255).clamp(0, 255).round().div(255)
					preds = np.clip(preds, 16/255, 235/255).astype(np.float64)
					# preds = np.clip(preds, 0, 1).astype(np.float64)

				preds = preds.squeeze()
				if len(preds.shape) == 3:
					preds = preds.transpose([1, 2, 0])
				preds = modcrop(preds.squeeze(), kwargs['upscale'])
				preds_bd = shave(preds.squeeze(), kwargs['upscale'])
				y = modcrop(y.squeeze(), kwargs['upscale'])
				# y = np.round(y * 255).astype(np.uint8)
				y_bd = shave(y.squeeze(), kwargs['upscale'])#/ 255.

				# print(preds_bd.shape, y_bd.shape)
				x = x.data.cpu().numpy().squeeze()
				# bic = x
				bic = imresize.imresize(x, scalar_scale=kwargs['upscale'])
				# bic = np.clip(bic, 16 / 255, 235 / 255).astype(np.float64)
				bic = np.round(bic*255).astype(np.uint8)
				bic = shave(bic.squeeze(), kwargs['upscale']) / 255.

				b_psnr = measure.compare_psnr(bic, y_bd, data_range=1)
				# b_ssim = measure.compare_ssim(bic, y_bd, data_range=1)
				b_ssim = self.calculate_ssim(bic* 255, y_bd* 255)
				# b_ssim = self.vifp_mscale(bic, y_bd)
				b_psnr_list.append(b_psnr)
				b_ssim_list.append(b_ssim)

				m_psnr = measure.compare_psnr(preds_bd, y_bd)

				# m_ssim = measure.compare_ssim(preds_bd, y_bd, multichannel=multichannel)
				m_ssim = self.calculate_ssim(preds_bd* 255, y_bd* 255)
				# print('image {} PSNR: {} SSIM: {}'.format(img_bundle['name'], m_psnr, m_ssim))
				gap_list[m_psnr-b_psnr] = img_bundle['name']
				psnr_list.append(m_psnr)
				ssim_list.append(m_ssim)
				test_value = '{}_{}'.format(m_psnr, m_ssim)
				# self.save_preds(save_dir, test_value, preds, img_bundle, True)

		diversity = diversity / len(data)
		print('Averaged Diversity is {}'.format(diversity))
		print('Averaged PSNR is {}, SSIM is {}'.format(np.mean(np.array(psnr_list)), np.mean(np.array(ssim_list))))
		print('Averaged BIC PSNR is {}, SSIM is {}'.format(np.mean(np.array(b_psnr_list)), np.mean(np.array(b_ssim_list))))
		# print(self.model.module.w_output)
		# print(self.model.module.w_inter)
		bigest_gap = sorted(gap_list, reverse=True)
		print(bigest_gap)
		print(gap_list[bigest_gap[0]], gap_list[bigest_gap[1]])
		print('Inference cost time {}s'.format(cost_time))

	def save_preds(self, save_dir, test_value, preds, img_bundle, is_color=False):
		if isinstance(preds, list):
			# TODO: not in use
			for i, cascade in enumerate(preds):
				cascade = cascade.data.cpu().numpy()
				cascade = cascade.clip(16 / 255, 235 / 255)     # color range of Y channel is [16, 235]
				io.imsave(os.path.join(save_dir,  '{}_{}_{}.png'.format(img_bundle['name'], i, test_value)), cascade)
		else:
			# preds = preds.data.cpu().numpy()
			# preds = preds.clip(16 / 255, 235 / 255)
			preds = preds.squeeze()
			if is_color:
				img_shape = preds.shape[:2]
				cb = misc.imresize(img_bundle['cb'], size=img_shape, mode='F')
				cr = misc.imresize(img_bundle['cr'], size=img_shape, mode='F')

				preds = colorize(preds, cb, cr)
			else:
				preds = preds*255
			io.imsave(os.path.join(save_dir, '{}_{}.png'.format(img_bundle['name'].split('.')[0], test_value)), preds.astype(np.uint8))

	def ssim(self, img1, img2):
		C1 = (0.01 * 255) ** 2
		C2 = (0.03 * 255) ** 2

		img1 = img1.astype(np.float64)
		img2 = img2.astype(np.float64)
		kernel = cv2.getGaussianKernel(11, 1.5)
		window = np.outer(kernel, kernel.transpose())

		mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
		mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
		mu1_sq = mu1 ** 2
		mu2_sq = mu2 ** 2
		mu1_mu2 = mu1 * mu2
		sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
		sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
		sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

		ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
		                                                        (sigma1_sq + sigma2_sq + C2))
		return ssim_map.mean()

	def calculate_ssim(self, img1, img2):
		'''calculate SSIM
		the same outputs as MATLAB's
		img1, img2: [0, 255]
		'''
		if not img1.shape == img2.shape:
			raise ValueError('Input images must have the same dimensions.')
		if img1.ndim == 2:
			return self.ssim(img1, img2)
		elif img1.ndim == 3:
			if img1.shape[2] == 3:
				ssims = []
				for i in range(3):
					ssims.append(self.ssim(img1, img2))
				return np.array(ssims).mean()
			elif img1.shape[2] == 1:
				return self.ssim(np.squeeze(img1), np.squeeze(img2))
		else:
			raise ValueError('Wrong input image dimensions.')
