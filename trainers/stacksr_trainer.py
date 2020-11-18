# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
# @Time    : 11/09/2018 11:13 PM
# @Author  : Boyu Zhang
# @Site    : 
# @File    : stacksr_trainer.py
# @Software: PyCharm
"""

from bases.trainer_base import TrainerBase
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from utils import utils
from trainers import get_optimizer
from trainers import get_loss_function
from skimage import measure

import torch.nn as nn
import torch
import tqdm
import os
import numpy as np
import contextlib
import sys


class DummyFile(object):
	def __init__(self, file):
		self.file = file

	def write(self, x):
		# Avoid print() second call (useless \n)
		if len(x.rstrip()) > 0:
			tqdm.tqdm.write(x, file=self.file)

@contextlib.contextmanager
def nostdout():
	save_stdout = sys.stdout
	sys.stdout = DummyFile(sys.stdout)
	yield
	sys.stdout = save_stdout


class SRTrainer(TrainerBase):
	def __init__(self, model, data, config):
		super(SRTrainer, self).__init__(model, data, config)
		if self.config['cuda'] and not torch.cuda.is_available():
			print("GPU is not available on this device! Running in CPU!")
			self.config['cuda'] = False

		if self.config['resume']:
			model_path = self.config['checkpoint']
			pretrained = torch.load(model_path, map_location=lambda storage, loc: storage)
			# self.model = torch.nn.DataParallel(self.model)
			self.model.load_state_dict(pretrained)

		self.optim, self.scheduler = get_optimizer.get_optimizer(model, config['optimizer'])
		self.loss_func = get_loss_function.get_loss_function(config['loss'])
		if 'pop' in config.keys():
			from models.stacksr import Em_Generator
		else:
			from models.stacksr_uni import Em_Generator
		self.em_generator = Em_Generator(config)

		if self.config['cuda']:
			self.model = self.model.cuda()
			if torch.cuda.device_count() > 1:
				self.em_generator = torch.nn.DataParallel(self.em_generator)
			self.em_generator = self.em_generator.cuda()
			self.loss_func = self.loss_func.cuda()

		self.highest_score = 0
		self.best_model = None
		self.draw_graph = False
		self.writer = SummaryWriter(self.config['tb_dir'])

	def feedforward(self, x, y, combine=False, just_w=False):
		preds = self.model(x, just_w)
		if combine:
			loss = self.loss_func(preds[0], y)+sum([self.loss_func(p, y) for p in preds[1]])
		else:
			loss = sum([self.loss_func(p, y) for p in preds[1]])

		return loss, preds

	@property
	def train(self):

		for epoch in range(1, self.config['num_epochs']+self.config['train_ow_epoch'] + 1):
			print('running epoch {}'.format(epoch))
			train_loss = 0.0
			self.model.train()
			combine = self.config['combine']

			for iteration, batch in enumerate(tqdm.tqdm(self.data['train'], file=sys.stdout)):
				with nostdout():
					step = len(self.data['train']) * (epoch - 1) + iteration
					x, y = batch[0], batch[1]
					if self.config['cuda']:
						# x = [img.cuda() for img in x]
						x = x.cuda()
						y = y.cuda()

					self.optim.zero_grad()
					with torch.no_grad():
						input_list = self.em_generator(x)

					loss, preds = self.feedforward(input_list, y, combine, self.config['just_w'])

					loss.backward()
					if 'clip' in self.config.keys():
						nn.utils.clip_grad_norm(self.model.parameters(), self.config['clip'])

					# for name, param in self.model.named_parameters():
					# 	if param.grad is not None:
					# 		a = param.grad.clone().cpu().data.numpy()
					# 		print('grad', name, a.max(), a.min())
					# print('\n')
					self.optim.step()

					# for name, param in self.em_generator.named_parameters():
					# 	a = param.clone().cpu().data.numpy()
					# 	print('weight after', name, a.max(), a.min())
					# print('\n')
					train_loss += loss.item()
					# tqdm.tqdm.write('batch loss is {}'.format(loss.item()))

					# add log for visualization
					if iteration % self.config['log_freq'] == 0:
						self.writer.add_scalar('train_loss', loss.item(), step)

						if self.config['net'] in ['ensemsr', 'stacksr']:
							# dummy_sub = vutils.make_grid(preds[0][:9], normalize=False, scale_each=True)
							# self.writer.add_image('Image_preds', dummy_sub, step)

							dummy_sub = vutils.make_grid(input_list[0][:9])
							self.writer.add_image('Image_input', dummy_sub, step)
						#
							# for idx, residuals in enumerate(preds[1]):
							# 	dummy_sub = vutils.make_grid(residuals[:9], normalize=False, scale_each=True)
							# 	self.writer.add_image('Image_residuals_{}'.format(idx), dummy_sub, step)

						if self.config['net'] == 'ensemplus':
							for idx, img in enumerate(preds[0]):
								dummy_sub = vutils.make_grid(img[:9], normalize=True, scale_each=True)
								self.writer.add_image('Image_preds_{}'.format(idx), dummy_sub, step)

							for idx, residuals in enumerate(preds[1]):
								dummy_sub = vutils.make_grid(residuals[:9], normalize=False, scale_each=True)
								self.writer.add_image('Image_residuals_{}'.format(idx), dummy_sub, step)

						weight_lr = self.optim.param_groups[0]['lr']
						self.writer.add_scalar('weight_lr', weight_lr, step)

						bias_lr = self.optim.param_groups[1]['lr']
						self.writer.add_scalar('bias_lr', bias_lr, step)

						for name, param in self.model.named_parameters():
							self.writer.add_histogram(name, param.clone().cpu().data.numpy(), step)
							if param.grad is not None:
								self.writer.add_histogram('{}_gradient'.format(name), param.grad.clone().cpu().data.numpy(), step)

						# for name, param in self.em_generator.named_parameters():
						# 	self.writer.add_histogram(name, param.clone().cpu().data.numpy(), step)
						# 	if param.grad is not None:
						# 		self.writer.add_histogram('{}_gradient'.format(name), param.grad.clone().cpu().data.numpy(), step)
						# break
				# break

			epoch_loss = train_loss / len(self.data['train'])

			# adjust learning rate according to training loss
			if self.scheduler is not None:
				if self.config["optimizer"]["lr_scheduler"] == "ReduceLROnPlateau":
					self.scheduler.step(metrics=epoch_loss)
				else:
					self.scheduler.step()
					# for param_group in self.optim.param_groups:
					#     param_group["lr"] = max(param_group["lr"],
					#                             param_group["initial_lr"]*self.config["optimizer"]["min_lr_fraction"])

			# validation
			average_psnr = self.validate(self.data['test'], combine)
			for i, ap in enumerate(average_psnr):
				self.writer.add_scalar('{}x'.format(i+2), ap, epoch)

			if len(average_psnr) == 1:
				msg = 'Net: {}, Epoch: {}, LR: {}, bias_LR: {}, Training Loss: {:.4f}, ' \
					  'Validation PSNR: {}x {:.4f}'.format(self.config['net'], epoch, self.optim.param_groups[0]['lr'],
				                                           self.optim.param_groups[1]['lr'],
				                                           epoch_loss, self.config['upscale'], average_psnr[0])
			else:
				msg = 'Net: {}, Epoch: {}, LR: {}, bias_LR: {} Training Loss: {:.4f}, Validation PSNR {}x:'.format(
					self.config['net'], epoch, self.optim.param_groups[0]['lr'],
					self.optim.param_groups[1]['lr'], epoch_loss, self.config['upscale'])
				for idx, psnr in enumerate(average_psnr):
					msg += '{:.4f},'.format(psnr)

			print(msg)
			# save checkpoint
			if epoch % self.config['save_freq'] == 0 or np.max(average_psnr) > self.highest_score:
				self.highest_score = np.max(average_psnr)
				self.checkpoint(epoch, average_psnr)
				print("saving checkpoint in {}".format(self.config['cp_dir']))

		self.writer.close()
		return self.highest_score, self.best_model

	def validate(self, data, combine=False):
		set_psnr = []
		self.model.eval()
		for x_list, y in data:
			# utils.preview(x_list[0])
			if self.config['cuda']:
				x_list = [x.cuda() for x in x_list]
			y = y.data.cpu().numpy()

			psnr_multi_scale = []
			for i, img in enumerate(x_list):
				if self.config['upscale'] and not self.config['progressive'] and i != self.config['upscale']-2:
					continue

				label = utils.modcrop(y.squeeze(), self.config['upscale'])
				with torch.no_grad():
					input_list = self.em_generator(img)
				results = self.model(input_list)
				if combine:
					preds = results[0]
				else:
					preds = results[1]
					preds.append(results[0])

				if isinstance(preds, list):
					preds = [np.clip(p.data.cpu().numpy(), 16 / 255, 235 / 255) for p in preds]
					psnr_multi_scale.extend([measure.compare_psnr(p.squeeze(), label) for p in preds])
				else:
					print(preds.shape)
					preds = np.clip(preds.data.cpu().numpy(), 16 / 255, 235 / 255)
					psnr_multi_scale.append(measure.compare_psnr(preds.squeeze(), label))

			set_psnr.append(psnr_multi_scale)

		set_psnr = np.array(set_psnr)

		average_psnr = np.mean(set_psnr, 0)
		return average_psnr

	def checkpoint(self, epoch, val_psnr):
		max_score = val_psnr[-1]
		model_name = 'weights.epoch_{}_mean_val_psnr_{:0.3f}.hdf5'.format(epoch, max_score)
		filepath = os.path.join(self.config['cp_dir'], model_name)
		if max_score > self.highest_score:
			self.highest_score = max_score
			self.best_model = model_name
		torch.save(self.model.state_dict(), filepath)

