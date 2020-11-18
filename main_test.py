#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by B. Y. Zhang on 2018/8/20
"""

from data_loaders.imageloader import TestImageLoader
from infers.sr_infer import SRInfer
from utils.config_utils import process_config, get_test_args
import importlib
import torch
import random
import numpy as np


def set_random_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def test_main():
	print('[INFO] Retrieving configuration...')
	parser = None
	config = None
	set_random_seed(0)
	try:
		args, parser = get_test_args()
		# args.config = 'experiments/wmcnn/wmcnn.json'
		# args.config = 'configs/stacksr.json'
		# args.config = 'configs/stacksr_3vdsr.json'
		config = process_config(args.config, False)
	except Exception as e:
		print('[Exception] Configuration is invalid, %s' % e)
		if parser:
			parser.print_help()
		print('[Exception] Refer to: python main_train.py -c configs/lapsrn.json')
		exit(0)

	print('[INFO] Building graph...')
	try:
		Net = importlib.import_module('models.{}'.format(config['trainer']['net'])).Net
		model = Net(config=config['model'])
		# model = Net()
		if torch.cuda.device_count() > 1:
			model = torch.nn.DataParallel(model)
		# print_network(model)
	except ModuleNotFoundError:
		raise RuntimeWarning("The model name is incorrect or does not exist! Please check!")

	for path in config['test_data_loader']['test_path']:
		test_config = config['test_data_loader']
		test_config['test_path'] = path
		print('[INFO] Loading data...')
		dl = TestImageLoader(config=test_config)

		print('[INFO] Predicting...')
		infer = SRInfer(model, config['trainer'])
		infer.predict(dl.get_test_mat(), testset=config['test_data_loader']['test_path'],
					  upscale=config['test_data_loader']['upscale'])


if __name__ == '__main__':
	test_main()
