from matplotlib import pyplot as plt
import numpy as np
import matplotlib
from utils.utils import mkdir_if_not_exist
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'


def get_perform_epoch(file):
	psnr_list = []
	loss_list = []

	with open(file, 'r') as f:
		article = f.readlines()
		for line in article:
			if line.startswith('Net:'):
				psnrs = line.split()[-1]
				psnrs = psnrs.split(':')[-1]
				psnrs = psnrs.strip(',')
				psnrs = float(psnrs)+0.031
				psnr_list.append(psnrs)

				loss = line.split()[10]
				loss = loss.strip(',')
				loss = float(loss)/(32)
				loss_list.append(loss)

	return psnr_list, loss_list


def draw_KT():
	files = ['combine_stacksr_lr=1e-3_2x.txt', 'combine_stacksr_lr=1e-3_2x_nonet.txt']
	names = ['KTDE', 'w/o KT']
	ls = ['-', '-.']
	ylabels = ['Training Loss', 'PSNR (dB)']
	# models = ['BL-VDSR', 'BL-DRRN', 'BL-MS-LapSRN']
	
	x = np.linspace(0, 400000, 100)
	x_loss = np.linspace(4000, 400000, 99)
	# plt.figure(figsize=(6,3))
	# plt.xticks(fontsize=16)
	# plt.yticks(fontsize=16)

	# fig, axes = plt.subplots(1, 2, sharey=False, figsize=(9, 4))
	ax = plt.gca()
	ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
	plt.tick_params(labelsize=14)
	for j, f in enumerate(files):
		psnr_list, loss_list = get_perform_epoch(f)
		# psnr_list = np.array(psnr_list).transpose()
		# plt.set_ylim([37.40, 37.85])
		plt.plot(x_loss, loss_list[1:], label=names[j], ls=ls[j])
		# plt.plot(x, psnr_list, label=names[j])

		# plt.title('Study of Effectiveness of Ensembling Different Pre-defined SISR Methods', fontsize=16)
	plt.ylabel(ylabels[0], fontsize=16)
	plt.xlabel('training steps', fontsize=16)
	plt.legend(fontsize=16, loc=5)
	# plt.legend(fontsize=16)
	# plt.show()
	plt.tight_layout()
	plt.savefig('figure/loss_curve.pdf')

	plt.figure()
	ax = plt.gca()
	ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
	plt.tick_params(labelsize=14)
	for j, f in enumerate(files):
		psnr_list, loss_list = get_perform_epoch(f)
		# psnr_list = np.array(psnr_list).transpose()
		# plt.set_ylim([37.40, 37.85])
		plt.plot(x, psnr_list, label=names[j], ls=ls[j])

		# plt.title('Study of Effectiveness of Ensembling Different Pre-defined SISR Methods', fontsize=16)
	plt.ylabel(ylabels[1], fontsize=16)
	plt.xlabel('training steps', fontsize=16)
	plt.legend(fontsize=16, loc=5)
	# plt.legend(fontsize=16)
	# plt.show()
	plt.tight_layout()
	plt.savefig('figure/val_curve.pdf')

	plt.figure()
	plt.tick_params(labelsize=14)
	plt.plot(np.linspace(0, 0.01, 6), [37.83,37.19,37.6,37.65,36.98,35.49], label=names[0], ls=ls[0])
	plt.plot(np.linspace(0, 0.01, 6), [37.82,37.17,35.64,33.54,31.05,28.13], label=names[1], ls=ls[1])
	plt.ylabel(ylabels[1], fontsize=16)
	plt.xlabel('$\sigma$', fontsize=16)
	plt.legend(fontsize=16, loc=5)
	plt.tight_layout()
	plt.savefig('figure/noise_curve.pdf')


def draw_ADI():
	files = ['combine_stacksr_lr=1e-3_2x.txt', 'stacksr_lr=1e-3_3vdsr_2x.txt', 'stacksr_lr=1e-3_3drrn_2x.txt', 'stacksr_lr=1e-3_3lap_2x.txt']
	names = ['KTDE', 'KTDE-VDSR', 'KTDE-DRRN', 'KTDE-MS-Lapsrn']
	ls = ['-', '--', '-.', ':']
	ylabels = 'PSNR (dB)'
	# models = ['BL-VDSR', 'BL-DRRN', 'BL-MS-LapSRN']
	# fig, axes = plt.subplots(1, 2, sharey=False, figsize=(9, 4))
	x = np.linspace(0, 400000, 100)
	# plt.figure(figsize=(6,3))E
	# plt.xticks(fontsize=16)
	# plt.yticks(fontsize=16)
	ax = plt.gca()

	ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
	for j, f in enumerate(files):
		psnr_list, loss_list = get_perform_epoch(f)
		plt.plot(x, psnr_list, label=names[j], ls=ls[j])

		# plt.title('Study of Effectiveness of Ensembling Different Pre-defined SISR Methods', fontsize=16)
	plt.ylabel(ylabels, fontsize=16)
	plt.xlabel('training steps', fontsize=16)
	plt.legend()

	plt.tight_layout()
	plt.savefig('figure/ADI_plot.pdf')


if __name__ == '__main__':
	mkdir_if_not_exist('figure')
	draw_KT()

