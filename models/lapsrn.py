import torch
import torch.nn as nn
import numpy as np
from models.base_module import ConvBlock, DeconvBlock
import utils.utils as utils
from bases.model_base import ModelBase
import torch.nn.functional as F
# import torch.nn.init as init


def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


def bilinear_upsample_weights(weights):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    TODO: This function needs to be refined!
    """
    f_out = weights.size(0)
    f_in = weights.size(1)
    filter_size = weights.size(2)
    weights = np.zeros((f_out,
                        f_in,
                        filter_size,
                        filter_size), dtype=np.float32)

    upsample_kernel = upsample_filt(filter_size)

    for i in range(f_out):
        for j in range(f_in):
            weights[i, j, :, :] = upsample_kernel
    return torch.Tensor(weights)


class FeatureEmbedding(nn.Module):
    def __init__(self, config):
        '''
        self.config['D'], self.config['R'], self.config['skip_connect']
        :param D: Number of conv layers per recursive block
        :param R: Number of recursive blocks per level
        :param skip_connect: the way of residual connection, 'NS', 'DS', 'SS'
        '''
        super(FeatureEmbedding, self).__init__()
        self.config = config
        rec_block = []
        for i in range(self.config['D']):
            rec_block.append(ConvBlock(config['num_filter'], config['num_filter'], 3, 1, 1, bias=True,
                                       activation='lrelu', order='ac'))
        self.recursive_block = nn.Sequential(*rec_block)

        self.convt_F = DeconvBlock(config['num_filter'], config['num_filter'], 3, 2, 0, bias=True,
                                    activation='lrelu', order='ac')

    def forward(self, x):
        for i in range(self.config['R']):
            if self.config['skip_connect'] == 'DS':
                block_input = x
                x = self.recursive_block(x)
                x = block_input + x
            elif self.config['skip_connect'] == 'SS':
                if i == 0:
                    level_input = x
                x = self.recursive_block(x)
                x = x + level_input
            else:  # 'NS'
                x = self.recursive_block(x)

        x = F.pad(self.convt_F(x), (0, -1, 0, -1))
        return x


class ImageReconstruction(nn.Module):
    def __init__(self, config):
        super(ImageReconstruction, self).__init__()
        self.convt_I = DeconvBlock(config['in_channels'], config['in_channels'], 4, 2, 0, bias=False)
        self.conv_R = ConvBlock(config['num_filter'], config['in_channels'], 3, 1, 1, bias=True, activation=None)

    def forward(self, LR, convt_F):
        convt_I = F.pad(self.convt_I(LR), (-1, -1, -1, -1))
        conv_R = self.conv_R(convt_F)

        HR = torch.add(convt_I, conv_R)
        return HR

    def init_tconv(self):
        self.convt_I.init_weights(bilinear_upsample_weights(self.convt_I.deconv.weight))


class Net(ModelBase):
    def __init__(self, config):
        super(Net, self).__init__(config)
        self.input_conv = ConvBlock(self.config['in_channels'], self.config['num_filter'], 3, 1, 1, bias=True, activation=None)
        self.FeatureExtraction = FeatureEmbedding(config)
        self.ImageReconstruction = ImageReconstruction(config)
        self.weight_init()

    def forward(self, x, **kwargs):
        '''
        feed-forward operation
        :param x: input
        :param kwargs: additional parameter
        :return: out
        '''
        out = []
        # step = len(kwargs['y_sizes'])
        step = kwargs['step']

        img = x
        x = self.input_conv(x)

        for i in range(step):
            x = self.FeatureExtraction(x)
            img = self.ImageReconstruction(img, x)
            out.append(img)

        return out

    def weight_init(self):
        for m in self.modules():
            utils.weights_init_kaming(m)

        if self.config['input_conv_sigma']:
            for m in self.input_conv.modules():
                utils.weights_init_normal(m, std=self.config['input_conv_sigma'])

        self.ImageReconstruction.init_tconv()
