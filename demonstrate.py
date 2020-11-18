import os
from os.path import join
from os import listdir
import numpy as np
from skimage import io, draw
from skimage.transform import rescale
import cv2
from matplotlib import pyplot as plt


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".PNG", ".jpg", ".jpeg", ".bmp", ".JPG"])


def bbox_expand(im, start, end):
    rr, cc = draw.rectangle(start=start, end=end)
    cv2.rectangle(im, start, end, (255, 0, 0), 2)
    im_bbox = im[cc, rr]
    crop_resized = rescale(im_bbox, im.shape[1]/im_bbox.shape[1], order=0)
    # io.imshow(crop_resized)
    # io.show()
    im_comb = np.concatenate((im/255., crop_resized), axis=0)
    return im_comb


if __name__ == '__main__':
    form = 'ktde'
    folder = './result/{}/'.format(form)
    # start = (141, 231)
    # end = (275, 330)
    start = (130, 150)
    end = (175, 195)
    hr_list = [join(folder, x) for x in sorted(listdir(folder)) if is_image_file(x)]
    for fname in hr_list:
        if 'woman' in fname:
            im = io.imread(fname)
            plt.imshow(im)
            plt.show()
            im_expand = bbox_expand(im, start, end)
            name = os.path.basename(fname)
            io.imsave(join(folder, 'bbox_{}'.format(name)), im_expand)
