#!/usr/bin/env python3
"""
Greg Marcil
CS280 UC Berkeley
Practice implementation of Burt and Adelson's "A Multiresolution Spline With
Application to Image Mosaics
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage as ndimage
import cv2

def subtract(a,b):
    return a - b
# Add something weird 
def im_reduce(img):
    '''
    Apply gaussian filter and drop every other pixel
    '''
    filter = 1.0 / 20 * np.array([1, 5, 8, 5, 1])
    lowpass = ndimage.filters.correlate1d(img, filter, 0)
    lowpass = ndimage.filters.correlate1d(lowpass, filter, 1)
    im_reduced = lowpass[::2, ::2, ...]
    return im_reduced

def add(a, b):
    return a + b
def im_expand(img, template):
    '''
    Re-expand a reduced image by interpolating according to gaussian kernel
    Include template parameter to match size, easy way to avoid off by 1 errors
    re-expanding a previous layer that may have had odd or even dimension
    '''
    # y_temp, x_temp = template.shape[:2]
    # im_expanded = np.zeros((y_temp, x_temp) + template.shape[2:], img.dtype)
    im_expanded = np.zeros(template.shape, img.dtype)
    im_expanded[::2, ::2, ...] = img

    filter = 1.0 / 10 * np.array([1, 5, 8, 5, 1])
    lowpass = ndimage.filters.correlate1d(
        im_expanded, filter, 0, mode="constant")
    lowpass = ndimage.filters.correlate1d(lowpass, filter, 1, mode="constant")
    return lowpass



def gaussian_pyramid(image, layers=7):
    '''
    pyramid of increasingly strongly low-pass filtered images,
    shrunk 2x h and w each layer
    '''
    pyr = [image]
    temp_img = image
    for i in range(layers):
        temp_img = im_reduce(temp_img)
        pyr.append(temp_img)
    return pyr


def laplacian_pyramid(gaussian_pyramid):
    '''
    laplacian pyramid is a band-pass filter pyramid, calculated by the
    difference between subsequent gaussian pyramid layers, terminating with top
    layer of gaussian. Laplacian pyramid can be summed to give back original
    image
    '''
    pyr = []
    for i in range(len(gaussian_pyramid) - 1):
        g_k = gaussian_pyramid[i]
        g_k_plus_1 = gaussian_pyramid[i + 1]
        g_k_1_expand = im_expand(g_k_plus_1, g_k)
        laplacian = g_k - g_k_1_expand
        pyr.append(laplacian)

    pyr.append(gaussian_pyramid[-1])
    return pyr


def laplacian_collapse(pyr):
    '''
    Rejoin all levels of a laplacian pyramid. As the pyramid is a spanning set
    of band-pass filter outputs (all frequencies represented once and only
    once), joining all levels will give back the original image, modulo
    compression loss
    '''
    ''' Start with lowest pass data, top of pyramid '''
    partial_img = pyr[-1]
    for i in range(len(pyr) - 1):
        next_lowest = pyr[-2 - i]
        expanded_partial = im_expand(partial_img, next_lowest)
        partial_img = expanded_partial + next_lowest
    return partial_img


def laplacian_pyr_join(pyr1, pyr2, mask_gp):
    pyr = []
    for i in range(len(pyr1)):
        mask = np.array([mask_gp[i], mask_gp[i], mask_gp[i]]).transpose(1, 2, 0)
        layer = np.multiply(pyr1[i], mask) + np.multiply(pyr2[i], 1 - mask)
        pyr.append(layer)
    return pyr



def main():
    plt.ion()
    im1 = cv2.imread('test.png')
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    im2 = cv2.imread('data/portrait_old/align/target/0303.png')
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    
    mask = matplotlib.image.imread('mask.png')
    # mask = matplotlib.image.imread('data/portrait_old/mask/Repoparsing_mask/0303.png')
    im1, im2 = np.int32(im1), np.int32(im2)
    mask = np.uint8(mask)

    gp_1, gp_2 = [gaussian_pyramid(im) for im in [im1, im2]]
    mask_gp  = [cv2.resize(mask, (gp.shape[1], gp.shape[0])) for gp in gp_1]
    
    lp_1, lp_2 = [laplacian_pyramid(gp) for gp in [gp_1, gp_2]]
    lp_join = laplacian_pyr_join(lp_1, lp_2, mask_gp)
    im_join = laplacian_collapse(lp_join)

    np.clip(im_join, 0, 255, out=im_join)
    im_join = np.uint8(im_join)
    plt.imsave('test7.png', im_join)


if __name__ == '__main__':
    main()