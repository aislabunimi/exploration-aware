#!/usr/bin/env python
# coding: utf-8


import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.arraysetops import unique
from PIL import Image
from sklearn import metrics
from sklearn.cluster import DBSCAN, MeanShift
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler


# get_ipython().run_line_magic('matplotlib', 'inline')


def write_result(image_adr, result, map_folder, run_number):
    for k in range(len(image_adr) - 1, 0, -1):
        if image_adr[k] == '/':
            name = image_adr[k + 1:]
            break
    dest_adr = '/PATH_TO_FOLDER_WITH_MAP_IMAGES/' + map_folder + '/z_resTests/' + run_number
    img = cv2.imread(image_adr)
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 150)
    fontScale = 4
    fontColor = (0, 254, 0)
    lineType = 3

    if (result == 1):
        cv2.putText(img, 'OK',
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
    else:
        cv2.putText(img, 'KO',
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    (254, 0, 0),
                    lineType)

    # Display the image
    # print("DISPLAY")
    # cv2.imshow("img",img)
    if not (os.path.exists(dest_adr)):
        os.makedirs(dest_adr)
    # if not(cv2.imwrite(dest_adr+name, img)):
    #	raise Exception("Could not write image")
    img = cv2.resize(img, (800, 800), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(dest_adr + '/' + name, img)


# cv2.waitKey(0)


def write_cluster_image(image_adr, result, map_folder, message, run_number):
    for k in range(len(image_adr) - 1, 0, -1):
        if image_adr[k] == '/':
            name = image_adr[k + 1:]
            break
    dest_adr = '/media/mrk/TOSHIBA EXT/runs/training/gmapping/realistic/' + map_folder + '/z_resTests/' + run_number
    img = cv2.imread(image_adr)
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 150)
    fontScale = 4
    fontColor = (0, 254, 0)
    lineType = 3

    if (result == 1):
        cv2.putText(img, message,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
    else:
        cv2.putText(img, message,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    (254, 0, 0),
                    lineType)

    # Display the image
    # print("DISPLAY")
    # cv2.imshow("img",img)
    if not (os.path.exists(dest_adr)):
        os.makedirs(dest_adr)
    # if not(cv2.imwrite(dest_adr+name, img)):
    #	raise Exception("Could not write image")
    img = cv2.resize(img, (800, 800), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(dest_adr + '/' + name, img)


# cv2.waitKey(0)

def write_cluster_image_2(image_adr, result, map_folder, message, plot_image, run_number):
    image_tmp = cv2.imread(image_adr)

    for k in range(len(image_adr) - 1, 0, -1):
        if image_adr[k] == '/':
            name = image_adr[k + 1:]
            break
    dest_adr = '/media/mrk/TOSHIBA EXT/runs/training/gmapping/realistic/' + map_folder + '/z_resTests/' + run_number
    img = plot_image
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 150)
    fontScale = 0.5
    fontColor = (0, 254, 0)
    lineType = 2

    if (result == 1):
        cv2.putText(img, message,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
    else:
        cv2.putText(img, message,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    (254, 0, 0),
                    lineType)

    # Display the image
    # print("DISPLAY")
    # cv2.imshow("img",img)
    if not (os.path.exists(dest_adr)):
        os.makedirs(dest_adr)
    # if not(cv2.imwrite(dest_adr+name, img)):
    #	raise Exception("Could not write image")
    cv2.imwrite(dest_adr + '/' + name, img)
    name = name[0:len(name) - 4]
    name = name + 'a.png'
    image_tmp = cv2.resize(image_tmp, (800, 800), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(dest_adr + '/' + name, image_tmp)


# cv2.waitKey(0)


def rotate_bound_modified(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(205, 205, 205))
