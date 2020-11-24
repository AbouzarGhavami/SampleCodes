# Author: Abouzar Ghavami
# Email: ghavamip@gmail.com
# This code is protected by copyright laws in US.
# Please do not reuse in any format without permission of Abouzar Ghavami.

import cv2
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

theta_min = -20
theta_max = 20
shift_min = -30
shift_max = 30
for i in range(10):
    path = ''.join(['./', str(i)])
    a = os.listdir(path)
    nums = []
    for x in a:
        s = ''
        for l in range(len(x)):
            if x[l] == '.':
                break
            else:
                s = ''.join([s, x[l]])
        nums.append(int(s))
    index = max(nums) + 1
    for x in a:
        image = cv2.imread(''.join([path, '/', x]), cv2.IMREAD_GRAYSCALE)
        for i in range(len(image)):
            for j in range(len(image[0])):
                if image[i][j] < 255:
                    image[i][j] = 252
        shift_i = int(shift_min + \
                           np.random.uniform() * (shift_max - shift_min))
        shift_j = int(shift_min + \
                           np.random.uniform() * (shift_max - shift_min))
        theta = theta_min + np.random.uniform() * (theta_max - theta_min)
        image2 = rotate_image(image, theta)
        image3 = 255 * np.ones((len(image2), len(image2[0])))
        for i in range(len(image2)):
            for j in range(len(image2[0])):
                if i + shift_i < len(image2) and i + shift_i >= 0:
                    if j + shift_j < len(image2) and j + shift_j >= 0:
                        if image2[i][j] != 252:
                            image3[i + shift_i][j + shift_j] = 255
                        if image2[i + shift_i][j + shift_j] == 252:
                            image3[i + shift_i][j + shift_j] = 0
        
        cv2.imwrite(''.join([path, '/', str(index), '.jpg']),image3)
        index = index + 1
