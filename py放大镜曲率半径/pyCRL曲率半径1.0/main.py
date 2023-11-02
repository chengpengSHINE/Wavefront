import numpy as np
from PIL import Image
from skimage.restoration import unwrap_phase
import matplotlib.pyplot as plt
from pylab import mpl
# import wavepy
import datetime
import os
import glob
import pandas as pd
import sys

sys.path.append('./code')
from jifeng import frankotchellappa
from jiequ import flybhxqyjgb, flybhsdxqyjgb

######          让标题中文
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
# 像素大小=6.5e-07
# 棋盘格光栅周期=4.8e-06
# 模式=对角半圆周
# 距离探测器到gr=0.33
# 光子能量=14000.0
# 源距离=32.0

energy=12400
p1 = 10e-6
d = 0.4845
R=32
pixel=6.5e-7

p2s = p1*(R+d)/R
λ = ((6.626e-34) * 299792458) / (energy * 1.602e-19)

I = Image.open("./grating_00012.tiff")  # 读取图片
# arr = I
# mean = np.mean(arr)
# std = np.std(arr)
# # 定义异常点阈值（例如，假设超过3倍标准差为异常点）
# threshold = 3 * std
# # 遍历二维数组，去除异常点
# cleaned_arr = np.where(np.abs(arr - mean) <= threshold, arr, mean)
# I = cleaned_arr

cut_x=720
cut_y=720
I=np.array(I)
I=I[cut_x:-cut_x,cut_y:-cut_y]
print(len(I),len(I[0]))
f = np.fft.fft2(I)
fshift = np.fft.fftshift(f)  # 将0频率分量移动到中心
###计算
central_place=int(len(I)/2)
distance=round(len(I)*pixel/p1/2/np.sqrt(2))
print(central_place,distance)
jiequ_fshift = flybhsdxqyjgb(fshift, central_place, distance)
jiequ_fshift_01 = jiequ_fshift[0]
ifshift_01 = np.fft.ifftshift(jiequ_fshift_01)
if_img_01 = np.fft.ifft2(ifshift_01)
jiequ_fshift_10 = jiequ_fshift[1]
ifshift_10 = np.fft.ifftshift(jiequ_fshift_10)
if_img_10 = np.fft.ifft2(ifshift_10)

noI = Image.open("./grating_00011.tiff")  # 读取图片
noI=np.array(noI)
noI=noI[cut_x:-cut_x,cut_y:-cut_y]
nof = np.fft.fft2(noI)
nofshift = np.fft.fftshift(nof)  # 将0频率分量移动到中心
nojiequ_fshift = flybhsdxqyjgb(nofshift, central_place, distance)
nojiequ_fshift_01 = nojiequ_fshift[0]
noifshift_01 = np.fft.ifftshift(nojiequ_fshift_01)
noif_img_01 = np.fft.ifft2(noifshift_01)
nojiequ_fshift_10 = nojiequ_fshift[1]
noifshift_10 = np.fft.ifftshift(nojiequ_fshift_10)
noif_img_10 = np.fft.ifft2(noifshift_10)

k_phi = p2s / (d * (2 * np.pi))
k2 = central_place / distance * pixel / λ * 2 * np.pi
z01 = (unwrap_phase(np.angle(if_img_01))-unwrap_phase(np.angle(noif_img_01))) * k_phi
z10 = (unwrap_phase(np.angle(if_img_10))-unwrap_phase(np.angle(noif_img_10))) * k_phi
phi = frankotchellappa(z01, z10) * k2
# phi=phi*λ/(2*np.pi)
phi2 = phi * (λ * d) / (2 * np.pi * (pixel) ** 2)
gradient_x = np.gradient(phi2, axis=1) / d
gradient_xx = np.gradient(gradient_x, axis=1)
gradient_xx=gradient_xx[3:-3,3:-3]
curvature_x=1/np.mean(gradient_xx)
print("平均值x：",np.mean(gradient_xx))
print("曲率半径Rx：", curvature_x)

gradient_y = np.gradient(phi2, axis=0) / d
gradient_yy = np.gradient(gradient_y, axis=0)
gradient_yy=gradient_yy[3:-3,3:-3]
curvature_y=1/np.mean(gradient_yy)
print("平均值y：",np.mean(gradient_yy))
print("曲率半径Ry：", curvature_y)
print("比值Ry/Rx：",curvature_y/curvature_x)

plt.figure(num=1)
plt.subplot(221)
plt.imshow(I, cmap='plasma')
plt.title('I')
plt.colorbar()
plt.subplot(222)
plt.imshow(np.abs(fshift), cmap='plasma')
plt.title('fshift')
plt.subplot(223)
plt.imshow(z10, cmap='plasma')
plt.colorbar()
plt.title('z10')
plt.subplot(224)
plt.imshow(z01, cmap='plasma')
plt.colorbar()
plt.title('z01')
plt.figure(num=2)
plt.subplot(221)
plt.imshow(phi, cmap='plasma')
plt.title('phi')
plt.colorbar()
plt.subplot(222)
plt.imshow(gradient_x, cmap='plasma')
plt.title('gradient_x')
plt.subplot(223)
plt.imshow(gradient_xx, cmap='plasma')
plt.title('gradient_xx')
plt.subplot(224)
plt.imshow(gradient_y, cmap='plasma')
plt.title('gradient_y')
plt.show()


