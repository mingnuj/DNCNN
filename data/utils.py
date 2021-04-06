import numpy as np
import torch
from torch.autograd import Variable
import random


def is_image_file(filename):
    img_extension = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in img_extension)


def gaussian_noise(img):
    mean = 0
    stddev = [15, 25, 50]
    sigma = random.choice(stddev)
    noise = Variable(torch.zeros(img.size()))
    noise = noise.data.normal_(mean, sigma/255.)

    return noise


def speckle_noise(img):
    mean = 0
    stddev = [15, 25, 50]
    sigma = random.choice(stddev)
    noise = Variable(torch.zeros(img.size()))
    noise = noise.data.normal_(mean, 1/sigma) * img

    return noise


def salt_and_pepper_noise(img, s_vs_p=0.5, amount=0.05, clip=True):
    noise = img
    # Salt mode
    s_n_p = int(np.ceil(amount * img.size[0] * img.size[1]))
    salt_ratio = int(s_n_p * s_vs_p)
    coords = [np.random.randint(0, i - 1, s_n_p) for i in img.size]

    for i in range(s_n_p):
        if i < salt_ratio:
            noise.putpixel((coords[0][i], coords[1][i]), (255, 255, 255))
        else:
            noise.putpixel((coords[0][i], coords[1][i]), (0, 0, 0))

    return noise

def poisson_noise(img):
    return torch.poisson(img) - img

def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])
