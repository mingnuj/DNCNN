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
    noise = noise.data.normal_(mean, 1/sigma)

    noise_img = img+noise
    noise_img = torch.clamp(noise_img, 0, 1)

    return noise_img - img


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


def speckle_noise(img):
    mean = 0
    stddev = [15, 25, 50]
    sigma = random.choice(stddev)
    noise = Variable(torch.zeros(img.size()))
    noise = noise.data.normal_(mean, 1/sigma)

    noise_img = img+noise
    noise_img = torch.clamp(noise_img, 0, 1)

    return noise_img - img


def poisson_noise(img):
    return torch.poisson(img) - img
