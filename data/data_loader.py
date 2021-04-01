from PIL import Image
import cv2
import os
import torch.utils.data as data
from torch.utils.data import DataLoader
from data.transform import CustomTransform

class ImageLoader(data.Dataset):
    def __init__(self, file_path, size=180, mode="train", noise="gaussian"):
        self.file_path = os.path.join(file_path, mode)
        self.noise = noise
        self.transforms = CustomTransform(size, noise)
        self.files = []
        for image in os.listdir(self.file_path):
            self.files.append(os.path.join(self.file_path, image))
        print(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = Image.open(self.files[index])

        gt, noise = self.transforms(image)

        return gt, noise
