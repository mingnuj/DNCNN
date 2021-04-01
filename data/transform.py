from torchvision import transforms
from data.utils import gaussian_noise, speckle_noise


class CustomTransform(object):
    def __init__(self, size, mode=None):
        self.size = size
        self.mode = mode
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.size, self.size), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        if self.mode == "gaussian":
            self.transform = transforms.Compose([
                transforms.Resize((self.size, self.size), interpolation=2),
                transforms.ToTensor(),
                transforms.Lambda(gaussian_noise),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        elif self.mode == "speckle":
            self.transform = transforms.Compose([
                transforms.Resize((self.size, self.size), interpolation=2),
                transforms.Lambda(speckle_noise),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.size, self.size), interpolation=2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __call__(self, image):
        return self.gt_transform(image), self.transform(image)
