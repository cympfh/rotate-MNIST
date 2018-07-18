import random
from torchvision import transforms
from torchvision.transforms import functional


class RandomRotation:

    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img, target = sample
        angle = random.uniform(-self.degree, self.degree)
        img_rot = functional.rotate(img, angle)
        return img_rot, (target, angle)


class Free:

    def __call__(self, sample):
        img, target = sample
        if type(target) == tuple and len(target) == 2:
            target = target[0]
        return img, target


class ToTensor:

    def __call__(self, sample):
        img, target = sample
        x = transforms.ToTensor()(img)
        return x, target


class Normalize:

    def __call__(self, sample):
        x, target = sample
        x = transforms.Normalize((0,), (1,))(x)
        return x, target


class Identity:

    def __call__(self, sample):
        return sample
