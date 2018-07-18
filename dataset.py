import torchvision


class MNIST(torchvision.datasets.MNIST):

    def __init__(self, root, train=True, transform=None, download=False):
        super().__init__(root, train=train, transform=None, target_transform=None, download=download)
        self.sample_transform = transform

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if self.sample_transform is not None:
            sample = self.sample_transform(sample)
        return sample
