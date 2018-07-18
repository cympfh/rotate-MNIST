from torch import nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, learn_angle):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5, stride=2)
        self.conv2 = nn.Conv2d(8, 16, 5, stride=2)
        self.lin = nn.Linear(256, 10)
        if learn_angle:
            self.angle = nn.Linear(256, 1)
        else:
            self.angle = None

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = x.view(-1, 256)
        y = self.lin(x)
        if self.angle:
            a = F.tanh(self.angle(x))
            return y, a
        else:
            return y


def loss(input, target, lr):

    if type(target) == list and len(target) == 2:  # when learn_angle=True
        label, angle = target
        return F.cross_entropy(input[0], label) + F.mse_loss(input[1], angle) * lr

    return F.cross_entropy(input, target)
