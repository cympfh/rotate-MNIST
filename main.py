import click
import torch
import torch.optim as optim
from torch.autograd import Variable

import dataset
import model
import torchvision
import transforms


def pack(sample, learn_angle):
    x, y = sample
    if torch.cuda.is_available():
        x = x.cuda()
        if learn_angle:
            y_label = y[0].cuda()
            y_angle = y[1].cuda().float().reshape((-1, 1)) / 30
            y = [y_label, y_angle]
        else:
            y = y.cuda()
    else:
        y[1] = y[1].float().reshape((-1, 1)) / 30
    return x, y


@click.command()
@click.option('--epochs', type=int, default=20)
@click.option('--batch-size', type=int, default=30)
@click.option('--learn-angle', type=bool, required=True)
@click.option('--angle-lr', type=float, default=0.1)
def main(epochs, batch_size, learn_angle, angle_lr):

    transform = torchvision.transforms.Compose([
        transforms.RandomRotation(30),
        transforms.Identity() if learn_angle else transforms.Free(),
        transforms.ToTensor(),
        transforms.Normalize()
    ])

    set_train = dataset.MNIST(root='./data', train=True, download=True,
                              transform=transform)
    set_test = dataset.MNIST(root='./data', train=False, download=True,
                             transform=transform)
    loader_train = torch.utils.data.DataLoader(set_train, batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    loader_test = torch.utils.data.DataLoader(set_test, batch_size=batch_size,
                                              shuffle=False, num_workers=2)

    nn = model.Net(learn_angle)
    if torch.cuda.is_available():
        nn.cuda()
    optimizer = optim.SGD(nn.parameters(), lr=0.001, momentum=0.9)

    VIEW_INTERVAL = 100
    for epoch in range(epochs):
        acc_loss = 0.0
        running_loss = 0.0
        for i, sample in enumerate(loader_train):

            x, y = pack(sample, learn_angle)

            optimizer.zero_grad()
            y_pred = nn(x)
            loss = model.loss(y_pred, y, angle_lr)
            loss.backward()
            optimizer.step()

            # report loss
            acc_loss += loss.item()
            if i % VIEW_INTERVAL == VIEW_INTERVAL - 1:
                running_loss = acc_loss / VIEW_INTERVAL
                click.secho(
                    f"\rEpoch {epoch+1}, iteration {i+1}; "
                    f"loss: {(running_loss):.3f}; ",
                    err=True, nl=False)
                acc_loss = 0.0

        # testing
        count_correct = 0
        count_total = 0
        for sample in loader_test:
            x, labels = pack(sample, learn_angle)
            y_pred = nn(Variable(x))
            if learn_angle:
                labels = labels[0]
                y_pred = y_pred[0]
            _, labels_pred = torch.max(y_pred.data, 1)
            c = (labels_pred == labels).squeeze()
            count_correct += c.sum().item()
            count_total += len(c)

        click.secho(
            f"\rEpoch {epoch+1}; loss: {(running_loss):.3f}; "
            f"Test Acc: {100.0 * count_correct / count_total :.2f}%",
            err=True, nl=False)
        running_loss = 0

        click.secho('', err=True)


if __name__ == '__main__':
    main()
