import torch
from torch import nn


class Lenet5(nn.Module):
    """
    for cifar 10 dataset
    """

    def __init__(self):
        super(Lenet5, self).__init__()

        self.model = nn.Sequential(
            # x: [b, 3, 32, 32] => [b, 6, 28, 28]
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0), # Feature map: 6, kernel: 5 * 5
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0), #  kenel: 2 * 2
            #
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            #
            nn.Flatten(),
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )


    def forward(self, x):
        """

        :param x: [b, 3, 32, 32]
        :return:
        """
        logits = self.model(x)
        return logits


def main():
    net = Lenet5()

    tmp = torch.randn(2, 3, 32, 32)
    out = net(tmp)
    print('out', out.shape)


if __name__ == '__main__':
    main()