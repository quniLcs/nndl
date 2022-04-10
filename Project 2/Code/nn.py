from torch import nn
from util import get_num_parameters


class NN(nn.Module):
    def __init__(self, in_channels = 3, hidden_channels = (16, 32),
                 hidden_neurons = (128, 128), num_classes = 10):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.extractor = nn.Sequential(
            # stage 1
            nn.Conv2d(in_channels = in_channels,
                      out_channels = hidden_channels[0],
                      kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            # stage 2
            nn.Conv2d(in_channels = hidden_channels[0],
                      out_channels = hidden_channels[1],
                      kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels[1] * 8 * 8, hidden_neurons[0]),
            nn.ReLU(),
            nn.Linear(hidden_neurons[0], hidden_neurons[1]),
            nn.ReLU(),
            nn.Linear(hidden_neurons[1], num_classes))

    def forward(self, inputs):
        hidden = self.extractor(inputs)
        outputs = \
            self.classifier(hidden.view(-1,
                                        self.hidden_channels[1] * 8 * 8))
        return outputs


class NN_tanh(nn.Module):
    def __init__(self, in_channels = 3, hidden_channels = (16, 32),
                 hidden_neurons = (128, 128), num_classes = 10):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.extractor = nn.Sequential(
            # stage 1
            nn.Conv2d(in_channels = in_channels,
                      out_channels = hidden_channels[0],
                      kernel_size = 3, padding = 1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            # stage 2
            nn.Conv2d(in_channels = hidden_channels[0],
                      out_channels = hidden_channels[1],
                      kernel_size = 3, padding = 1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels[1] * 8 * 8, hidden_neurons[0]),
            nn.Tanh(),
            nn.Linear(hidden_neurons[0], hidden_neurons[1]),
            nn.Tanh(),
            nn.Linear(hidden_neurons[1], num_classes))

    def forward(self, inputs):
        hidden = self.extractor(inputs)
        outputs = \
            self.classifier(hidden.view(-1,
                                        self.hidden_channels[1] * 8 * 8))
        return outputs


class NN_softplus(nn.Module):
    def __init__(self, in_channels = 3, hidden_channels = (16, 32),
                 hidden_neurons = (128, 128), num_classes = 10):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.extractor = nn.Sequential(
            # stage 1
            nn.Conv2d(in_channels = in_channels,
                      out_channels = hidden_channels[0],
                      kernel_size = 3, padding = 1),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            # stage 2
            nn.Conv2d(in_channels = hidden_channels[0],
                      out_channels = hidden_channels[1],
                      kernel_size = 3, padding = 1),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels[1] * 8 * 8, hidden_neurons[0]),
            nn.Softplus(),
            nn.Linear(hidden_neurons[0], hidden_neurons[1]),
            nn.Softplus(),
            nn.Linear(hidden_neurons[1], num_classes))

    def forward(self, inputs):
        hidden = self.extractor(inputs)
        outputs = \
            self.classifier(hidden.view(-1,
                                        self.hidden_channels[1] * 8 * 8))
        return outputs


if __name__ == '__main__':
    print(get_num_parameters(NN()))
