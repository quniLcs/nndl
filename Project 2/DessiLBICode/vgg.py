from torch import nn


class VGG(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 10):
        super().__init__()

        self.extractor = nn.Sequential(
            # stage 1
            nn.Conv2d(in_channels = in_channels, out_channels = 64,
                      kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            # stage 2
            nn.Conv2d(in_channels = 64, out_channels = 128,
                      kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            # stage 3
            nn.Conv2d(in_channels = 128, out_channels = 256,
                      kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256, out_channels = 256,
                      kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            # stage 4
            nn.Conv2d(in_channels = 256, out_channels = 512,
                      kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 512, out_channels = 512,
                      kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            # stage5
            nn.Conv2d(in_channels = 512, out_channels = 512,
                      kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 512, out_channels = 512,
                      kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes))

    def forward(self, inputs):
        hidden = self.extractor(inputs)
        outputs = self.classifier(hidden.view(-1, 512 * 1 * 1))
        return outputs


class VGG_BN(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 10):
        super().__init__()

        self.extractor = nn.Sequential(
            # stage 1
            nn.Conv2d(in_channels = in_channels, out_channels = 64,
                      kernel_size = 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            # stage 2
            nn.Conv2d(in_channels = 64, out_channels = 128,
                      kernel_size = 3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            # stage 3
            nn.Conv2d(in_channels = 128, out_channels = 256,
                      kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256, out_channels = 256,
                      kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            # stage 4
            nn.Conv2d(in_channels = 256, out_channels = 512,
                      kernel_size = 3, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels = 512, out_channels = 512,
                      kernel_size = 3, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            # stage5
            nn.Conv2d(in_channels = 512, out_channels = 512,
                      kernel_size = 3, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels = 512, out_channels = 512,
                      kernel_size = 3, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes))

    def forward(self, inputs):
        hidden = self.extractor(inputs)
        outputs = self.classifier(hidden.view(-1, 512 * 1 * 1))
        return outputs


class VGG_Light(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 10):
        super().__init__()

        self.extractor = nn.Sequential(
            # stage 1
            nn.Conv2d(in_channels = in_channels, out_channels = 16,
                      kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            # stage 2
            nn.Conv2d(in_channels = 16, out_channels = 32,
                      kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))

        '''
            # stage 3
            nn.Conv2d(in_channels = 32, out_channels = 64, 
                      kernel_size = 3, padding = 1),
            nn.Conv2d(in_channels = 64, out_channels = 64, 
                      kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            # stage 4
            nn.Conv2d(in_channels = 64, out_channels = 128, 
                      kernel_size = 3, padding = 1),
            nn.Conv2d(in_channels = 128, out_channels = 128, 
                      kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        '''

        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, inputs):
        hidden = self.extractor(inputs)
        outputs = self.classifier(hidden.view(-1, 32 * 8 * 8))
        return outputs


class VGG_Dropout(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 10):
        super().__init__()

        self.extractor = nn.Sequential(
            # stage 1
            nn.Conv2d(in_channels = in_channels, out_channels = 64,
                      kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            # stage 2
            nn.Conv2d(in_channels = 64, out_channels = 128,
                      kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            # stage 3
            nn.Conv2d(in_channels = 128, out_channels = 256,
                      kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256, out_channels = 256,
                      kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            # stage 4
            nn.Conv2d(in_channels = 256, out_channels = 512,
                      kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 512, out_channels = 512,
                      kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            # stage5
            nn.Conv2d(in_channels = 512, out_channels = 512,
                      kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 512, out_channels = 512,
                      kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes))

    def forward(self, inputs):
        hidden = self.extractor(inputs)
        outputs = self.classifier(hidden.view(-1, 512 * 1 * 1))
        return outputs
