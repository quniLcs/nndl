import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 6, kernel_size = 5, padding = 2)
		self.conv2 = nn.Conv2d(6, 16, kernel_size = 5)
		self.conv3 = nn.Conv2d(16, 120, kernel_size = 5)
		self.fc1 = nn.Linear(120, 84)
		self.fc2 = nn.Linear(84, 10)

	def forward(self, inputs):
		hiddens = F.max_pool2d(F.relu(self.conv1(inputs)), 2)
		hiddens = F.max_pool2d(F.relu(self.conv2(hiddens)), 2)
		hiddens = F.relu(self.conv3(hiddens))
		hiddens = hiddens.view(-1, 120)
		hiddens = F.relu(self.fc1(hiddens))
		outputs = self.fc2(hiddens)
		return outputs