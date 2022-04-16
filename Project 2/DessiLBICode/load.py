from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load(dataset = 'MNIST', root = '../Data/', train = True, batch_size = 1, shuffle = True):
	if dataset == 'MNIST':
		data_loader = DataLoader(datasets.MNIST(root = root, train = train, download = True,
												transform = transforms.ToTensor()),
								 batch_size = batch_size, shuffle = shuffle)
	elif dataset == 'CIFAR-10':
		transform = transforms.Compose([transforms.ToTensor(),
										transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])])
		data_loader = DataLoader(datasets.CIFAR10(root = root, train = train, download = True, transform = transform),
								 batch_size = batch_size, shuffle = shuffle)
	else:
		data_loader = None
	return data_loader
