from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load(dataset = 'MNIST', root = '../Data/', train = True, batch_size = 1, shuffle = True):
	if dataset == 'MNIST':
		data_loader = DataLoader(datasets.MNIST(root = root, train = train, download = True,
												transform = transforms.ToTensor()),
								 batch_size = batch_size, shuffle = shuffle)
	else:
		data_loader = None
	return data_loader
