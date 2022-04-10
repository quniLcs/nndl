import torch
from torch import nn

from nn import NN
from load import load
from util import set_random_seeds, train


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    set_random_seeds(seed = 0, device = device)

    lr = 0.001
    num_epochs = 1
    model = NN()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()
    train_loader = load(train = True)
    test_loader = load(train = False)

    losses, train_error, test_error = \
        train(model, optimizer, criterion, train_loader, test_loader, num_epochs = num_epochs, device = device,
              wrap_tqdms = True, print_errors = True,
              best_model_file = 'model.pt', losses_file = 'losses.pt',
              train_errors_file = 'train_errors.pt', test_errors_file = 'test_errors.pt')
