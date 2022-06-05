import torch
from torchvision.transforms import Resize


def optimize(classifier, augmentor, train_loader, lr, device):
    classifier.train()
    resizer = Resize((224, 224))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr = lr)

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        if augmentor:
            inputs = augmentor(inputs)
        inputs = resizer(inputs)
        inputs = inputs.repeat(1, 3, 1, 1)
        outputs = classifier(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(classifier, data_loader, device):
    classifier.eval()
    resizer = Resize((224, 224))
    criterion = torch.nn.CrossEntropyLoss()

    count = 0
    losses = 0
    correct_t1 = 0
    correct_t5 = 0

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        inputs = resizer(inputs)
        inputs = inputs.repeat(1, 3, 1, 1)
        outputs = classifier(inputs)
        loss = criterion(outputs, targets)

        count += inputs.shape[0]
        _, pred_t1 = outputs.max(dim = 1)
        _, pred_t5 = outputs.topk(k = 5, dim = 1)
        correct_t1 += pred_t1.eq(targets).sum().item()
        correct_t5 += pred_t5.eq(targets.unsqueeze(dim = 1).repeat(1, 5)).sum().item()
        losses += loss.item() * inputs.shape[0]

    return correct_t1 / count, correct_t5 / count, losses / count


def save_status(model, path):
    save_dict = model.state_dict()
    torch.save(save_dict, path)


def load_status(model, path):
    load_dict = torch.load(path)
    model.load_state_dict(load_dict)


def train(classifier, augmentor, train_loader, train_img_loader, test_img_loader, writer, logger, args):
    logger.info('Epoch\tTrain top1\tTrain top5\tTest top1\tTest top5\t')
    for ind_epoch in range(args.num_epoch):
        optimize(classifier, augmentor, train_loader, args.lr, args.device)

        train_acc_t1, train_acc_t5, train_loss = evaluate(classifier, train_img_loader, args.device)
        test_acc_t1, test_acc_t5, test_loss = evaluate(classifier, test_img_loader, args.device)

        save_status(classifier, 'results/classifier_%s_%d.pth' % (args.file, ind_epoch + 1))

        writer.add_scalars('top1', {'train': train_acc_t1, 'test': test_acc_t1}, ind_epoch)
        writer.add_scalars('top5', {'train': train_acc_t5, 'test': test_acc_t5}, ind_epoch)
        writer.add_scalars('loss', {'train': train_loss, 'test': test_loss}, ind_epoch)

        logger.info('%3d\t%.5f\t\t%.5f\t\t%.5f\t\t%.5f' %
                    (ind_epoch + 1, train_acc_t1, train_acc_t5, test_acc_t1, test_acc_t5))
