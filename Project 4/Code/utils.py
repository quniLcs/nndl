import torch
from torchvision.transforms import Resize


def pretrain_optimize(augmentor, train_loader, lr, device):
    augmentor.train()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(augmentor.parameters(), lr = lr)

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        mask = torch.rand(inputs.shape[0], 300).to(device)
        mask = (mask > 0.15) * 1

        outputs = augmentor(inputs, mask)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def pretrain_evaluate(augmentor, data_loader, device):
    augmentor.eval()
    criterion = torch.nn.MSELoss()

    count = 0
    losses = 0

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        mask = torch.rand(inputs.shape[0], 300).to(device)
        mask = (mask > 0.15) * 1

        outputs = augmentor(inputs, mask)
        loss = criterion(outputs, targets)

        count += inputs.shape[0]
        losses += loss.item() * inputs.shape[0]

    return losses / count


def train_optimize(classifier, augmentor, train_loader, lr, device):
    classifier.train()
    resizer = Resize((224, 224))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr = lr)

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        if augmentor:
            prob = torch.randint(high = 80, size = (1, )) / 200 + 0.1
            mask = torch.rand(inputs.shape[0], 300).to(device)
            mask = (mask > prob) * 1
            inputs = augmentor(inputs, mask).reshape((inputs.shape[0], 1, 50, 50))

        inputs = resizer(inputs).repeat(1, 3, 1, 1)
        outputs = classifier(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_evaluate(classifier, data_loader, device):
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

        inputs = resizer(inputs).repeat(1, 3, 1, 1)
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


def pretrain(augmentor, train_loader, test_loader, writer, logger, args):
    logger.info('Epoch\tTrain loss\tTest loss')
    best_test_loss = torch.inf
    best_test_loss_epoch = 0

    for idx_epoch in range(args.num_epoch):
        pretrain_optimize(augmentor, train_loader, args.lr, args.device)

        train_loss = pretrain_evaluate(augmentor, train_loader, args.device)
        test_loss = pretrain_evaluate(augmentor, test_loader, args.device)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_test_loss_epoch = idx_epoch + 1
            save_status(augmentor, 'results/augmentor_%s_%d.pth' % (args.file, idx_epoch + 1))

        writer.add_scalars('loss', {'train': train_loss, 'test': test_loss}, idx_epoch)
        logger.info('%3d\t%.5f\t\t%.5f' % (idx_epoch + 1, train_loss, test_loss))

    logger.info('Best test loss is %.5f at epoch %d' % (best_test_loss, best_test_loss_epoch))


def train(classifier, augmentor, train_loader, train_img_loader, test_img_loader, writer, logger, args):
    logger.info('Epoch\tTrain top1\tTrain top5\tTest top1\tTest top5\t')
    best_test_acc_t1 = 0
    best_test_acc_t1_epoch = 0

    for idx_epoch in range(args.num_epoch):
        train_optimize(classifier, augmentor, train_loader, args.lr, args.device)

        train_acc_t1, train_acc_t5, train_loss = train_evaluate(classifier, train_img_loader, args.device)
        test_acc_t1, test_acc_t5, test_loss = train_evaluate(classifier, test_img_loader, args.device)

        if test_acc_t1 > best_test_acc_t1:
            best_test_acc_t1 = test_acc_t1
            best_test_acc_t1_epoch = idx_epoch + 1
            save_status(classifier, 'results/classifier_%s_%d.pth' % (args.file, idx_epoch + 1))

        writer.add_scalars('top1', {'train': train_acc_t1, 'test': test_acc_t1}, idx_epoch)
        writer.add_scalars('top5', {'train': train_acc_t5, 'test': test_acc_t5}, idx_epoch)
        writer.add_scalars('loss', {'train': train_loss, 'test': test_loss}, idx_epoch)

        logger.info('%3d\t%.5f\t\t%.5f\t\t%.5f\t\t%.5f' %
                    (idx_epoch + 1, train_acc_t1, train_acc_t5, test_acc_t1, test_acc_t5))

    logger.info('Best test top1 accuracy is %.5f at epoch %d' % (best_test_acc_t1, best_test_acc_t1_epoch))
