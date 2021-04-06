from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from torchvision import datasets, transforms
from admm import ADMM_pruning
from tensorboardX import SummaryWriter



Net = nn.Sequential(
    nn.Conv2d(1, 20, 3, 1, 1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(20, 50, 3, 1, 1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(7*7*50, 300),
    nn.ReLU(),
    nn.Linear(300, 10)
)



def train(args, model, device, train_loader, optimizer, epoch, writer, admm):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.cross_entropy(output, target)
        optimizer.zero_grad()
        loss.backward()
        if epoch >= args.admm_update_start and epoch < args.admm_finetune_start:
            admm.loss_update(loss)
        if epoch >= args.admm_finetune_start:
            admm.grad_mask()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    if epoch >= args.admm_update_start and epoch < args.admm_finetune_start:
        admm.update(epoch)


def test(args, model, device, test_loader, epoch, writer, admm):
    if epoch >= args.admm_update_start and epoch < args.admm_finetune_start:
        admm.apply_projW()

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    if epoch >= args.admm_update_start and epoch < args.admm_finetune_start:
        admm.restoreW()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--admm-update-interval', type=int, default=3, metavar='N',
                        help='update interval of admm iteration')
    parser.add_argument('--admm-update-start', type=int, default=0, metavar='N',
                        help='starting epoch number of admm iteration')
    parser.add_argument('--admm-finetune-start', type=int, default=15, metavar='N',
                        help='starting epoch number of finetune after admm iteration')
    parser.add_argument('--admm-pruning-type', type=int, default=0, metavar='N',
                        help='pruning type: 0/1/2 for normal/channel/filter pruning')
    parser.add_argument('--admm-l1', action='store_true', default=False,
                        help='using l1 norm with admm')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda:2" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    logdir = './summaries/mnist_pruning_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    writer = SummaryWriter(logdir)

    admm = ADMM_pruning(model, update_interval=args.admm_update_interval, l1=args.admm_l1)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, writer, admm)
        test(args, model, device, test_loader, epoch, writer, admm)
        if epoch == args.admm_finetune_start:
            admm.apply_projW()

    total_zero_count = 0
    total_nonzero_count = 0
    for index, m in enumerate(model.modules()):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            print(m)
            zero_count = np.sum( m.weight.cpu().detach().numpy() != 0 )
            nonzero_count = np.sum( m.weight.cpu().detach().numpy() == 0 )
            total_zero_count += zero_count
            total_nonzero_count += nonzero_count
            print(f"Zero count:{zero_count} Non-zero count:{nonzero_count}")
            print(f"Prune ratio: {zero_count / (nonzero_count+zero_count)}")
    print(f"Total zero count: {total_zero_count} Total non-zero count:{total_nonzero_count}")
    print(f"Total prune ratio: {total_zero_count / (total_zero_count + total_nonzero_count)}")

    if (args.save_model):
        torch.save(model.state_dict(), "./tmp/mnist/mnist_prune.pt")


if __name__ == '__main__':
    main()
