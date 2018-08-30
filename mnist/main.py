from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import noisy_datasets

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def Sample(model: nn.Module, data: torch.Tensor, target: torch.Tensor, R):
    output = model(data)
    loss = F.nll_loss(output, target, reduce=False)
    return loss.sort()[1][:(int(data.shape[0] * R))]


def train(args, modelf, modelg, device, train_loaderf, train_loaderg, epoch):
    modelg.train()
    modelf.train()
    optimizerf = optim.SGD(modelf.parameters(), lr=args.lr, momentum=args.momentum)
    optimizerg = optim.SGD(modelg.parameters(), lr=args.lr, momentum=args.momentum)
    R = 1 - min(float(epoch-1)/10*0.5,0.5) 
    print("percentage of samples kept: {}".format(R))
    for batch_idx, ((dataf, targetf, noisy_targetf), 
            (datag, targetg, noisy_targetg)) in enumerate(zip(train_loaderf, train_loaderg)):
        dataf, noisy_targetf, targetf = dataf.to(device), noisy_targetf.to(device), \
                                        targetf.to(device)
        datag, noisy_targetg, targetg = datag.to(device), noisy_targetg.to(device), \
                                    targetg.to(device)
        # outputf = modelf(noisy_targetf)
        # outputg = modelg(noisy_targetg)
        # lossf = F.nll_loss(outputf, noisy_targetf, reduce=False)
        # lossg = F.nll_loss(outputg, noisy_targetg, reduce=False)
        optimizerf.zero_grad()
        optimizerg.zero_grad()

        train_f = Sample(modelf, dataf, noisy_targetf, R)
        train_g = Sample(modelg, datag, noisy_targetg, R)

        outputf = modelf(datag[train_g])
        outputg = modelg(dataf[train_f])
    
        lossf = F.nll_loss(outputf, noisy_targetg[train_g])
        lossg = F.nll_loss(outputg, noisy_targetf[train_f])
        lossg.backward()
        lossf.backward()
        
        optimizerf.step()
        optimizerg.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrainLoss: {:.6f}\tTrueLoss: {:.6f}'.format(
                epoch, batch_idx * len(dataf), len(train_loaderf.dataset),
                100. * batch_idx / len(train_loaderf), lossf.item(), 
                F.nll_loss(outputf, targetf[train_f].to(device)).item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loaderf = torch.utils.data.DataLoader(
        noisy_datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    train_loaderg = torch.utils.data.DataLoader(
        noisy_datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]), err_rate=50),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    modelf = Net().to(device)
    modelg = Net().to(device)
    
    for epoch in range(1, args.epochs + 1):
        train(args, modelf, modelg, device, train_loaderf, train_loaderg, epoch)
        test(args, modelf, device, test_loader)


if __name__ == '__main__':
    main()
