import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from vgg import VGG

def generate_noise_data(label: torch.Tensor, eta: int, method):
    '''
    eta是错误的概率
    method是论文中使用的两种制造噪声的方法
    '''
    if method == 'pair':
        sta = torch.randint_like(label, 1, 101)
        ret = torch.empty_like(label)
        ret.data = label.clone()
        ret[sta < eta] += 1
        ret[ret > label.max()] = label.min()
        return ret
    elif method == 'symmetric' or method == 'symmetry':
        cnt = ((label.max() - label.min()).float() / (float(eta)/100)).int()
        sta = torch.randint_like(label, 0, cnt)
        ret = torch.empty_like(label)
        ret.data = label.clone()
        ret[sta < label.max()] = sta[sta < label.max()]
        return ret
    else: assert(0)

class NOISYCIFAR10(datasets.CIFAR100):
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                download=False, **kargs):
        datasets.CIFAR10.__init__(self, root, train, 
                                  transform, target_transform, 
                                  download)
        self.err_rate = kargs.get('err_rate', 45)
        self.method = kargs.get('method', 'pair')
        if self.train:
            self.noisy_target = generate_noise_data(torch.tensor(self.train_labels),
                                                    self.err_rate, self.method) 
    def __getitem__(self, index):
        ret = datasets.CIFAR10.__getitem__(self, index)
        if self.train:
            ret = ret + (self.noisy_target[index], )
        return ret

def Sample(model: nn.Module, data: torch.Tensor, target: torch.Tensor, R):
    output = model(data)
    loss = F.nll_loss(output, target, reduce=False)
    return loss.sort()[1][:(int(data.shape[0] * R))]

def train(args, modelf, modelg, device, train_loaderf, train_loaderg, epoch):
    modelg.train()
    modelf.train()
    optimizerf = optim.SGD(modelf.parameters(), lr=args.lr, momentum=args.momentum)
    optimizerg = optim.SGD(modelg.parameters(), lr=args.lr, momentum=args.momentum)
    tao = 1 - 0.5#eta
    R = 1 - min(float(epoch-1)/10*tao, tao) 
    print("percentage of samples kept: {}".format(R))
    for batch_idx, ((dataf, targetf, noisy_targetf), 
            (datag, targetg, noisy_targetg)) in enumerate(zip(train_loaderf, train_loaderg)):
        dataf, noisy_targetf, targetf = dataf.to(device), noisy_targetf.to(device), \
                                        targetf.to(device)
        datag, noisy_targetg, targetg = datag.to(device), noisy_targetg.to(device), \
                                    targetg.to(device)

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
                F.nll_loss(outputf, targetg[train_g].to(device)).item()))
            print("\t\t\tTrainLoss: {:.6f}\tTrueLoss: {:.6f}".format(lossg.item(), 
                F.nll_loss(outputg, targetf[train_f].to(device)).item()))

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
    return 100. * correct / len(test_loader.dataset)


def main():
    parser = argparse.ArgumentParser(description = 'Pytorch CIFAR10 Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training(default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing(default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                            help='number of epochs to train (default: 20)')
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
    parser.add_argument('--err_rate', type=int, default=50,
                            help='the percentage of noisy data')
    parser.add_argument('--method', type=str, default='symmetry',
                            help='how the noisy data is generated')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()


    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    cifarkargs = {'err_rate': args.err_rate, 'method': args.method}
    
    train_loaderf = torch.utils.data.DataLoader(
                    NOISYCIFAR10('../data/CIFAR100', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]), **cifarkargs),
                    batch_size=args.batch_size, shuffle=True, **kwargs)
    
    train_loaderg = torch.utils.data.DataLoader(
                    NOISYCIFAR10('../data/CIFAR100', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]), **cifarkargs),
            batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
            NOISYCIFAR10('../data/CIFAR100', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    modelf = VGG('VGG11').to(device)
    modelg = VGG('VGG11').to(device)

    errf = []
    errg = []
    for epoch in range(1, args.epochs + 1):
        train(args, modelf, modelg, device, train_loaderf, train_loaderg, epoch)
        errf.append(test(args, modelf, device, test_loader))
        errg.append(test(args, modelg, device, test_loader))
    print(errf)
    print(errg)


if __name__ == '__main__':
    main()
