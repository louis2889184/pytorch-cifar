'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar, set_seed
from fisher import create_all_ones_mask, create_mask_gradient_list
from copy import deepcopy


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--mask_method', default="label-square", type=str)
parser.add_argument('--num_samples', default=1024, type=int)
parser.add_argument('--update_mask_epochs', default=500, type=int)
parser.add_argument('--save_file', default="default_accs.bin", type=str)
parser.add_argument('--max_epoch', default=100, type=int)
parser.add_argument('--merge_steps', default=10, type=int)
parser.add_argument('--recalculate_interval', default=10, type=int)
parser.add_argument('--split', default=2, type=int)
parser.add_argument('--diff_aggr_method', default="sum", type=str)

args = parser.parse_args()


sample_type, grad_type = args.mask_method.split("-")

set_seed(args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = ResNet50()


class Trainer:

    def __init__(self, args, net, mask, trainset, testset):
        self.args = args
        self.net = net

        self.mask = mask

        self.trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2)

        self.testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(net.parameters(), lr=args.lr,
                            momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.args.max_epoch
        )

        self.global_steps = 0

    def set_mask(self, mask):
        self.mask = mask

    def set_weight(self, weight):
        for name, params in self.net.named_parameters():
            device = params.device
            params.data.copy_(weight[name].data)

    def train(self, epoch):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0

        # if epoch % self.args.update_mask_epochs == 0:
        #     print("update mask...")
        #     mask = create_mask_gradient(net, trainset, args.num_samples, 0.005, sample_type, grad_type)

        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()

            if self.mask is not None:
                for name, params in self.net.named_parameters():
                    self.mask[name] = self.mask[name].to(device)

                    params.grad.data.copy_(params.grad.data * self.mask[name].data)

            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

            self.global_steps += 1

            if self.global_steps % self.args.merge_steps == 0:
                yield 100.*correct/total

        self.scheduler.step()

        yield 100.*correct/total

    def test(self, epoch):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(self.testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Save checkpoint.
        acc = 100.*correct/total

        return acc


def merge_models(models, merged_model, diff_aggr_method, same_classifier):
    # save the weight difference
    weight_diff = {}

    for model in models:
        diff = {}
        for n, p in model.named_parameters():
            # the classifiers have different shapes
            if "classifier" in n and not same_classifier:
                continue
            pretrained_p = merged_model.state_dict()[n]
            diff[n] = p - pretrained_p

        if len(weight_diff) == 0:
            weight_diff.update(diff)
        else:
            for n, p in diff.items():
                weight_diff[n] += diff[n]

    if diff_aggr_method == 'mean':
        for n, p in weight_diff.items():
            weight_diff[n] = weight_diff[n] / len(models)

    for n, p in merged_model.named_parameters():
        if n not in weight_diff:
            continue

        diff_p = weight_diff[n]

        p.data.copy_(p.data + diff_p.data)


net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

merged_model = net

merged_trainer = Trainer(args, merged_model, None, trainset, testset)

masks = create_mask_gradient_list(
    merged_model, 
    trainset, 
    args.num_samples, 
    0.005, 
    sample_type, 
    grad_type,
    split=args.split
)

trainers = []

for i in range(args.split):
    trainer = Trainer(args, deepcopy(merged_model), masks[i], trainset, testset)
    trainers.append(trainer)

train_accs = []
test_accs = []

step = 1

for epoch in range(start_epoch, start_epoch+args.max_epoch):
    train_jobs = [trainer.train(epoch) for trainer in trainers]
    for train_results in zip(*train_jobs):
        # extract models
        models = [trainer.net for trainer in trainers]

        # merge models
        merge_models(
            models, 
            merged_model, 
            args.diff_aggr_method, 
            True
        )

        if step % args.recalculate_interval == 0:
            # re-calulate masks
            masks = create_mask_gradient_list(
                merged_model, 
                trainset, 
                args.num_samples, 
                0.005, 
                sample_type, 
                grad_type,
                split=args.split
            )

            # re-assign the masks
            for trainer, mask in zip(trainers, masks):
                trainer.set_mask(mask)

        # re-assign weights
        for trainer in trainers:
            trainer.set_weight(merged_model.state_dict())

        step += 1
        
    test_accs.append(merged_trainer.test(epoch))

torch.save({"train_accs": train_accs, "test_accs": test_accs}, args.save_file)
