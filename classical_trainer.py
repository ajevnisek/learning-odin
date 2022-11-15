"""Classical Neural Network Trainer."""
import os
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

from utils import get_logger
from resnets import ResNet18, ResNet34
from madrys_resnet import MadrysResnet
from imagenet_to_imagenet30 import imagenet30_classes_to_imagenet_classes


TEXT_LOGS = 'text-logs'
ARGUMENTS_LOGS = 'arguments-logs'
CHECKPOINTS_ROOT = 'checkpoints'
TENSORBOARD_LOGS = 'tensorboard-logs'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


class ClassicalTrainer:
    """Train and evaluate Resnet network classically."""
    def __init__(self, args):
        self.arguments = args
        self._create_artifact_folders()
        self.store_arguments_in_json()
        self.id_dataset_train_loader, self.id_dataset_test_loader = \
            self._prepare_train_and_test_dataset_loaders(args.in_dataset,
                                                         args.batch_size)
        self.net = self._load_network(args.network_name, args.id_num_classes,
                                      args.load_network_checkpoint,
                                      args.network_checkpoint_path)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs)

        self.logger = get_logger(self.logfile_path)
        self.tb_writer = SummaryWriter(log_dir=self.tensorboard_log_dir)
        self.best_accuracy = 0
        self.current_train_accuracy = 0
        self.current_test_accuracy = 0
        self.train_loss = 0
        self.test_loss = 0

    def _get_optimizer(self):
        if self.arguments.optimizer_name == 'SGD':
            return optim.SGD(self.net.parameters(), lr=self.arguments.lr,
                             momentum=0.9, weight_decay=5e-4)
        elif self.arguments.optimizer_name == 'Adam':
            return optim.Adam(self.net.parameters(), lr=self.arguments.lr,)
        else:
            return optim.SGD(self.net.parameters(), lr=self.arguments.lr,
                             momentum=0.9, weight_decay=5e-4)

    def _create_artifact_folders(self):
        datetime = time.strftime("%Y_%m_%d_%H_%M_%S")
        experiment_name = f"ID_{self.arguments.in_dataset}_" \
                          f"model_{self.arguments.network_name}"
        full_experiment_name = f"{datetime}_{experiment_name}"
        self.checkpoint_path = os.path.join(CHECKPOINTS_ROOT,
                                            full_experiment_name)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        self.tensorboard_log_dir = os.path.join(TENSORBOARD_LOGS,
                                                full_experiment_name)
        os.makedirs(self.tensorboard_log_dir, exist_ok=True)
        self.logfile_path = os.path.join(TEXT_LOGS, full_experiment_name,
                                         'trainer_log.log')
        os.makedirs(os.path.dirname(self.logfile_path), exist_ok=True)
        self.arguments_log_path = os.path.join(ARGUMENTS_LOGS,
                                               full_experiment_name,
                                               'arguments.json')
        os.makedirs(os.path.dirname(self.arguments_log_path), exist_ok=True)

    def store_arguments_in_json(self):
        with open(self.arguments_log_path, 'w') as f:
            json.dump(vars(self.arguments), f, indent=4)

    @staticmethod
    def _prepare_train_and_test_dataset_loaders(
            dataset_name: str = 'CIFAR-10', batch_size: int = 128) -> tuple:
        """Return dataset loaders for train and test loaders."""
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        if dataset_name == 'CIFAR-10':
            trainset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True,
                transform=transform_train)
            testset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True,
                transform=transform_test)
        elif dataset_name == 'CIFAR-100':
            trainset = torchvision.datasets.CIFAR100(
                root='./data', train=True, download=True,
                transform=transform_train)
            testset = torchvision.datasets.CIFAR100(
                root='./data', train=False, download=True,
                transform=transform_test)
        elif dataset_name == 'Imagenet30':
            transform_train = transforms.Compose([
                transforms.Resize(254),
                transforms.RandomCrop(254, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            transform_test = transforms.Compose([
                transforms.Resize((254, 254)),
                transforms.ToTensor(),
            ])
            trainset = torchvision.datasets.ImageFolder(
                "data/ImageNet-30/one_class_train",
                transform=transform_train)
            testset = torchvision.datasets.ImageFolder(
                "data/ImageNet-30/one_class_test",
                transform=transform_test)

        else:
            trainset = torchvision.datasets.CIFAR100(
                root='./data', train=True, download=True,
                transform=transform_train)
            testset = torchvision.datasets.CIFAR100(
                root='./data', train=False, download=True,
                transform=transform_test)

        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=8)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=8)
        return train_loader, test_loader

    @staticmethod
    def _load_network(network_name: str = 'MadrysResnet',
                      num_classes: int = 10,
                      is_load_net_weights: bool = False,
                      network_weights_path: str = 'path/to/classifier.pth')\
            -> nn.Module:
        """Return pytorch model."""
        if network_name == 'MadrysResnet':
            net = MadrysResnet(num_classes).to(device)
        elif network_name == 'ResNet18':
            net = ResNet18(num_classes).to(device)
        elif network_name == 'ResNet34':
            net = ResNet34(num_classes).to(device)
        elif network_name == 'PretrainedResNet18Imagenet':
            import torchvision.models as models
            net = models.resnet18(pretrained=True)
        elif network_name == 'PretrainedResNet101Imagenet':
            import torchvision.models as models
            net = models.resnet101(pretrained=True)
        else:
            assert False, f"{network_name} not supported"
        if is_load_net_weights:
            net.load_state_dict(torch.load(network_weights_path)['net'])
        net = net.to(device)
        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True
        return net

    def _load_model_checkpoint(self, checkpoint_path: str =
                               './checkpoint/madrys_classifier.pth'):
        """Load model checkpoint from path."""
        assert os.path.isdir(
            'checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(checkpoint_path)
        self.net.load_state_dict(checkpoint['net'])

    def save_checkpoint_if_better(self, epoch):
        """Save model checkpoint along with the epoch and train accuracy."""
        if self.current_train_accuracy > self.best_accuracy:
            state = {
                'net': self.net.state_dict(),
                'train_accuracy': self.current_train_accuracy,
                'test_accuracy': self.current_test_accuracy,
                'epoch': epoch,
            }
            torch.save(state, os.path.join(self.checkpoint_path,
                                           'classifier.pth'))
            self.best_accuracy = self.current_train_accuracy

    def save_checkpoint_every_epoch(self, epoch):
        """Save model checkpoint along with the epoch and train accuracy."""
        state = {
            'net': self.net.state_dict(),
            'train_accuracy': self.current_train_accuracy,
            'test_accuracy': self.current_test_accuracy,
            'epoch': epoch,
        }
        torch.save(state,
                   os.path.join(self.checkpoint_path,
                                f'epoch_{epoch}_classifier.pth'))

    def train_one_epoch(self, epoch):
        """Train Resnet network classically."""
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(
                self.id_dataset_train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        nof_batches = len(self.id_dataset_train_loader)
        self.train_loss = train_loss / nof_batches
        self.print_and_log_train_statistics(epoch, correct, total,
                                            self.train_loss,
                                            self.current_train_accuracy)

    def print_and_log_train_statistics(self, epoch, correctly_classified,
                                       total_nof_samples, train_loss,
                                       train_accuracy, ):
        self.current_train_accuracy = (100. * correctly_classified /
                                       total_nof_samples)
        self.tb_writer.add_scalar('Loss/train', train_loss, epoch)
        self.tb_writer.add_scalar('Accuracy/train',
                                  train_accuracy, epoch)
        message = (f"[{epoch}] | "
                   f"Train Loss: {train_loss:.3f} | "
                   f"Train Accuracy: {self.current_train_accuracy:.2f} [%] "
                   f"({correctly_classified}/{total_nof_samples})")
        print(message)
        self.logger.debug(message)

    def test_one_epoch(self, epoch):
        """Evaluate Resnet network classically."""
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(
                    self.id_dataset_test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                if self.arguments.in_dataset == 'Imagenet30' and \
                        'Pretrained' in self.arguments.network_name:
                    targets = torch.tensor([
                        imagenet30_classes_to_imagenet_classes[x.item()] for x
                        in
                        targets]).to(device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        nof_batches = len(self.id_dataset_test_loader)
        self.test_loss = test_loss / nof_batches
        self.current_test_accuracy = 100. * correct / total
        self.tb_writer.add_scalar('Loss/test', self.test_loss, epoch)
        self.tb_writer.add_scalar('Accuracy/test',
                                  self.current_test_accuracy, epoch)

        message = (f"[{epoch}] | "
                   f"Test Loss: {test_loss:.3f} | "
                   f"Test Accuracy: {self.current_test_accuracy:.2f} [%] "
                   f"({correct}/{total})")
        print(message)
        self.logger.debug(message)

    def run_trainer(self):
        """Run train and evaluation phases for number of epochs."""
        for epoch in range(1, 1 + self.arguments.epochs):
            self.train_one_epoch(epoch)
            self.test_one_epoch(epoch)
            self.scheduler.step()
            self.save_checkpoint_if_better(epoch)
            if self.arguments.save_checkpoint_every_epoch:
                self.save_checkpoint_every_epoch(epoch)
