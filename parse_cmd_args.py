"""Parse Command-line arguments."""
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Robust OOD Detection')
    parser.add_argument('--gpu', default='0', type=str, help='which gpu to use')
    parser.add_argument('--data', default='./data', type=str,
                        help='datasets root folder')
    parser.add_argument('--which-robust-optimization',
                        default="None", type=str,
                        choices=['Classical', 'ODIN-Optimization'],
                        help='Which robust optimization to use: '
                             '[Classical,'
                             'ODIN-Optimization,]')
    parser.add_argument('--optimizer-name', type=str, default='SGD',
                        choices=['SGD', 'Adam'],
                        help='which optimizer to use for training')

    parser.add_argument('--in-dataset', default="CIFAR-10", type=str,
                        choices=['CIFAR-10', 'CIFAR-100', 'Imagenet30'],
                        help='in-distribution dataset')
    parser.add_argument('--id-num-classes', default=10, type=int,
                        help='The number of classes in the In-Distribution '
                             'dataset')
    parser.add_argument('--network-name', default='MadrysResnet', type=str,
                        choices=['MadrysResnet', 'ResNet18', 'ResNet34',
                                 'PretrainedResNet18Imagenet',
                                 'PretrainedResNet101Imagenet'],
                        help='Classifier Model name')
    parser.add_argument('--load-network-checkpoint', action='store_true',
                        help='Do you want to load a classifier checkpoint')
    parser.add_argument('--network-checkpoint-path',
                        default='./checkpoint/madrys_classifier.pth',
                        type=str,
                        help='Do you want to load a classifier checkpoint')

    parser.add_argument('--which-odin-reg', default='grad-over-grad', type=str,
                        choices=['grad-over-grad',
                                 'grad-over-grad-to-softmax',
                                 'one-epoch-gq'],
                        help='Which ODIN regularization to use')

    parser.add_argument('--lambda-odin', default=1.0, type=float,
                        help='lambda which scales the ODIN part compared to '
                             'CE loss.')

    parser.add_argument('--epochs', default=100, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--save-epoch', default=10, type=int,
                        help='save the model every save_epoch')
    parser.add_argument('--save-checkpoint-every-epoch', action='store_true',
                        help='if passed, saves the model every epoch')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        help='mini-batch size (default: 64)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--lr-scheduler', default='cosine_annealing', help='learning rate scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0.0001, type=float,
                        help='weight decay (default: 0.0001)')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--name', required=True, type=str,
                        help='name of experiment')
    parser.add_argument('--tensorboard',
                        help='Log progress to TensorBoard', action='store_true')

    args = parser.parse_args()
    return args

