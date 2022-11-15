python main.py --epochs 200 --name ID-CIFAR-10-MadrysResnet-Classical --which-robust-optimization Classical --in-dataset CIFAR-10 --id-num-classes 10 --network-name MadrysResnet
python main.py --epochs 200 --name ID-CIFAR-10-MadrysResnet-Gradient-Quotient --which-robust-optimization ODIN-Optimization --lambda-odin=1e-6 --which-odin-reg grad-over-grad --in-dataset CIFAR-10 --id-num-classes 10 --network-name MadrysResnet
python main.py --epochs 200 --name ID-CIFAR-10-ResNet18-Classical --which-robust-optimization Classical --in-dataset CIFAR-10 --id-num-classes 10 --network-name ResNet18
python main.py --epochs 200 --name ID-CIFAR-10-ResNet18-Gradient-Quotient --which-robust-optimization ODIN-Optimization --lambda-odin=1e-6 --which-odin-reg grad-over-grad --in-dataset CIFAR-10 --id-num-classes 10 --network-name ResNet18
python main.py --epochs 200 --name ID-CIFAR-10-ResNet34-Classical --which-robust-optimization Classical --in-dataset CIFAR-10 --id-num-classes 10 --network-name ResNet34
python main.py --epochs 200 --name ID-CIFAR-10-ResNet34-Gradient-Quotient --which-robust-optimization ODIN-Optimization --lambda-odin=1e-6 --which-odin-reg grad-over-grad --in-dataset CIFAR-10 --id-num-classes 10 --network-name ResNet34


python main.py --epochs 200 --name ID-CIFAR-100-MadrysResnet-Classical --which-robust-optimization Classical --in-dataset CIFAR-100 --id-num-classes 100 --network-name MadrysResnet
python main.py --epochs 200 --name ID-CIFAR-100-MadrysResnet-Gradient-Quotient --which-robust-optimization ODIN-Optimization --lambda-odin=1e-6 --which-odin-reg grad-over-grad --in-dataset CIFAR-100 --id-num-classes 100 --network-name MadrysResnet
python main.py --epochs 200 --name ID-CIFAR-100-ResNet18-Classical --which-robust-optimization Classical --in-dataset CIFAR-100 --id-num-classes 100 --network-name ResNet18
python main.py --epochs 200 --name ID-CIFAR-100-ResNet18-Gradient-Quotient --which-robust-optimization ODIN-Optimization --lambda-odin=1e-6 --which-odin-reg grad-over-grad --in-dataset CIFAR-100 --id-num-classes 100 --network-name ResNet18
python main.py --epochs 200 --name ID-CIFAR-100-ResNet34-Classical --which-robust-optimization Classical --in-dataset CIFAR-100 --id-num-classes 100 --network-name ResNet34
python main.py --epochs 200 --name ID-CIFAR-100-ResNet34-Gradient-Quotient --which-robust-optimization ODIN-Optimization --lambda-odin=1e-6 --which-odin-reg grad-over-grad --in-dataset CIFAR-100 --id-num-classes 100 --network-name ResNet34

python main.py --epochs 200 --name ID-Imagenet30-PretrainedResNet18Imagenet-One-Epoch-Just-GQ --which-robust-optimization ODIN-Optimization --lambda-odin=1e-6 --which-odin-reg one-epoch-gq --in-dataset Imagenet30 --id-num-classes 1000 --network-name PretrainedResNet18Imagenet
python main.py --epochs 200 --name ID-Imagenet30-PretrainedResNet101Imagenet-One-Epoch-Just-GQ --which-robust-optimization ODIN-Optimization --lambda-odin=1e-6 --which-odin-reg one-epoch-gq --in-dataset Imagenet30 --id-num-classes 1000 --network-name PretrainedResNet101Imagenet
