# About EfficientNet PyTorch
EfficientNet PyTorch is a PyTorch re-implementation of EfficientNet. It is consistent with the original TensorFlow implementation, such that it is easy to load weights from a TensorFlow checkpoint. At the same time, we aim to make our PyTorch implementation as simple, flexible, and extensible as possible.

# Installation
Install via pip:

pip install efficientnet_pytorch
Or install from source:

git clone https://github.com/lukemelas/EfficientNet-PyTorch
cd EfficientNet-Pytorch
pip install -e .

# Usage
Load an EfficientNet:

from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_name('efficientnet-b3')

Load a pretrained EfficientNet:

from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b3')
