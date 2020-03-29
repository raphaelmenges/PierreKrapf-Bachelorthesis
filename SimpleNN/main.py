from .net import Net
import torch.optim as optim
import torch.nn as nn
import os.path as path
import torchvision.transforms as transforms

# TODO: name of .pth file as arg
PATH = path.join(".", "checkpoints", "simple_net.pth")

if __name__ == "__main__":
    # TODO: proper device handling for GPU
    device = "cpu"
    net = Net()
    net.to(device)
    net.load_state_dict(PATH)
    # TODO: lr as args or depending on Epoch
    optimizer = optim.SGD(net.parameters(), lr=.001, momentum=.9)
    criterion = nn.CrossEntropyLoss()
    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(1),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
    ])

    # TODO: Load data
    # TODO: Train
    # TODO: Checkpoints
