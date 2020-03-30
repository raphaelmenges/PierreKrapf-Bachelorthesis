from net import Net
import os
import torch
import torch.optim as optim
import torch.nn as nn
import os.path as path
import torchvision.transforms as transforms
from data import SimpleDataset

# TODO: name of .pth file as arg
CHECKPOINT_PATH = path.join("checkpoints", "simple_net.pth")
CLASSES = ("arrow_backward", "more", "menu", "avatar", "search",
           "star", "arrow_forward", "close", "add", "play")
DATA_DIR = os.path.join("G:", "BA", "Data", "Out")

if __name__ == "__main__":
    # TODO: proper device handling for GPU
    device = "cpu"
    net = Net()
    net.to(device)
    if os.path.isfile(CHECKPOINT_PATH):
        net.load_state_dict(CHECKPOINT_PATH)
    # TODO: lr as args or depending on Epoch
    optimizer = optim.SGD(net.parameters(), lr=.01, momentum=.9)
    criterion = nn.CrossEntropyLoss()
    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.RandomAffine(0, translate=(.1, .1)),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
        # ZCA Whiten TODO: figure out what to pass to this function
        # transforms.LinearTransformation()
    ])

    # TODO: Load data
    dataset = SimpleDataset(DATA_DIR, transform=transform, classes=CLASSES)
    dataset.showOne()
    # TODO: Train
    # TODO: Checkpoints
