import os
from torchvision import transforms, datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
from tqdm import tqdm
from net import Net
from dataset import SimpleDataset
from training import Training


def run():
    tr = Training(savepoint_dir=os.path.join(
        "drive", "My Drive", "savepoints"))
    tr.run(epochs=10)


def main():
    run()


if __name__ == "__main__":
    main()
