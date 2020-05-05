import numpy as np
import os
import datetime
import torch
import torch.optim as optim
import torch.nn.functional as F
# from tqdm import tqdm
from torchvision import transforms, datasets
from net import Net
from itertools import takewhile


class Training():
    def __init__(self, lr=0.1, momentum=0.9, savepoint_dir="savepoints", sp_serial=-1):
        self.sp_serial = sp_serial
        self.savepoint_dir = savepoint_dir
        self.net = Net()
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'
        self.net.to(self.device)
        print(f"Device :: {self.device}")
        # TODO: dynamic learning rate
        self.optimizer = optim.RMSprop(
            self.net.parameters(), lr=lr, momentum=momentum)
        self.transforms = transforms.Compose([
            transforms.Grayscale(1),
            transforms.RandomAffine(0, translate=(.1, .1)),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,)),
            # TODO: ZCA whitening with: transforms.LinearTransformation()
        ])

        # load savepoints if available
        savepoints = os.listdir(self.savepoint_dir) if os.path.isdir(
            self.savepoint_dir) else []
        if not savepoints == []:
            self._loadSavepoint(savepoints)

        # TODO: Use actual dataset
        # Using CIFAR10 to test
        self.trainset = datasets.CIFAR10(
            os.path.join("drive", "data"), train=True, download=True, transform=self.transforms)
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=4, shuffle=True, num_workers=2)

        self.testset = datasets.CIFAR10(
            os.path.join("drive", "data"), train=False, download=True, transform=self.transforms)
        self.testloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=4, shuffle=False, num_workers=2)

    def run(self, epochs=1):
        print("Starting training!")
        self.net.train()
        for epoch in range(epochs):
            print(f"Epoch {epoch+1} of {epochs}:")
            running_loss = 0.0
            for i, data in enumerate(self.trainloader):
                X, y = data
                if self.device == "cuda":
                    X = X.cuda()
                    y = y.cuda()
                self.optimizer.zero_grad()
                output = self.net(X)
                loss = F.cross_entropy(output, y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i % 2000 == 1999:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
            self._makeSavepoint()
        print("Finished training!")

    def _loadSavepoint(self, savepoints):
        if not os.path.isdir(self.savepoint_dir):
            return
        target_file = None
        ser_files = self._getSavepointList()
        if len(ser_files) == 0:
            print("No existing savepoints!")
            return

        if self.sp_serial > -1:
            for n, f in ser_files:
                if n == self.sp_serial:
                    target_file = f
        else:
            self.sp_serial, target_file = ser_files[-1]

        print(f"Loading progress from {target_file}!")
        self.net.load_state_dict(torch.load(
            os.path.join(self.savepoint_dir, target_file)))
        self.net.eval()

    def _makeSavepoint(self):
        if not os.path.isdir(self.savepoint_dir):
            os.mkdir(self.savepoint_dir)
        target_path = os.path.join(
            self.savepoint_dir, self._getNextSavepointPath())
        print(f"Saving progress in {target_path}!")
        torch.save(self.net.state_dict(), target_path)

    def _getSavepointList(self):
        # only look @ .pt and .pth files
        path_files = [f for f in os.listdir(self.savepoint_dir) if f[-4:]
                      == ".pth" or f[-3:] == ".pt"]
        # parse serial number
        ser_files = [(int(''.join([t for t in takewhile(lambda x: x != '_', f)])), f)
                     for f in path_files]
        # sort in place
        ser_files.sort()
        return ser_files

    def _getNextSavepointPath(self):
        sn = self.sp_serial + 1
        fn = f"{sn}_savepoint.pth"
        current_files = os.listdir(self.savepoint_dir)
        while fn in current_files:
            sn = self.sp_serial + 1
            fn = f"{sn}_savepoint.pth"
        return fn
