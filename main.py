# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from torchvision import transforms, datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# %%
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 384, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(384, 384, 1)
        self.conv3 = nn.Conv2d(384, 384, 2)
        self.conv4 = nn.Conv2d(384, 640, 2)
        self.conv5 = nn.Conv2d(640, 640, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv6 = nn.Conv2d(640, 640, 1)
        self.conv7 = nn.Conv2d(640, 768, 2)
        self.conv8 = nn.Conv2d(768, 768, 2)
        self.conv9 = nn.Conv2d(768, 768, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv10 = nn.Conv2d(768, 768, 1)
        self.conv11 = nn.Conv2d(768, 896, 2)
        self.conv12 = nn.Conv2d(896, 896, 2)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv13 = nn.Conv2d(896, 896, 3)
        self.conv14 = nn.Conv2d(896, 1024, 2)
        self.conv15 = nn.Conv2d(1024, 1024, 2)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.conv16 = nn.Conv2d(1024, 1024, 1)
        self.conv17 = nn.Conv2d(1024, 1152, 2)
        self.pool6 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1152, 512)
        self.fc2 = nn.Linear(512, 99)

    def forward(self, x):
        x = self.conv1(x)
        print(x.shape)
        x = F.elu(x)
        x = self.pool1(x)
        print(x.shape)
        x = F.dropout2d(x, p=0)
        x = self.conv2(x)
        print(x.shape)
        x = F.elu(x)
        x = self.conv3(x)
        print(x.shape)
        x = F.elu(x)
        x = self.conv4(x)
        print(x.shape)
        x = F.elu(x)
        x = self.conv5(x)
        print(x.shape)
        x = F.elu(x)
        x = self.pool2(x)
        print(x.shape)
        x = F.dropout2d(x, p=0.1)
        x = self.conv6(x)
        print(x.shape)
        x = F.elu(x)
        x = self.conv7(x)
        print(x.shape)
        x = F.elu(x)
        x = self.conv8(x)
        print(x.shape)
        x = F.elu(x)
        x = self.conv9(x)
        print(x.shape)
        x = F.elu(x)
        x = self.pool3(x)
        print(x.shape)
        x = F.dropout2d(x, p=0.2)
        x = self.conv10(x)
        print(x.shape)
        x = F.elu(x)
        x = self.conv11(x)
        print(x.shape)
        x = F.elu(x)
        x = self.conv12(x)
        print(x.shape)
        x = F.elu(x)
        x = self.pool4(x)
        print(x.shape)
        x = F.dropout2d(x, p=0.3)
        x = self.conv13(x)
        print(x.shape)
        x = F.elu(x)
        x = self.conv14(x)
        print(x.shape)
        x = F.elu(x)
        x = self.conv15(x)
        print(x.shape)
        x = F.elu(x)
        x = self.pool5(x)
        print(x.shape)
        x = F.dropout2d(x, p=0.4)
        x = self.conv16(x)
        print(x.shape)
        x = F.elu(x)
        x = self.conv17(x)
        print(x.shape)
        x = F.elu(x)
        x = self.pool6(x)
        print(x.shape)
        x = F.dropout2d(x, p=0.5)
        x = x.view(-1, 1152)  # 1152 might be wrong?
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = F.softmax(x)
        return x


# %%


# %%
train = datasets.CIFAR100("./data", train=True, download=True, transform=transforms.Compose(
    [transforms.Grayscale(), transforms.Pad(4), transforms.ToTensor()]))
test = datasets.CIFAR100("./data", train=False, download=True, transform=transforms.Compose(
    [transforms.Grayscale(), transforms.Pad(4), transforms.ToTensor()]))
print(train[0][0].shape)


# %%
trainset = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)


# %%
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(net.parameters(), lr=0.001, momentum=0.9)
for epoch in range(1):
    for data in trainset:
        X, y = data
        net.zero_grad()
        output = net(X.view(-1, 1, 40, 40))
        # stops @conv 11 with: "RuntimeError: Calculated padded input size per channel: (1 x 1). Kernel size: (2 x 2). Kernel size can't be greater than actual input size"
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
    print(loss)
