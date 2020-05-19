import math
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce

class Net(nn.Module):
    def __init__(self, classes_number):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, classes_number)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

'''
class Net(nn.Module):
    def __init__(self, classes_number):
        super().__init__()

        # Stack 1
        self.stack1_conv1 = nn.Conv2d(1, 384,
                                      kernel_size=3, padding=1)
        # self.stack1 = [self.stack1_conv1]

        # Stack 2
        self.stack2_conv1 = nn.Conv2d(384, 384,
                                      kernel_size=1, padding=0)
        self.stack2_conv2 = nn.Conv2d(384, 384,
                                      kernel_size=2, padding=1)
        self.stack2_conv3 = nn.Conv2d(384, 640,
                                      kernel_size=2, padding=1)
        self.stack2_conv4 = nn.Conv2d(640, 640,
                                      kernel_size=2, padding=1)
        # self.stack2 = [self.stack2_conv1, self.stack2_conv2,
        #                self.stack2_conv3, self.stack2_conv4]

        # Stack 3
        self.stack3_conv1 = nn.Conv2d(640, 640,
                                      kernel_size=1, padding=0)
        self.stack3_conv2 = nn.Conv2d(640, 768,
                                      kernel_size=2, padding=1)
        self.stack3_conv3 = nn.Conv2d(768, 768,
                                      kernel_size=2, padding=1)
        self.stack3_conv4 = nn.Conv2d(768, 768,
                                      kernel_size=2, padding=1)
        # self.stack3 = [self.stack3_conv1, self.stack3_conv2,
        #                self.stack3_conv3, self.stack3_conv4]

        # Stack 4
        self.stack4_conv1 = nn.Conv2d(768, 768,
                                      kernel_size=1, padding=0)
        self.stack4_conv2 = nn.Conv2d(768, 896,
                                      kernel_size=2, padding=1)
        self.stack4_conv3 = nn.Conv2d(896, 896,
                                      kernel_size=2, padding=1)
        # self.stack4 = [self.stack4_conv1, self.stack4_conv2, self.stack4_conv3]

        # Stack 5
        self.stack5_conv1 = nn.Conv2d(896, 896,
                                      kernel_size=3, padding=1)
        self.stack5_conv2 = nn.Conv2d(896, 1024,
                                      kernel_size=2, padding=1)
        self.stack5_conv3 = nn.Conv2d(1024, 1024,
                                      kernel_size=2, padding=1)
        # self.stack5 = [self.stack5_conv1, self.stack5_conv2, self.stack5_conv3]

        # Stack 6
        self.stack6_conv1 = nn.Conv2d(1024, 1024,
                                      kernel_size=1, padding=0)
        self.stack6_conv2 = nn.Conv2d(1024, 1152,
                                      kernel_size=1, padding=0)
        # self.stack6 = [self.stack6_conv1, self.stack6_conv2]

        # Fully connected
        self.dense1 = nn.Linear(1152, 512)
        self.dense2 = nn.Linear(512, classes_number)

        # Max-Pool
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.elu(self.stack1_conv1(x))
        x = self.pool(x)

        x = F.elu(self.stack2_conv1(x))
        x = F.elu(self.stack2_conv2(x))
        x = F.elu(self.stack2_conv3(x))
        x = F.elu(self.stack2_conv4(x))
        x = self.pool(x)
        x = F.dropout2d(x, 0.1)

        x = F.elu(self.stack3_conv1(x))
        x = F.elu(self.stack3_conv2(x))
        x = F.elu(self.stack3_conv3(x))
        x = F.elu(self.stack3_conv4(x))
        x = self.pool(x)
        x = F.dropout2d(x, 0.2)

        x = F.elu(self.stack4_conv1(x))
        x = F.elu(self.stack4_conv2(x))
        x = F.elu(self.stack4_conv3(x))
        x = self.pool(x)
        x = F.dropout2d(x, 0.3)

        x = F.elu(self.stack5_conv1(x))
        x = F.elu(self.stack5_conv2(x))
        x = F.elu(self.stack5_conv3(x))
        x = self.pool(x)
        x = F.dropout2d(x, 0.4)

        x = F.elu(self.stack6_conv1(x))
        x = F.elu(self.stack6_conv2(x))
        x = self.pool(x)
        x = F.dropout2d(x, 0.5)

        x = x.view(x.size()[0], -1)
        x = F.elu(self.dense1(x))
        x = self.dense2(x)
        # print(x)
        # x = F.softmax(x, dim=0)

        return x

'''