from helper import isPng, filenameToLabel
from torch.utils.data.dataset import Dataset
import torch
import torch.nn.functional as F
import os
import csv
import re
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
import random


def find(arr, el):
    for i, e in enumerate(arr):
        if e == el:
            return i
    return None


class SimpleDataset(Dataset):
    def __init__(self, root_dir, transform=None, labels="data.csv", classes=None):
        super().__init__()

        self.root_dir = root_dir
        self.transform = transform if transform != None else transforms.ToTensor()
        self.labels = labels
        self.csv_path = os.path.join(root_dir, labels)
        self.classes = classes

        if not os.path.isfile(self.csv_path):
            self.genCSV()

        df = pd.read_csv(self.csv_path)
        self.data_info = df if self.classes == None else df[df.label.isin(
            self.classes)]
        self.image_arr = np.asarray(self.data_info.iloc[:, 1])
        self.label_arr = self.data_info.iloc[:, 2]
        if not self.classes:
            self.classes = np.unique(self.label_arr)
        self.data_len = len(self.data_info.nr)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        image_name = self.image_arr[index]
        label = self.label_arr[index]
        img = Image.open(os.path.join(self.root_dir, image_name))
        img_tensor = self.transform(img)
        i = find(self.classes, label)
        return (img_tensor, i)

    def genCSV(self):
        print("Generating csv")
        files = os.listdir(self.root_dir)
        pngs = filter(isPng, files)
        with open(self.csv_path, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["nr", "file_name", "label"])
            for (nr, file_name) in tqdm(enumerate(pngs)):
                label = filenameToLabel(file_name)
                if label == None:
                    print(f"Faulty file name {file_name}.")
                else:
                    writer.writerow([nr, file_name, label])

    def showOne(self, index=None):
        index = index if index != None else random.randint(0, self.data_len-1)
        (image, label) = self[index]
        # print(f"{label}: {image}")

        plt.figure(self.image_arr[index])
        if image.shape[0] == 1:
            plt.imshow(transforms.ToPILImage()(image), cmap="gray")
        else:
            plt.imshow(transforms.ToPILImage()(image))

        plt.title(label)
        plt.show()


if __name__ == "__main__":
    ds = SimpleDataset(os.path.join("G:", "BA", "Data", "Out"))
    ds.showOne()
    # pass
