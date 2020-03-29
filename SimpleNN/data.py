import torch.utils.data.dataset as Dataset
import torch

classes = ("arrow_backward", "more", "menu", "avatar", "search",
           "star", "arrow_forward", "close", "add", "play")


class SimpleDataset(Dataset):
    def __init__(self, root_dir, transform):
        super().__init__()

        self.root_dir = root_dir
        self.transform = self.transform

    def __len__(self, idx):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pass
        # TODO: implementation
