from torch.utils.data import Dataset
from torchvision.transforms import transforms


class SubsetTransformWrapper(Dataset):
    def __init__(self, subset: Dataset, transform: transforms.Compose = None):
        super().__init__()
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx: int):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.subset)
