import pickle as pkl

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class SubsetGraphCombiner(Dataset):
    def __init__(
        self,
        subset: Dataset,
        embedding_path: str,
        adjacency_path: str,
        corr_thres: float = 0.5,
        weight: float = 0.25,
        transform: transforms.Compose = None,
    ):
        super().__init__()
        self.subset = subset
        self.transform = transform
        self.label_embedding = self.extract_label_embedding(embedding_path)
        self.adj = self.get_adj_matrix(
            adjacency_path, corr_thres, weight
        )

    def __getitem__(self, idx: int):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return (img, self.label_embedding, self.adj), label

    def __len__(self):
        return len(self.subset)

    def extract_label_embedding(self, path: str):
        with open(path, "rb") as f:
            embedding = torch.tensor(pkl.load(f))
        return embedding

    def extract_adjancency(self, path: str, corr_thres: float, weight: float):
        # load pickle file's content
        loader = pkl.load(open(path, "rb"))
        counter, coappearance = loader['nums'], loader['adj']

        # compute adjancency matrix
        corr = coappearance / counter[:, np.newaxis]
        corr = np.where(corr < corr_thres, 0, 1)

        # re-weight adjancency matrix
        filter_mat = np.ones_like(corr)
        np.fill_diagonal(filter_mat, 0)
        adj = corr * weight / (
            np.sum(filter_mat * corr, axis=0, keepdims=True) + 1e-6)
        np.fill_diagonal(adj, 1 - weight)

        return adj

    def normalize_adj(self, adj: np.ndarray):
        adj = adj + np.eye(adj.shape[0])
        degree = np.power(adj.sum(1), -0.5)
        degree[np.isinf(degree)] = 0
        degree = np.diag(degree)
        return degree.dot(adj).dot(degree)

    def get_adj_matrix(self, path: str, corr_thres: float, weight: float):
        adj = self.normalize_adj(
            self.extract_adjancency(path, corr_thres, weight)
        )
        return torch.tensor(adj, dtype=torch.float)
