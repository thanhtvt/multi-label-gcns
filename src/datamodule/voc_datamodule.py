from typing import Tuple, List, Union, Optional, Dict, Any

import torch
import pytorch_lightning as pl
from torchvision.datasets import VOCDetection
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision.transforms import transforms

from src.datamodule.utils import SubsetGraphCombiner


class VOCDataModule(pl.LightningDataModule):
    """LightningDataModule for VOC dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        year: int = 2007,
        download: bool = True,
        data_dir: str = "./data",
        embedding_path: str = "./data/voc_glove.pkl",
        adjacency_path: str = "./data/voc_adj.pkl",
        correlation_threshold: float = 0.5,
        correlation_weight: float = 0.2,
        train_val_test_split: Union[Tuple, List] = (0.8, 0.1, 0.1),
        img_size: Union[int, Tuple, List] = (448, 448),
        img_norm_mean: Union[Tuple, List] = (0.485, 0.456, 0.406),
        img_norm_std: Union[Tuple, List] = (0.229, 0.224, 0.225),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__()
        assert year == 2007, "Only VOC2007 is supported for now."
        # assert year in range(2007, 2013)

        self.save_hyperparameters(logger=False)

        self.object_classes = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
            'dog', 'horse', 'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor'
        ]

        self.train_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=img_norm_mean, std=img_norm_std),
            transforms.RandomErasing(),
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=img_norm_mean, std=img_norm_std),
        ])

        self.data_train, self.data_val, self.data_test = None, None, None

    def prepare_data(self):
        """Download data and prepare for processing.
        """
        VOCDetection(
            root=self.hparams.data_dir,
            year=str(self.hparams.year),
            image_set="trainval",
            download=self.hparams.download,
        )
        if self.hparams.year == 2007:
            VOCDetection(
                root=self.hparams.data_dir,
                year=str(self.hparams.year),
                image_set="test",
                download=self.hparams.download,
            )

    def target_transform(self, target: Dict[str, Any]):
        """Transform target to be a torch.Tensor
        """
        labels = []
        for obj in target["annotation"]["object"]:
            labels.append(obj["name"])
        # convert to tensor
        labels = [self.object_classes.index(label) for label in labels]
        labels = [1 if idx in labels else 0 for idx in range(len(self.object_classes))]
        labels = torch.tensor(labels, dtype=torch.int32)
        return labels

    def setup(self, stage: Optional[str] = None):
        """Load data, set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`,
        so be careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = VOCDetection(
                root=self.hparams.data_dir,
                year=str(self.hparams.year),
                image_set="trainval",
                target_transform=self.target_transform,
            )
            testset = None
            if self.hparams.year == "2007":
                testset = VOCDetection(
                    root=self.hparams.data_dir,
                    year=str(self.hparams.year),
                    image_set="test",
                    target_transform=self.target_transform,
                )
            self.split_dataset(trainset, testset)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

    def split_dataset(self, trainset, testset):
        dataset = ConcatDataset([trainset, testset]) if testset else trainset
        splited_datasets = random_split(
            dataset=dataset,
            lengths=self.hparams.train_val_test_split,
            generator=torch.Generator().manual_seed(self.hparams.seed),
        )
        if testset:
            data_train, data_val = splited_datasets
        else:
            data_train, data_val, testset = splited_datasets

        self.data_train = SubsetGraphCombiner(
            data_train,
            self.hparams.embedding_path,
            self.hparams.adjacency_path,
            self.hparams.correlation_threshold,
            self.hparams.correlation_weight,
            self.train_transform
        )
        self.data_val = SubsetGraphCombiner(
            data_val,
            self.hparams.embedding_path,
            self.hparams.adjacency_path,
            self.hparams.correlation_threshold,
            self.hparams.correlation_weight,
            self.test_transform
        )
        self.data_test = SubsetGraphCombiner(
            testset,
            self.hparams.embedding_path,
            self.hparams.adjacency_path,
            self.hparams.correlation_threshold,
            self.hparams.correlation_weight,
            self.test_transform,
        )
