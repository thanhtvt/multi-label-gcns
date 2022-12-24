import torch.nn as nn
from torchvision import models


class ResNet(nn.Module):
    def __init__(
        self,
        num_layers: int = 50,
        weights: str = "IMAGENET1K_V2",
        progress: bool = True
    ):
        super(ResNet, self).__init__()
        assert num_layers in [34, 50, 101]
        assert weights in ["IMAGENET1K_V1", "IMAGENET1K_V2"]

        params = {"weights": weights, "progress": progress}
        if num_layers == 34:
            self.resnet = models.resnet34(**params)
        elif num_layers == 50:
            self.resnet = models.resnet50(**params)
        else:
            self.resnet = models.resnet101(**params)

    def get_backbone(self):
        return nn.Sequential(*list(self.resnet.children())[:-2])

    def forward(self, x):
        return self.resnet(x)
