
import torch
import torchvision
from torch import nn


def create_effnetb3_model(num_classes):

    weights = torchvision.models.EfficientNet_B3_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.efficientnet_b3(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1536, out_features=num_classes),
    )

    return model, transforms
