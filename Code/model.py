import torch
import torch.nn as nn
from torchvision.models import vgg16

# Setting up device
device = torch.device('cpu')

class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()

        # Load a pre-trained VGG16 model
        self.model = vgg16(pretrained=True)

        # Modify the first convolutional layer to accept 1 channel instead of 3
        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)

        # Modify the classifier layer to match the number of classes
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)