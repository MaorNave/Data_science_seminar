import torch.nn as nn
from torchvision.models import vgg16, vgg19

class SegmentationModel(nn.Module):
    def __init__(self, num_classes, pretrained_state, backbone_name):
        super(SegmentationModel, self).__init__()

        # Dictionary to map string to backbone function
        backbone_dict = {
            "VGG16": vgg16,
            "VGG19": vgg19
        }

        backbone = backbone_dict[backbone_name]

        if pretrained_state == True:
            self.features = backbone(pretrained=True).features
        else:
            self.features = backbone(pretrained=False).features

        # Add upsampling layers to match the input spatial dimensions
        self.upsample = nn.Sequential(
                    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # Upsample by a factor of 2
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Upsample by a factor of 2
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Upsample by a factor of 2
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(64, num_classes, kernel_size=4, stride=2, padding=1),  # Upsample by a factor of 2
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1),
                    # Upsample by a factor of 2
                )

    def forward(self, x):
        x = self.features(x)
        x = self.upsample(x)
        return x