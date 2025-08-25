import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

class SwinUNet(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(SwinUNet, self).__init__()

        # Load Swin Transformer as Encoder
        self.encoder = timm.create_model(
            "swinv2_cr_small_ns_224.sw_in1k", pretrained=pretrained, features_only=True
        )

        # Get the number of channels from Swin Transformer encoder outputs
        encoder_channels = [info['num_chs'] for info in self.encoder.feature_info]
        decoder_channels = [512, 256, 128, 64] # efficientnet = 256, 128, 64, 32

        # Decoder
        self.decoder1 = self.conv_block(encoder_channels[-1], decoder_channels[0])
        self.decoder2 = self.conv_block(decoder_channels[0], decoder_channels[1])
        self.decoder3 = self.conv_block(decoder_channels[1], decoder_channels[2])
        self.decoder4 = self.conv_block(decoder_channels[2], decoder_channels[3])

        # Final Segmentation Head
        self.segmentation_head = nn.Conv2d(decoder_channels[3], num_classes, kernel_size=1)

        self.skip_conv3 = nn.Conv2d(encoder_channels[2], decoder_channels[0], kernel_size=1)
        self.skip_conv2 = nn.Conv2d(encoder_channels[1], decoder_channels[1], kernel_size=1)
        self.skip_conv1 = nn.Conv2d(encoder_channels[0], decoder_channels[2], kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )

    def forward(self, x):
        # Encoder
        enc_features = self.encoder(x)

        # enc1, enc2, enc3, enc4 = [feat.permute(0, 3, 1, 2) for feat in enc_features]
        enc1, enc2, enc3, enc4 = enc_features

        dec1 = self.decoder1(enc4)
        dec2 = self.decoder2(dec1 + self.skip_conv3(enc3))
        dec3 = self.decoder3(dec2 + self.skip_conv2(enc2))
        dec4 = self.decoder4(dec3 + self.skip_conv1(enc1))

        # Segmentation Head
        out = self.segmentation_head(dec4)
        out = nn.functional.interpolate(out, scale_factor=2, mode="bilinear", align_corners=True)
        return out


def model_builder(model_name):
  model = None
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  if model_name == 'CNN':
    model = smp.Unet(
    encoder_name="resnet34",       # Backbone
    encoder_weights="imagenet",    # Pre-trained weights
    in_channels=3,                 # Input channels (RGB)
    classes=1,
    )# Output classes (binary mask)
  elif model_name == "Swin U-Net":
    model = SwinUNet().to(device)
  elif model_name == "EfficientNet":
    model = smp.Unet( encoder_name="timm-efficientnet-b4", encoder_weights="imagenet", in_channels=3, classes=1 ).to(device)

  return model, device

