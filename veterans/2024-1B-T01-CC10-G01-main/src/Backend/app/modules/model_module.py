import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

class DoubleConv(nn.Module):
    """
    Applies two convolution layers with batch normalization and ReLU activation.

    Attributes:
        double_conv (nn.Sequential): Sequential container for the double convolution operation.
    """

    def __init__(self, in_channels, out_channels):
        """
        Initializes the DoubleConv layer with given input and output channels.

        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass through the double convolution layers.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.double_conv(x)

class Down(nn.Module):
    """
    Downsamples the input using max pooling followed by double convolution.

    Attributes:
        maxpool_conv (nn.Sequential): Sequential container for max pooling and double convolution.
    """

    def __init__(self, in_channels, out_channels):
        """
        Initializes the Down layer with given input and output channels.

        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        """
        Forward pass through the downsampling layer.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.maxpool_conv(x)

class Up(nn.Module):
    """
    Upsamples the input and then applies double convolution.

    Attributes:
        up (nn.Upsample): Upsampling layer.
        conv (DoubleConv): Double convolution layer.
    """

    def __init__(self, in_channels, out_channels):
        """
        Initializes the Up layer with given input and output channels.

        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Forward pass through the upsampling layer.

        Parameters:
            x1 (torch.Tensor): Input tensor from the previous layer.
            x2 (torch.Tensor): Input tensor from the corresponding down layer.

        Returns:
            torch.Tensor: Output tensor.
        """
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """
    Applies a 1x1 convolution to reduce the number of channels.

    Attributes:
        conv (nn.Conv2d): Convolution layer.
    """

    def __init__(self, in_channels, out_channels):
        """
        Initializes the OutConv layer with given input and output channels.

        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through the output convolution layer.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.conv(x)

class ResUNet(nn.Module):
    """
    Residual UNet architecture for image segmentation.

    Attributes:
        inc (DoubleConv): Initial double convolution layer.
        down1 (Down): Downsampling layer 1.
        down2 (Down): Downsampling layer 2.
        down3 (Down): Downsampling layer 3.
        down4 (Down): Downsampling layer 4.
        up1 (Up): Upsampling layer 1.
        up2 (Up): Upsampling layer 2.
        up3 (Up): Upsampling layer 3.
        up4 (Up): Upsampling layer 4.
        outc (OutConv): Output convolution layer.
    """

    def __init__(self, n_channels, n_classes):
        """
        Initializes the ResUNet with given input channels and number of classes.

        Parameters:
            n_channels (int): Number of input channels.
            n_classes (int): Number of output classes.
        """
        super(ResUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        """
        Forward pass through the ResUNet model.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor (logits).
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class Model:
    def __init__(self, model_path):
        """
        Inicializa o modelo carregando-o do arquivo .pth especificado e configura o dispositivo de processamento.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.model = torch.load(model_path, map_location=self.device)  
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((120, 120)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, images):
        """
        Realiza inferências em uma lista de imagens em formato numpy array, retornando máscaras binárias das predições.
        :param images: lista de imagens, cada uma como um numpy array de shape (X, Y, 3)
        :return: lista de máscaras binárias (numpy array)
        """
        masks = []
        for image in images:
            image = Image.fromarray(image.astype('uint8'), 'RGB')
            image_t = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(image_t)
                _, preds = torch.max(output, dim=1)
            
            masks.append(preds.cpu().numpy()[0])
        return masks

if __name__ == '__main__':
    model_name = "../models/model_92.pth"
    model = Model(model_name)
    print(f"Loaded model {model_name}!")

    folder = "../../../Imagens/Cropped/"
    images = []

    for filename in os.listdir(folder):
        if filename.endswith('.tif'):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert('RGB')
            images.append(np.array(img))  
    
    print(f"Tipo da lista de entrada: {type(images)}")
    if images:
        print(f"Tipo dos elementos da lista de entrada: {type(images[0])}")

    if images:
        masks = model.predict(images)

        print(f"Tipo da lista de saída: {type(masks)}")
        if masks:
            print(f"Tipo dos elementos da lista de saída: {type(masks[0])}")

        fig, axes = plt.subplots(len(images), 2, figsize=(10, 5 * len(images)))
        for i, (orig, mask) in enumerate(zip(images, masks)):
            ax = axes[i]
            ax[0].imshow(orig)
            ax[0].set_title(f'Original Image {i}')
            ax[0].axis('off')

            ax[1].imshow(mask, cmap='gray')
            ax[1].set_title(f'Predicted Mask {i}')
            ax[1].axis('off')

        plt.tight_layout()
        plt.show()
    else:
        print("No images were loaded.")