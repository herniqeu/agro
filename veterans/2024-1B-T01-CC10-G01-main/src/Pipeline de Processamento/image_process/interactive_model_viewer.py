import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

class ImageTransformer:
    """
    Handles image transformations including resizing, conversion to tensor, and normalization.

    Attributes:
        transform (transforms.Compose): Composed transformations to apply to the images.
    """

    def __init__(self, resize_dim=(120, 120), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """
        Initializes the image transformer with given resize dimensions, mean, and standard deviation.

        Parameters:
            resize_dim (tuple): Dimensions to resize the image.
            mean (list): Mean for normalization.
            std (list): Standard deviation for normalization.
        """
        self.transform = transforms.Compose([
            transforms.Resize(resize_dim),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, image):
        """
        Applies the composed transformations to the image.

        Parameters:
            image (PIL.Image.Image): Input image.

        Returns:
            torch.Tensor: Transformed image tensor.
        """
        return self.transform(image)

class ImageCropper:
    """
    Handles image cropping with various offset options.

    Attributes:
        crop_width (int): Width of the crop.
        crop_height (int): Height of the crop.
    """

    def __init__(self, crop_width=120, crop_height=120):
        """
        Initializes the image cropper with given crop width and height.

        Parameters:
            crop_width (int): Width of the crop.
            crop_height (int): Height of the crop.
        """
        self.crop_width = crop_width
        self.crop_height = crop_height

    def crop_image(self, image_array):
        """
        Crops the image into multiple smaller images.

        Parameters:
            image_array (numpy.ndarray): Array representation of the image.

        Returns:
            tuple: List of cropped images and their coordinates.
        """
        original_height, original_width = image_array.shape[:2]
        if self.crop_width > original_width or self.crop_height > original_height:
            raise ValueError("Crop dimensions must be smaller than the original dimensions.")

        cropped_images, crop_coords = [], []

        num_height_crops = original_height // self.crop_height
        num_width_crops = original_width // self.crop_width

        for row in range(num_height_crops):
            for col in range(num_width_crops):
                top = row * self.crop_height
                left = col * self.crop_width
                cropped_img = image_array[top:top + self.crop_height, left:left + self.crop_width]
                cropped_images.append(cropped_img)
                crop_coords.append(((left, top), (left + self.crop_width, top + self.crop_height)))

        return cropped_images, crop_coords

    def crop_image_with_x_offset(self, image_array):
        """
        Crops the image with a horizontal offset.

        Parameters:
            image_array (numpy.ndarray): Array representation of the image.

        Returns:
            tuple: List of cropped images and their coordinates.
        """
        original_height, original_width = image_array.shape[:2]
        if self.crop_width > original_width or self.crop_height > original_height:
            raise ValueError("Crop dimensions must be smaller than the original dimensions.")

        cropped_images, crop_coords = [], []

        num_height_crops = original_height // self.crop_height
        num_width_crops = original_width // self.crop_width

        offset_width = self.crop_width // 2

        for row in range(num_height_crops):
            for col in range(num_width_crops):
                top = row * self.crop_height
                left = col * self.crop_width + offset_width
                if top + self.crop_height > original_height or left + self.crop_width > original_width:
                    continue
                cropped_img = image_array[top:top + self.crop_height, left:left + self.crop_width]
                cropped_images.append(cropped_img)
                crop_coords.append(((left, top), (left + self.crop_width, top + self.crop_height)))

        return cropped_images, crop_coords

    def crop_image_with_xy_offset(self, image_array):
        """
        Crops the image with both horizontal and vertical offsets.

        Parameters:
            image_array (numpy.ndarray): Array representation of the image.

        Returns:
            tuple: List of cropped images and their coordinates.
        """
        original_height, original_width = image_array.shape[:2]
        if self.crop_width > original_width or self.crop_height > original_height:
            raise ValueError("Crop dimensions must be smaller than the original dimensions.")

        cropped_images, crop_coords = [], []

        num_height_crops = original_height // self.crop_height
        num_width_crops = original_width // self.crop_width

        offset_height = self.crop_height // 2
        offset_width = self.crop_width // 2

        for row in range(num_height_crops):
            for col in range(num_width_crops):
                top = row * self.crop_height + offset_height
                left = col * self.crop_width + offset_width
                if top + self.crop_height > original_height or left + self.crop_width > original_width:
                    continue
                cropped_img = image_array[top:top + self.crop_height, left:left + self.crop_width]
                cropped_images.append(cropped_img)
                crop_coords.append(((left, top), (left + self.crop_width, top + self.crop_height)))

        return cropped_images, crop_coords

class ImagePredictor:
    """
    Handles image prediction using a pre-trained model.

    Attributes:
        model (torch.nn.Module): The pre-trained model for prediction.
        device (torch.device): Device to run the model on.
        transformer (ImageTransformer): Transformer to preprocess the images.
    """

    def __init__(self, model, device, transformer):
        """
        Initializes the image predictor with the model, device, and transformer.

        Parameters:
            model (torch.nn.Module): The pre-trained model for prediction.
            device (torch.device): Device to run the model on.
            transformer (ImageTransformer): Transformer to preprocess the images.
        """
        self.model = model
        self.device = device
        self.transformer = transformer

    def predict(self, image):
        """
        Predicts the output for a given image.

        Parameters:
            image (PIL.Image.Image): Input image.

        Returns:
            numpy.ndarray: Predicted output.
        """
        image_tensor = self.transformer(image)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            output = self.model(image_tensor)
            _, preds = torch.max(output, dim=1)
            preds = preds.cpu().numpy()
        return preds[0]

class ImageProcessor:
    """
    Handles processing of cropped images and combines them into a single image.

    Attributes:
        cropper (ImageCropper): Image cropper for cropping operations.
        predictor (ImagePredictor): Image predictor for predictions.
    """

    def __init__(self, cropper, predictor):
        """
        Initializes the image processor with a cropper and predictor.

        Parameters:
            cropper (ImageCropper): Image cropper for cropping operations.
            predictor (ImagePredictor): Image predictor for predictions.
        """
        self.cropper = cropper
        self.predictor = predictor

    def process_cropped_images(self, cropped_images, crop_coords, combined_image):
        """
        Processes cropped images and combines them into a single image.

        Parameters:
            cropped_images (list): List of cropped images.
            crop_coords (list): List of coordinates for the cropped images.
            combined_image (numpy.ndarray): Combined image to update with predictions.

        Returns:
            numpy.ndarray: Updated combined image with predictions.
        """
        for cropped_img, coord in zip(cropped_images, crop_coords):
            prediction = self.predictor.predict(Image.fromarray(cropped_img))
            combined_image[coord[0][1]:coord[1][1], coord[0][0]:coord[1][0]] = prediction
        return combined_image

def main(args):
    """
    Main function to load the model, process the image, and display the results.

    Parameters:
        args (argparse.Namespace): Command-line arguments.
    """
    model = torch.load(args.model_path)
    model.eval()

    transformer = ImageTransformer()
    cropper = ImageCropper()
    predictor = ImagePredictor(model, device, transformer)
    processor = ImageProcessor(cropper, predictor)

    image = Image.open(args.image_path).convert("RGB")

    cropped_images, crop_coords = cropper.crop_image(np.array(image))
    cropped_images_offset_x, crop_coords_offset_x = cropper.crop_image_with_x_offset(np.array(image))
    cropped_images_offset_xy, crop_coords_offset_xy = cropper.crop_image_with_xy_offset(np.array(image))

    combined_image = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
    combined_image = processor.process_cropped_images(cropped_images, crop_coords, combined_image)
    combined_image = processor.process_cropped_images(cropped_images_offset_x, crop_coords_offset_x, combined_image)
    combined_image = processor.process_cropped_images(cropped_images_offset_xy, crop_coords_offset_xy, combined_image)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[1].imshow(combined_image, cmap='gray')
    ax[1].set_title('Combined Image')

    def sync_zoom_or_pan(event):
        xlim0, ylim0 = ax[0].get_xlim(), ax[0].get_ylim()
        xlim1, ylim1 = ax[1].get_xlim(), ax[1].get_ylim()

        if event.inaxes == ax[0]:
            ax[1].set_xlim(xlim0)
            ax[1].set_ylim(ylim0)
        elif event.inaxes == ax[1]:
            ax[0].set_xlim(xlim1)
            ax[0].set_ylim(ylim1)

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_release_event', sync_zoom_or_pan)
    fig.canvas.mpl_connect('scroll_event', sync_zoom_or_pan)

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Segmentation with ResUNet")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the pre-trained model")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image")
    args = parser.parse_args()
    main(args)
