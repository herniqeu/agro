from PIL import Image
import numpy as np


class ImageProcessor:
    """
    Handles processing of cropped images and combines them into a single image.

    Attributes:
        cropper (ImageCropper): Image cropper for cropping operations.
        predictor (ImagePredictor): Image predictor for predictions.
    """

    def __init__(self, height, width, predicted_imgs, cropped_coords):
        """
        Initializes the image processor with a cropper and predictor.

        Parameters:
            cropper (ImageCropper): Image cropper for cropping operations.
            predictor (ImagePredictor): Image predictor for predictions.
        """
        self.height = height
        self.width = width
        self.predicted_imgs = predicted_imgs
        self.cropped_coords = cropped_coords
        self.combined_image = np.zeros((self.height, self.width), dtype=np.uint8)

    def process_cropped_images(self):
        """
        Processes cropped images and combines them into a single image.

        Parameters:
            cropped_images (list): List of cropped images.
            crop_coords (list): List of coordinates for the cropped images.
            combined_image (numpy.ndarray): Combined image to update with predictions.

        Returns:
            numpy.ndarray: Updated combined image with predictions.
        """
        for predicted_img, coord in zip(self.predicted_imgs, self.cropped_coords):
            self.combined_image[coord[0][1]:coord[1][1], coord[0][0]:coord[1][0]] = predicted_img
        return self.combined_image
