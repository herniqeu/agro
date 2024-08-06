# Image Data Manager Class

# The `ImageDataManager` class is designed to simplify the handling of image 
# datasets. It loads images from specified directories, processes them into a 
# usable format, and stores them for further analysis.

import os
import cv2 as cv
import numpy as np

class ImageDataManager:
    """
    A class used to manage image data for processing and analysis.

    Attributes
    ----------
    objects : dict
        A dictionary to store processed image data with mask IDs as keys.
    base_masks_path : str
        Path to the directory containing mask images.
    base_inputs_path : str
        Path to the directory containing input images.

    Methods
    -------
    process_images()
        Processes all images found in the base paths and populates the objects dictionary.
    process_input_folder(input_folder_path)
        Processes all images within a given input folder path and returns a list of images.
    """

    def __init__(self, base_masks_path, base_inputs_path):
        """
        Constructs all the necessary attributes for the ImageDataManager object.

        Parameters
        ----------
        base_masks_path : str
            Path to the directory containing mask images.
        base_inputs_path : str
            Path to the directory containing input images.
        """
        self.objects = {}
        self.base_masks_path = base_masks_path
        self.base_inputs_path = base_inputs_path
        self.process_images()

    @staticmethod
    def load_image(image_path, flags=cv.IMREAD_COLOR):
        """
        Loads an image from a specified path.

        Parameters
        ----------
        image_path : str
            Path to the image file to be loaded.
        flags : int
            Flags for image color format to be read.

        Returns
        -------
        ndarray
            The image loaded into memory.
        """
        return cv.imread(image_path, flags)

    @staticmethod
    def split_channels(image_path):
        """
        Splits the channels of an image at the given path.

        Parameters
        ----------
        image_path : str
            Path to the image file.

        Returns
        -------
        list
            A list containing the channels of the image.
        """
        image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
        channels = cv.split(image)
        if len(channels) == 4:
            return channels[:3]  # Ignore alpha channel
        return channels

    def process_images(self):
        """
        Processes the images found in the base paths and stores them in the objects dictionary.
        """
        for mask_filename in os.listdir(self.base_masks_path):
            if mask_filename.endswith('.png'):
                mask_id = os.path.splitext(mask_filename)[0]
                mask_path = os.path.join(self.base_masks_path, mask_filename)
                mask = self.load_image(mask_path, cv.IMREAD_GRAYSCALE)
                input_folder_path = os.path.join(self.base_inputs_path, mask_id)
                images = self.process_input_folder(input_folder_path)
                self.objects[int(mask_id)] = {
                    'mask': mask,
                    'input': input_folder_path,
                    'images': images
                }

    def process_input_folder(self, input_folder_path):
        """
        Processes all images within the given input folder path.

        Parameters
        ----------
        input_folder_path : str
            Path to the input folder containing image files.

        Returns
        -------
        list
            A list of processed images.
        """
        images = []
        if os.path.isdir(input_folder_path):
            for image_filename in os.listdir(input_folder_path):
                image_path = os.path.join(input_folder_path, image_filename)
                if image_filename.endswith('.png'):
                    channels = self.split_channels(image_path)
                    images.append(np.array(channels))
                elif image_filename.endswith(('.tif', '.tiff')):
                    tif_image = self.load_image(image_path, cv.IMREAD_UNCHANGED)
                    images.append(np.array([tif_image]))
        return images