import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import random

from PIL import Image, ImageTk

from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QTabWidget,
    QWidget,
    QGridLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QMainWindow,
)


class ProcessingPipeline:
    """
    Manages the application of image processing filters and augmentations.

    Attributes:
        filters (list): A list of filter objects to apply to the images.
        augmentations (list): A list of augmentation objects to apply to the images.
        history (list): Records outcomes of applied filters and augmentations for visualization.
    """

    def __init__(self, plot_storyline=False):
        """
        Initializes the processing pipeline with empty lists for filters, augmentations, and history.
        """
        self.filters = []
        self.augmentations = []
        self.history = []

    def add_filters(self, filters):
        """
        Adds multiple filter objects to the pipeline.

        Parameters:
            filters (list): List of filter objects to be added.
        """
        self.filters.extend(filters)

    def clear_filters(self):
        """
        Clears all filter objects from the pipeline.
        """
        self.filters = []

    def add_augmentations(self, augmentations):
        """
        Adds multiple augmentation objects to the pipeline.

        Parameters:
            augmentations (list): List of augmentation objects to be added.
        """
        self.augmentations.extend(augmentations)

    def clear_augmentations(self):
        """
        Clears all augmentation objects from the pipeline.
        """
        self.augmentations = []

    def apply_filters(self, images, filters):
        """
        Applies each filter in sequence to the image.

        Parameters:
            img (numpy.ndarray): The original image to be processed.

        Returns:
            numpy.ndarray: The image processed by all filters.
        """
        filtered_images = []

        if len(filters) <= 0:
            return images

        for img in images:
            filtered_channels = [img]
            for filter_func in filters:
                filtered_channel = filter_func(img)
                if len(filtered_channel.shape) == 2:
                    filtered_channel = np.expand_dims(filtered_channel, axis=-1)
                filtered_channels.append(filtered_channel)
            filtered_img = np.concatenate(filtered_channels, axis=-1)
            filtered_images.append(filtered_img)

        return filtered_images

    def add_channels(self, imgs, channels):
        """
        Adds grayscale channels to RGB images and resizes them to 1200x1200.

        Args:
            imgs (list of numpy.ndarray): List of RGB images.
            channels (list of list of numpy.ndarray): List of lists of grayscale channels associated with each image.

        Returns:
            numpy.ndarray: An array of images with additional channels combined.
        """
        if (len(channels) <= 0):
            return imgs
        
        result_images = []

        for img, extra_channels in zip(imgs, channels):
            resized_img = cv.resize(img, (1200, 1200), interpolation=cv.INTER_LINEAR)
            resized_channels = [cv.resize(c, (1200, 1200), interpolation=cv.INTER_LINEAR) for c in extra_channels]
            all_channels = [resized_img] + resized_channels
            full_image = np.dstack(all_channels)
            result_images.append(full_image)
        
        return result_images
    
    def apply_crop(self, imgs, masks, crop_size=120):
        """
        Sequentially crops the given image and mask arrays based on the specified crop size.
        
        Parameters:
            imgs (list of numpy.ndarray): List of numpy arrays representing the original images.
            masks (list of numpy.ndarray): List of numpy arrays representing the original masks.
            crop_size (int, optional): The size (both width and height) of each square crop. Defaults to 120.
        
        Returns:
            tuple: A tuple containing two elements:
                   - A list of numpy.ndarray representing the cropped images.
                   - A list of numpy.ndarray representing the cropped masks.
        
        Raises:
            ValueError: If the crop size is larger than the dimensions of the original images.
        """
        if len(imgs[0]) < crop_size or len(imgs[0][0]) < crop_size:
          raise ValueError("Crop size must be smaller than the dimensions of the original images.")
        
        cropped_images = []
        cropped_masks = []

        for img, mask in zip(imgs, masks):
            height, width = img.shape[:2]
            for i in range(0, height, crop_size):
                for j in range(0, width, crop_size):
                    if i + crop_size <= height and j + crop_size <= width:
                        cropped_img = img[i:i+crop_size, j:j+crop_size]
                        cropped_msk = mask[i:i+crop_size, j:j+crop_size]
                        cropped_images.append(cropped_img)
                        cropped_masks.append(cropped_msk)

        return cropped_images, cropped_masks

    def apply_augmentations(self, images, masks, augmentation):
        """
        Applies data augmentation to a list of images and their corresponding masks.

        Parameters:
            images (list of numpy.ndarray): The list of numpy arrays representing the original images.
            masks (list of numpy.ndarray): The list of numpy arrays representing the masks for the images.
            n (int): Number of augmentations to apply to each image.

        Returns:
            tuple: A tuple containing two elements:
                  - List of numpy.ndarray representing the augmented images.
                  - List of numpy.ndarray representing the augmented masks.
        """
        if not augmentation:
            return images, masks
        
        def rotate_image(image, angle):
            center = (image.shape[1] // 2, image.shape[0] // 2)
            matrix = cv.getRotationMatrix2D(center, angle, 1.0)
            return cv.warpAffine(image, matrix, (image.shape[1], image.shape[0]))

        def translate_image(image, x, y):
            matrix = np.float32([[1, 0, x], [0, 1, y]])
            return cv.warpAffine(image, matrix, (image.shape[1], image.shape[0]))

        def flip_image(image, axis):
            return cv.flip(image, axis)

        augmentations = [
            lambda img: rotate_image(img, 180),
            lambda img: translate_image(img, image.shape[:2][0] // 2, image.shape[:2][1] // 2),
            lambda img: flip_image(img, -1)
        ]

        augmented_imgs = []
        augmented_masks = []

        for image, mask in zip(images, masks):
            for augmentation in augmentations:
                augmented_img = augmentation(image)
                augmented_mask = augmentation(mask)
                augmented_imgs.append(augmented_img)
                augmented_masks.append(augmented_mask)

        return augmented_imgs, augmented_masks

    def apply_normalization(self, imgs, masks):
        """
        Normalizes the pixel values of images and masks to the range [0, 1].

        Parameters:
            imgs (list of numpy.ndarray): The list of images to be normalized.
            masks (list of numpy.ndarray): The list of masks to be normalized.

        Returns:
            tuple: A tuple containing two elements:
                   - List of numpy.ndarray representing the normalized images.
                   - List of numpy.ndarray representing the normalized masks.
        """

        normalized_images = []
        normalized_masks = []
        for img, mask in zip(imgs, masks):
            normalized_img = img.astype('float32') / 255.0
            normalized_mask = mask.astype('float32') / 255.0
            normalized_images.append(normalized_img)
            normalized_masks.append(normalized_mask)

        return normalized_images, normalized_masks

    def run(self, imgs, masks, channels, filters, augmentation=False, crop_size=120):
        """
        Executes the entire image processing pipeline.

        Parameters:
            img (numpy.ndarray): The original image to process.
            mask (numpy.ndarray): The associated mask for the image.
            n_augmented (int): The number of augmented images to generate.
            crop_size (int): The size for cropping the images.
            n_crop (int): The number of crops to produce.

        Returns:
            tuple: A tuple containing three elements:
                   - List of numpy.ndarray representing the normalized images.
                   - List of numpy.ndarray representing the normalized masks.
                   - List of tuples containing the coordinates of cropped areas.
        """

        imgs_with_filters = self.apply_filters(
            imgs, filters
        )

        imgs_with_channels = self.add_channels(
            imgs_with_filters, channels
        )

        cropped_imgs, cropped_masks = self.apply_crop(
            imgs_with_channels, masks, crop_size
        )

        augmented_imgs, augmented_masks = self.apply_augmentations(
            cropped_imgs, cropped_masks, augmentation
        )

        normalized_imgs, normalized_masks = self.apply_normalization(
            augmented_imgs, augmented_masks
        )

        return normalized_imgs, normalized_masks

    def get_history(self):
        return self.history


class PipelineViewer:
    def __init__(self, history):
        """
        Initializes the PipelineViewer with a history of image processing steps.

        Parameters:
            history (list): A list of tuples detailing the processed images, their descriptions, and the stage of processing.
        """
        self.history = history

    def plot_history(self):
        """
        Sets up and displays the GUI application for visualizing image processing history.
        The application includes tabs for different stages like filtering, cropping, and augmentation.
        """
        app = QApplication(sys.argv)
        main_window = QMainWindow()
        main_window.setWindowTitle("PipelineViewer")
        tab_widget = QTabWidget()
        tabs = {"Filter": QWidget(), "Crop": QWidget(), "Augmentation": QWidget()}
        grid_layouts = {}

        for name, tab in tabs.items():
            scroll_area = QScrollArea()
            grid_layout = QGridLayout()
            widget = QWidget()
            widget.setLayout(grid_layout)
            scroll_area.setWidget(widget)
            scroll_area.setWidgetResizable(True)
            scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            tab.setLayout(QVBoxLayout())
            tab.layout().addWidget(scroll_area)
            tab_widget.addTab(tab, name)
            grid_layouts[name] = grid_layout

        self._add_images_to_grid(grid_layouts)

        main_window.setCentralWidget(tab_widget)
        main_window.show()
        app.exec()

    def _add_images_to_grid(self, grid_layouts):
        image_counts = {name: 0 for name in grid_layouts}
        for image, name, stage in self.history:
            if stage not in grid_layouts:
                continue
            idx = image_counts[stage]
            row, col = idx // 3, idx % 3
            self._add_image_to_grid(
                image, f"{name} ({stage})", stage, grid_layouts[stage], row, col
            )
            image_counts[stage] += 1

    def _add_image_to_grid(self, image, text, stage, grid_layout, row, col):
        if image.ndim > 2:
            height, width, _ = image.shape
        else:
            height, width = image.shape
        bytes_per_line = 3 * width
        if image.ndim == 3:
            q_image = QImage(
                image.data.tobytes(),
                width,
                height,
                bytes_per_line,
                QImage.Format_RGB888,
            )
            q_image = q_image.rgbSwapped()
        else:
            q_image = QImage(
                image.data.tobytes(), width, height, width, QImage.Format_Grayscale8
            )

        pixmap = QPixmap.fromImage(q_image)
        label_image = QLabel()
        label_image.setPixmap(pixmap.scaled(250, 250, Qt.KeepAspectRatio))

        label_text = QLabel(text)
        label_text.setAlignment(Qt.AlignCenter)

        grid_layout.addWidget(label_text, row * 2, col, 1, 1, Qt.AlignCenter)
        grid_layout.addWidget(label_image, (row * 2) + 1, col, 1, 1, Qt.AlignCenter)
