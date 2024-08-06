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

    def apply_filters(self, img):
        """
        Applies each filter in sequence to the image.

        Parameters:
            img (numpy.ndarray): The original image to be processed.

        Returns:
            numpy.ndarray: The image processed by all filters.
        """
        _img = img
        for _filter in self.filters:
            _img, _ = _filter.apply(_img, None)
            self.history.append((_img.copy(), type(_filter).__name__, "Filter"))
        return _img

    def apply_crop(self, img, mask, new_width=120, new_height=120):
        """
        Randomly crops the given image and mask arrays into 'n' new images and masks with dimensions 'new_width' x 'new_height'.

        Parameters:
            img (numpy.ndarray): The numpy array representing the original image.
            mask (numpy.ndarray): The numpy array representing the original mask.
            new_width (int): The width of the new images and masks.
            new_height (int): The height of the new images and masks.

        Returns:
            tuple: A tuple containing three elements:
                   - A list of numpy.ndarray representing the cropped images.
                   - A list of numpy.ndarray representing the cropped masks.
                   - A list of tuples containing the coordinates of each crop.
        """
        original_height, original_width = img.shape[:2]
        if new_width > original_width or new_height > original_height:
            raise ValueError(
                "New dimensions must be smaller than the original dimensions."
            )

        cropped_images, cropped_masks, crop_coordinates = [], [], []
        
        number_of_height_crops = original_height // new_height
        number_of_width_crops = original_width // new_width
                
        for row in range(0, number_of_height_crops):
            for col in range(0, number_of_width_crops):
                top = row * new_height
                left = col * new_width
                cropped_img = img[top : top + new_height, left : left + new_width]
                cropped_mask = mask[top : top + new_height, left : left + new_width]
                cropped_images.append(cropped_img)
                cropped_masks.append(cropped_mask)
                crop_coordinates.append(
                    ((left, top), (left + new_width, top + new_height))
                )
                
                self.history.append(
                    (cropped_img, f"Cropped Image at ({left}, {top})", "Crop")
                )

        return cropped_images, cropped_masks, crop_coordinates

    def apply_augmentations(self, images, masks, n=3):
        """
        Applies data augmentation to a list of images and their corresponding masks.

        Parameters:
            images (list of numpy.ndarray): The list of numpy arrays representing the original images.
            masks (list of numpy.ndarray): The list of numpy arrays representing the masks for the images.
            n (int): Number of augmentations to apply to each image.
            filters (list): List of instantiated filter classes to apply.

        Returns:
            tuple: A tuple containing two elements:
                   - List of numpy.ndarray representing the original and augmented images.
                   - List of numpy.ndarray representing the original and augmented masks.
        """
        all_images = []
        all_masks = []

        for image, mask in zip(images, masks):
            augmented_images = [image]
            augmented_masks = [mask]
            previous_transformations = set()

            while len(augmented_images) - 1 < n:
                selected_filter = random.choice(self.augmentations)
                transformation_key = (
                    type(selected_filter).__name__,
                    tuple(selected_filter.__dict__.values()),
                )

                if transformation_key not in previous_transformations:
                    augmented_image, augmented_mask = selected_filter.apply(image, mask)
                    augmented_images.append(augmented_image)
                    augmented_masks.append(augmented_mask)
                    previous_transformations.add(transformation_key)

                    augmentation_name_str = "Augmented with {}".format(
                        {type(selected_filter).__name__}
                    )

                    self.history.append(
                        (
                            augmented_image,
                            augmentation_name_str,
                            "Augmentation",
                        )
                    )
                    self.history.append(
                        (
                            augmented_mask,
                            augmentation_name_str,
                            "Augmentation",
                        )
                    )

            all_images.extend(augmented_images)
            all_masks.extend(augmented_masks)

        return all_images, all_masks

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
        _imgs = []
        _masks = []
        for index, _img in enumerate(imgs):
            if _img.ndim > 2:
                height, width, _ = _img.shape
            else:
                height, width = _img.shape
            m_height, m_width = masks[index].shape
            norm_img = np.zeros((height, width))
            norm_mask = np.zeros((m_height, m_width))
            norm_img = cv.normalize(_img, norm_img, 0, 255, cv.NORM_MINMAX)
            norm_mask = cv.normalize(masks[index], norm_mask, 0, 255, cv.NORM_MINMAX)
            norm_img = norm_img / 255
            norm_mask = norm_mask / 255
            _imgs.append(norm_img)
            _masks.append(norm_mask)
        return _imgs, _masks

    def run(self, img, mask, n_augmented, crop_size=120):
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
        highlighted_img = self.apply_filters(img)
        cropped_imgs, cropped_masks, cropped_coordinates = self.apply_crop(
            highlighted_img,
            mask,
            new_height=crop_size,
            new_width=crop_size,
        )
    
        augmented_imgs, augmented_masks = self.apply_augmentations(
            cropped_imgs, cropped_masks, n_augmented
        )
        normalized_imgs, normalized_masks = self.apply_normalization(
            augmented_imgs, augmented_masks
        )
        return normalized_imgs, normalized_masks, cropped_coordinates

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
