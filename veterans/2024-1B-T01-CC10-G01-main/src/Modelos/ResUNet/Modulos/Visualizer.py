# Visualizer Class

# The `Visualizer` class contains methods for displaying images and their 
# channels. It helps us understand the data and the effects of processing steps 
# visually.

import math
import matplotlib.pyplot as plt

class Visualizer:
    """
    A class used to visualize image data.

    Methods
    -------
    visualize_objects(objects)
        Visualizes the processed image data from the objects dictionary.
    calculate_subplots(image_sets)
        Calculates the number of subplots needed for the visualization based on the image sets.
    display_images(num_rows, num_columns, image_sets)
        Displays images in a grid layout as subplots.
    """

    @staticmethod
    def visualize_objects(objects):
        """
        Visualizes the processed image data from the objects dictionary.

        Parameters
        ----------
        objects : dict
            Dictionary containing processed image data to visualize.
        """
        for mask_id, data in objects.items():
            mask = data['mask']
            image_sets = data['images']
            num_subplots = Visualizer.calculate_subplots(image_sets)
            num_columns = min(num_subplots, 4)
            num_rows = math.ceil(num_subplots / num_columns)
            plt.figure(figsize=(4 * num_columns, 3 * num_rows))
            plt.subplot(num_rows, num_columns, 1)
            plt.title(f"Mask {mask_id}")
            plt.imshow(mask, cmap='gray')
            plt.axis('off')
            Visualizer.display_images(num_rows, num_columns, image_sets)
            plt.tight_layout()
            plt.show()

    @staticmethod
    def calculate_subplots(image_sets):
        """
        Calculates the number of subplots needed based on the image sets.

        Parameters
        ----------
        image_sets : list
            A list of image sets, where each image set corresponds to a group of channels.

        Returns
        -------
        int
            The total number of subplots needed.
        """
        return 1 + sum(len(images) for images in image_sets)

    @staticmethod
    def display_images(num_rows, num_columns, image_sets):
        """
        Displays images in a grid layout as subplots.

        Parameters
        ----------
        num_rows : int
            Number of rows in the grid layout.
        num_columns : int
            Number of columns in the grid layout.
        image_sets : list
            A list of image sets to be displayed.
        """
        subplot_idx = 2
        for image_set in image_sets:
            for channel in image_set:
                plt.subplot(num_rows, num_columns, subplot_idx)
                plt.imshow(channel, cmap='gray')
                plt.axis('off')
                title = f"TIFF Image {subplot_idx-1}" if len(image_set) == 1 else f"PNG Ch {subplot_idx-1}"
                plt.title(title)
                subplot_idx += 1