# Helper Functions

# Auxiliary functions such as `get_dataset` are defined here. These functions 
# support the main pipeline by providing image loading capabilities and 
# visualizing the processing history.

import torch
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import data as tf_data

def get_dataset(batch_size,
                img_size,
                input_img_arr,
                target_img_arr,
                max_dataset_len=None):
    """
    Returns a TensorFlow Dataset.

    Args:
        batch_size (int): The batch size for the dataset.
        img_size (tuple): The size of the images in the dataset.
        input_img_arr (numpy.array): Array of input images.
        target_img_arr (numpy.array): Array of target images.
        max_dataset_len (int, optional): Maximum length of the dataset. Defaults to None.

    Returns:
        tf.data.Dataset: A TensorFlow dataset batched with the specified batch size.
    """
    if max_dataset_len:
        input_img_arr = input_img_arr[:max_dataset_len]
        target_img_arr = target_img_arr[:max_dataset_len]
    dataset = tf_data.Dataset.from_tensor_slices((input_img_arr, target_img_arr))
    return dataset.batch(batch_size)

def visualize_dataset(dataset, num_samples, title):
    """
    Visualize the dataset with images and their corresponding masks.

    Parameters:
    - dataset: The TensorFlow dataset to visualize.
    - num_samples: Number of samples to visualize.
    - title: Title for the plot.
    """
    plt.figure(figsize=(15, num_samples * 3))
    for i, (image, mask) in enumerate(dataset.take(num_samples)):
        plt.subplot(num_samples, 2, 2 * i + 1)
        plt.imshow(np.squeeze(image.numpy()), cmap='gray')
        plt.title(f'{title} Image {i+1}')
        plt.axis('off')

        plt.subplot(num_samples, 2, 2 * i + 2)
        plt.imshow(np.squeeze(mask.numpy()), cmap='gray')
        plt.title(f'{title} Mask {i+1}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def calculate_coverage(true_masks, pred_masks):
    true_masks = (true_masks > 0.5).float()
    pred_masks = (pred_masks > 0.5).float()

    intersection = torch.logical_and(true_masks, pred_masks).sum().item()
    union = torch.logical_or(true_masks, pred_masks).sum().item()

    coverage = intersection / union if union != 0 else 0
    return coverage * 100