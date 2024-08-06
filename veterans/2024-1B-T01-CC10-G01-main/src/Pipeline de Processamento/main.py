import cv2 as cv
import numpy as np
import logging
import argparse
import os

from image_process.pipeline import PipelineViewer, ProcessingPipeline

from image_process.processes import (
    MorphDilate,
    Rotate,
    Translate,
    Flip,
    BrightnessContrast,
    RandomGaussianBlur,
    MedianBlur,
    BilateralFilter,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Args passed to the script call
# Image path
# Mask path
# Number of augmented images to generate
# Width of the crop
# Height of the crop
# Number of crops to generate
# Number of filters to apply
# Number of augmentations to apply
# Seed for random number generator

def parse_args():
    parser = argparse.ArgumentParser(description="Process images with a pipeline")
    parser.add_argument("--img_path", required=True, type=str, help="Images path")
    parser.add_argument("--output_path", required=True, type=str, help="Output path")
    parser.add_argument("--crop_size", required=True, type=int, help="Width of the crop")
    parser.add_argument("--n_augmented", type=int, default=5, help="Number of augmented images to generate")
    args = parser.parse_args()
    return args

def load_image(path, color_mode=cv.IMREAD_COLOR):
    """Loads an image from the given path with specified color mode."""
    image = cv.imread(path, color_mode)
    if image is None:
        raise FileNotFoundError(f"Could not load image from {path}")
    return image

def main(args):
    try:
        img_path = args.img_path
        n_augmented = args.n_augmented
        crop_size = args.crop_size
        output_path = args.output_path
        output_images_path = os.path.join(output_path, "images")
        output_masks_path = os.path.join(output_path, "masks")
        
        pipeline = ProcessingPipeline()
        pipeline.add_filters([])
        pipeline.add_augmentations(
            [
                Rotate(),
                Translate(),
                Flip(),
                BrightnessContrast(),
                RandomGaussianBlur(),
                MedianBlur(),
                BilateralFilter(),
            ]
        )
        
        normalized_imgs = []
        normalized_masks = []
        cropped_coordinates = []
        
        for img_folder in os.listdir(img_path):
            img_folder_path = os.path.join(img_path, img_folder, "img.tif")
            mask_folder_path = os.path.join(img_path, img_folder, "mask.png")
            img = load_image(img_folder_path, cv.IMREAD_COLOR)
            mask = load_image(mask_folder_path, cv.IMREAD_GRAYSCALE)
            
            n_crop = img.shape[0] // crop_size
            
            _normalized_imgs, _normalized_masks, _cropped_coordinates = pipeline.run(img, mask, n_augmented, crop_size, n_crop)
            normalized_imgs.extend(_normalized_imgs)
            normalized_masks.extend(_normalized_masks)
            cropped_coordinates.extend(_cropped_coordinates)
        
        if not os.path.exists(output_images_path):
            os.mkdir(output_images_path)
        
        if not os.path.exists(output_masks_path):
            os.mkdir(output_masks_path)

        for i, (img, mask) in enumerate(zip(normalized_imgs, normalized_masks)):
            img_path = os.path.join(output_images_path, f"img_{i}.tif")
            mask_path = os.path.join(output_masks_path, f"mask_{i}.png")
            cv.imwrite(img_path, img)
            cv.imwrite(mask_path, mask)
        
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return

if __name__ == "__main__":
    args = parse_args()
    main(args)
