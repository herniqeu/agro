# Preprocessing Auxiliary Classes

# Here we define a suite of image preprocessing classes that extend 
# `BaseImageProcess`. Each class implements a specific image preprocessing 
# technique, such as rotation, blurring, and thresholding, which are essential 
# for the feature extraction phase of plot segmentation.

import random
import cv2 as cv
import numpy as np
from scipy.ndimage import convolve

class BaseImageProcess:
    """
    BaseImageProcess: A base class for image processing algorithms.

    This class provides a basic framework for implementing image processing algorithms and is intended to be subclassed.
    Subclasses should implement the `apply` method to perform specific image processing operations on an input image.
    """

    def apply(self, img, mask=None):
        """
        Placeholder for applying an image processing algorithm.

        Args:
            img: The input image to process.

        Returns:
            The processed image.
        """
        pass


class Rotate(BaseImageProcess):
    def __init__(self):
        self.angle = random.choice(list(range(-180, 181, 10)))

    def apply(self, img, mask=None):
        height, width = img.shape[:2]
        rotation_matrix = cv.getRotationMatrix2D((width / 2, height / 2), self.angle, 1)
        return cv.warpAffine(img, rotation_matrix, (width, height)), (
            cv.warpAffine(mask, rotation_matrix, (width, height))
            if mask is not None
            else None
        )


class BilateralFilter(BaseImageProcess):
    """
    BilateralFilter: Applies bilateral filtering to an image to reduce noise while keeping edges sharp.
    """

    def __init__(self, d=9, sigmaColor=75, sigmaSpace=75):
        self.d = d
        self.sigmaColor = sigmaColor
        self.sigmaSpace = sigmaSpace

    def apply(self, img, mask=None):
        return cv.bilateralFilter(img, self.d, self.sigmaColor, self.sigmaSpace), mask


class Translate(BaseImageProcess):
    """
    Applies translation to an image using random horizontal and vertical shifts.

    Attributes:
        dx (int): Horizontal shift, chosen randomly from a specified range.
        dy (int): Vertical shift, chosen randomly from a specified range.
    """

    def __init__(self):
        self.dx = random.choice([-10, -5, 0, 5, 10])
        self.dy = random.choice([-10, -5, 0, 5, 10])

    def apply(self, img, mask=None):
        translation_matrix = np.float32([[1, 0, self.dx], [0, 1, self.dy]])
        height, width = img.shape[:2]
        return cv.warpAffine(img, translation_matrix, (width, height)), (
            cv.warpAffine(mask, translation_matrix, (width, height))
            if mask is not None
            else None
        )


class Flip(BaseImageProcess):
    """
    Flips an image either horizontally, vertically, or both, based on a randomly selected flip type.

    Attributes:
        flip_type (int): Type of flip to apply; -1 for both axes, 0 for vertical, 1 for horizontal.
    """

    def __init__(self):
        self.flip_type = random.choice([-1, 0, 1])

    def apply(self, img, mask=None):
        return cv.flip(img, self.flip_type), (
            cv.flip(mask, self.flip_type) if mask is not None else None
        )


class BrightnessContrast(BaseImageProcess):
    """
    Adjusts the brightness and contrast of an image using random values.

    Attributes:
        alpha (float): Factor by which the contrast will be adjusted.
        beta (int): Value that will be added to the pixels for brightness adjustment.
    """

    def __init__(self):
        self.alpha = random.uniform(0.5, 1.5)
        self.beta = random.randint(-50, 50)

    def apply(self, img, mask=None):
        return cv.convertScaleAbs(img, alpha=self.alpha, beta=self.beta), mask


class MedianBlur(BaseImageProcess):
    """
    Applies median blurring to an image using a randomly chosen kernel size.

    Attributes:
        kernel_size (int): The size of the kernel used, selected randomly from a set of possible odd sizes.
    """

    def __init__(self):
        self.kernel_size = random.choice([3, 5, 7, 9, 11])

    def apply(self, img, mask=None):
        return cv.medianBlur(img, self.kernel_size), mask


class RandomGaussianBlur(BaseImageProcess):
    """
    Applies Gaussian blur filtering to an image with a randomly chosen kernel size.

    Attributes:
        kernel_size (int): Size of the Gaussian blur kernel, selected randomly.
    """

    def __init__(self):
        self.kernel_size = random.choice([3, 5, 7, 9, 11])

    def apply(self, img, mask=None):
        return cv.GaussianBlur(img, (self.kernel_size, self.kernel_size), 0), mask


class GaussianBlur(BaseImageProcess):
    """
    GaussianBlur: Applies Gaussian blur filtering to an image.

    This class provides an implementation of Gaussian blur filtering, commonly used to reduce image noise and detail.

    Attributes:
        kernel_size (int): Size of the kernel used for the Gaussian filter.
    """

    def __init__(self, kernel_size=5):
        self.kernel_size = kernel_size

    def apply(self, img, mask=None):
        return cv.GaussianBlur(img, (self.kernel_size, self.kernel_size), 0), mask


class BinaryThresh(BaseImageProcess):
    """
    BinaryThresh: Applies binary thresholding to an image.

    Binary thresholding converts an image to binary (black and white) based on a threshold value. Pixels above the
    threshold are set to the maximum value, and those below are set to zero.

    Attributes:
        thresh (int): Threshold value.
        max_val (int): Maximum value to use with the threshold.
    """

    def __init__(self, thresh=127, max_val=255):
        self.thresh = thresh
        self.max_val = max_val

    def apply(self, img, mask=None):
        _img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) > 2 else img
        _, _img = cv.threshold(_img, self.thresh, self.max_val, cv.THRESH_BINARY)
        return _img, mask


class AdaptiveMeanThresh(BaseImageProcess):
    """
    AdaptiveMeanThresh: Applies adaptive mean thresholding to an image.

    Unlike simple thresholding, adaptive thresholding changes the threshold dynamically over the image based on local
    image characteristics.

    Attributes:
        block_size (int): Size of a pixel neighborhood used to calculate the threshold.
        c (int): Constant subtracted from the calculated mean or weighted mean.
    """

    def __init__(self, block_size=11, c=2):
        self.block_size = block_size
        self.c = c

    def apply(self, img, mask=None):
        _img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) > 2 else img
        return (
            cv.adaptiveThreshold(
                _img,
                255,
                cv.ADAPTIVE_THRESH_MEAN_C,
                cv.THRESH_BINARY,
                self.block_size,
                self.c,
            ),
            mask,
        )


class AdaptiveGaussThresh(BaseImageProcess):
    """
    AdaptiveGaussThresh: Applies adaptive Gaussian thresholding to an image.

    This method uses a weighted sum of neighbourhood values where weights are a Gaussian window, which provides
    a more natural thresholding, especially under varying illumination.

    Attributes:
        block_size (int): Size of a pixel neighborhood used to calculate the threshold.
        c (int): Constant subtracted from the calculated weighted sum.
    """

    def __init__(self, block_size=11, c=2):
        self.block_size = block_size
        self.c = c

    def apply(self, img, mask=None):
        _img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) > 2 else img
        return (
            cv.adaptiveThreshold(
                _img,
                255,
                cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv.THRESH_BINARY,
                self.block_size,
                self.c,
            ),
            mask,
        )


class OtsuThresh(BaseImageProcess):
    """
    OtsuThresh: Applies Otsu's thresholding to automatically perform histogram shape-based image thresholding.

    This method is useful when the image contains two prominent pixel intensities and calculates an optimal threshold
    separating these two classes so that their combined spread (intra-class variance) is minimal.
    """

    def apply(self, img, mask=None):
        _img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) > 2 else img
        _, _img = cv.threshold(_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        return _img, mask


class MorphDilate(BaseImageProcess):
    """
    MorphDilate: Applies morphological dilation to an image.

    Dilation increases the white region in the image or size of the foreground object. Commonly used to accentuate
    features.

    Attributes:
        kernel_size (int): Size of the structuring element.
        iterations (int): Number of times dilation is applied.
    """

    def __init__(self, kernel_size=3, iterations=2):
        self.kernel_size = kernel_size
        self.iterations = iterations
        self.kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)

    def apply(self, img, mask=None):
        return cv.dilate(img, self.kernel, iterations=self.iterations), mask


class MorphErode(BaseImageProcess):
    """
    MorphErode: Applies morphological erosion to an image.

    Erosion erodes away the boundaries of the foreground object and is used to diminish the features of an image.

    Attributes:
        kernel_size (int): Size of the structuring element.
        iterations (int): Number of times erosion is applied.
    """

    def __init__(self, kernel_size=3, iterations=2):
        self.kernel_size = kernel_size
        self.iterations = iterations
        self.kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)

    def apply(self, img, mask=None):
        return cv.erode(img, self.kernel, iterations=self.iterations), mask


class LoG(BaseImageProcess):
    """
    LoG: Applies Laplacian of Gaussian filtering to an image.

    This method is used to highlight regions of rapid intensity change and is therefore often used for edge detection.
    First, it applies a Gaussian blur, then computes the Laplacian of the result.

    Attributes:
        sigma (float): Standard deviation of the Gaussian filter.
        size (int): Size of the filter kernel.
    """

    def __init__(self, sigma=2.0, size=None):
        self.sigma = sigma
        self.size = (
            size
            if size is not None
            else int(6 * self.sigma + 1) if self.sigma >= 1 else 7
        )
        if self.size % 2 == 0:
            self.size += 1

    def apply(self, img, mask=None):
        x, y = np.meshgrid(
            np.arange(-self.size // 2 + 1, self.size // 2 + 1),
            np.arange(-self.size // 2 + 1, self.size // 2 + 1),
        )
        kernel = (
            -(1 / (np.pi * self.sigma**4))
            * (1 - ((x**2 + y**2) / (2 * self.sigma**2)))
            * np.exp(-(x**2 + y**2) / (2 * self.sigma**2))
        )
        kernel = kernel / np.sum(np.abs(kernel))
        return cv.filter2D(img, -1, kernel), mask


class LoGConv(BaseImageProcess):
    """
    LoGConv: Implements convolution with a Laplacian of Gaussian kernel to an image.

    Similar to the LoG class, but tailored for applying custom convolution operations directly with a manually
    crafted LoG kernel.

    Attributes:
        sigma (float): Standard deviation of the Gaussian filter.
        size (int): Size of the filter kernel.
    """

    def __init__(self, sigma=2.0, size=None):
        self.sigma = sigma
        self.size = size if size is not None else int(6 * sigma + 1)
        if self.size % 2 == 0:
            self.size += 1

    def apply(self, img, mask=None):
        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        x, y = np.meshgrid(
            np.arange(-self.size // 2 + 1, self.size // 2 + 1),
            np.arange(-self.size // 2 + 1, self.size // 2 + 1),
        )
        kernel = (
            -(1 / (np.pi * self.sigma**4))
            * (1 - ((x**2 + y**2) / (2 * self.sigma**2)))
            * np.exp(-(x**2 + y**2) / (2 * self.sigma**2))
        )
        kernel = kernel / np.sum(np.abs(kernel))
        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            img = convolve(img, kernel)
            img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img = convolve(img, kernel)
        return img, mask