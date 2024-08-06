import cv2 as cv
import numpy as np
import random
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
    """
    Rotates an image by a specified angle.

    Attributes:
        angle (float): The angle by which the image will be rotated.
    """

    def __init__(self, angle):
        """
        Initializes the Rotate process with the given angle.

        Args:
            angle (float): The angle by which the image will be rotated.
        """
        self.angle = angle

    def apply(self, img, mask=None):
        """
        Applies the rotation to the image and optionally to the mask.

        Args:
            img: The image to rotate.
            mask: The mask to rotate (optional).

        Returns:
            The rotated image and the rotated mask (if provided).
        """
        height, width = img.shape[:2]
        rotation_matrix = cv.getRotationMatrix2D((width / 2, height / 2), self.angle, 1)
        rotated_img = cv.warpAffine(img, rotation_matrix, (width, height))
        rotated_mask = cv.warpAffine(mask, rotation_matrix, (width, height)) if mask is not None else None
        return rotated_img, rotated_mask



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
    Applies translation to an image using user-defined horizontal and vertical shifts.

    Attributes:
        dx (int): Horizontal shift defined by the user.
        dy (int): Vertical shift defined by the user.
    """

    def __init__(self, dx, dy):
        """
        Initializes the Translate class with user-defined shifts.

        Args:
            dx (int): Horizontal shift.
            dy (int): Vertical shift.
        """
        self.dx = dx
        self.dy = dy

    def apply(self, img, mask=None):
        """
        Applies the translation to the image and mask.

        Args:
            img (np.ndarray): The input image.
            mask (np.ndarray, optional): The input mask. Defaults to None.

        Returns:
            tuple: Translated image and translated mask (if provided).
        """
        translation_matrix = np.float32([[1, 0, self.dx], [0, 1, self.dy]])
        height, width = img.shape[:2]
        translated_img = cv.warpAffine(img, translation_matrix, (width, height))
        translated_mask = (
            cv.warpAffine(mask, translation_matrix, (width, height))
            if mask is not None
            else None
        )
        return translated_img, translated_mask

class Flip(BaseImageProcess):
    """
    Flips an image either horizontally, vertically, or both, based on the specified flip type.

    Attributes:
        flip_type (int): Type of flip to apply; -1 for both axes, 0 for vertical, 1 for horizontal.
    """

    def __init__(self, flip_type):
        """
        Initializes the Flip process with the given flip type.

        Args:
            flip_type (int): Type of flip to apply; -1 for both axes, 0 for vertical, 1 for horizontal.
        """
        if flip_type not in [-1, 0, 1]:
            raise ValueError("flip_type must be -1, 0, or 1.")
        self.flip_type = flip_type

    def apply(self, img, mask=None):
        flipped_img = cv.flip(img, self.flip_type)
        flipped_mask = cv.flip(mask, self.flip_type) if mask is not None else None
        return flipped_img, flipped_mask



class BrightnessContrast(BaseImageProcess):
    """
    Adjusts the brightness and contrast of an image using specified values.

    Attributes:
        alpha (float): Factor by which the contrast will be adjusted.
        beta (int): Value that will be added to the pixels for brightness adjustment.
    """

    def __init__(self, alpha, beta):
        """
        Initializes the BrightnessContrast process with the given alpha and beta values.

        Args:
            alpha (float): Factor by which the contrast will be adjusted.
            beta (int): Value that will be added to the pixels for brightness adjustment.
        """
        self.alpha = alpha
        self.beta = beta

    def apply(self, img, mask=None):
        return cv.convertScaleAbs(img, alpha=self.alpha, beta=self.beta), mask



class MedianBlur(BaseImageProcess):
    """
    Applies median blurring to an image using a specified kernel size.

    Attributes:
        kernel_size (int): The size of the kernel used for median blurring.
    """

    def __init__(self, kernel_size):
        """
        Initializes the MedianBlur process with the given kernel size.

        Args:
            kernel_size (int): The size of the kernel used for median blurring. Must be an odd integer.
        """
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer.")
        self.kernel_size = kernel_size

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
    Applies Gaussian blur filtering to an image with a specified kernel size.

    Attributes:
        kernel_size (int): Size of the Gaussian blur kernel.
    """

    def __init__(self, kernel_size):
        """
        Initializes the GaussianBlur process with the given kernel size.

        Args:
            kernel_size (int): Size of the Gaussian blur kernel.
        """
        if kernel_size not in [3, 5, 7, 9, 11]:
            raise ValueError("kernel_size must be one of the following: 3, 5, 7, 9, 11.")
        self.kernel_size = kernel_size

    def apply(self, img, mask=None):
        """
        Applies the Gaussian blur to the image.

        Args:
            img (np.ndarray): The input image.
            mask (np.ndarray, optional): The input mask. Defaults to None.

        Returns:
            tuple: Blurred image and the mask (if provided).
        """
        blurred_img = cv.GaussianBlur(img, (self.kernel_size, self.kernel_size), 0)
        return blurred_img, mask


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
