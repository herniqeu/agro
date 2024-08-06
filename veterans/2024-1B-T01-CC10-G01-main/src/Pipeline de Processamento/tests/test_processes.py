import unittest
from unittest.mock import patch
import numpy as np
import cv2 as cv
from image_process.processes import Rotate, BilateralFilter, Translate, GaussianBlur, BinaryThresh, AdaptiveMeanThresh, AdaptiveGaussThresh, OtsuThresh, MorphDilate, MorphErode, LoG, LoGConv

class TestRotate(unittest.TestCase):
    def test_rotate_image(self):
        # Mocking random.choice to always return a specific angle
        with patch('random.choice', return_value=10):
            rotator = Rotate()
            self.assertEqual(rotator.angle, 10)

            # Create a simple test image (100x100 black square)
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            img[25:75, 25:75] = 255  # Add a white square in the center

            # Rotate the image
            rotated_img, _ = rotator.apply(img)

            # Check if image is rotated correctly
            # Note: This check can be more complex depending on how exact you need the match to be.
            # Using OpenCV to find the white square might be necessary if exact pixel matching isn't feasible.
            self.assertNotEqual(np.sum(img), np.sum(rotated_img))

    def test_rotate_image_with_mask(self):
        with patch('random.choice', return_value=-5):
            rotator = Rotate()
            self.assertEqual(rotator.angle, -5)

            # Create a test image and a mask
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            mask = np.zeros((100, 100), dtype=np.uint8)
            img[25:75, 25:75] = 255  # White square
            mask[25:75, 25:75] = 255  # Same mask as image

            # Rotate both image and mask
            rotated_img, rotated_mask = rotator.apply(img, mask)

            # Check both image and mask are rotated
            self.assertNotEqual(np.sum(img), np.sum(rotated_img))
            self.assertNotEqual(np.sum(mask), np.sum(rotated_mask))
            self.assertIsNotNone(rotated_mask)

class TestBilateralFilter(unittest.TestCase):
    
    def setUp(self):
        self.bilateral_filter = BilateralFilter()

    def test_apply_without_mask(self):
        """
            This test ensures that the method runs and returns output in the expected
            format (a tuple containing the filtered image and the mask
        """
        # Create a dummy image
        img = np.random.rand(100, 100, 3).astype(np.uint8)
        
        # Apply the bilateral filter
        filtered_image, mask = self.bilateral_filter.apply(img)
        
        # Assert the returned objects are not None
        self.assertIsNotNone(filtered_image, "Filtered image should not be None.")
        self.assertIsNone(mask, "Mask should be None when not provided.")

        # Assert the shape of the returned image is the same as the input
        self.assertEqual(img.shape, filtered_image.shape, "Filtered image should have the same shape as input image.")

    def test_apply_with_mask(self):
        # Create a dummy image and a dummy mask
        img = np.random.rand(100, 100, 3).astype(np.uint8)
        mask = np.ones((100, 100), dtype=np.uint8)
        
        # Apply the bilateral filter with a mask
        filtered_image, returned_mask = self.bilateral_filter.apply(img, mask)
        
        # Assert the returned objects are not None
        self.assertIsNotNone(filtered_image, "Filtered image should not be None.")
        self.assertIsNotNone(returned_mask, "Returned mask should not be None.")
        
        # Assert the returned mask is the same as the input mask
        self.assertIs(mask, returned_mask, "Returned mask should be the same object as the input mask.")

class TestTranslate(unittest.TestCase):
    """
        Since the translations dx and dy are randomly chosen, the random.choice function
        is mocked to control the randomness for testing purposes
    """

    @patch('random.choice', side_effect=[5, 10])
    def test_apply_translation(self, mock_random_choice):
        # Initialize the Translate object which should call random.choice
        translator = Translate()
        self.assertEqual(translator.dx, 5)
        self.assertEqual(translator.dy, 10)

        # Create a dummy image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[25:75, 25:75] = 255  # Add a white square in the center for easy visibility of translation

        # Apply the translation
        translated_img, mask = translator.apply(img)

        # Assert the mask is None
        self.assertIsNone(mask, "Mask should be None when not provided.")

        # Assert the translated image is not the same as the original
        self.assertFalse(np.array_equal(img, translated_img), "Translated image should differ from the input image.")

    @patch('random.choice', side_effect=[-5, -10])
    def test_apply_translation_with_mask(self, mock_random_choice):
        # Initialize the Translate object which should call random.choice
        translator = Translate()
        self.assertEqual(translator.dx, -5)
        self.assertEqual(translator.dy, -10)

        # Create a dummy image and mask
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.ones((100, 100), dtype=np.uint8)
        img[25:75, 25:75] = 255  # White square
        mask[25:75, 25:75] = 255  # Same mask as image

        # Apply the translation with mask
        translated_img, translated_mask = translator.apply(img, mask)

        # Assert the translated mask is not None
        self.assertIsNotNone(translated_mask, "Returned mask should not be None.")

        # Assert the translated image and mask are not the same as the originals
        self.assertFalse(np.array_equal(img, translated_img), "Translated image should differ from the input image.")
        self.assertFalse(np.array_equal(mask, translated_mask), "Translated mask should differ from the input mask.")

class TestGaussianBlur(unittest.TestCase):

    def setUp(self):
        self.gaussian_blur = GaussianBlur(kernel_size=5)

    def test_apply_gaussian_blur(self):
        # Create a dummy image with noise
        img = np.random.rand(100, 100, 3).astype(np.uint8)

        # Apply the Gaussian blur
        blurred_img, mask = self.gaussian_blur.apply(img)

        # Assert the returned objects are not None
        self.assertIsNotNone(blurred_img, "Blurred image should not be None.")
        self.assertIsNone(mask, "Mask should be None when not provided.")

        # Assert the shape of the returned image is the same as the input
        self.assertEqual(img.shape, blurred_img.shape, "Blurred image should have the same shape as input image.")

    def test_apply_gaussian_blur_with_mask(self):
        # Create a dummy image with noise and a mask
        img = np.random.rand(100, 100, 3).astype(np.uint8)
        mask = np.ones((100, 100), dtype=np.uint8)

        # Apply the Gaussian blur with a mask
        blurred_img, returned_mask = self.gaussian_blur.apply(img, mask)

        # Assert the returned objects are not None
        self.assertIsNotNone(blurred_img, "Blurred image should not be None.")
        self.assertIsNotNone(returned_mask, "Returned mask should not be None.")

        # Assert the returned mask is the same as the input mask
        self.assertIs(mask, returned_mask, "Returned mask should be the same object as the input mask.")

class TestBinaryThresh(unittest.TestCase):

    def setUp(self):
        self.binary_thresh = BinaryThresh()

    def test_apply_binary_threshold(self):
        # Create a grayscale image with a gradient
        img = np.linspace(0, 255, 100*100, dtype=np.uint8).reshape((100, 100))

        # Apply the binary threshold
        thresholded_img, mask = self.binary_thresh.apply(img)

        # Assert the returned objects are not None
        self.assertIsNotNone(thresholded_img, "Thresholded image should not be None.")
        self.assertIsNone(mask, "Mask should be None when not provided.")

        # Assert the shape of the returned image is the same as the input
        self.assertEqual(img.shape, thresholded_img.shape, "Thresholded image should have the same shape as input image.")

        # Check if all pixels are either 0 or max_val
        unique_values = np.unique(thresholded_img)
        self.assertTrue(np.array_equal(unique_values, [0, self.binary_thresh.max_val]),
                        "Thresholded image pixels should be only 0 or max_val.")

    def test_apply_binary_threshold_with_mask(self):
        # Create a grayscale image with a gradient and a mask
        img = np.linspace(0, 255, 100*100, dtype=np.uint8).reshape((100, 100))
        mask = np.ones((100, 100), dtype=np.uint8)

        # Apply the binary threshold with a mask
        thresholded_img, returned_mask = self.binary_thresh.apply(img, mask)

        # Assert the returned objects are not None
        self.assertIsNotNone(thresholded_img, "Thresholded image should not be None.")
        self.assertIsNotNone(returned_mask, "Returned mask should not be None.")

        # Assert the returned mask is the same as the input mask
        self.assertIs(mask, returned_mask, "Returned mask should be the same object as the input mask.")

class TestAdaptiveMeanThresh(unittest.TestCase):

    def setUp(self):
        self.adaptive_thresh = AdaptiveMeanThresh()

    def test_apply_adaptive_mean_threshold(self):
        # Create a grayscale image with varying intensities
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

        # Apply the adaptive mean threshold
        thresholded_img, mask = self.adaptive_thresh.apply(img)

        # Assert the returned objects are not None
        self.assertIsNotNone(thresholded_img, "Thresholded image should not be None.")
        self.assertIsNone(mask, "Mask should be None when not provided.")

        # Assert the shape of the returned image is the same as the input
        self.assertEqual(img.shape, thresholded_img.shape, "Thresholded image should have the same shape as input image.")

        # Check the image type is binary (0 or 255)
        unique_values = np.unique(thresholded_img)
        self.assertTrue(set(unique_values).issubset({0, 255}),
                        "All pixels in the thresholded image should be either 0 or 255 after thresholding.")

    def test_apply_adaptive_mean_threshold_with_mask(self):
        # Create a grayscale image with varying intensities and a mask
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        mask = np.ones((100, 100), dtype=np.uint8)

        # Apply the adaptive mean threshold with a mask
        thresholded_img, returned_mask = self.adaptive_thresh.apply(img, mask)

        # Assert the returned objects are not None
        self.assertIsNotNone(thresholded_img, "Thresholded image should not be None.")
        self.assertIsNotNone(returned_mask, "Returned mask should not be None.")

        # Assert the returned mask is the same as the input mask
        self.assertIs(mask, returned_mask, "Returned mask should be the same object as the input mask.")

class TestAdaptiveGaussThresh(unittest.TestCase):

    def setUp(self):
        self.adaptive_gauss_thresh = AdaptiveGaussThresh()

    def test_apply_adaptive_gaussian_threshold(self):
        # Create a grayscale image with varying intensities
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

        # Apply the adaptive Gaussian threshold
        thresholded_img, mask = self.adaptive_gauss_thresh.apply(img)

        # Assert the returned objects are not None
        self.assertIsNotNone(thresholded_img, "Thresholded image should not be None.")
        self.assertIsNone(mask, "Mask should be None when not provided.")

        # Assert the shape of the returned image is the same as the input
        self.assertEqual(img.shape, thresholded_img.shape, "Thresholded image should have the same shape as input image.")

        # Check the image type is binary (0 or 255)
        unique_values = np.unique(thresholded_img)
        self.assertTrue(set(unique_values).issubset({0, 255}),
                        "All pixels in the thresholded image should be either 0 or 255 after thresholding.")

    def test_apply_adaptive_gaussian_threshold_with_mask(self):
        # Create a grayscale image with varying intensities and a mask
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        mask = np.ones((100, 100), dtype=np.uint8)

        # Apply the adaptive Gaussian threshold with a mask
        thresholded_img, returned_mask = self.adaptive_gauss_thresh.apply(img, mask)

        # Assert the returned objects are not None
        self.assertIsNotNone(thresholded_img, "Thresholded image should not be None.")
        self.assertIsNotNone(returned_mask, "Returned mask should not be None.")

        # Assert the returned mask is the same as the input mask
        self.assertIs(mask, returned_mask, "Returned mask should be the same object as the input mask.")

class TestOtsuThresh(unittest.TestCase):

    def setUp(self):
        self.otsu_thresh = OtsuThresh()

    def test_apply_otsu_threshold(self):
        # Create a grayscale image with bimodal histogram, ideal for Otsu's method
        img = np.zeros((100, 100), dtype=np.uint8)
        img[25:75, 25:75] = 255

        # Apply Otsu's threshold
        thresholded_img, mask = self.otsu_thresh.apply(img)

        # Assert the returned objects are not None
        self.assertIsNotNone(thresholded_img, "Thresholded image should not be None.")
        self.assertIsNone(mask, "Mask should be None when not provided.")

        # Assert the shape of the returned image is the same as the input
        self.assertEqual(img.shape, thresholded_img.shape, "Thresholded image should have the same shape as input image.")

        # Check the image type is binary (0 or 255)
        unique_values = np.unique(thresholded_img)
        self.assertTrue(set(unique_values).issubset({0, 255}),
                        "All pixels in the thresholded image should be either 0 or 255 after Otsu's thresholding.")

    def test_apply_otsu_threshold_with_mask(self):
        # Create a grayscale image with bimodal histogram and a mask
        img = np.zeros((100, 100), dtype=np.uint8)
        img[25:75, 25:75] = 255
        mask = np.ones((100, 100), dtype=np.uint8)

        # Apply Otsu's threshold with a mask
        thresholded_img, returned_mask = self.otsu_thresh.apply(img, mask)

        # Assert the returned objects are not None
        self.assertIsNotNone(thresholded_img, "Thresholded image should not be None.")
        self.assertIsNotNone(returned_mask, "Returned mask should not be None.")

        # Assert the returned mask is the same as the input mask
        self.assertIs(mask, returned_mask, "Returned mask should be the same object as the input mask.")

class TestMorphDilate(unittest.TestCase):

    def setUp(self):
        self.morph_dilate = MorphDilate()

    def test_apply_morph_dilate(self):
        # Create an image with a small white square in the center
        img = np.zeros((100, 100), dtype=np.uint8)
        cv.rectangle(img, (40, 40), (60, 60), 255, -1)

        # Apply morphological dilation
        dilated_img, mask = self.morph_dilate.apply(img)

        # Check that dilated image is not the same as the original
        self.assertFalse(np.array_equal(img, dilated_img), "Dilated image should differ from the input image.")

        # Ensure the dilation has actually increased the white region
        original_white_area = np.sum(img == 255)
        dilated_white_area = np.sum(dilated_img == 255)
        self.assertGreater(dilated_white_area, original_white_area, "Dilated image should have more white area than the original.")

        # Assert the mask is None if not provided
        self.assertIsNone(mask, "Mask should be None when not provided.")

    def test_apply_morph_dilate_with_mask(self):
        # Create an image with a small white square and a mask
        img = np.zeros((100, 100), dtype=np.uint8)
        mask = np.ones((100, 100), dtype=np.uint8)
        cv.rectangle(img, (40, 40), (60, 60), 255, -1)

        # Apply morphological dilation with a mask
        dilated_img, returned_mask = self.morph_dilate.apply(img, mask)

        # Ensure the dilation has increased the white region
        original_white_area = np.sum(img == 255)
        dilated_white_area = np.sum(dilated_img == 255)
        self.assertGreater(dilated_white_area, original_white_area, "Dilated image should have more white area than the original.")

        # Assert the returned mask is the same as the input mask
        self.assertIs(mask, returned_mask, "Returned mask should be the same object as the input mask.")

class TestMorphErode(unittest.TestCase):

    def setUp(self):
        self.morph_erode = MorphErode()

    def test_apply_morph_erode(self):
        # Create an image with a larger white square in the center
        img = np.zeros((100, 100), dtype=np.uint8)
        cv.rectangle(img, (30, 30), (70, 70), 255, -1)  # A 40x40 white square

        # Apply morphological erosion
        eroded_img, mask = self.morph_erode.apply(img)

        # Ensure the erosion has actually decreased the white region
        original_white_area = np.sum(img == 255)
        eroded_white_area = np.sum(eroded_img == 255)
        self.assertLess(eroded_white_area, original_white_area, "Eroded image should have less white area than the original.")

        # Check that eroded image is not the same as the original
        self.assertFalse(np.array_equal(img, eroded_img), "Eroded image should differ from the input image.")

        # Assert the mask is None if not provided
        self.assertIsNone(mask, "Mask should be None when not provided.")

    def test_apply_morph_erode_with_mask(self):
        # Create an image with a larger white square and a mask
        img = np.zeros((100, 100), dtype=np.uint8)
        mask = np.ones((100, 100), dtype=np.uint8)
        cv.rectangle(img, (30, 30), (70, 70), 255, -1)

        # Apply morphological erosion with a mask
        eroded_img, returned_mask = self.morph_erode.apply(img, mask)

        # Ensure the erosion has decreased the white region
        original_white_area = np.sum(img == 255)
        eroded_white_area = np.sum(eroded_img == 255)
        self.assertLess(eroded_white_area, original_white_area, "Eroded image should have less white area than the original.")

        # Assert the returned mask is the same as the input mask
        self.assertIs(mask, returned_mask, "Returned mask should be the same object as the input mask.")

class TestLoG(unittest.TestCase):

    def setUp(self):
        self.log_filter = LoG(sigma=2.0)

    def test_apply_log_filter(self):
        # Create an image with a distinct edge
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:, 50:] = 255  # Half black, half white

        # Apply the LoG filter
        log_img, mask = self.log_filter.apply(img)

        # Ensure the output is processed (should not be the same as the input)
        self.assertFalse(np.array_equal(img, log_img), "LoG-filtered image should differ from the input image.")

        # Check for expected properties in the output
        # For example, zero-crossings or negative values near the edges
        central_row = log_img[50, :]
        sign_changes = np.where(np.diff(np.sign(central_row)))[0]
        self.assertTrue(len(sign_changes) > 0, "There should be sign changes in the output, indicating zero-crossings.")

        # Assert the mask is None if not provided
        self.assertIsNone(mask, "Mask should be None when not provided.")

    def test_apply_log_filter_with_mask(self):
        # Create an image with a distinct edge and a mask
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:, 50:] = 255
        mask = np.ones((100, 100), dtype=np.uint8)

        # Apply the LoG filter with a mask
        log_img, returned_mask = self.log_filter.apply(img, mask)

        # Ensure the output is processed
        self.assertFalse(np.array_equal(img, log_img), "LoG-filtered image should differ from the input image.")

        # Assert the returned mask is the same as the input mask
        self.assertIs(mask, returned_mask, "Returned mask should be the same object as the input mask.")

class TestLoGConv(unittest.TestCase):

    def setUp(self):
        self.log_conv = LoGConv(sigma=2.0)

    def test_apply_log_conv_filter(self):
        # Create an image with distinct edges
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[25:75, 25:75, :] = 255  # A white square in the center of a black background

        # Apply the LoG convolution filter
        filtered_img, mask = self.log_conv.apply(img)

        # Check that the filtered image is different from the original
        self.assertFalse(np.array_equal(img, filtered_img), "Filtered image should differ from the input image.")

        # Check the type and range of the filtered image
        self.assertTrue(np.all((filtered_img >= 0) & (filtered_img <= 255)), "Filtered image pixels should be within 0-255.")

        # Check for the presence of edges
        edge_detected = np.any(filtered_img[25:75, :] != filtered_img[0, 0])
        self.assertTrue(edge_detected, "Edges should be enhanced or visible in the filtered image.")

        # Assert the mask is None if not provided
        self.assertIsNone(mask, "Mask should be None when not provided.")

    def test_apply_log_conv_filter_with_mask(self):
        # Create an image with distinct edges and a mask
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[25:75, 25:75, :] = 255
        mask = np.ones((100, 100), dtype=np.uint8)

        # Apply the LoG convolution filter with a mask
        filtered_img, returned_mask = self.log_conv.apply(img, mask)

        # Check the type and range of the filtered image
        self.assertTrue(np.all((filtered_img >= 0) & (filtered_img <= 255)), "Filtered image pixels should be within 0-255.")

        # Assert the returned mask is the same as the input mask
        self.assertIs(mask, returned_mask, "Returned mask should be the same object as the input mask.")

if __name__ == '__main__':
    unittest.main()