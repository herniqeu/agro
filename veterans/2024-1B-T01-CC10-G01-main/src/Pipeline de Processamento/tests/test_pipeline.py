import unittest
import numpy as np

from image_process.pipeline import ProcessingPipeline

class TestProcessingPipeline(unittest.TestCase):
    def setUp(self):
        # Initialize the pipeline
        self.pipeline = ProcessingPipeline()
        
        # Create a dummy image and mask (10x10 pixels)
        self.dummy_img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        self.dummy_mask = np.random.randint(0, 2, (10, 10), dtype=np.uint8)
        
        class MockFilter:
            def apply(self, img, mask):
                return img * 0.9, np.zeros((10, 10), dtype=np.uint8) # Return filtered image and mask
        class MockAugmentation:
            def apply(self, img, mask):
                return img * 1.1, np.zeros((10, 10), dtype=np.uint8) # Return augmented image and mask
        
        self.mock_filter = MockFilter()
        self.mock_augmentation = MockAugmentation()

    def test_add_and_clear_filters(self):
        # Add filters and check
        self.pipeline.add_filters([self.mock_filter])
        self.assertEqual(len(self.pipeline.filters), 1)
        # Clear filters and check
        self.pipeline.clear_filters()
        self.assertEqual(len(self.pipeline.filters), 0)

    def test_add_and_clear_augmentations(self):
        # Add augmentations and check
        self.pipeline.add_augmentations([self.mock_augmentation])
        self.assertEqual(len(self.pipeline.augmentations), 1)
        # Clear augmentations and check
        self.pipeline.clear_augmentations()
        self.assertEqual(len(self.pipeline.augmentations), 0)

    def test_apply_filters(self):
        # Apply filters
        self.pipeline.add_filters([self.mock_filter])
        processed_img = self.pipeline.apply_filters(self.dummy_img)
        np.testing.assert_array_almost_equal(processed_img, self.dummy_img * 0.9)

    def test_apply_crop(self):
        # Testing cropping
        cropped_images, cropped_masks, crop_coordinates = self.pipeline.apply_crop(
            self.dummy_img, self.dummy_mask, new_width=5, new_height=5, n=1
        )
        self.assertEqual(len(cropped_images), 1)
        self.assertEqual(cropped_images[0].shape, (5, 5, 3))
        self.assertEqual(cropped_masks[0].shape, (5, 5))

    def test_apply_augmentations(self):
        # Apply augmentations
        self.pipeline.add_augmentations([self.mock_augmentation])
        images, masks = self.pipeline.apply_augmentations(
            [self.dummy_img], [self.dummy_mask], n=1
        )
        np.testing.assert_array_almost_equal(images[1], self.dummy_img * 1.1)

    def test_run_pipeline(self):
        # Run the entire pipeline
        self.pipeline.add_filters([self.mock_filter])
        self.pipeline.add_augmentations([self.mock_augmentation])
        normalized_imgs, normalized_masks, coords = self.pipeline.run(
            self.dummy_img, self.dummy_mask, n_augmented=1, crop_size=5, n_crop=1
        )
        # Assert the size of outputs
        self.assertTrue(len(normalized_imgs) > 0)
        self.assertTrue(len(normalized_masks) > 0)
        self.assertTrue(len(coords) > 0)

if __name__ == '__main__':
    unittest.main()
