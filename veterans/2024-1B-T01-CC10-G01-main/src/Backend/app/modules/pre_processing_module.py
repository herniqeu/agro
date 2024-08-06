class ImageCropper:
    """
    Handles image cropping with various offset options.

    Attributes:
        crop_width (int): Width of the crop.
        crop_height (int): Height of the crop.
    """

    def __init__(self, crop_width=120, crop_height=120):
        """
        Initializes the image cropper with given crop width and height.

        Parameters:
            crop_width (int): Width of the crop.
            crop_height (int): Height of the crop.
        """
        self.crop_width = crop_width
        self.crop_height = crop_height

    def crop_image(self, image_array):
        """
        Crops the image into multiple smaller images.

        Parameters:
            image_array (numpyx.ndarray): Array representation of the image.

        Returns:
            tuple: List of cropped images and their coordinates.
        """
        original_height, original_width = image_array.shape[:2]
        if self.crop_width > original_width or self.crop_height > original_height:
            raise ValueError("Crop dimensions must be smaller than the original dimensions.")

        cropped_images, crop_coords = [], []

        num_height_crops = original_height // self.crop_height
        num_width_crops = original_width // self.crop_width

        for row in range(num_height_crops):
            for col in range(num_width_crops):
                top = row * self.crop_height
                left = col * self.crop_width
                cropped_img = image_array[top:top + self.crop_height, left:left + self.crop_width]
                cropped_images.append(cropped_img)
                crop_coords.append(((left, top), (left + self.crop_width, top + self.crop_height)))

        return cropped_images, crop_coords

    def crop_image_with_x_offset(self, image_array):
        """
        Crops the image with a horizontal offset.

        Parameters:
            image_array (numpy.ndarray): Array representation of the image.

        Returns:
            tuple: List of cropped images and their coordinates.
        """
        original_height, original_width = image_array.shape[:2]
        if self.crop_width > original_width or self.crop_height > original_height:
            raise ValueError("Crop dimensions must be smaller than the original dimensions.")

        cropped_images, crop_coords = [], []

        num_height_crops = original_height // self.crop_height
        num_width_crops = original_width // self.crop_width

        offset_width = self.crop_width // 2

        for row in range(num_height_crops):
            for col in range(num_width_crops):
                top = row * self.crop_height
                left = col * self.crop_width + offset_width
                if top + self.crop_height > original_height or left + self.crop_width > original_width:
                    continue
                cropped_img = image_array[top:top + self.crop_height, left:left + self.crop_width]
                cropped_images.append(cropped_img)
                crop_coords.append(((left, top), (left + self.crop_width, top + self.crop_height)))

        return cropped_images, crop_coords

    def crop_image_with_xy_offset(self, image_array):
        """
        Crops the image with both horizontal and vertical offsets.

        Parameters:
            image_array (numpy.ndarray): Array representation of the image.

        Returns:
            tuple: List of cropped images and their coordinates.
        """
        original_height, original_width = image_array.shape[:2]
        if self.crop_width > original_width or self.crop_height > original_height:
            raise ValueError("Crop dimensions must be smaller than the original dimensions.")

        cropped_images, crop_coords = [], []

        num_height_crops = original_height // self.crop_height
        num_width_crops = original_width // self.crop_width

        offset_height = self.crop_height // 2
        offset_width = self.crop_width // 2

        for row in range(num_height_crops):
            for col in range(num_width_crops):
                top = row * self.crop_height + offset_height
                left = col * self.crop_width + offset_width
                if top + self.crop_height > original_height or left + self.crop_width > original_width:
                    continue
                cropped_img = image_array[top:top + self.crop_height, left:left + self.crop_width]
                cropped_images.append(cropped_img)
                crop_coords.append(((left, top), (left + self.crop_width, top + self.crop_height)))

        return cropped_images, crop_coords