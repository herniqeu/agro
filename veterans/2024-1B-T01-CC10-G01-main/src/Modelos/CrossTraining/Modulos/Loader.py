import numpy as np

def get_masks_and_images_as_np_array(image_data_manager, pipeline, RAW_IMAGE_SIZE, CROP_SIZE):
    masks_as_np_array = []
    images_as_np_array = []

    for key in image_data_manager.objects.keys():
        images = image_data_manager.objects[key]['images']
        mask = image_data_manager.objects[key]['mask']

        filtered_images = []

        for image in images:
            if image.shape[1] == RAW_IMAGE_SIZE:
                filtered_images.append(np.transpose(image, (1, 2, 0)))

        image_data_manager.objects[key]['images'] = filtered_images

        image = filtered_images[-1]
        n_crop = image.shape[0] // CROP_SIZE
        images, masks, coordinates = pipeline.run(image, mask, crop_size=CROP_SIZE, n_crop=n_crop, n_augmented=0)

        masks_as_np_array.extend(masks)
        images_as_np_array.extend(images)

    print(f"Loaded {len(images_as_np_array)} images")

    return masks_as_np_array, images_as_np_array
