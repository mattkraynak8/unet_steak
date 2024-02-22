import os
from PIL import Image

# Try to set the resampling filter based on the Pillow version
try:
    # For newer versions of Pillow (>= 9.0.0)
    resampling_filter = Image.Resampling.LANCZOS
except AttributeError:
    try:
        # For Pillow versions that have LANCZOS but not ImageResampling
        resampling_filter = Image.LANCZOS
    except AttributeError:
        # For older versions of Pillow, where ANTIALIAS is available and LANCZOS is equivalent
        resampling_filter = 1  # ANTIALIAS

class BulkImageResizer:
    def __init__(self, input_path, output_path, size=(512, 512)):
        self.input_path = input_path
        self.output_path = output_path
        self.size = size
        # Ensure the output directory exists
        os.makedirs(self.output_path, exist_ok=True)
        # Check if the input directory exists
        if not os.path.exists(self.input_path):
            raise ValueError(f"Source directory {self.input_path} does not exist.")

    def resize_image(self, image_name):
        input_file_path = os.path.join(self.input_path, image_name)
        output_file_path = os.path.join(self.output_path, image_name)

        if not os.path.exists(output_file_path):
            try:
                with Image.open(input_file_path) as img:
                    # Resize the image using the correct resampling filter
                    img = img.resize(self.size, resampling_filter)
                    img.save(output_file_path, img.format, quality=95)
            except IOError as e:
                print(f"Unable to resize {image_name}: {e}")

    def resize_images_in_directory(self):
        for image_name in os.listdir(self.input_path):
            if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.resize_image(image_name)

# Example usage
images_path = './data/unet_1/raw/images'
output_images_path = './data/unet_1/processed/images'
masks_path = './data/unet_1/raw/masks'
output_masks_path = './data/unet_1/processed/masks'

image_resizer = BulkImageResizer(images_path, output_images_path)
mask_resizer = BulkImageResizer(masks_path, output_masks_path)

image_resizer.resize_images_in_directory()
mask_resizer.resize_images_in_directory()
