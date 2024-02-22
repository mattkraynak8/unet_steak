import os
import numpy as np
from patchify import patchify
from PIL import Image
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import logging
import shutil

class ImagePatcher:
    def __init__(self, patch_size=(512, 512, 3), step=512):
        self.patch_size = patch_size
        self.step = step

    def create_patches(self, src, dest_path):
        if not os.path.exists(src):
            logging.error(f"Source directory {src} does not exist.")
            return

        if os.path.exists(dest_path):
            shutil.rmtree(dest_path)
        os.makedirs(dest_path, exist_ok=True)
        
        # Use a pool of workers to process images in parallel
        with ProcessPoolExecutor() as executor:
            for file_name in os.listdir(src):
                executor.submit(self.process_image, src, dest_path, file_name)

    def process_image(self, src, dest_path, file_name):
        if not file_name.endswith(('png', 'jpg', 'jpeg')):  # Skip non-image files
            return

        file_path = os.path.join(src, file_name)
        image = Image.open(file_path)
        image_array = np.asarray(image)

        if image_array.ndim == 2:
            image_array = image_array[:, :, np.newaxis]

        if image_array.shape[0] < self.patch_size[0] or image_array.shape[1] < self.patch_size[1]:
            logging.warning(f"Skipping {file_name}: Image size is smaller than the minimum patch size.")
            return

        patches = patchify(image_array, (self.patch_size[0], self.patch_size[1], image_array.shape[2]), step=self.step)
        
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                patch = patches[i, j, 0, :, :, :]
                if patch.ndim == 3 and patch.shape[2] == 1:
                    patch = patch[..., 0]
                patch_image = Image.fromarray(patch)
                patch_filename = f"{Path(file_name).stem}_patch{i}_{j}.png"
                patch_image.save(os.path.join(dest_path, patch_filename))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test = ImagePatcher()
    test.create_patches(src='/steak_image_segmentation/data/unet_1/raw/images', dest_path='/steak_image_segmentation/data/unet_1/processed/images')
    test.create_patches(src='/steak_image_segmentation/data/unet_1/raw/masks', dest_path='/steak_image_segmentation/data/unet_1/processed/masks')

    logging.basicConfig(level=logging.INFO)
    test = ImagePatcher()
    test.create_patches(src='/steak_image_segmentation/data/unet_2/raw/images', dest_path='/steak_image_segmentation/data/unet_2/processed/images')
    test.create_patches(src='/steak_image_segmentation/data/unet_2/raw/masks', dest_path='/steak_image_segmentation/data/unet_2/processed/masks')
