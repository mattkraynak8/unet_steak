import cv2
import numpy as np
import os

# Define the directories
image_dir = './data/unet_1/raw/images'
mask_dir = './data/unet_1/raw/masks'
output_dir = './data/unet_2/raw/images'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to resize mask to match image's size with maintaining aspect ratio
def resize_mask_to_image(image, mask):
    image_h, image_w = image.shape[:2]
    mask_resized = cv2.resize(mask, (image_w, image_h), interpolation=cv2.INTER_AREA)
    return mask_resized

# Function to apply mask to the original image
def apply_mask_to_image(image, mask):
    # Make sure the mask is a binary mask
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Create an alpha channel based on the mask for transparency
    alpha_channel = np.where(binary_mask == 255, 255, 0).astype(np.uint8)

    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, image, mask=binary_mask)

    # If the image has 3 channels, merge them with the alpha channel
    if len(image.shape) == 3:
        b_channel, g_channel, r_channel = cv2.split(masked_image)
        transparent_image = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    else: # For grayscale images, just combine the image with the alpha channel
        transparent_image = cv2.merge((masked_image, alpha_channel))

    return transparent_image

# Get the list of image names from both directories
image_names = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
mask_names = [f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f))]

# Loop through all the images
for image_name in image_names:
    image_path = os.path.join(image_dir, image_name)
    mask_path = os.path.join(mask_dir, image_name.replace('.jpg', '.png'))  # Assuming mask extension is .png
    output_path = os.path.join(output_dir, image_name)

    # Skip processing if there is no corresponding mask file
    if image_name.replace('.jpg', '.png') not in mask_names:
        print(f"No corresponding mask for image {image_name}, skipping.")
        continue

    try:
        # Load the original image and the mask
        original_image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Check if both original image and mask were loaded successfully
        if original_image is None or mask is None:
            print(f"Could not load image or mask for {image_name}, skipping.")
            continue

        # Resize mask to match the original image's size
        resized_mask = resize_mask_to_image(original_image, mask)

        # Apply the mask to the original image
        result_image = apply_mask_to_image(original_image, resized_mask)

        # Save the result
        cv2.imwrite(output_path, result_image)

        print(f"Processed and saved {image_name}")

    except Exception as e:
        print(f"Failed to process {image_name} due to an error: {e}")
