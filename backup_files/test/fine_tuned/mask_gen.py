import numpy as np
import cv2

# Paths to input and output files
inference_output_file = "./output_data.npy"  # Path to the inference results
original_image_file = "./3_lane.png"        # Path to the original image
overlay_output_file = "./overlay.png"       # Path to save the overlay image
mask_output_file = "./mask.png"             # Path to save the generated mask

# Load the inference output
output_data = np.load(inference_output_file)
print("Loaded inference output shape:", output_data.shape)

# Load the original image
original_image = cv2.imread(original_image_file)
if original_image is None:
    raise FileNotFoundError(f"Original image not found at {original_image_file}")

# Resize the original image to match the model's input size
input_size = (output_data.shape[2], output_data.shape[1])  # (width, height)
original_image_resized = cv2.resize(original_image, input_size)

# Process the output to generate a mask
mask = np.squeeze(output_data[0], axis=-1)  # Remove batch and channel dimensions
if mask.ndim != 2:
    raise ValueError(f"Unexpected mask shape: {mask.shape}")
print("Generated mask shape:", mask.shape)

unique_classes = np.unique(mask)
print("Unique classes in the mask:", unique_classes)

# # Optionally, map class indices to colors for a colored mask
# color_map = {
#     40: [255, 0, 0], # Red for class 40
#     41: [0, 255, 0], # Green for class 41
#     43: [0, 0, 255], # Blue for class 43
# }

# Option 2: Assign colors to all classes
color_map = {class_id: np.random.randint(0, 256, size=3).tolist() for class_id in unique_classes}


# Create a colored mask
colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
for class_id, color in color_map.items():
    colored_mask[mask == class_id] = color

# Save the mask as an image
cv2.imwrite(mask_output_file, colored_mask)
print(f"Mask saved to {mask_output_file}")

# Resize the colored mask to match the original image dimensions
if colored_mask.shape[:2] != original_image.shape[:2]:
    colored_mask = cv2.resize(colored_mask, (original_image.shape[1], original_image.shape[0]))

# Create an overlay by blending the mask with the original image
overlay = original_image.copy()

# Identify non-black pixels in the mask
non_zero_mask = np.any(colored_mask != [0, 0, 0], axis=-1)  # Correctly identify non-black pixels

# Ensure non_zero_mask is valid
if non_zero_mask is None or not np.any(non_zero_mask):
    raise ValueError("No non-black pixels found in the mask. Check the color map or mask generation.")

# Blend the non-black pixels
overlay[non_zero_mask] = cv2.addWeighted(
    original_image[non_zero_mask], 0.5, colored_mask[non_zero_mask], 0.5, 0
)

# Save the overlay image
cv2.imwrite(overlay_output_file, overlay)
print(f"Overlay saved to {overlay_output_file}")



# import numpy as np
# import cv2

# # Paths to input and output files
# inference_output_file = "./output_data.npy"  # Path to the inference results
# original_image_file = "./3_lane.png"        # Path to the original image
# mask_output_file = "./mask.png"             # Path to save the generated mask
# overlay_output_file = "./overlay.png"       # Path to save the overlay image

# # Load the inference output
# output_data = np.load(inference_output_file)
# print("Loaded inference output shape:", output_data.shape)

# # Load the original image
# original_image = cv2.imread(original_image_file)
# if original_image is None:
#     raise FileNotFoundError(f"Original image not found at {original_image_file}")

# # Process the output to generate a mask
# mask = np.squeeze(output_data[0], axis=-1)  # Remove batch and channel dimensions
# if mask.ndim != 2:
#     raise ValueError(f"Unexpected mask shape: {mask.shape}")
# print("Generated mask shape:", mask.shape)

# # Map class indices to colors
# color_map = {
#     0: [0, 0, 0],       # Black for background
#     1: [255, 0, 0],     # Red for class 1
#     2: [0, 255, 0],     # Green for class 2
#     3: [0, 0, 255],     # Blue for class 3
# }

# # Create a colored mask
# colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
# for class_id, color in color_map.items():
#     colored_mask[mask == class_id] = color

# # Resize the colored mask to match the original image dimensions
# colored_mask_resized = cv2.resize(colored_mask, (original_image.shape[1], original_image.shape[0]))

# # Save the mask as an image
# cv2.imwrite(mask_output_file, colored_mask_resized)
# print(f"Mask saved to {mask_output_file}")

# # Overlay the mask on the original image
# overlay = cv2.addWeighted(original_image, 0.5, colored_mask_resized, 0.5, 0)
# cv2.imwrite(overlay_output_file, overlay)
# print(f"Overlay saved to {overlay_output_file}")