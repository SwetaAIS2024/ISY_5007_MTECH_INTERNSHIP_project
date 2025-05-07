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
input_size = (output_data.shape[3], output_data.shape[2])  # (width, height)
original_image_resized = cv2.resize(original_image, input_size)

# Process the output to generate a mask
# Assuming the output is a 4D tensor: (batch_size, num_classes, height, width)
# Use argmax to get the class with the highest probability for each pixel
mask = np.argmax(output_data[0], axis=0)  # Remove batch dimension and get class indices
print("Generated mask shape:", mask.shape)

# Optionally, map class indices to colors for a colored mask
# Define a color map (e.g., for 3 classes: background, lane, road)
color_map = {
    0: [0, 0, 0],       # Black for background
    1: [255, 0, 0],     # Red for lane
    2: [0, 255, 0],     # Green for road
}

# Create a colored mask
colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
for class_id, color in color_map.items():
    colored_mask[mask == class_id] = color

# Save the mask as an image
cv2.imwrite(mask_output_file, colored_mask)
print(f"Mask saved to {mask_output_file}")

print("Original image resized shape:", original_image_resized.shape)
print("Colored mask shape:", colored_mask.shape)

# Ensure the mask is resized to match the original image dimensions
if colored_mask.shape[:2] != original_image.shape[:2]:
    colored_mask = cv2.resize(colored_mask, (original_image.shape[1], original_image.shape[0]))

# Create an overlay by blending the mask with the original image
overlay = original_image.copy()
non_zero_mask = (colored_mask != [0, 0, 0]).any(axis=2)  # Identify non-black pixels in the mask
overlay[non_zero_mask] = cv2.addWeighted(original_image[non_zero_mask], 0.5, colored_mask[non_zero_mask], 0.5, 0)

# Save the overlay image
cv2.imwrite(overlay_output_file, overlay)
print(f"Overlay saved to {overlay_output_file}")