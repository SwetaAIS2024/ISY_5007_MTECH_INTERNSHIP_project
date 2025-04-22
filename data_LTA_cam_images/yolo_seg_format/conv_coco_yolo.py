# import json
# import os

# # Input COCO JSON file
# coco_json_path = "/home/rubesh/Desktop/sweta/Mtech_internship/ISY_5007_MTECH_INTERNSHIP_project/data_LTA_cam_images/yolo_seg_format/4708_20250421_160658_coco.json"

# # Output directory for YOLO labels
# output_dir = "/home/rubesh/Desktop/sweta/Mtech_internship/ISY_5007_MTECH_INTERNSHIP_project/data_LTA_cam_images/yolo_labels/"
# os.makedirs(output_dir, exist_ok=True)

# # Load COCO JSON
# with open(coco_json_path, "r") as f:
#     coco_data = json.load(f)

# # Get the single image metadata
# image = coco_data["images"][0]  # Access the first (and only) image
# image_id = image["id"]
# file_name = os.path.splitext(image["file_name"])[0]
# label_path = os.path.join(output_dir, f"{file_name}.txt")

# # Collect annotations for this image
# labels = []
# for annotation in coco_data["annotations"]:
#     if annotation["image_id"] == image_id:
#         class_id = annotation["category_id"] - 1  # YOLO class IDs start from 0
#         segmentation = annotation["segmentation"][0]

#         # Normalize segmentation points
#         normalized_points = []
#         for i in range(0, len(segmentation), 2):
#             x = segmentation[i] / image["width"]
#             y = segmentation[i + 1] / image["height"]
#             normalized_points.extend([x, y])

#         # Create YOLO label line
#         labels.append(f"{class_id} " + " ".join(map(str, normalized_points)))

# # Write labels to file
# with open(label_path, "w") as f:
#     f.write("\n".join(labels))

# print(f"YOLO label saved to: {label_path}")

import json
import os

# Input folder containing COCO JSON files
coco_json_folder = "/home/rubesh/Desktop/sweta/Mtech_internship/ISY_5007_MTECH_INTERNSHIP_project/data_LTA_cam_images/annotated_images/coco_json"

# Output folder for YOLO labels
output_dir = "/home/rubesh/Desktop/sweta/Mtech_internship/ISY_5007_MTECH_INTERNSHIP_project/data_LTA_cam_images/annotated_images/yolo_labels"
os.makedirs(output_dir, exist_ok=True)  # Create output folder if it doesn't exist

# Loop through all COCO JSON files in the input folder
for file_name in os.listdir(coco_json_folder):
    if file_name.endswith(".json"):  # Process only JSON files
        coco_json_path = os.path.join(coco_json_folder, file_name)

        # Load COCO JSON
        with open(coco_json_path, "r") as f:
            coco_data = json.load(f)

        # Get the single image metadata
        image = coco_data["images"][0]  # Access the first (and only) image
        image_id = image["id"]
        file_name_without_ext = os.path.splitext(image["file_name"])[0]
        label_path = os.path.join(output_dir, f"{file_name_without_ext}.txt")

        # Collect annotations for this image
        labels = []
        for annotation in coco_data["annotations"]:
            if annotation["image_id"] == image_id:
                class_id = annotation["category_id"] - 1  # YOLO class IDs start from 0
                segmentation = annotation["segmentation"][0]

                # Normalize segmentation points
                normalized_points = []
                for i in range(0, len(segmentation), 2):
                    x = segmentation[i] / image["width"]
                    y = segmentation[i + 1] / image["height"]
                    normalized_points.extend([x, y])

                # Create YOLO label line
                labels.append(f"{class_id} " + " ".join(map(str, normalized_points)))

        # Write labels to file
        with open(label_path, "w") as f:
            f.write("\n".join(labels))

        print(f"YOLO label saved to: {label_path}")