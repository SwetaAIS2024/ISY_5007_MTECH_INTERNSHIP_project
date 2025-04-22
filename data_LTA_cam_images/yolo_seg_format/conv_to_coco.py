# import json

# # Input JSON file
# input_folder_path = "/home/rubesh/Desktop/sweta/Mtech_internship/ISY_5007_MTECH_INTERNSHIP_project/data_LTA_cam_images/annotated_images"

# # Output COCO JSON file
# output_coco_folder_path = "/home/rubesh/Desktop/sweta/Mtech_internship/ISY_5007_MTECH_INTERNSHIP_project/data_LTA_cam_images/annotated_images/coco_json"

# # Load the original JSON
# with open(input_json_path, "r") as f:
#     data = json.load(f)

# #print("loaded json file", data )
# # Convert to COCO format
# coco_format = {
#     "images": [
#         {
#             "id": 1,
#             "file_name": data["imagePath"].split("/")[-1],
#             "height": data["imageHeight"],
#             "width": data["imageWidth"]
#         }
#     ],
#     "annotations": [],
#     "categories": [
#         {"id": 1, "name": "lane", "supercategory": "none"}
#     ]
# }

# # Add annotations (segmentation only)
# for idx, shape in enumerate(data["shapes"]):
#     annotation = {
#         "id": idx + 1,
#         "image_id": 1,
#         "category_id": 1,
#         "segmentation": [list(sum(shape["points"], []))],  # Flatten points
#         "iscrowd": 0
#     }
#     coco_format["annotations"].append(annotation)

# # Save the COCO JSON
# with open(output_coco_json_path, "w") as f:
#     json.dump(coco_format, f, indent=4)

# print(f"COCO JSON saved to: {output_coco_json_path}")

import json
import os

# Input folder containing JSON files
input_folder_path = "/home/rubesh/Desktop/sweta/Mtech_internship/ISY_5007_MTECH_INTERNSHIP_project/data_LTA_cam_images/annotated_images"

# Output folder for COCO JSON files
output_coco_folder_path = "/home/rubesh/Desktop/sweta/Mtech_internship/ISY_5007_MTECH_INTERNSHIP_project/data_LTA_cam_images/annotated_images/coco_json"
os.makedirs(output_coco_folder_path, exist_ok=True)  # Create output folder if it doesn't exist

# Loop through all JSON files in the input folder
for file_name in os.listdir(input_folder_path):
    if file_name.endswith(".json"):  # Process only JSON files
        input_json_path = os.path.join(input_folder_path, file_name)
        output_coco_json_path = os.path.join(output_coco_folder_path, file_name)

        # Load the original JSON
        with open(input_json_path, "r") as f:
            data = json.load(f)

        # Convert to COCO format
        coco_format = {
            "images": [
                {
                    "id": 1,
                    "file_name": data["imagePath"].split("/")[-1],
                    "height": data["imageHeight"],
                    "width": data["imageWidth"]
                }
            ],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "lane", "supercategory": "none"}
            ]
        }

        # Add annotations (segmentation only)
        for idx, shape in enumerate(data["shapes"]):
            annotation = {
                "id": idx + 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [list(sum(shape["points"], []))],  # Flatten points
                "iscrowd": 0
            }
            coco_format["annotations"].append(annotation)

        # Save the COCO JSON
        with open(output_coco_json_path, "w") as f:
            json.dump(coco_format, f, indent=4)

        print(f"COCO JSON saved to: {output_coco_json_path}")