import os
import json
import glob

# Directories
root_dir = "./synthetic_data"
images_dir = os.path.join(root_dir, "images")
annotations_dir = os.path.join(root_dir, "annotations")

# Output COCO structure
coco_output = {
    "images": [],
    "annotations": [],
    "categories": []
}

# ID trackers
category_name_to_id = {}
next_image_id = 1
next_annotation_id = 1
next_category_id = 1

# Process each annotation file
for ann_path in glob.glob(os.path.join(annotations_dir, "*.json")):
    with open(ann_path, "r") as f:
        data = json.load(f)

    filename = data["image_id"] + ".png"
    width = data["image_width"]
    height = data["image_height"]
    category_name = data["category_name"]

    # Add to images
    coco_output["images"].append({
        "id": next_image_id,
        "file_name": filename,
        "width": width,
        "height": height
    })

    # Map category name to ID
    if category_name not in category_name_to_id:
        category_name_to_id[category_name] = next_category_id
        coco_output["categories"].append({
            "id": next_category_id,
            "name": category_name,
            "supercategory": "object"
        })
        next_category_id += 1

    category_id = category_name_to_id[category_name]
    bbox_data = data["bbox"]

    coco_output["annotations"].append({
        "id": next_annotation_id,
        "image_id": next_image_id,
        "category_id": category_id,
        "bbox": [
            bbox_data["min_x"],
            bbox_data["min_y"],
            bbox_data["width"],
            bbox_data["height"]
        ],
        "area": bbox_data["width"] * bbox_data["height"],
        "iscrowd": 0
    })

    next_image_id += 1
    next_annotation_id += 1

# Save result
with open("coco_annotations.json", "w") as f:
    json.dump(coco_output, f, indent=2)

print("COCO conversion complete: output_coco_annotations.json")