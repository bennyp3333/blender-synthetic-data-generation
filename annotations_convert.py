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
processed_images = set()  # Track processed images to avoid duplicates

# Process each annotation file
for ann_path in glob.glob(os.path.join(annotations_dir, "*.json")):
    with open(ann_path, "r") as f:
        data = json.load(f)

    image_id = data["image_id"]
    width = data["image_width"]
    height = data["image_height"]
    filename = image_id + ".png"
    
    # Add image only once
    if image_id not in processed_images:
        coco_output["images"].append({
            "id": next_image_id,
            "file_name": filename,
            "width": width,
            "height": height
        })
        processed_images.add(image_id)
        
        # Map the image filename to its COCO ID for annotation references
        image_coco_id = next_image_id
        next_image_id += 1
    else:
        # Find the COCO image ID for this image
        image_coco_id = next(img["id"] for img in coco_output["images"] if img["file_name"] == filename)
    
    # Process all annotations for this image
    for annotation in data["annotations"]:
        category_name = annotation["category_name"]
        
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
        bbox_data = annotation["bbox"]
        
        # The new format uses min_x, min_y, max_x, max_y instead of just width/height
        # COCO format expects [x, y, width, height]
        bbox = [
            bbox_data["min_x"],
            bbox_data["min_y"],
            bbox_data["width"],
            bbox_data["height"]
        ]
        
        coco_output["annotations"].append({
            "id": next_annotation_id,
            "image_id": image_coco_id,
            "category_id": category_id,
            "bbox": bbox,
            "area": bbox_data["width"] * bbox_data["height"],
            "iscrowd": 0
        })
        
        next_annotation_id += 1

# Save result
output_path = os.path.join(root_dir, "coco_annotations.json")
with open(output_path, "w") as f:
    json.dump(coco_output, f, indent=2)

print(f"COCO conversion complete: {output_path}")
print(f"Processed {len(processed_images)} images with {len(coco_output['annotations'])} annotations across {len(category_name_to_id)} categories.")