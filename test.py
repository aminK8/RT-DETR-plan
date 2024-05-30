# import torch
# import torch.nn as nn


# # Specify the file path where you saved the model
# file_path = "/Users/amin/Desktop/higharc/RT-DETR-ZOO/rtdetr_r50vd_2x_coco_objects365_from_paddle.pth"

# # Load the model from file
# state_dict = torch.load(file_path)

# print(state_dict['ema']['module'].keys())
# print("amin")

import json
import os

def convert_coco_to_panoptic(coco_json):
    res = dict()
    res['info'] = coco_json['info']
    res['licenses'] = coco_json['licenses']
    res['categories'] = coco_json['categories']
    res['images'] = coco_json['images']
    
    panoptic_annotations = []
    for image_info in coco_json['images']:
        panoptic_annotation = {
            "file_name": image_info['file_name'],
            "image_id": image_info['id'],
            "segments_info": []
        }
        for annotation in coco_json['annotations']:
            if annotation['image_id'] == image_info['id']:
                segment_info = {
                    "id": annotation['id'],
                    "category_id": annotation['category_id'],
                    "iscrowd": annotation['iscrowd'],
                    "bbox": annotation['bbox'],
                    "segmentation": annotation['segmentation'],
                    "area": annotation['area']
                }
                panoptic_annotation["segments_info"].append(segment_info)
        panoptic_annotations.append(panoptic_annotation)
        
    res['annotations'] = panoptic_annotations
    return res


for t in ["test", "valid", "train"]:

    base_path = "~/dataset/seg_object_detection/auto_translate_v4-3/{}/"

    anno_file = os.path.join(base_path, "_annotations.coco.json").format(t)
    # Load the JSON file
    with open(anno_file, 'r') as f:
        coco_json = json.load(f)


    panoptic_json = convert_coco_to_panoptic(coco_json)

    # Save to a JSON file
    out_file = os.path.join(base_path, "_panoptic_annotations.coco.json").format(t)
    with open(out_file, 'w') as f:
        json.dump(panoptic_json, f, indent=2)
