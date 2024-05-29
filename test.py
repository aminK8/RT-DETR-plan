import torch
import torch.nn as nn


# Specify the file path where you saved the model
file_path = "/Users/amin/Desktop/higharc/RT-DETR-ZOO/rtdetr_r50vd_2x_coco_objects365_from_paddle.pth"

# Load the model from file
state_dict = torch.load(file_path)

print(state_dict['ema']['module'].keys())
print("amin")
