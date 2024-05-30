import torch
import torch.utils.data
import torchvision
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from torchvision import transforms


class CocoSegmentation(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms=None):
        super(CocoSegmentation, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, idx):
        img, target = super(CocoSegmentation, self).__getitem__(idx)
        image_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        masks = self._convert_poly_to_mask(anns, img.size[1], img.size[0])
        category_ids = [item['category_id'] for item in anns]

        target = {
            'image_id': image_id,
            'annotations': anns,
            'masks': masks,
            'labels': category_ids
        }
        
        if self._transforms is not None:
            img = self._transforms(img)
        
        return img, target

    def _convert_poly_to_mask(self, segmentations, height, width):
        masks = []
        for polygons in segmentations:
            rles = coco_mask.frPyObjects(polygons['segmentation'], height, width)
            mask = coco_mask.decode(rles)
            if len(mask.shape) < 3:
                mask = mask[..., None]
            mask = torch.as_tensor(mask, dtype=torch.uint8)
            mask = mask.any(dim=2)
            masks.append(mask)
        if masks:
            masks = torch.stack(masks, dim=0)
        else:
            masks = torch.zeros((0, height, width), dtype=torch.uint8)
        return masks

    def extra_repr(self):
        return f"Root: {self.root}, Transforms: {self._transforms}"

def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
    ])

def main():
    img_folder = '/Users/amin/Desktop/higharc/Datasets/Laleled-2024-05-29/auto_translate_v4.v3i.coco-segmentation/test/'
    ann_file = '/Users/amin/Desktop/higharc/Datasets/Laleled-2024-05-29/auto_translate_v4.v3i.coco-segmentation/test/_annotations.coco.json'
    dataset = CocoSegmentation(img_folder, ann_file, transforms=get_transform())

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1)

    for images, targets in dataloader:
        print(images, targets)

if __name__ == '__main__':
    main()



mscoco_category2name = {
    1: "bath",
    2: "bed_closet",
    3: "bed_pass",
    4: "bedroom",
    5: "chase",
    6: "closet",
    7: "dining",
    8: "entry",
    9: "fireplace",
    10: "flex",
    11: "foyer",
    12: "front_porch",
    13: "garage",
    14: "general",
    15: "hall",
    16: "hall_cased_opening",
    17: "kitchen",
    18: "laundry",
    19: "living",
    20: "master_bed",
    21: "master_closet",
    22: "master_hall",
    23: "master_vestibule",
    24: "mech",
    25: "mudroom",
    26: "office",
    27: "pantry",
    28: "patio",
    29: "portico",
    30: "powder",
    31: "reach_closet",
    32: "reading_nook",
    33: "rear_porch",
    34: "solarium",
    35: "stairs_editor",
    36: "util_hall",
    37: "walk",
    38: "water_closet",
    39: "workshop"
}


# mscoco_category2name = {
#     1: 'person',
#     2: 'bicycle',
#     3: 'car',
#     4: 'motorcycle',
#     5: 'airplane',
#     6: 'bus',
#     7: 'train',
#     8: 'truck',
#     9: 'boat',
#     10: 'traffic light',
#     11: 'fire hydrant',
#     13: 'stop sign',
#     14: 'parking meter',
#     15: 'bench',
#     16: 'bird',
#     17: 'cat',
#     18: 'dog',
#     19: 'horse',
#     20: 'sheep',
#     21: 'cow',
#     22: 'elephant',
#     23: 'bear',
#     24: 'zebra',
#     25: 'giraffe',
#     27: 'backpack',
#     28: 'umbrella',
#     31: 'handbag',
#     32: 'tie',
#     33: 'suitcase',
#     34: 'frisbee',
#     35: 'skis',
#     36: 'snowboard',
#     37: 'sports ball',
#     38: 'kite',
#     39: 'baseball bat',
#     40: 'baseball glove',
#     41: 'skateboard',
#     42: 'surfboard',
#     43: 'tennis racket',
#     44: 'bottle',
#     46: 'wine glass',
#     47: 'cup',
#     48: 'fork',
#     49: 'knife',
#     50: 'spoon',
#     51: 'bowl',
#     52: 'banana',
#     53: 'apple',
#     54: 'sandwich',
#     55: 'orange',
#     56: 'broccoli',
#     57: 'carrot',
#     58: 'hot dog',
#     59: 'pizza',
#     60: 'donut',
#     61: 'cake',
#     62: 'chair',
#     63: 'couch',
#     64: 'potted plant',
#     65: 'bed',
#     67: 'dining table',
#     70: 'toilet',
#     72: 'tv',
#     73: 'laptop',
#     74: 'mouse',
#     75: 'remote',
#     76: 'keyboard',
#     77: 'cell phone',
#     78: 'microwave',
#     79: 'oven',
#     80: 'toaster',
#     81: 'sink',
#     82: 'refrigerator',
#     84: 'book',
#     85: 'clock',
#     86: 'vase',
#     87: 'scissors',
#     88: 'teddy bear',
#     89: 'hair drier',
#     90: 'toothbrush'
# }

mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}