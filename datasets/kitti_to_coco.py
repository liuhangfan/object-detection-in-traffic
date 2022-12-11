import functools
import json
import os
import random
import shutil
from collections import defaultdict

import matplotlib.pyplot as plt
import pycocotools
from pycocotools import mask
import numpy as np
import pandas as pd
import uuid
import cv2

CROP_SIZE = 120

categories = set()
categories_dist = defaultdict(int)
fixed_category = {
    k: idx for idx, k in
    list(zip([1, 2, 3, 4, 5, 6, 7, 8, 9],
             ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']))
}


def parse_label(path, img_id):
    """
    convert KITTI label to coco format
    :param path:
    :param img_id:
    :return: annotations
    """
    try:
        labels = pd.read_csv(path, delimiter=' ', header=None).values
    except Exception:
        print('Empty Label: {}'.format(path))
        return []
    annotations = []

    for idx, label in enumerate(labels):
        category_name = label[0]
        category_id = fixed_category[category_name]
        top_x, top_y, bottom_x, bottom_y = label[4:8]
        poly = [[top_x, top_y],
                [bottom_x, top_y],
                [bottom_x, bottom_y],
                [top_x, bottom_y]]
        width = bottom_x - top_x
        height = bottom_y - top_y
        area = width * height
        annotations.append({
            'id': '{}_{}'.format(img_id, idx),
            'image_id': img_id,
            "segmentation": list([poly]),
            'category_name': category_name,
            'category_id': category_id,
            'area': float(area),
            "iscrowd": 0,
            'bbox': [float(top_x), float(top_y), float(width), float(height)],
        })
        categories.add(category_id)
    return annotations


def process_dataset(dataset_root, name):
    """
    process dataset
    :param dataset_root:
    :param name: train or test
    """
    images = []
    annotations = []
    img_list = os.listdir(os.path.join(dataset_root, name))
    if len(img_list) == 0:
        print(img_list, "path is empty")
        return
    for idx, img_name in enumerate(sorted(img_list)):
        img_path = os.path.join(dataset_root, name, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        height, width = img.shape[:2]
        img_name_wo_suffix = img_name[:img_name.rfind('.')].replace('/', '_')
        img_id = int(img_name_wo_suffix)
        label_path = os.path.join(dataset_root, "labels", "{}.txt".format(img_name_wo_suffix))

        annotation = parse_label(label_path, img_id=img_id)

        images.append({
            'file_name': img_name.replace('/', '_'),
            'width': width,
            'height': height,
            'id': img_id,
        })
        annotations.extend(annotation)

        print('{}/{} {}'.format(idx, len(img_list), img_name))
        print(categories)

    dataset = {
        "categories": [{
            "id": int(fixed_category[name]),
            "name": str(name),
            "supercategory": 'str',
        } for name in fixed_category.keys()],
        'images': images,
        'annotations': annotations,
    }

    json.dump(dataset, open(
        os.path.join(dataset_root, 'KITTI_{}.json')
        .format(name), 'w'), indent=0)


def convert_kitti_to_coco(dataset_root):
    process_dataset(dataset_root, 'train')
    process_dataset(dataset_root, 'val')
