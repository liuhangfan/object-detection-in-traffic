import os
import random
import shutil


def split_dataset(dataset_root, rate=0.8):
    """
    split dataset in train and val
    :param dataset_root: root path of dataset
    :param rate: rate of dataset for training
    :return:
    """
    img_list = os.listdir(os.path.join(dataset_root, 'images'))
    val_path = os.path.join(dataset_root, 'val')

    # split in 20/80
    random.shuffle(img_list)
    split = int(len(img_list) * rate)
    val_list = img_list[split:]
    for img_name in val_list:
        from_path = os.path.join(dataset_root, "images", img_name)
        to_path = os.path.join(val_path, img_name)
        shutil.move(from_path, to_path)
    os.rename(os.path.join(dataset_root, 'images'), os.path.join(dataset_root, 'train'))
