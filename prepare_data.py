import os
import glob
import json
import tqdm
import cv2
import pandas as pd

def prepare_imagenet_classification(dirpath):
    mapping_path = os.path.join(dirpath, "LOC_synset_mapping.txt")

    # class dict
    class_dict = dict()
    with open(mapping_path, "r") as f:
        lines = f.read().strip().split("\n")
    
    for class_num, line in enumerate(lines):
        ind = line.find(" ")
        k = line[:ind]
        v = line[ind:].strip().split(",")[0]
        class_dict[k] = [class_num, v]

    # train solution
    train_solution_path = os.path.join(dirpath, "LOC_train_solution.csv")
    train_anno = pd.read_csv(train_solution_path)
    
    # val solution
    val_solution_path = os.path.join(dirpath, "LOC_val_solution.csv")
    val_anno = pd.read_csv(val_solution_path)

    # test solution
    test_solution_path = os.path.join(dirpath, "LOC_sample_submission.csv")
    test_anno = pd.read_csv(test_solution_path)

    lst = []
    for row in tqdm.tqdm(train_anno.itertuples(), total=len(train_anno)):
        _, image_name, val = row
        val = val.split()
        image_id = image_name.split("_")[0]
        image_path = os.path.join(dirpath, f"ILSVRC/Data/CLS-LOC/train/", image_id, image_name) + ".JPEG"
        img = cv2.imread(image_path)
        h, w, c = img.shape
        if h < 224 or w < 224:
            lst.append(image_path)
    print(lst)
    print(len(lst))

def make_coco_small_anno():
    data_path = "/data/coco/annotations/instances_train2017.json"

    with open(data_path, 'r') as f:
        val = json.load(f)

    with open('/data/coco/annotations/small_train_anno.txt', 'w') as f:
        for anno in tqdm.tqdm(val['annotations']):
            image_id = anno['image_id']
            bbox = anno['bbox']
            category_id = anno['category_id']

            image_path = os.path.join('train2017', f'{image_id:012}.jpg')
            x1, y1 = bbox[:2]
            x2, y2 = x1 + bbox[2], y1 + bbox[3]

            f.write(f'{image_path} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {category_id}\n')

if __name__ == "__main__":
    prepare_imagenet_classification("/data/imagenet")