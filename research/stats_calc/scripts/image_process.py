
import cv2
from collections import defaultdict
import numpy as np
from scipy.ndimage import label
import json


def calc_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    classes = defaultdict(lambda: {"counts": 0, "surface": 0})

    for cl, s in zip(*np.unique(img, return_counts=True)):
        classes[cl]["surface"] = s

    for cl in classes.keys():
        mask = np.zeros(img.shape)
        mask[img == cl] = 1
        _, num_objects = label(mask)
        classes[cl]["counts"] = num_objects
    
    return dict(classes)


def calc_images(images, file=None):
    result = []
    for p in images:
        result.append({"path": p, "classes": calc_image(p)})

    data = {"images": result}

    if file is not None:
        with open(file, "w") as f:
            json.dump(data, f, indent=2)   

    return data
