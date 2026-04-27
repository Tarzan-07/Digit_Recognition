"""
This performs the MSER to detect potential regions.
"""

import cv2
import numpy as np
from PIL import Image

def image_pyramid(image, scales=[1.0, 0.75, 0.5]):
    for scale in scales:
        h, w = image.shape[:2]
        resized = cv2.resize(image, (int(w * scale), int(h * scale)))
        yield resized, scale

def get_potential_regions(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mser = cv2.MSER_create()

    regions, _ = mser.detectRegions(gray)

    boxes = []
    for p in regions:
        x, y, w, h = cv2.boundingRect(p)

        # filtering (VERY IMPORTANT)
        if w < 10 or h < 10:
            continue
        if w > 120 or h > 120:
            continue

        aspect_ratio = w / float(h)
        if aspect_ratio < 0.2 or aspect_ratio > 1.2:
            continue

        boxes.append((x, y, w, h))

    return boxes


def extract_rois(image, boxes):
    rois = []
    for (x, y, w, h) in boxes:
        crop = image[y:y+h, x:x+w]
        rois.append((crop, (x, y, w, h)))
    return rois


def simple_nms(boxes, thresh=15):
    filtered = []
    for box in boxes:
        x, y, w, h = box
        keep = True
        for fx, fy, fw, fh in filtered:
            if abs(x - fx) < thresh and abs(y - fy) < thresh:
                keep = False
                break
        if keep:
            filtered.append(box)
    return filtered