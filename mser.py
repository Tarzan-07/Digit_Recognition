"""
This performs the MSER to detect potential regions.
"""

import cv2
import numpy as np
from PIL import Image

import cv2
import numpy as np

def normalize_digit(crop):
    h, w = crop.shape[:2]

    # make square
    size = max(h, w)
    square = np.zeros((size, size, 3), dtype=np.uint8)

    # center the digit
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = crop

    # resize to model input
    square = cv2.resize(square, (32, 32))

    return square


def image_pyramid(image, scales=[1.0, 0.8, 0.6, 0.4]):
    for scale in scales:
        h, w = image.shape[:2]
        resized = cv2.resize(image, (int(w * scale), int(h * scale)))
        yield resized, scale

# def get_potential_regions(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     mser = cv2.MSER_create()

#     regions, _ = mser.detectRegions(gray)

#     boxes = []
#     for p in regions:
#         x, y, w, h = cv2.boundingRect(p)

#         # filtering (VERY IMPORTANT)
#         if w < 10 or h < 10:
#             continue
#         if w > 120 or h > 120:
#             continue

#         aspect_ratio = w / float(h)
#         if aspect_ratio < 0.2 or aspect_ratio > 1.2:
#             continue

#         boxes.append((x, y, w, h))

#     return boxes

def get_potential_regions(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 🔥 improve contrast
    gray = cv2.equalizeHist(gray)

    # # 🔥 tuned MSER
    # mser = cv2.MSER_create(
    #     _min_area=30,
    #     _max_area=3000,
    #     _max_variation=0.25
    # )

    mser = cv2.MSER_create()

    mser.setMinArea(30)
    mser.setMaxArea(3000)
    mser.setMaxVariation(0.25)

    regions, _ = mser.detectRegions(gray)

    boxes = []
    for p in regions:
        x, y, w, h = cv2.boundingRect(p)

        area = w * h

        # 🔥 better filtering
        if area < 150 or area > 3000:
            continue

        aspect_ratio = w / float(h)

        if aspect_ratio < 0.4 or aspect_ratio > 0.9:
            continue

        boxes.append((x, y, w, h))

    return boxes


# def extract_rois(image, boxes):
#     rois = []
#     for (x, y, w, h) in boxes:
#         crop = image[y:y+h, x:x+w]
#         rois.append((crop, (x, y, w, h)))
#     return rois

def extract_rois(image, boxes):
    rois = []

    for (x, y, w, h) in boxes:
        pad = 4

        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)

        crop = image[y1:y2, x1:x2]
        rois.append((crop, (x, y, w, h)))

    return rois


# def simple_nms(boxes, thresh=15):
#     filtered = []
#     for box in boxes:
#         x, y, w, h = box
#         keep = True
#         for fx, fy, fw, fh in filtered:
#             if abs(x - fx) < thresh and abs(y - fy) < thresh:
#                 keep = False
#                 break
#         if keep:
#             filtered.append(box)
#     return filtered


def simple_nms(boxes, iou_thresh=0.5):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = x1 + boxes[:,2]
    y2 = y1 + boxes[:,3]

    areas = boxes[:,2] * boxes[:,3]
    order = areas.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(tuple(boxes[i]))

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou < iou_thresh)[0]
        order = order[inds + 1]

    return keep


