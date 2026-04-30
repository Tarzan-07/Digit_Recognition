"""
This performs the MSER to detect potential regions.
"""

import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

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

def normalize_digit(crop):
    h, w = crop.shape[:2]
    size = max(h, w)
    square = np.zeros((size, size, 3), dtype=np.uint8)
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = crop
    square = cv2.resize(square, (32, 32))

    return square

def filter_by_row(boxes):
    if len(boxes) == 0:
        return []

    ys = np.array([y + h/2 for (x, y, w, h) in boxes])  # center y
    hs = np.array([h for (_, _, _, h) in boxes])

    mean_h = np.mean(hs)
    bands = []
    for i, y in enumerate(ys):
        placed = False
        for band in bands:
            if abs(y - band[0]) < mean_h:
                band.append(y)
                placed = True
                break
        if not placed:
            bands.append([y])

    best_band = max(bands, key=lambda b: len(b))
    band_center = np.mean(best_band)

    filtered = []
    for (x, y, w, h) in boxes:
        cy = y + h/2
        if abs(cy - band_center) < mean_h:
            filtered.append((x, y, w, h))

    return filtered

def load_the_image_and_test(file):
    image = cv2.imread(file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    if np.mean(gray) < 120:
        gray = cv2.bitwise_not(gray)

    h_img, w_img = gray.shape
    img_area = h_img * w_img

    mser = cv2.MSER_create()
    mser.setMinArea(int(0.001 * img_area))
    mser.setMaxArea(int(0.15 * img_area)) 

    regions, _ = mser.detectRegions(gray)
    boxes = []
    scores = []

    for region in regions:
        x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))

        if w < 5 or h < 5:
            continue

        area = w * h
        aspect_ratio = h / float(w)

        if area < 40 or area > 0.08 * img_area:
            continue
        if not (0.5 < aspect_ratio < 8):
            continue

        boxes.append([x, y, w, h])
        score = 1.0 - abs(aspect_ratio - 2.0) * 0.2
        scores.append(score)

    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=0.5)
        boxes = [boxes[i[0] if isinstance(i, (list, np.ndarray)) else i] for i in indices]
    else:
        boxes = []

    def cluster_boxes(boxes, y_thresh=20):
        boxes = sorted(boxes, key=lambda b: b[0])
        clusters = []

        for box in boxes:
            x, y, w, h = box
            placed = False

            for cluster in clusters:
                cx, cy, cw, ch = cluster[0]
                if abs(y - cy) < y_thresh:
                    cluster.append(box)
                    placed = True
                    break

            if not placed:
                clusters.append([box])

        return clusters

    def cluster_score(cluster):
        xs = np.array([b[0] for b in cluster], dtype=np.float32)
        ys = np.array([b[1] + b[3]/2 for b in cluster], dtype=np.float32)
        hs = np.array([b[3] for b in cluster], dtype=np.float32)

        if len(cluster) == 0:
            return -1e9

        y_jitter = np.std(ys) / (np.mean(hs) + 1e-6)
        xs_sorted = np.sort(xs)
        gaps = np.diff(xs_sorted) if len(xs_sorted) > 1 else np.array([0.0])
        gap_jitter = np.std(gaps) / (np.mean(gaps) + 1e-6) if np.mean(gaps) > 0 else 0
        return len(cluster) * 2.0 - y_jitter * 2.0 - gap_jitter * 1.5

    boxes = filter_by_row(boxes)
    clusters = cluster_boxes(boxes)
    if len(clusters) > 0:
        best_cluster = max(clusters, key=cluster_score)
    else:
        best_cluster = []
    best_cluster = sorted(best_cluster, key=lambda b: b[0])

    for x, y, w, h in best_cluster:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    os.makedirs("graded_images", exist_ok=True)
    # save_path = os.path.join(
    #     "graded_images",
    #     f"{os.path.splitext(os.path.basename(file))[0]}_mser.png"
    # )

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"{file} - Detected digits: {len(best_cluster)}")
    plt.axis("off")
    # plt.savefig(save_path)
    plt.show()

    print(f"MSER raw: {len(regions)} | Final boxes: {len(best_cluster)}")

    return best_cluster