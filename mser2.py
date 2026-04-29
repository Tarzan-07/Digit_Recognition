import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# def load_the_f_image_and_test(file):
#     image = cv2.imread(file)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Contrast normalization + mild smoothing for MSER stability
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
#     gray = clahe.apply(gray)
#     gray = cv2.GaussianBlur(gray, (3, 3), 0)

#     # If image is globally dark, invert to favor bright text regions
#     gray = cv2.bitwise_not(gray) if np.mean(gray) < 120 else gray

#     h_img, w_img = gray.shape[:2]
#     img_area = h_img * w_img

#     mser = cv2.MSER_create(
#         delta=4,
#         min_area=max(40, int(0.002 * img_area)),
#         max_area=max(250, int(0.06 * img_area)),
#         max_variation=0.35
#     )
#     regions, bboxes = mser.detectRegions(gray)

#     # Geometric filtering with soft shape prior
#     candidates = []
#     cand_scores = []
#     for i, bbox in enumerate(bboxes):
#         x, y, w, h = [int(v) for v in bbox]
#         if w <= 0 or h <= 0:
#             continue

#         area = w * h
#         if area < max(35, int(0.0015 * img_area)) or area > int(0.08 * img_area):
#             continue

#         ar = h / float(w)
#         if not (0.9 <= ar <= 5.5):
#             continue

#         region_area = len(regions[i])
#         solidity = region_area / float(area)
#         if not (0.12 <= solidity <= 0.92):
#             continue

#         # Score candidates by how digit-like they are
#         score = 1.0
#         score -= min(abs(ar - 2.0) / 3.0, 1.0) * 0.35
#         score -= min(abs(solidity - 0.5) / 0.5, 1.0) * 0.35
#         score -= min(abs((h / float(h_img)) - 0.28) / 0.28, 1.0) * 0.30

#         candidates.append((x, y, w, h))
#         cand_scores.append(score)

#     # NMS to deduplicate overlaps
#     if len(candidates) > 0:
#         rects = [[x, y, w, h] for (x, y, w, h) in candidates]
#         indices = cv2.dnn.NMSBoxes(rects, cand_scores, score_threshold=0.0, nms_threshold=0.35)
#         nms_boxes = []
#         nms_scores = []
#         for idx_raw in indices:
#             idx = idx_raw[0] if isinstance(idx_raw, (list, np.ndarray)) else idx_raw
#             nms_boxes.append(candidates[idx])
#             nms_scores.append(cand_scores[idx])
#     else:
#         nms_boxes, nms_scores = [], []

#     # Group into horizontal text lines by y-center + similar height
#     def cluster_boxes(boxes, scores, height_tol=0.45, y_tol=0.55):
#         if not boxes:
#             return []

#         clusters = []
#         used = [False] * len(boxes)
#         order = np.argsort([b[0] for b in boxes])

#         for i0 in order:
#             i = int(i0)
#             if used[i]:
#                 continue
#             used[i] = True
#             cluster_idx = [i]
#             x1, y1, w1, h1 = boxes[i]
#             yc1 = y1 + h1 / 2.0

#             for j0 in order:
#                 j = int(j0)
#                 if used[j]:
#                     continue
#                 x2, y2, w2, h2 = boxes[j]
#                 yc2 = y2 + h2 / 2.0
#                 ref_h = max(h1, h2)

#                 if abs(h1 - h2) / float(ref_h) < height_tol and abs(yc1 - yc2) / float(ref_h) < y_tol:
#                     used[j] = True
#                     cluster_idx.append(j)

#             clusters.append(cluster_idx)

#         # Convert index clusters to (boxes, scores)
#         out = []
#         for c in clusters:
#             c_boxes = [boxes[k] for k in c]
#             c_scores = [scores[k] for k in c]
#             out.append((c_boxes, c_scores))
#         return out

#     clusters = cluster_boxes(nms_boxes, nms_scores)

#     # Keep the best line: large cluster + good average score + low vertical jitter
#     def cluster_quality(c_boxes, c_scores):
#         ys = np.array([y + h / 2.0 for (_, y, _, h) in c_boxes], dtype=np.float32)
#         hs = np.array([h for (_, _, _, h) in c_boxes], dtype=np.float32)
#         if len(hs) == 0:
#             return -1e9
#         jitter = float(np.std(ys) / (np.mean(hs) + 1e-6))
#         return len(c_boxes) * 1.0 + float(np.mean(c_scores)) * 2.0 - jitter * 2.0

#     if clusters:
#         clusters_sorted = sorted(clusters, key=lambda cs: cluster_quality(cs[0], cs[1]), reverse=True)
#         best_boxes, best_scores = clusters_sorted[0]
#     else:
#         best_boxes, best_scores = [], []

#     # Optional prior from filename: use digit-count in filename as expected count
#     base = os.path.splitext(os.path.basename(file))[0]
#     expected_count = sum(ch.isdigit() for ch in base)
#     if expected_count == 0:
#         expected_count = None

#     best_cluster = []
#     if best_boxes:
#         pairs = sorted(zip(best_boxes, best_scores), key=lambda bs: bs[0][0])
#         boxes_sorted = [p[0] for p in pairs]
#         scores_sorted = [p[1] for p in pairs]

#         # If too many boxes, select the most self-consistent contiguous subsequence
#         if expected_count is not None and len(boxes_sorted) > expected_count:
#             win = expected_count
#             best_val = -1e9
#             best_slice = (0, win)
#             for s in range(0, len(boxes_sorted) - win + 1):
#                 e = s + win
#                 sub_b = boxes_sorted[s:e]
#                 sub_s = scores_sorted[s:e]

#                 ys = np.array([y + h / 2.0 for (_, y, _, h) in sub_b], dtype=np.float32)
#                 hs = np.array([h for (_, _, _, h) in sub_b], dtype=np.float32)
#                 xs = np.array([x for (x, _, _, _) in sub_b], dtype=np.float32)

#                 y_jitter = float(np.std(ys) / (np.mean(hs) + 1e-6))
#                 h_jitter = float(np.std(hs) / (np.mean(hs) + 1e-6))
#                 x_gaps = np.diff(xs) if len(xs) > 1 else np.array([0.0], dtype=np.float32)
#                 gap_jitter = float(np.std(x_gaps) / (np.mean(x_gaps) + 1e-6)) if np.mean(x_gaps) > 0 else 0.0

#                 val = float(np.mean(sub_s)) * 2.0 - y_jitter * 2.0 - h_jitter * 1.5 - gap_jitter * 1.0
#                 if val > best_val:
#                     best_val = val
#                     best_slice = (s, e)

#             s, e = best_slice
#             best_cluster = boxes_sorted[s:e]
#         else:
#             best_cluster = boxes_sorted

#     for x, y, w, h in best_cluster:
#         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     plt.figure(figsize=(12, 8))
#     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     plt.title(f"{file} - Detected: {len(best_cluster)} regions")
#     plt.show()

#     print(f"MSER raw: {len(bboxes)} | Geometric: {len(candidates)} | NMS: {len(nms_boxes)}")
#     print(f"Line clusters: {[len(c[0]) for c in clusters]} | Kept: {len(best_cluster)}")
#     if expected_count is not None:
#         print(f"Expected (from filename): {expected_count}")

#     return best_cluster

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

def load_the_f_image_and_test(file):
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