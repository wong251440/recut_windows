from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional
    cv2 = None  # type: ignore


@dataclass
class MaskBox:
    x: int
    y: int
    w: int
    h: int


def apply_mask(image: np.ndarray, boxes: List[MaskBox]) -> np.ndarray:
    if not boxes:
        return image
    mask = np.ones(image.shape[:2], dtype=np.uint8)
    for b in boxes:
        x2, y2 = b.x + b.w, b.y + b.h
        mask[b.y : y2, b.x : x2] = 0
    if image.ndim == 3:
        return image * mask[..., None]
    return image * mask


def detect_text_boxes(image: np.ndarray, east_model_path: Optional[str] = None) -> List[MaskBox]:
    """
    Return detected text boxes using EAST if available, otherwise empty.
    The caller can always supply custom boxes externally.
    """
    if cv2 is None or east_model_path is None:
        return []

    H, W = image.shape[:2]
    newW, newH = (W // 32) * 32, (H // 32) * 32
    rW, rH = W / float(newW), H / float(newH)
    resized = cv2.resize(image, (newW, newH))
    blob = cv2.dnn.blobFromImage(
        resized, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False
    )
    net = cv2.dnn.readNet(east_model_path)
    net.setInput(blob)
    (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

    numRows, numCols = scores.shape[2:4]
    boxes: List[Tuple[int, int, int, int]] = []
    confidences: List[float] = []
    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        for x in range(0, numCols):
            if scoresData[x] < 0.5:
                continue
            offsetX, offsetY = x * 4.0, y * 4.0
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            boxes.append((startX, startY, int(w), int(h)))
            confidences.append(float(scoresData[x]))

    # Non-maximum suppression
    if cv2 is None:
        sel = list(range(len(boxes)))
    else:
        rects = [(x, y, x + w, y + h) for (x, y, w, h) in boxes]
        sel = cv2.dnn.NMSBoxes(rects, confidences, score_threshold=0.5, nms_threshold=0.4)
        if isinstance(sel, np.ndarray):
            sel = sel.flatten().tolist()

    out: List[MaskBox] = []
    for i in sel:
        x, y, w, h = boxes[i]
        x = max(0, int(x * rW))
        y = max(0, int(y * rH))
        w = int(w * rW)
        h = int(h * rH)
        out.append(MaskBox(x, y, w, h))
    return out

