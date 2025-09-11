from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


def _video_meta(path: Path) -> tuple[float, int]:
    import cv2  # type: ignore
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return float(fps if fps > 0 else 30.0), int(frames)


def detect_shots_transnet(video_path: Path, threshold: float = 0.5, min_scene_len: float = 0.1) -> List[tuple[float, float]]:
    """
    使用 TransNet V2 偵測切點。
    需要安裝第三方套件與權重，例如：
      - pip install transnetv2
      - 或其他移植（需提供相同 API）
    偵測不到或未安裝時會拋錯。
    回傳 [(start_sec, end_sec), ...]
    """
    try:
        # 官方/社群版本 API 可能不同，嘗試多種呼叫方式
        try:
            from transnetv2 import TransNetV2  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "TransNet V2 not available. Please install 'transnetv2' and its dependencies (TensorFlow or PyTorch port)."
            ) from e

        model = TransNetV2()
        # 嘗試常見 API：predict_video 回傳 (predictions, single_frame_predictions)
        try:
            pred, _ = model.predict_video(str(video_path))  # type: ignore
        except TypeError:
            # 其他版本可能是返回單一 ndarray
            pred = model.predict_video(str(video_path))  # type: ignore

        pred = np.asarray(pred)
        if pred.ndim == 2 and pred.shape[1] >= 2:
            # 取 cut 機率通道
            scores = pred[:, 1]
        else:
            scores = pred.reshape(-1)

        fps, total_frames = _video_meta(video_path)
        # 以門檻取候選切點
        cut_idx = np.where(scores >= float(threshold))[0].astype(int)
        if cut_idx.size == 0:
            # 若沒有切點，視為整段一個 shot
            return [(0.0, float(total_frames) / fps)]

        # 後處理：NMS 合併相鄰近切點（2 幀以內）
        merged = []
        last = -9999
        for i in cut_idx.tolist():
            if i - last <= 2:
                # 距離過近，保留較高分者（此處簡化為後者）
                merged[-1] = i
            else:
                merged.append(i)
            last = i

        # 由切點拆分為 shots，並套用最短鏡頭長度（以秒為單位）
        min_len_frames = max(1, int(round(min_scene_len * fps)))
        shots: List[tuple[int, int]] = []
        start = 0
        for ci in merged:
            end = int(ci)
            if end - start >= min_len_frames:
                shots.append((start, end))
            start = end
        # 最後一段
        if total_frames - start >= min_len_frames:
            shots.append((start, total_frames))

        # 轉為秒
        out: List[tuple[float, float]] = [(s / fps, e / fps) for (s, e) in shots]
        if not out:
            out = [(0.0, float(total_frames) / fps)]
        return out
    except Exception as e:
        raise

