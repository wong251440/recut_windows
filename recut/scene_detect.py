from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:  # PySceneDetect（可選）
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector
except Exception:  # pragma: no cover
    VideoManager = None  # type: ignore
    SceneManager = None  # type: ignore
    ContentDetector = None  # type: ignore


@dataclass
class Shot:
    start: float
    end: float


def simple_scene_detect(video_path: Path, thresh: float = 0.4, min_len: float = 0.5) -> List[Shot]:
    """
    Fallback scene detection using HSV histogram difference.
    thresh: higher -> fewer cuts.
    """
    if cv2 is None:
        # No OpenCV, return single shot.
        return [Shot(0.0, 1e9)]

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    prev = None
    shots: List[Shot] = []
    start_t = 0.0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        if prev is not None:
            d = cv2.compareHist(prev, hist, cv2.HISTCMP_BHATTACHARYYA)
            if d > thresh:
                end_t = frame_idx / fps
                if end_t - start_t >= min_len:
                    shots.append(Shot(start_t, end_t))
                start_t = end_t
        prev = hist
        frame_idx += 1
    dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or frame_idx) / (fps or 25.0)
    cap.release()
    # last shot
    if not shots or dur - shots[-1].end >= min_len:
        shots.append(Shot(start_t, dur))
    return shots


def pyscenedetect(video_path: Path, threshold: float = 27.0, min_scene_len: float = 0.5) -> List[Shot]:
    """
    使用 PySceneDetect ContentDetector 進行場景偵測。
    threshold: 內容變化靈敏度（越大越不易切）。
    """
    if VideoManager is None or SceneManager is None or ContentDetector is None:
        return simple_scene_detect(video_path)

    video_manager = VideoManager([str(video_path)])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=int(round(min_scene_len * 1000))))
    base_timecode = None
    try:
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()
    finally:
        video_manager.release()

    shots: List[Shot] = []
    # scene_list 為 [(start_timecode, end_timecode), ...]
    for start_tc, end_tc in scene_list:
        s = start_tc.get_seconds()
        e = end_tc.get_seconds()
        if e - s >= min_scene_len:
            shots.append(Shot(float(s), float(e)))
    if not shots:
        # 若未偵測到，退回為整支影片單場景
        shots.append(Shot(0.0, 1e9))
    return shots


def detect_shots_auto(video_path: Path) -> List[Shot]:
    """優先使用 PySceneDetect，失敗則退回簡易 HSV 偵測。"""
    try:
        return pyscenedetect(video_path)
    except Exception:
        return simple_scene_detect(video_path)
