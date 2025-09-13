from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:  # PySceneDetect（必須）
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector
except Exception as e:  # pragma: no cover
    VideoManager = None  # type: ignore
    SceneManager = None  # type: ignore
    ContentDetector = None  # type: ignore
    _IMPORT_ERR = e


@dataclass
class Shot:
    start: float
    end: float


def simple_scene_detect(video_path: Path, thresh: float = 0.4, min_len: float = 0.5) -> List[Shot]:
    raise RuntimeError("Only PySceneDetect is supported for scene detection. Please install 'scenedetect'.")


def pyscenedetect(video_path: Path, threshold: float = 27.0, min_scene_len: float = 0.5) -> List[Shot]:
    """
    使用 PySceneDetect ContentDetector 進行場景偵測。
    threshold: 內容變化靈敏度（越大越不易切）。
    """
    if VideoManager is None or SceneManager is None or ContentDetector is None:
        raise RuntimeError(
            f"PySceneDetect is required but not available: {_IMPORT_ERR if '_IMPORT_ERR' in globals() else 'unknown import error'}"
        )

    video_manager = VideoManager([str(video_path)])
    scene_manager = SceneManager()
    # PySceneDetect 的 min_scene_len 以「影格數」為單位，這裡將秒數轉為影格數
    try:
        base_tc = video_manager.get_base_timecode()
        fps = float(getattr(base_tc, "framerate", 30.0))
    except Exception:
        fps = 30.0
    min_frames = max(1, int(round(min_scene_len * fps)))
    scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_frames))
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
        # 沒偵測到視為錯誤（不退回 HSV）
        raise RuntimeError("No scenes detected by PySceneDetect; please adjust threshold/min_scene_len.")
    return shots


def detect_shots_auto(video_path: Path) -> List[Shot]:
    # 僅允許 PySceneDetect
    return pyscenedetect(video_path)


def transnet_detect(video_path: Path, threshold: float = 0.5, min_scene_len: float = 0.1) -> List[Shot]:
    """使用 TransNet V2 偵測切點（需要額外套件）。"""
    try:
        from .transnet_integration import detect_shots_transnet
    except Exception as e:  # pragma: no cover
        raise RuntimeError("TransNet V2 integration missing.") from e

    shots_se = detect_shots_transnet(video_path, threshold=threshold, min_scene_len=min_scene_len)
    shots: List[Shot] = [Shot(float(s), float(e)) for (s, e) in shots_se]
    if not shots:
        raise RuntimeError("No scenes detected by TransNet V2; adjust threshold/min_scene_len.")
    return shots
