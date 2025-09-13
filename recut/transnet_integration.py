from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _video_meta(path: Path) -> tuple[float, int]:
    import cv2  # type: ignore
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return float(fps if fps > 0 else 30.0), int(frames)


def _read_frames_resized(video_path: Path, size: Tuple[int, int] = (48, 27)) -> np.ndarray:
    import cv2  # type: ignore
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames: List[np.ndarray] = []
    ok, frame = cap.read()
    idx = 0
    last_pct = -1
    print("[TransNetV2] Loading & resizing frames…")
    while ok and frame is not None:
        # resize to (W,H)=(48,27)
        fr = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
        idx += 1
        if total > 0:
            pct = int(idx * 100 / total)
            if pct != last_pct and pct % 5 == 0:
                print(f"[TransNetV2] Read {pct}% ({idx}/{total})")
                last_pct = pct
        ok, frame = cap.read()
    cap.release()
    if not frames:
        return np.zeros((0, 27, 48, 3), dtype=np.uint8)
    arr = np.stack(frames, axis=0).astype(np.uint8)
    # shape [N, 27, 48, 3]
    return arr


def _predict_scores_pytorch(frames: np.ndarray) -> np.ndarray:
    import torch  # type: ignore
    from vendor.TransNetV2.inference_pytorch.transnetv2_pytorch import TransNetV2 as TorchTN  # type: ignore
    if frames.size == 0:
        return np.zeros((0,), dtype=np.float32)
    model = TorchTN()
    model.eval()
    preds: List[np.ndarray] = []
    # sliding windows of 100 with 25-overlap on both sides, stride 50
    # pad start/end by frame replication like TF version
    no_padded_start = 25
    no_padded_end = 25 + 50 - (len(frames) % 50 if len(frames) % 50 != 0 else 50)
    start_frame = frames[0:1]
    end_frame = frames[-1:]
    padded = np.concatenate([start_frame] * no_padded_start + [frames] + [end_frame] * no_padded_end, axis=0)
    ptr = 0
    print("[TransNetV2][PyTorch] Sliding-window inference…")
    total_core = len(frames)
    last = -1
    while ptr + 100 <= len(padded):
        chunk = padded[ptr:ptr + 100]
        ptr += 50
        inp = torch.from_numpy(chunk[np.newaxis, ...])  # [1,100,27,48,3], uint8
        with torch.no_grad():
            out = model(inp)
            if isinstance(out, tuple):
                one_hot = out[0]
            else:
                one_hot = out
            prob = torch.sigmoid(one_hot)[0, 25:75, 0].cpu().numpy()
            preds.append(prob.astype(np.float32))
        done = min(len(preds) * 50, total_core)
        pct = int(done * 100 / max(1, total_core))
        if pct != last and pct % 5 == 0:
            print(f"[TransNetV2][PyTorch] {pct}% ({done}/{total_core})")
            last = pct
    if not preds:
        return np.zeros((0,), dtype=np.float32)
    scores = np.concatenate(preds, axis=0)
    return scores[: len(frames)]


def _predict_scores_tensorflow(frames: np.ndarray) -> np.ndarray:
    from vendor.TransNetV2.inference.transnetv2 import TransNetV2 as TFTN  # type: ignore
    if frames.size == 0:
        return np.zeros((0,), dtype=np.float32)
    model = TFTN()
    print("[TransNetV2][TensorFlow] Sliding-window inference…")
    single_frame, all_frames = model.predict_frames(frames)
    # 取單幀預測通道
    return np.asarray(single_frame).reshape(-1).astype(np.float32)


def detect_shots_transnet(video_path: Path, threshold: float = 0.5, min_scene_len: float = 0.1) -> List[tuple[float, float]]:
    """
    使用 vendor/TransNetV2 實作偵測切點，優先使用 PyTorch 版本；
    若 torch 不可用，退回 TensorFlow 版本（需 tensorflow）。
    回傳 [(start_sec, end_sec), ...]
    """
    frames = _read_frames_resized(video_path)
    # 優先 TensorFlow（vendor 內含 SavedModel 權重）；若不可用，再嘗試 PyTorch 版本（需自行轉權重）
    scores: np.ndarray
    try:
        import tensorflow  # noqa: F401
        scores = _predict_scores_tensorflow(frames)
    except Exception:
        try:
            import torch  # noqa: F401
            scores = _predict_scores_pytorch(frames)
        except Exception as e:
            raise RuntimeError(
                "TransNet V2 vendor not usable. Please install 'tensorflow' (prefer) or provide PyTorch with converted weights."
            ) from e

    fps, total_frames = _video_meta(video_path)
    scores = np.asarray(scores).reshape(-1)
    cut_idx = np.where(scores >= float(threshold))[0].astype(int)
    if cut_idx.size == 0:
        return [(0.0, float(total_frames) / fps)]
    # merge near cuts (<=2 frames)
    merged: List[int] = []
    last = -9999
    for i in cut_idx.tolist():
        if i - last <= 2:
            if merged:
                merged[-1] = i
            else:
                merged.append(i)
        else:
            merged.append(i)
        last = i
    # build shots respecting min length
    min_len_frames = max(1, int(round(min_scene_len * fps)))
    shots_f: List[Tuple[int, int]] = []
    start = 0
    for ci in merged:
        end = int(ci)
        if end - start >= min_len_frames:
            shots_f.append((start, end))
        start = end
    if total_frames - start >= min_len_frames:
        shots_f.append((start, total_frames))
    out: List[Tuple[float, float]] = [(s / fps, e / fps) for (s, e) in shots_f]
    if not out:
        out = [(0.0, float(total_frames) / fps)]
    print(f"[TransNetV2] Detected {len(out)} scenes; threshold={threshold}, min_len={min_scene_len}s")
    return out

