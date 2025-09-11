from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import torch  # type: ignore
    from transformers import CLIPModel  # type: ignore
    try:
        from transformers import CLIPImageProcessor  # type: ignore
    except Exception:  # older versions name it CLIPProcessor
        CLIPImageProcessor = None  # type: ignore
        from transformers import CLIPProcessor as _CLIPProcessor  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    CLIPModel = None  # type: ignore
    CLIPImageProcessor = None  # type: ignore

from .masking import MaskBox, apply_mask


@dataclass
class FeatureConfig:
    method: str = "auto"  # "clip" | "hsv" | "auto"
    mask_boxes: Optional[List[MaskBox]] = None
    east_model_path: Optional[str] = None
    use_fast_processor: bool = True


def load_frame_at(video_path: Path, t: float) -> Optional[np.ndarray]:
    if cv2 is None:
        return None
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return frame


class HSVHistExtractor:
    def __init__(self) -> None:
        if cv2 is None:
            raise RuntimeError("opencv-python is required for HSV features")
        self.dim = 64

    def __call__(self, image: np.ndarray) -> np.ndarray:
        img = image
        if img is None:
            return np.zeros((64,), dtype=np.float32)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 1], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
        return hist


class CLIPExtractor:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", use_fast: bool = True) -> None:
        if torch is None or CLIPModel is None:
            raise RuntimeError("torch+transformers is required for CLIP features")
        self.model = CLIPModel.from_pretrained(model_name)
        # 將輸出維度記錄下來（避免 None frame 時維度不一致）
        try:
            self.dim = int(getattr(self.model.config, "projection_dim", 512))
        except Exception:
            self.dim = 512
        if CLIPImageProcessor is not None:
            # Prefer fast image processor if available
            try:
                self.proc = CLIPImageProcessor.from_pretrained(model_name, use_fast=use_fast)
            except TypeError:
                self.proc = CLIPImageProcessor.from_pretrained(model_name)
        else:
            # Fallback to legacy processor API
            self.proc = _CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        import PIL.Image

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil = PIL.Image.fromarray(rgb)
        with torch.no_grad():
            inputs = self.proc(images=pil, return_tensors="pt")
            out = self.model.get_image_features(**inputs)
        vec = out[0].detach().cpu().numpy().astype(np.float32)
        return vec


def build_extractor(method: str, use_fast: bool = True) -> object:
    if method == "clip":
        return CLIPExtractor(use_fast=use_fast)
    if method == "hsv":
        return HSVHistExtractor()
    # auto
    try:
        return CLIPExtractor(use_fast=use_fast)
    except Exception:
        return HSVHistExtractor()


def sample_times(duration: float, start: float, end: float, step: float) -> List[float]:
    t = start
    times = []
    while t <= min(duration, end):
        times.append(t)
        t += step
    if not times or times[-1] < end:
        times.append(min(end, duration))
    return times


def video_duration(video_path: Path) -> float:
    if cv2 is None:
        return 0.0
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    if fps <= 0:
        return 0.0
    return float(frames / fps)


def video_fps(video_path: Path) -> float:
    if cv2 is None:
        return 30.0
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return float(fps if fps > 0 else 30.0)


def extract_sequence_features(
    video_path: Path,
    times: List[float],
    cfg: FeatureConfig,
) -> np.ndarray:
    extractor = build_extractor(cfg.method if cfg.method != "auto" else "auto", use_fast=cfg.use_fast_processor)
    out_dim = int(getattr(extractor, "dim", 64))
    feats: List[np.ndarray] = []
    for t in times:
        frame = load_frame_at(video_path, t)
        if frame is None:
            feats.append(np.zeros((out_dim,), dtype=np.float32))
            continue
        if cfg.mask_boxes:
            frame = apply_mask(frame, cfg.mask_boxes)
        vec = extractor(frame)
        if vec is None or (hasattr(vec, "size") and vec.size == 0):
            feats.append(np.zeros((out_dim,), dtype=np.float32))
            continue
        if out_dim != len(vec):
            # 第一次有效特徵時校正 out_dim，並將先前填補的向量重新調整
            out_dim = int(len(vec))
            for i in range(len(feats)):
                if feats[i].shape[0] != out_dim:
                    feats[i] = np.pad(feats[i], (0, out_dim - feats[i].shape[0]))
        feats.append(vec.astype(np.float32))
    return np.stack(feats, axis=0)
