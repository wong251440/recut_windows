from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Callable

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
    method: str = "clip"  # 強制僅允許 clip
    mask_boxes: Optional[List[MaskBox]] = None
    east_model_path: Optional[str] = None
    use_fast_processor: bool = True
    batch_size: int = 16


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


# 已移除 HSV extractor：系統僅支援 CLIP，缺少 CLIP 會拋錯。


class CLIPExtractor:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", use_fast: bool = True) -> None:
        if torch is None or CLIPModel is None:
            raise RuntimeError("torch+transformers is required for CLIP features")
        # 選擇裝置（優先 CUDA，其次 macOS MPS，最後 CPU）
        dev = "cpu"
        try:
            if torch.cuda.is_available():
                dev = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                dev = "mps"
        except Exception:
            dev = "cpu"
        self.device = torch.device(dev)
        # 載入模型到裝置
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        # 高速化選項
        try:
            if self.device.type == "cuda":
                torch.backends.cudnn.benchmark = True  # 動態最佳化卷積設定
                torch.set_float32_matmul_precision("high")
        except Exception:
            pass
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

    def __call__(self, image: np.ndarray) -> np.ndarray:
        # 單張影像推理
        import PIL.Image
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil = PIL.Image.fromarray(rgb)
        with torch.no_grad():
            inputs = self.proc(images=pil, return_tensors="pt").to(self.device)
            # 自動半精度加速（CUDA/MPS）
            if self.device.type in ("cuda", "mps"):
                try:
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                        out = self.model.get_image_features(**inputs)
                except Exception:
                    out = self.model.get_image_features(**inputs)
            else:
                out = self.model.get_image_features(**inputs)
        vec = out[0].detach().float().cpu().numpy().astype(np.float32)
        return vec

    def encode_images(self, images: List[np.ndarray], batch_size: int = 16) -> np.ndarray:
        # 多張影像批次推理（更快）
        import PIL.Image
        feats: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i : i + batch_size]
                pils = [PIL.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in batch]
                inputs = self.proc(images=pils, return_tensors="pt").to(self.device)
                if self.device.type in ("cuda", "mps"):
                    try:
                        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                            out = self.model.get_image_features(**inputs)
                    except Exception:
                        out = self.model.get_image_features(**inputs)
                else:
                    out = self.model.get_image_features(**inputs)
                feats.append(out.detach().float().cpu().numpy().astype(np.float32))
        if feats:
            return np.concatenate(feats, axis=0)
        return np.zeros((0, self.dim), dtype=np.float32)


def build_extractor(method: str, use_fast: bool = True) -> object:
    if method != "clip":
        raise RuntimeError("Only CLIP features are supported. Set feature method to 'clip'.")
    return CLIPExtractor(use_fast=use_fast)


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
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    if cfg.method != "clip":
        raise RuntimeError("Feature method must be 'clip'.")
    extractor = build_extractor("clip", use_fast=cfg.use_fast_processor)
    out_dim = int(getattr(extractor, "dim", 64))
    feats: List[np.ndarray] = []
    # 若為 CLIPExtractor，改用串流批次以提升效能並支援進度回報
    if isinstance(extractor, CLIPExtractor):
        total = len(times)
        batch_frames: List[np.ndarray] = []
        batch_indices: List[int] = []
        done = 0

        def report():
            # 僅在上層（pipeline）負責節流
            pass

        for idx, t in enumerate(times):
            frame = load_frame_at(video_path, t)
            if frame is None:
                feats.append(np.zeros((out_dim,), dtype=np.float32))
                done += 1
                if on_progress:
                    on_progress(done, total)
                continue
            if cfg.mask_boxes:
                frame = apply_mask(frame, cfg.mask_boxes)
            # 先放置 placeholder，稍後填回
            feats.append(None)
            batch_frames.append(frame)
            batch_indices.append(len(feats) - 1)

            bs = max(1, int(getattr(cfg, "batch_size", 16)))
            if len(batch_frames) >= bs:
                out = extractor.encode_images(batch_frames, batch_size=bs)
                for k, pos in enumerate(batch_indices):
                    vec = out[k] if k < len(out) else np.zeros((out_dim,), dtype=np.float32)
                    feats[pos] = vec.astype(np.float32)
                done += len(batch_frames)
                if on_progress:
                    on_progress(done, total)
                batch_frames.clear()
                batch_indices.clear()
        # 處理尾批
        if batch_frames:
            bs = max(1, int(getattr(cfg, "batch_size", 16)))
            out = extractor.encode_images(batch_frames, batch_size=bs)
            for k, pos in enumerate(batch_indices):
                vec = out[k] if k < len(out) else np.zeros((out_dim,), dtype=np.float32)
                feats[pos] = vec.astype(np.float32)
            done += len(batch_frames)
            if on_progress:
                on_progress(done, total)

        # 收尾：替換任何殘留 None
        feats = [f if f is not None else np.zeros((out_dim,), dtype=np.float32) for f in feats]
        return np.stack(feats, axis=0)

    # 逐張（非批次）後備：理論上不會走到 HSV，僅保留單張 CLIP 推論以防萬一
    for t in times:
        frame = load_frame_at(video_path, t)
        if frame is None:
            feats.append(np.zeros((out_dim,), dtype=np.float32))
            continue
        if cfg.mask_boxes:
            frame = apply_mask(frame, cfg.mask_boxes)
        vec = extractor(frame)
        if vec is None or (hasattr(vec, "size") and getattr(vec, "size", 0) == 0):
            feats.append(np.zeros((out_dim,), dtype=np.float32))
            continue
        if out_dim != len(vec):
            out_dim = int(len(vec))
            for i in range(len(feats)):
                if feats[i].shape[0] != out_dim:
                    feats[i] = np.pad(feats[i], (0, out_dim - feats[i].shape[0]))
        feats.append(np.asarray(vec, dtype=np.float32))
    return np.stack(feats, axis=0)


def clip_runtime_info() -> dict:
    info = {"device": "cpu", "vram_gb": None, "suggest_batch": 8}
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            info["device"] = "cuda"
            try:
                props = torch.cuda.get_device_properties(0)
                vram_gb = float(getattr(props, "total_memory", 0) / (1024 ** 3))
            except Exception:
                vram_gb = None
            info["vram_gb"] = vram_gb
            # 粗略建議：依 VRAM 估批次
            if vram_gb is None:
                info["suggest_batch"] = 16
            elif vram_gb >= 20:
                info["suggest_batch"] = 64
            elif vram_gb >= 12:
                info["suggest_batch"] = 48
            elif vram_gb >= 8:
                info["suggest_batch"] = 32
            elif vram_gb >= 6:
                info["suggest_batch"] = 24
            else:
                info["suggest_batch"] = 16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["device"] = "mps"
            info["suggest_batch"] = 32
        else:
            info["device"] = "cpu"
            info["suggest_batch"] = 8
    except Exception:
        pass
    return info
