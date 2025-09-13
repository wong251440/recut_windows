from __future__ import annotations

import json
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .dtw import best_alignment_offset, cdist
from .features import (
    FeatureConfig,
    extract_sequence_features,
    sample_times,
    video_duration,
    video_fps,
    load_frame_idx,
)
from .ffmpeg_utils import (
    cut_segment,
    cut_segment_frames,
    concat_segments,
    ensure_dir,
    write_concat_file,
    render_segments_single_pass,
)
from .scene_detect import Shot, detect_shots_auto


@dataclass
class PipelineConfig:
    sample_step: float = 0.0  # seconds; 0 表示逐幀取樣（1/min(fps)）
    feature_method: str = "clip"  # 僅允許 clip
    east_model_path: Optional[str] = None
    auto_mask_text: bool = False
    search_margin: float = 30.0  # seconds around last match
    dtw_window: int | None = None
    sc_threshold: float | None = 0.5  # 預設給 TransNet；PySceneDetect 時可調大（如 18）
    min_scene_len: float = 0.10  # seconds, to capture dense jump cuts
    sc_detector: str = "transnet"  # transnet | pyscenedetect
    # 非順序對齊（全域搜尋）
    global_search: bool = True
    topk_candidates: int = 12
    anchor_count: int = 5
    candidate_window: float = 12.0
    use_fast_processor: bool = True
    # 邊界精修
    refine_boundaries: bool = True
    refine_window: float = 0.5
    refine_metric: str = "clip"  # 僅允許 clip
    cache_source_features: bool = True
    # CLIP 批次大小（同時處理影格數）
    clip_batch_size: int = 16
    # 從已存在的場景清單續跑（若提供，將跳過場景偵測）
    scenes_path: Optional[Path] = None


def _source_cache_path(out_dir: Path, source_video: Path, cfg: PipelineConfig) -> Path:
    try:
        st = os.stat(source_video)
        key = {
            "path": str(Path(source_video).resolve()),
            "size": st.st_size,
            "mtime": int(st.st_mtime),
            "step": cfg.sample_step,
            "feat": cfg.feature_method,
            "fast": cfg.use_fast_processor,
        }
        key_s = json.dumps(key, sort_keys=True).encode("utf-8")
        h = hashlib.sha1(key_s).hexdigest()[:16]
    except Exception:
        h = "nocache"
    return out_dir / f"src_feats_{h}.npz"


def detect_shots(reference_video: Path, cfg: PipelineConfig) -> List[Shot]:
    # 依選擇的偵測器偵測場景
    if cfg.sc_detector == "pyscenedetect":
        from .scene_detect import pyscenedetect
        return pyscenedetect(reference_video, threshold=(cfg.sc_threshold or 27.0), min_scene_len=cfg.min_scene_len)
    elif cfg.sc_detector == "transnet":
        from .scene_detect import transnet_detect
        thr = float(cfg.sc_threshold if cfg.sc_threshold is not None else 0.5)
        return transnet_detect(reference_video, threshold=thr, min_scene_len=cfg.min_scene_len)
    else:
        raise RuntimeError(f"Unknown scene detector: {cfg.sc_detector}")


def _load_scenes_from_json(path: Path) -> List[Shot]:
    data = json.load(open(path, "r", encoding="utf-8"))
    shots = data.get("shots") if isinstance(data, dict) else None
    if not isinstance(shots, list):
        raise RuntimeError(f"Invalid scenes file: {path}")
    out: List[Shot] = []
    for it in shots:
        try:
            s, e = float(it.get("start")), float(it.get("end"))
            out.append(Shot(s, e))
        except Exception:
            continue
    if not out:
        raise RuntimeError(f"No shots in scenes file: {path}")
    return out


def scene_detect_only(reference_video: Path, out_dir: Path, cfg: PipelineConfig, log: Optional[callable] = None) -> Path:
    ensure_dir(out_dir)
    shots = detect_shots(reference_video, cfg)
    scenes = {
        "reference": str(reference_video),
        "detector": cfg.sc_detector,
        "threshold": cfg.sc_threshold,
        "min_scene_len": cfg.min_scene_len,
        "shots": [{"start": s.start, "end": s.end} for s in shots],
    }
    out_path = out_dir / "shots.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(scenes, f, ensure_ascii=False, indent=2)
    if log:
        log(f"已輸出場景清單：{out_path}")
    return out_path


def align(
    reference_video: Path,
    source_video: Path,
    out_dir: Path,
    cfg: PipelineConfig,
    log: Optional[callable] = None,
    progress: Optional[callable] = None,
    resume: bool = False,
) -> Dict:
    ensure_dir(out_dir)
    if log:
        log("讀取影片資訊…")
    ref_dur = video_duration(reference_video)
    src_dur = video_duration(source_video)

    # 場景偵測或讀取既有場景
    shots: List[Shot]
    if cfg.scenes_path is not None and Path(cfg.scenes_path).exists():
        if log:
            log(f"使用既有場景清單：{Path(cfg.scenes_path).name}")
        shots = _load_scenes_from_json(Path(cfg.scenes_path))
    else:
        shots = detect_shots(reference_video, cfg)
    result: Dict = {
        "reference": str(reference_video),
        "source": str(source_video),
        "shots": [{"start": s.start, "end": s.end} for s in shots],
        "matches": [],
        "meta": {"ref_duration": ref_dur, "src_duration": src_dur, "step": cfg.sample_step},
    }

    # CLIP 執行環境資訊與批次建議
    try:
        from .features import clip_runtime_info
        info = clip_runtime_info()
        if log:
            dev = info.get("device")
            vram = info.get("vram_gb")
            sugg = info.get("suggest_batch")
            if vram is not None:
                log(f"CLIP 裝置：{dev}, VRAM≈{float(vram):.1f}GB, 目前批次={cfg.clip_batch_size}，建議批次≈{sugg}")
            else:
                log(f"CLIP 裝置：{dev}, 目前批次={cfg.clip_batch_size}，建議批次≈{sugg}")
    except Exception:
        pass

    # 決定取樣步長（逐幀或固定秒數）
    eff_step = float(cfg.sample_step) if (cfg.sample_step and cfg.sample_step > 0) else (
        1.0 / max(1.0, min(video_fps(reference_video), video_fps(source_video)))
    )
    # Precompute source sequence features at uniform grid（支援快取）
    src_times = list(np.arange(0, max(1e-6, src_dur), eff_step).astype(float))
    cache_path = _source_cache_path(out_dir, source_video, cfg)
    src_feats: np.ndarray
    # 進度回報器（每+5%列印一次）
    def _src_prog(done: int, total: int) -> None:
        if not log:
            return
        pct = int(done * 100 / max(1, total))
        last = getattr(_src_prog, "_last", -1)
        if pct - last >= 1 or pct == 100:
            _src_prog._last = pct  # type: ignore[attr-defined]
            log(f"母帶特徵抽取進度：{pct}% ({done}/{total})")
    if cfg.cache_source_features and cache_path.exists():
        try:
            data = np.load(cache_path)
            src_times_cached = data["times"]
            src_feats = data["feats"]
            if len(src_times_cached) == len(src_times):
                if log:
                    log(f"命中快取：{cache_path.name}（{len(src_times)} 取樣）")
            else:
                raise ValueError("cache mismatch")
        except Exception:
            if log:
                log("快取讀取失敗，重新抽取母帶特徵…")
            src_feats = extract_sequence_features(
                source_video,
                src_times,
                FeatureConfig(
                    method="clip",
                    use_fast_processor=cfg.use_fast_processor,
                    batch_size=cfg.clip_batch_size,
                    east_model_path=cfg.east_model_path,
                    auto_mask_text=cfg.auto_mask_text,
                ),
                on_progress=_src_prog,
            )
            if cfg.cache_source_features:
                try:
                    np.savez_compressed(cache_path, times=np.array(src_times, dtype=np.float32), feats=src_feats)
                    if log:
                        log(f"已寫入快取：{cache_path.name}")
                except Exception:
                    if log:
                        log("寫入快取失敗（略過）")
    else:
        if log:
            log("預先抽取母帶特徵（整段）…")
        src_feats = extract_sequence_features(
            source_video,
            src_times,
            FeatureConfig(
                method="clip",
                use_fast_processor=cfg.use_fast_processor,
                batch_size=cfg.clip_batch_size,
                east_model_path=cfg.east_model_path,
                auto_mask_text=cfg.auto_mask_text,
            ),
            on_progress=_src_prog,
        )
        if cfg.cache_source_features:
            try:
                np.savez_compressed(cache_path, times=np.array(src_times, dtype=np.float32), feats=src_feats)
                if log:
                    log(f"已寫入快取：{cache_path.name}")
            except Exception:
                if log:
                    log("寫入快取失敗（略過）")

    # Resume support
    alignment_path = out_dir / "alignment.json"
    start_shot_index = 0
    last_end_idx = 0
    if resume and alignment_path.exists():
        try:
            with alignment_path.open("r", encoding="utf-8") as f:
                prev = json.load(f)
            if (
                prev.get("reference") == str(reference_video)
                and prev.get("source") == str(source_video)
                and isinstance(prev.get("matches"), list)
            ):
                # Use previous result and continue appending
                result = prev
                start_shot_index = len(prev["matches"])
                if start_shot_index > 0:
                    last_end_t = float(prev["matches"][-1].get("src_end", 0.0))
                    last_end_idx = max(0, int(last_end_t / eff_step))
                if log:
                    log(f"偵測到續跑狀態，將從鏡頭 {start_shot_index+1} 繼續…")
        except Exception:
            if log:
                log("既有 alignment.json 讀取失敗，將從頭開始。")

    total = max(1, len(shots))
    for idx, s in enumerate(shots[start_shot_index:], start=start_shot_index + 1):
        if log:
            log(f"對齊鏡頭 {idx}/{total}: {s.start:.2f}-{s.end:.2f}s")
        ref_times = sample_times(ref_dur, s.start, s.end, eff_step)
        ref_feats = extract_sequence_features(
            reference_video,
            ref_times,
            FeatureConfig(
                method="clip",
                use_fast_processor=cfg.use_fast_processor,
                batch_size=cfg.clip_batch_size,
                east_model_path=cfg.east_model_path,
                auto_mask_text=cfg.auto_mask_text,
            ),
        )

        if cfg.global_search:
            # 全域搜尋：以多個錨點找候選，再局部 DTW 評分
            A = max(1, int(cfg.anchor_count))
            anchor_idx = np.unique(np.linspace(0, len(ref_feats) - 1, num=A, dtype=int))
            cand: List[int] = []
            for ai in anchor_idx:
                d = cdist(ref_feats[ai:ai+1], src_feats)[0]
                topk = np.argsort(d)[: max(1, int(cfg.topk_candidates))].tolist()
                cand.extend(topk)
            # 去重與排序
            cand = sorted(set(cand))
            if log:
                log(f"  候選中心 {len(cand)}，anchors={len(anchor_idx)}, topk={cfg.topk_candidates}")
            best_cost = float('inf')
            best_off = 0
            win = max(1, int(cfg.candidate_window / eff_step))
            for c in cand:
                s_from = max(0, c - win)
                s_to = min(len(src_times) - 1, c + win)
                off_c, cost_c = best_alignment_offset(ref_feats, src_feats, s_from, s_to, window=cfg.dtw_window)
                if cost_c < best_cost:
                    best_cost, best_off = cost_c, off_c
            off, cost = best_off, best_cost
        else:
            # 單調往前：使用上一段附近的搜尋窗
            margin_idx = int(cfg.search_margin / eff_step)
            start_idx = max(0, last_end_idx - margin_idx)
            end_idx = min(len(src_times) - 1, last_end_idx + margin_idx + int((s.end - s.start) / eff_step) + 1)
            off, cost = best_alignment_offset(ref_feats, src_feats, start_idx, end_idx, window=cfg.dtw_window)
        match_start_t = src_times[off]
        # 為了與參照段長完全一致，結束點直接以參照段長決定
        ref_len = float(s.end - s.start)
        match_end_t = min(src_dur, match_start_t + ref_len)
        result["matches"].append(
            {
                "ref_start": s.start,
                "ref_end": s.end,
                "src_start": float(match_start_t),
                "src_end": float(match_end_t),
                "cost": float(cost),
            }
        )
        if not cfg.global_search:
            last_end_idx = off + max(1, int((s.end - s.start) / eff_step))
        if log:
            log(f"  選定 idx={off}, cost={cost:.4f}, 時間 {match_start_t:.2f}-{match_end_t:.2f}s")
        if progress:
            progress(idx / float(total))
        # Incremental save after each shot for resume
        with alignment_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    # Optional: boundary refinement to align cuts at frame-level
    if cfg.refine_boundaries and result["matches"]:
        if log:
            log("開始邊界精修（幀級）…")
        refined = _refine_boundaries(
            reference_video,
            source_video,
            result["matches"],
            window=cfg.refine_window,
            method="clip",
            use_fast=cfg.use_fast_processor,
            east_model_path=cfg.east_model_path,
            auto_mask_text=cfg.auto_mask_text,
            log=log,
        )
        result["matches"] = refined
        with alignment_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    # Save alignment json
    with alignment_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def _refine_boundaries(
    ref_video: Path,
    src_video: Path,
    matches: List[Dict],
    *,
    window: float = 0.5,
    method: str = "clip",
    use_fast: bool = True,
    east_model_path: Optional[str] = None,
    auto_mask_text: bool = False,
    log: Optional[callable] = None,
) -> List[Dict]:
    # Frame-accurate refinement at both boundaries
    fps_ref = max(1.0, video_fps(ref_video))
    fps_src = max(1.0, video_fps(src_video))
    if method != "clip":
        raise RuntimeError("Boundary refinement only supports 'clip' features.")

    def best_offset_frame(ref_idx: int, src_idx0: int) -> int:
        from .features import build_extractor, load_frame_idx
        import numpy as np
        from .masking import apply_mask, detect_text_boxes
        extractor = build_extractor("clip", use_fast)
        ref_frame = load_frame_idx(ref_video, max(0, int(ref_idx)))
        if ref_frame is None:
            return src_idx0
        # optional masking on ref frame
        if auto_mask_text and east_model_path:
            try:
                boxes = detect_text_boxes(ref_frame, east_model_path)
                if boxes:
                    ref_frame = apply_mask(ref_frame, boxes)
            except Exception:
                pass
        ref_vec = extractor(ref_frame).astype(np.float32)
        win = max(0, int(round(window * fps_src)))
        s_i = max(0, int(src_idx0 - win))
        e_i = int(src_idx0 + win)
        best_i, best_d = src_idx0, float("inf")
        for i in range(s_i, e_i + 1):
            src_frame = load_frame_idx(src_video, i)
            if src_frame is None:
                continue
            if auto_mask_text and east_model_path:
                try:
                    boxes_s = detect_text_boxes(src_frame, east_model_path)
                    if boxes_s:
                        src_frame = apply_mask(src_frame, boxes_s)
                except Exception:
                    pass
            vec = extractor(src_frame).astype(np.float32)
            a = ref_vec / (np.linalg.norm(ref_vec) + 1e-8)
            b = vec / (np.linalg.norm(vec) + 1e-8)
            d = float(1.0 - float(a @ b))
            if d < best_d:
                best_d, best_i = d, i
        return best_i

    refined: List[Dict] = []
    for i, m in enumerate(matches, start=1):
        rs, re = float(m.get("ref_start", 0.0)), float(m.get("ref_end", 0.0))
        ss0, se0 = float(m.get("src_start", 0.0)), float(m.get("src_end", 0.0))
        ref_start_idx = int(round(rs * fps_ref))
        ref_end_idx = int(round(re * fps_ref))
        src_start_idx0 = int(round(ss0 * fps_src))
        # refine start against ref_start frame
        src_start_idx = best_offset_frame(ref_start_idx, src_start_idx0)
        # refine end against ref_end frame; fall back to duration mapping if missing
        if se0 <= ss0:
            se0 = ss0 + max(0.0, (re - rs))
        src_end_idx0 = int(round(se0 * fps_src))
        src_end_idx = best_offset_frame(ref_end_idx, max(src_end_idx0, src_start_idx))
        if src_end_idx <= src_start_idx:
            src_end_idx = src_start_idx + max(1, int(round((re - rs) * fps_src)))
        ss = float(src_start_idx / fps_src)
        se = float(src_end_idx / fps_src)
        nm = dict(m)
        nm["src_start"], nm["src_end"] = float(ss), float(se)
        refined.append(nm)
        if log and (i % 25 == 0 or i == len(matches)):
            log(f"  精修 {i}/{len(matches)} 完成")
    return refined


def render_from_alignment(
    source_video: Path,
    alignment: Dict,
    out_dir: Path,
    run: bool = False,
    *,
    accurate_cut: bool = True,
    vcodec: str = "libx264",
    crf: int = 18,
    preset: str = "veryfast",
    vbitrate: Optional[str] = None,
    abitrate: str = "192k",
    force_ref_durations: bool = True,
    stabilize_audio: bool = True,
    concat_duration_mode: str = "actual",  # ref | none | actual
    gpu: Optional[int] = None,
) -> Path:
    """
    New rendering path: precise frame-accurate single-pass video-only export.
    - Always precise (ignores fast-copy)
    - Drops audio entirely
    - Performs cut+concat in one ffmpeg pass using select over frame indices
    """
    ensure_dir(out_dir)
    final_out = out_dir / "recut_output.mp4"
    # Build frame ranges from alignment
    ranges: List[Tuple[int, int]] = []
    for m in alignment.get("matches", []):
        ss0 = float(m.get("src_start", 0.0))
        se0 = float(m.get("src_end", 0.0))
        if force_ref_durations and ("ref_start" in m and "ref_end" in m):
            ref_len = max(0.0, float(m["ref_end"]) - float(m["ref_start"]))
            se0 = ss0 + ref_len
        if se0 <= ss0:
            continue
        fps_src = max(1.0, video_fps(source_video))
        sf = int(round(ss0 * fps_src))
        ef = int(round(se0 * fps_src))
        if ef > sf:
            ranges.append((sf, ef))
    if run:
        render_segments_single_pass(
            source_video,
            final_out,
            ranges,
            vcodec=vcodec,
            crf=crf,
            preset=preset,
            vbitrate=vbitrate,
        )
    return final_out
