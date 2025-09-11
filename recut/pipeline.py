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
from .features import FeatureConfig, extract_sequence_features, sample_times, video_duration, video_fps
from .ffmpeg_utils import cut_segment, concat_segments, ensure_dir, write_concat_file
from .scene_detect import Shot, detect_shots_auto


@dataclass
class PipelineConfig:
    sample_step: float = 0.2  # seconds (default tuned for tight jump cuts)
    feature_method: str = "clip"  # clip | hsv | auto
    east_model_path: Optional[str] = None
    search_margin: float = 30.0  # seconds around last match
    dtw_window: int | None = None
    sc_threshold: float | None = 18.0  # for PySceneDetect if used (more sensitive than default 27)
    min_scene_len: float = 0.10  # seconds, to capture dense jump cuts
    # 非順序對齊（全域搜尋）
    global_search: bool = True
    topk_candidates: int = 12
    anchor_count: int = 5
    candidate_window: float = 12.0
    use_fast_processor: bool = True
    # 邊界精修
    refine_boundaries: bool = True
    refine_window: float = 0.5
    refine_metric: str = "auto"  # auto|clip|hsv
    cache_source_features: bool = True


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
    # Try PySceneDetect with provided sensitivity, fallback otherwise
    try:
        from .scene_detect import pyscenedetect
        return pyscenedetect(reference_video, threshold=(cfg.sc_threshold or 27.0), min_scene_len=cfg.min_scene_len)
    except Exception:
        return detect_shots_auto(reference_video)


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

    shots = detect_shots(reference_video, cfg)
    result: Dict = {
        "reference": str(reference_video),
        "source": str(source_video),
        "shots": [{"start": s.start, "end": s.end} for s in shots],
        "matches": [],
        "meta": {"ref_duration": ref_dur, "src_duration": src_dur, "step": cfg.sample_step},
    }

    # Precompute source sequence features at uniform grid（支援快取）
    src_times = list(np.arange(0, max(1e-6, src_dur), cfg.sample_step).astype(float))
    cache_path = _source_cache_path(out_dir, source_video, cfg)
    src_feats: np.ndarray
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
                FeatureConfig(method=cfg.feature_method, use_fast_processor=cfg.use_fast_processor),
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
            FeatureConfig(method=cfg.feature_method, use_fast_processor=cfg.use_fast_processor),
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
                    last_end_idx = max(0, int(last_end_t / cfg.sample_step))
                if log:
                    log(f"偵測到續跑狀態，將從鏡頭 {start_shot_index+1} 繼續…")
        except Exception:
            if log:
                log("既有 alignment.json 讀取失敗，將從頭開始。")

    total = max(1, len(shots))
    for idx, s in enumerate(shots[start_shot_index:], start=start_shot_index + 1):
        if log:
            log(f"對齊鏡頭 {idx}/{total}: {s.start:.2f}-{s.end:.2f}s")
        ref_times = sample_times(ref_dur, s.start, s.end, cfg.sample_step)
        ref_feats = extract_sequence_features(
            reference_video,
            ref_times,
            FeatureConfig(method=cfg.feature_method, use_fast_processor=cfg.use_fast_processor),
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
            win = max(1, int(cfg.candidate_window / cfg.sample_step))
            for c in cand:
                s_from = max(0, c - win)
                s_to = min(len(src_times) - 1, c + win)
                off_c, cost_c = best_alignment_offset(ref_feats, src_feats, s_from, s_to, window=cfg.dtw_window)
                if cost_c < best_cost:
                    best_cost, best_off = cost_c, off_c
            off, cost = best_off, best_cost
        else:
            # 單調往前：使用上一段附近的搜尋窗
            margin_idx = int(cfg.search_margin / cfg.sample_step)
            start_idx = max(0, last_end_idx - margin_idx)
            end_idx = min(len(src_times) - 1, last_end_idx + margin_idx + int((s.end - s.start) / cfg.sample_step) + 1)
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
            last_end_idx = off + max(1, int((s.end - s.start) / cfg.sample_step))
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
            method=(cfg.refine_metric if cfg.refine_metric != "auto" else ("clip" if cfg.feature_method == "clip" else "hsv")),
            use_fast=cfg.use_fast_processor,
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
    method: str = "hsv",
    use_fast: bool = True,
    log: Optional[callable] = None,
) -> List[Dict]:
    # Compare single frames around boundaries using chosen feature metric
    fps_ref = max(30.0, video_fps(ref_video))
    fps_src = max(30.0, video_fps(src_video))
    step = 1.0 / max(30.0, min(fps_ref, fps_src))  # ~1/30s granularity
    ref_extractor_cfg = FeatureConfig(method=method, use_fast_processor=use_fast)
    src_extractor_cfg = FeatureConfig(method=method, use_fast_processor=use_fast)

    def best_offset(t_ref: float, t_src0: float) -> float:
        # Sample candidate times around t_src0 and pick min cosine distance
        from .features import load_frame_at, build_extractor
        import numpy as np
        extractor = build_extractor(method, use_fast)
        ref_frame = load_frame_at(ref_video, max(0.0, t_ref))
        if ref_frame is None:
            return t_src0
        ref_vec = extractor(ref_frame).astype(np.float32)
        # Candidates
        times = np.arange(t_src0 - window, t_src0 + window + 1e-6, step, dtype=np.float32)
        best_t, best_d = t_src0, float("inf")
        for ts in times:
            src_frame = load_frame_at(src_video, float(max(0.0, ts)))
            if src_frame is None:
                continue
            vec = extractor(src_frame).astype(np.float32)
            # cosine distance
            a = ref_vec / (np.linalg.norm(ref_vec) + 1e-8)
            b = vec / (np.linalg.norm(vec) + 1e-8)
            d = float(1.0 - float(a @ b))
            if d < best_d:
                best_d, best_t = d, float(ts)
        return best_t

    refined: List[Dict] = []
    for i, m in enumerate(matches, start=1):
        rs, re = float(m.get("ref_start", 0.0)), float(m.get("ref_end", 0.0))
        ss0, se0 = float(m.get("src_start", 0.0)), float(m.get("src_end", 0.0))
        # refine start against ref_start frame
        ss = best_offset(rs + 1e-3, ss0)
        # maintain reference duration for end
        ref_len = max(0.0, re - rs)
        se = ss + ref_len
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
) -> Path:
    ensure_dir(out_dir)
    tmp_dir = out_dir / "segments"
    ensure_dir(tmp_dir)
    out_files: List[Path] = []
    durations: List[float] = []
    # 若啟用，強制每段長度與參照一致（避免舊 alignment 內的 src_end 造成總長漂移）
    for i, m in enumerate(alignment.get("matches", []), start=1):
        out_file = tmp_dir / f"seg_{i:04d}.mp4"
        src_start = float(m["src_start"]) if "src_start" in m else 0.0
        if force_ref_durations and "ref_start" in m and "ref_end" in m:
            ref_len = max(0.0, float(m["ref_end"]) - float(m["ref_start"]))
            src_end = src_start + ref_len
        else:
            src_end = float(m.get("src_end", src_start))
        if src_end <= src_start:
            continue
        cut_segment(
            source_video,
            out_file,
            src_start,
            src_end,
            accurate=accurate_cut,
            vcodec=vcodec,
            crf=crf,
            preset=preset,
            vbitrate=vbitrate,
            abitrate=abitrate,
        )
        out_files.append(out_file)
        durations.append(max(0.0, src_end - src_start))

    concat_txt = out_dir / "concat.txt"
    write_concat_file(out_files, concat_txt, durations=durations)
    final_out = out_dir / "recut_output.mp4"
    if run:
        total_ref = float(sum(durations)) if durations else None
        concat_segments(concat_txt, final_out, total_duration=total_ref, stabilize_audio=stabilize_audio, abitrate=abitrate)
    return final_out
