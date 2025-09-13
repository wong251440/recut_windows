import subprocess
import platform
from pathlib import Path
from typing import List, Tuple, Optional


def run_ffmpeg(cmd: List[str]) -> int:
    return subprocess.call(["ffmpeg", "-y", *cmd])


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# NVENC 預檢已移除：改為預設軟編碼（libx264），僅保留 macOS VideoToolbox。


def normalize_video(input_path: Path, output_path: Path, fps: float = 30.0, width: int | None = None, height: int | None = None) -> int:
    """
    Normalize fps and optionally resize to given width/height (keeping aspect if only one given).
    """
    cmd = ["-i", str(input_path), "-r", str(fps)]
    if width and height:
        cmd += ["-vf", f"scale={width}:{height}"]
    elif width and not height:
        cmd += ["-vf", f"scale={width}:-2"]
    elif height and not width:
        cmd += ["-vf", f"scale=-2:{height}"]
    cmd += [str(output_path)]
    return run_ffmpeg(cmd)


def cut_segment(
    src: Path,
    out_file: Path,
    start: float,
    end: float,
    *,
    accurate: bool = True,
    vcodec: str = "libx264",
    crf: int = 18,
    preset: str = "veryfast",
    vbitrate: Optional[str] = None,
    acodec: str = "aac",
    abitrate: str = "192k",
    gpu: Optional[int] = None,
) -> int:
    duration = max(0.0, end - start)
    sysname = platform.system().lower()
    is_windows = sysname.startswith("win")
    is_macos = sysname == "darwin"
    if not accurate:
        # 快速但不精準（可能因 keyframe 導致段長累積誤差）
        cmd = [
            "-ss",
            f"{start:.3f}",
            "-i",
            str(src),
            "-t",
            f"{duration:.3f}",
            "-c",
            "copy",
            str(out_file),
        ]
        return run_ffmpeg(cmd)
    # 精準裁切（重編碼）：將 -ss 放在 -i 之前，快速靠近目標時間，再重編碼確保幀級精準
    cmd = [
        "-ss",
        f"{start:.3f}",
        "-i",
        str(src),
        "-t",
        f"{duration:.3f}",
        "-fflags",
        "+genpts",
        "-avoid_negative_ts",
        "make_zero",
        "-muxpreload",
        "0",
        "-muxdelay",
        "0",
    ]
    # 視訊編碼策略：
    # - macOS 且選擇 h264_videotoolbox：保留 VideoToolbox（可搭配 vbitrate）
    # - 其他情況：使用軟體 libx264（CRF + preset），忽略 GPU 參數
    if is_macos and vcodec == "h264_videotoolbox":
        # 啟用 VideoToolbox 硬體路徑（僅 macOS）
        cmd = ["-hwaccel", "videotoolbox", *cmd]
        cmd += ["-c:v", "h264_videotoolbox", "-pix_fmt", "yuv420p"]
        if vbitrate:
            cmd += ["-b:v", vbitrate]
    else:
        # 軟體編碼（跨平台）：libx264
        cmd += [
            "-c:v",
            "libx264",
            "-preset",
            str(preset),
            "-crf",
            str(int(crf)),
            "-pix_fmt",
            "yuv420p",
        ]
        if vbitrate:
            # 若使用者仍提供位元率，則套用（可與 CRF 併用作上限）
            cmd += ["-b:v", vbitrate]
    # Audio encode options
    cmd += [
        "-c:a",
        acodec,
        "-b:a",
        abitrate,
        str(out_file),
    ]
    return run_ffmpeg(cmd)


def cut_segment_frames(
    src: Path,
    out_file: Path,
    start_frame: int,
    end_frame: int,
    *,
    vcodec: str = "libx264",
    crf: int = 18,
    preset: str = "veryfast",
    vbitrate: Optional[str] = None,
    acodec: str = "aac",
    abitrate: str = "192k",
) -> int:
    """
    Frame-accurate cut using select/trim by frame index and matching audio trim by time.
    end_frame is exclusive (i.e., cut frames [start_frame, end_frame)).
    """
    import math
    from .features import video_fps
    fps = float(video_fps(src))
    sf = max(0, int(start_frame))
    ef = max(sf, int(end_frame))
    dur = max(0.0, (ef - sf) / fps if fps > 0 else 0.0)
    # Video select by frame number; Audio trim by time to the same duration
    vf = f"select='between(n,{sf},{ef-1})',setpts=PTS-STARTPTS"
    af = f"atrim=start={sf/ fps if fps>0 else 0.0}:duration={dur},asetpts=PTS-STARTPTS"
    cmd: List[str] = [
        "-i", str(src),
        "-filter_complex", f"[0:v]{vf}[v];[0:a]{af}[a]",
        "-map", "[v]",
        "-map", "[a]",
    ]
    sysname = platform.system().lower()
    is_macos = sysname == "darwin"
    if is_macos and vcodec == "h264_videotoolbox":
        cmd += ["-c:v", "h264_videotoolbox", "-pix_fmt", "yuv420p"]
        if vbitrate:
            cmd += ["-b:v", vbitrate]
    else:
        cmd += [
            "-c:v", "libx264",
            "-preset", str(preset),
            "-crf", str(int(crf)),
            "-pix_fmt", "yuv420p",
        ]
        if vbitrate:
            cmd += ["-b:v", vbitrate]
    cmd += ["-c:a", acodec, "-b:a", abitrate, str(out_file)]
    return run_ffmpeg(cmd)


def render_segments_single_pass(
    src: Path,
    output_path: Path,
    frame_ranges: List[Tuple[int, int]],
    *,
    vcodec: str = "libx264",
    crf: int = 18,
    preset: str = "veryfast",
    vbitrate: Optional[str] = None,
) -> int:
    """
    Single-pass precise render using trim+setpts+concat filtergraph (video only).
    - Avoids concat demuxer timing quirks (inpoint/outpoint) on MP4/H.264.
    - Ensures continuous PTS across segments; reduces dup/drop-induced freeze。
    """
    # Prepare segments
    segs: List[Tuple[int, int]] = []
    for s, e in frame_ranges:
        s0 = max(0, int(s))
        e0 = max(s0, int(e))
        if e0 > s0:
            segs.append((s0, e0))
    if not segs:
        return -1

    # If segments are too many, process in batches to reduce filtergraph cost
    BATCH_THRESHOLD = 120
    CHUNK_SIZE = 80
    if len(segs) > BATCH_THRESHOLD:
        tmp_files: List[Path] = []
        sysname = platform.system().lower()
        is_macos = sysname == "darwin"
        for gi in range(0, len(segs), CHUNK_SIZE):
            chunk = segs[gi : gi + CHUNK_SIZE]
            trim_lines: List[str] = []
            seg_labels: List[str] = []
            for i, (s0, e0) in enumerate(chunk):
                lbl = f"v{i}"
                trim_lines.append(
                    f"[0:v]trim=start_frame={s0}:end_frame={e0},setpts=PTS-STARTPTS[{lbl}]"
                )
                seg_labels.append(f"[{lbl}]")
            group_out = "g0"
            filtergraph = ";".join(trim_lines + ["".join(seg_labels) + f"concat=n={len(seg_labels)}:v=1:a=0[{group_out}]"])
            out_file = output_path.parent / f"_chunk_{gi//CHUNK_SIZE:04d}.mp4"
            cmd: List[str] = [
                "-i", str(src),
                "-filter_complex", filtergraph,
                "-map", f"[{group_out}]",
                "-an",
                "-fflags", "+genpts",
                "-fps_mode", "passthrough",
            ]
            if is_macos and vcodec == "h264_videotoolbox":
                cmd += ["-c:v", "h264_videotoolbox", "-pix_fmt", "yuv420p"]
                if vbitrate:
                    cmd += ["-b:v", vbitrate]
            else:
                cmd += [
                    "-c:v", "libx264",
                    "-preset", str(preset),
                    "-crf", str(int(crf)),
                    "-pix_fmt", "yuv420p",
                ]
                if vbitrate:
                    cmd += ["-b:v", vbitrate]
            cmd += [str(out_file)]
            rc = run_ffmpeg(cmd)
            if rc != 0:
                return rc
            tmp_files.append(out_file)
        # Concat chunks by copy (video-only), continuous PTS generated
        concat_txt = output_path.parent / "chunks_concat.txt"
        write_concat_file(tmp_files, concat_txt, durations=None)
        cmd2: List[str] = [
            "-f", "concat", "-safe", "0", "-i", str(concat_txt),
            "-c", "copy", "-movflags", "+faststart",
            str(output_path),
        ]
        return run_ffmpeg(cmd2)

    # Build filtergraph with grouping (single-pass path)
    trim_lines: List[str] = []
    seg_labels: List[str] = []
    for i, (s0, e0) in enumerate(segs):
        lbl = f"v{i}"
        trim_lines.append(
            f"[0:v]trim=start_frame={s0}:end_frame={e0},setpts=PTS-STARTPTS[{lbl}]"
        )
        seg_labels.append(f"[{lbl}]")

    group_size = 40
    group_labels: List[str] = []
    concat_lines: List[str] = []
    for gi in range(0, len(seg_labels), group_size):
        chunk = seg_labels[gi : gi + group_size]
        out_lbl = f"g{gi//group_size}"
        concat_lines.append("".join(chunk) + f"concat=n={len(chunk)}:v=1:a=0[{out_lbl}]")
        group_labels.append(f"[{out_lbl}]")

    final_label = "vout"
    parts = trim_lines + concat_lines
    if len(group_labels) > 1:
        parts.append("".join(group_labels) + f"concat=n={len(group_labels)}:v=1:a=0[{final_label}]")
        map_target = final_label
    else:
        map_target = group_labels[0].strip("[]")
    filtergraph = ";".join(parts)

    sysname = platform.system().lower()
    is_macos = sysname == "darwin"
    cmd: List[str] = [
        "-i", str(src),
        "-filter_complex", filtergraph,
        "-map", f"[{map_target}]",
        "-an",
        "-fflags", "+genpts",
        "-fps_mode", "passthrough",  # avoid CFR re-timing (no dup/drop)
    ]
    if is_macos and vcodec == "h264_videotoolbox":
        cmd += ["-c:v", "h264_videotoolbox", "-pix_fmt", "yuv420p"]
        if vbitrate:
            cmd += ["-b:v", vbitrate]
    else:
        cmd += [
            "-c:v", "libx264",
            "-preset", str(preset),
            "-crf", str(int(crf)),
            "-pix_fmt", "yuv420p",
        ]
        if vbitrate:
            cmd += ["-b:v", vbitrate]
    cmd += [str(output_path)]
    return run_ffmpeg(cmd)


def write_concat_file(files: List[Path], concat_txt: Path, durations: Optional[List[float]] = None) -> None:
    """
    產生 ffconcat 清單。
    - 若未提供 durations：輸出簡單的 concat 列表。
    - 若提供 durations：使用 demuxer 的 duration 欄位；注意 duration 是套用在「前一個 file」。
      正確格式需為：file A -> duration dA -> file B -> duration dB -> ... -> file Z（重複一次）。
    """
    with concat_txt.open("w", encoding="utf-8") as f:
        if not files:
            return
        if durations is None:
            for p in files:
                f.write(f"file '{p.as_posix()}'\n")
            return

        # 帶 duration 的 ffconcat 格式
        f.write("ffconcat version 1.0\n")
        n = len(files)
        # 第一個 file
        f.write(f"file '{files[0].as_posix()}'\n")
        # 之後的每個 file 之前，寫上「上一段的 duration」
        for i in range(1, n):
            d_prev = max(0.0, float(durations[i - 1]))
            f.write(f"duration {d_prev:.6f}\n")
            f.write(f"file '{files[i].as_posix()}'\n")
        # 重複最後一段 file 以讓最後一個 duration 生效
        d_last = max(0.0, float(durations[-1]))
        f.write(f"duration {d_last:.6f}\n")
        f.write(f"file '{files[-1].as_posix()}'\n")


def concat_segments(
    concat_txt: Path,
    output_path: Path,
    *,
    total_duration: Optional[float] = None,
    stabilize_audio: bool = True,
    abitrate: str = "192k",
) -> int:
    cmd: List[str] = [
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_txt),
        "-fflags",
        "+genpts",
        "-reset_timestamps",
        "1",
    ]
    if total_duration is not None and total_duration > 0:
        cmd += ["-t", f"{total_duration:.3f}"]
    if stabilize_audio:
        # 保留視訊不重編，音訊重編並修正時間戳，降低 DTS 警告與長度漂移
        cmd += [
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            abitrate,
            "-af",
            "aresample=async=1:first_pts=0",
            "-movflags",
            "+faststart",
        ]
    else:
        cmd += ["-c", "copy"]
    cmd += [str(output_path)]
    return run_ffmpeg(cmd)
