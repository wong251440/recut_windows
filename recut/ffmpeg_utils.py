import subprocess
from pathlib import Path
from typing import List, Tuple, Optional


def run_ffmpeg(cmd: List[str]) -> int:
    return subprocess.call(["ffmpeg", "-y", *cmd])


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


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
) -> int:
    duration = max(0.0, end - start)
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
    # Video encode options
    if vcodec == "h264_videotoolbox":
        cmd += [
            "-c:v",
            vcodec,
            "-pix_fmt",
            "yuv420p",
        ]
        if vbitrate:
            cmd += ["-b:v", vbitrate]
    else:
        cmd += [
            "-c:v",
            vcodec,
            "-crf",
            str(crf),
            "-preset",
            preset,
        ]
    # Audio encode options
    cmd += [
        "-c:a",
        acodec,
        "-b:a",
        abitrate,
        str(out_file),
    ]
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
