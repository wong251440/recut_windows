from __future__ import annotations

import os
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional

import xml.etree.ElementTree as ET


def _file_url(p: Path) -> str:
    ap = str(Path(p).absolute())
    ap = ap.replace("\\", "/")
    return "file:///" + urllib.parse.quote(ap, safe="/:.")


def _frames(t: float, fps: float) -> int:
    return int(round(max(0.0, float(t)) * float(fps)))


def export_fcp7_xml(
    alignment: Dict,
    source_video: Path,
    out_path: Path,
    *,
    timeline_fps: float = 30.0,
    ntsc: bool = False,
    sequence_name: str = "Recut",
) -> Path:
    """
    輸出 Final Cut Pro 7 XML（Premiere Pro 可匯入）。
    - 僅單視訊軌直切；音訊省略。
    - 座標以影格為單位；時間軸從 00:00:00:00 開始。
    """
    matches: List[Dict] = list(alignment.get("matches", []))
    timebase = int(round(timeline_fps))

    # XML 結構
    fcpx = ET.Element("xmeml", version="5")
    seq = ET.SubElement(fcpx, "sequence", id="sequence-1")
    ET.SubElement(seq, "name").text = sequence_name
    rate = ET.SubElement(seq, "rate")
    ET.SubElement(rate, "timebase").text = str(timebase)
    ET.SubElement(rate, "ntsc").text = "TRUE" if ntsc else "FALSE"
    ET.SubElement(seq, "in").text = "0"
    # 計算總長度（以參照段長疊加）
    total_frames = 0
    for m in matches:
        ref_len = float(m.get("ref_end", 0.0)) - float(m.get("ref_start", 0.0))
        total_frames += _frames(ref_len, timeline_fps)
    ET.SubElement(seq, "out").text = str(total_frames)

    media = ET.SubElement(seq, "media")
    video = ET.SubElement(media, "video")
    track = ET.SubElement(video, "track")

    # 檔案定義（所有片段共用）
    file_id = "file-1"
    # clipitem 需要嵌入 file 區塊（FCP7 XML 容忍重複 file, 為簡化在每個 clipitem 內嵌入同一 file）
    src_rate = ET.Element("rate")
    ET.SubElement(src_rate, "timebase").text = str(timebase)
    ET.SubElement(src_rate, "ntsc").text = "TRUE" if ntsc else "FALSE"

    src_pathurl = _file_url(Path(source_video))
    src_name = Path(source_video).name

    rec_cursor = 0
    for i, m in enumerate(matches, start=1):
        src_start = float(m.get("src_start", 0.0))
        ref_len = float(m.get("ref_end", 0.0)) - float(m.get("ref_start", 0.0))
        if ref_len <= 0:
            continue
        seg_len_f = _frames(ref_len, timeline_fps)
        if seg_len_f <= 0:
            continue
        src_in = _frames(src_start, timeline_fps)
        src_out = src_in + seg_len_f
        rec_in = rec_cursor
        rec_out = rec_in + seg_len_f

        clip = ET.SubElement(track, "clipitem", id=f"clipitem-{i}")
        ET.SubElement(clip, "name").text = f"seg_{i:04d}"
        cr = ET.SubElement(clip, "rate")
        ET.SubElement(cr, "timebase").text = str(timebase)
        ET.SubElement(cr, "ntsc").text = "TRUE" if ntsc else "FALSE"
        ET.SubElement(clip, "start").text = str(rec_in)
        ET.SubElement(clip, "end").text = str(rec_out)
        ET.SubElement(clip, "in").text = str(src_in)
        ET.SubElement(clip, "out").text = str(src_out)

        file_el = ET.SubElement(clip, "file", id=file_id)
        ET.SubElement(file_el, "name").text = src_name
        f_rate = ET.SubElement(file_el, "rate")
        ET.SubElement(f_rate, "timebase").text = str(timebase)
        ET.SubElement(f_rate, "ntsc").text = "TRUE" if ntsc else "FALSE"
        ET.SubElement(file_el, "pathurl").text = src_pathurl

        rec_cursor = rec_out

    # 輸出
    tree = ET.ElementTree(fcpx)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path

