from __future__ import annotations

import argparse
from pathlib import Path
import platform

from .pipeline import PipelineConfig, align, render_from_alignment


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto recut HD source to match reference cuts")
    p.add_argument("--ref", type=Path, required=True, help="參照（低清解說）影片路徑")
    p.add_argument("--src", type=Path, required=True, help="高清母帶影片路徑")
    p.add_argument("--out", type=Path, default=Path("out"), help="輸出資料夾")
    p.add_argument("--step", type=float, default=0.2, help="取樣間隔（秒）")
    p.add_argument("--feature", choices=["clip"], default="clip", help="特徵方法（僅支援 clip）")
    p.add_argument("--search-margin", type=float, default=30.0, help="每段搜尋範圍（秒）")
    p.add_argument("--dtw-window", type=int, default=0, help="DTW 視窗（0 表示自動）")
    p.add_argument("--global-search", action="store_true", help="非順序對齊：在整個母帶全域搜尋")
    p.add_argument("--topk", type=int, default=12, help="全域搜尋每個錨點的候選數")
    p.add_argument("--anchors", type=int, default=5, help="每段錨點數（頭/中/尾）")
    p.add_argument("--cand-window", type=float, default=12.0, help="候選窗大小（秒）")
    # Scene detection & refinement
    p.add_argument("--sc-threshold", type=float, default=18.0, help="PySceneDetect ContentDetector threshold（越小越敏感）")
    p.add_argument("--min-shot", dest="min_shot", type=float, default=0.10, help="最短鏡頭長度（秒）")
    p.add_argument("--refine-boundaries", action="store_true", help="啟用邊界幀級精修")
    p.add_argument("--refine-window", type=float, default=0.5, help="精修視窗（秒）")
    p.add_argument("--refine-metric", choices=["clip"], default="clip", help="精修用特徵（僅支援 clip）")
    p.add_argument("--resume", action="store_true", help="中斷後續跑已完成的鏡頭")
    p.add_argument("--slow-processor", action="store_true", help="停用 CLIP fast 影像處理器（較慢）")
    p.add_argument("--no-cache", action="store_true", help="停用母帶特徵快取")
    p.add_argument("--render", action="store_true", help="完成後用 ffmpeg 合成輸出")
    p.add_argument("--fast-copy", action="store_true", help="以快速但不精準的 copy 模式裁切（可能累積段長誤差）")
    p.add_argument("--from-alignment", action="store_true", help="跳過對齊，直接讀取 out/alignment.json 進行合成")
    p.add_argument("--alignment", type=Path, default=None, help="指定 alignment.json 路徑（優先於 --from-alignment）")
    # Premiere 匯出
    p.add_argument("--export-xml", action="store_true", help="輸出 Premiere 可匯入的 FCP7 XML（recut_premiere.xml）")
    p.add_argument("--timeline-fps", type=float, default=30.0, help="XML 時間軸 FPS（預設 30）")
    p.add_argument("--ntsc", action="store_true", help="XML 使用 NTSC (drop-frame) 記號，用於 29.97/59.94")
    # 編碼相關
    sysname = platform.system().lower()
    default_vcodec = "h264_videotoolbox" if sysname == "darwin" else ("h264_nvenc" if sysname.startswith("win") else "h264_videotoolbox")
    p.add_argument(
        "--vcodec",
        choices=["h264_videotoolbox", "h264_nvenc"],
        default=default_vcodec,
        help="視訊編碼器（Windows: h264_nvenc, macOS: h264_videotoolbox）",
    )
    p.add_argument("--crf", type=int, default=18, help="CRF 品質（libx264 適用）")
    p.add_argument("--preset", type=str, default="veryfast", help="x264 編碼預設檔（ultrafast..veryslow，libx264 適用）")
    p.add_argument("--vbitrate", type=str, default=None, help="視訊位元率（h264_videotoolbox 建議設置，例如 5M）")
    p.add_argument("--abitrate", type=str, default="192k", help="音訊位元率")
    p.add_argument("--concat-copy", action="store_true", help="合併時完全 copy（不重編音訊）；可能出現 DTS 警告/長度漂移")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig(
        sample_step=args.step,
        feature_method=args.feature,
        search_margin=args.search_margin,
        dtw_window=None if args.dtw_window <= 0 else args.dtw_window,
        global_search=bool(args.global_search),
        topk_candidates=args.topk,
        anchor_count=args.anchors,
        candidate_window=args.cand_window,
        use_fast_processor=not args.slow_processor,
        cache_source_features=not args.no_cache,
        sc_threshold=args.sc_threshold,
        min_scene_len=args.min_shot,
        refine_boundaries=args.refine_boundaries,
        refine_window=args.refine_window,
        refine_metric=args.refine_metric,
    )
    # Render-only path: load existing alignment and export
    if args.from_alignment or args.alignment is not None:
        align_path = args.alignment if args.alignment is not None else (args.out / "alignment.json")
        if not align_path.exists():
            raise SystemExit(f"找不到 alignment.json：{align_path}")
        import json as _json
        alignment = _json.load(open(align_path, "r", encoding="utf-8"))
        print(f"[INFO] 讀取現有對齊：{align_path}")
        final_out = render_from_alignment(
            args.src,
            alignment,
            args.out,
            run=True,
            accurate_cut=not args.fast_copy,
            vcodec=args.vcodec,
            crf=args.crf,
            preset=args.preset,
            vbitrate=args.vbitrate,
            abitrate=args.abitrate,
            stabilize_audio=not args.concat_copy,
        )
        print(f"已輸出影片：{final_out.as_posix()}")
        if args.export_xml:
            from .export_premiere import export_fcp7_xml
            xml_path = export_fcp7_xml(
                alignment,
                args.src,
                args.out / "recut_premiere.xml",
                timeline_fps=float(args.timeline_fps),
                ntsc=bool(args.ntsc),
                sequence_name="Recut",
            )
            print(f"已輸出 Premiere XML：{xml_path.as_posix()}")
        return

    print("[INFO] 開始對齊流程…")
    print(
        f"[INFO] 參數 step={cfg.sample_step}, feature={cfg.feature_method}, margin={cfg.search_margin}, dtw_window={cfg.dtw_window}, fast_processor={cfg.use_fast_processor}, global_search={cfg.global_search}, topk={cfg.topk_candidates}, anchors={cfg.anchor_count}, cand_window={cfg.candidate_window}"
    )
    alignment = align(
        args.ref,
        args.src,
        args.out,
        cfg,
        log=lambda m: print(f"[LOG] {m}"),
        progress=lambda v: print(f"[PROGRESS] {int(v*100)}%"),
        resume=args.resume,
    )
    final_out = render_from_alignment(
        args.src,
        alignment,
        args.out,
        run=args.render,
        accurate_cut=not args.fast_copy,
        vcodec=args.vcodec,
        crf=args.crf,
        preset=args.preset,
        vbitrate=args.vbitrate,
        abitrate=args.abitrate,
        stabilize_audio=not args.concat_copy,
    )
    print(f"完成。對齊結果：{(args.out / 'alignment.json').as_posix()}")
    if args.render:
        print(f"已輸出影片：{final_out.as_posix()}")
    else:
        print("若要直接合成輸出，請加上 --render 參數。")
    if args.export_xml:
        from .export_premiere import export_fcp7_xml
        xml_path = export_fcp7_xml(
            alignment,
            args.src,
            args.out / "recut_premiere.xml",
            timeline_fps=float(args.timeline_fps),
            ntsc=bool(args.ntsc),
            sequence_name="Recut",
        )
        print(f"已輸出 Premiere XML：{xml_path.as_posix()}")


if __name__ == "__main__":
    main()
