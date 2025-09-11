from __future__ import annotations

import threading
import traceback
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from .pipeline import PipelineConfig, align, render_from_alignment


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Recut 對齊重剪")
        self.geometry("720x520")

        pad = {"padx": 8, "pady": 6}
        frm = ttk.Frame(self)
        frm.pack(fill=tk.BOTH, expand=True)

        # Paths
        self.ref_var = tk.StringVar()
        self.src_var = tk.StringVar()
        self.out_var = tk.StringVar(value=str(Path("out").absolute()))

        self._row_path(frm, "參照影片（低清）", self.ref_var, self._browse_ref)
        self._row_path(frm, "母帶影片（高清）", self.src_var, self._browse_src)
        self._row_path(frm, "輸出資料夾", self.out_var, self._browse_out)

        # Params
        self.step_var = tk.DoubleVar(value=0.2)
        self.feature_var = tk.StringVar(value="clip")
        self.margin_var = tk.DoubleVar(value=30.0)
        self.dtw_var = tk.IntVar(value=0)
        self.render_var = tk.BooleanVar(value=True)
        self.resume_var = tk.BooleanVar(value=True)
        self.global_var = tk.BooleanVar(value=True)
        self.topk_var = tk.IntVar(value=12)
        self.anchors_var = tk.IntVar(value=5)
        self.candwin_var = tk.DoubleVar(value=12.0)
        self.cache_var = tk.BooleanVar(value=True)
        self.fastcopy_var = tk.BooleanVar(value=False)
        self.from_align_var = tk.BooleanVar(value=False)
        # 編碼選項
        self.vcodec_var = tk.StringVar(value="h264_videotoolbox")
        self.crf_var = tk.IntVar(value=18)
        self.preset_var = tk.StringVar(value="veryfast")
        self.vbitrate_var = tk.StringVar(value="5M")
        self.abitrate_var = tk.StringVar(value="192k")
        self.concat_copy_var = tk.BooleanVar(value=False)
        # Scene detect & refine
        self.sc_threshold_var = tk.DoubleVar(value=18.0)
        self.min_shot_var = tk.DoubleVar(value=0.10)
        self.refine_var = tk.BooleanVar(value=True)
        self.refine_window_var = tk.DoubleVar(value=0.5)
        self.refine_metric_var = tk.StringVar(value="auto")

        grid = ttk.Frame(frm)
        grid.pack(fill=tk.X, **pad)

        def add_field(col: int, label: str, widget: tk.Widget):
            ttk.Label(grid, text=label).grid(row=0, column=col * 2, sticky=tk.W)
            widget.grid(row=0, column=col * 2 + 1, sticky=tk.EW, padx=6)
            grid.columnconfigure(col * 2 + 1, weight=1)

        add_field(0, "取樣步長(s)", ttk.Entry(grid, textvariable=self.step_var, width=8))
        feat = ttk.Combobox(grid, textvariable=self.feature_var, values=["auto", "clip", "hsv"], width=8, state="readonly")
        add_field(1, "特徵", feat)
        add_field(2, "搜尋範圍(s)", ttk.Entry(grid, textvariable=self.margin_var, width=8))
        add_field(3, "DTW 視窗", ttk.Entry(grid, textvariable=self.dtw_var, width=8))

        chk = ttk.Checkbutton(frm, text="完成後直接合成輸出 (--render)", variable=self.render_var)
        chk.pack(anchor=tk.W, **pad)
        chk2 = ttk.Checkbutton(frm, text="允許中斷續跑 (--resume)", variable=self.resume_var)
        chk2.pack(anchor=tk.W, **pad)
        chk3 = ttk.Checkbutton(frm, text="非順序對齊（全域搜尋）", variable=self.global_var)
        chk3.pack(anchor=tk.W, **pad)
        chk4 = ttk.Checkbutton(frm, text="啟用母帶特徵快取", variable=self.cache_var)
        chk4.pack(anchor=tk.W, **pad)
        chk5 = ttk.Checkbutton(frm, text="快速裁切（可能不精準）", variable=self.fastcopy_var)
        chk5.pack(anchor=tk.W, **pad)
        chk6 = ttk.Checkbutton(frm, text="直接從現有 alignment.json 合成", variable=self.from_align_var)
        chk6.pack(anchor=tk.W, **pad)

        # Scene detect & refine controls
        scf = ttk.LabelFrame(frm, text="鏡頭偵測/邊界精修")
        scf.pack(fill=tk.X, **pad)
        rowsc = ttk.Frame(scf)
        rowsc.pack(fill=tk.X, **pad)
        ttk.Label(rowsc, text="Content threshold").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(rowsc, textvariable=self.sc_threshold_var, width=8).grid(row=0, column=1, sticky=tk.W, padx=6)
        ttk.Label(rowsc, text="最短鏡頭(秒)").grid(row=0, column=2, sticky=tk.W)
        ttk.Entry(rowsc, textvariable=self.min_shot_var, width=8).grid(row=0, column=3, sticky=tk.W, padx=6)
        ttk.Checkbutton(scf, text="啟用邊界精修", variable=self.refine_var).pack(anchor=tk.W, padx=8)
        rowrf = ttk.Frame(scf)
        rowrf.pack(fill=tk.X, **pad)
        ttk.Label(rowrf, text="精修視窗(秒)").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(rowrf, textvariable=self.refine_window_var, width=8).grid(row=0, column=1, sticky=tk.W, padx=6)
        ttk.Label(rowrf, text="精修特徵").grid(row=0, column=2, sticky=tk.W)
        ttk.Combobox(rowrf, textvariable=self.refine_metric_var, values=["auto","clip","hsv"], width=8, state="readonly").grid(row=0, column=3, sticky=tk.W, padx=6)

        # 編碼參數區
        enc = ttk.LabelFrame(frm, text="輸出編碼")
        enc.pack(fill=tk.X, **pad)
        rowe = ttk.Frame(enc)
        rowe.pack(fill=tk.X, **pad)
        ttk.Label(rowe, text="視訊編碼").grid(row=0, column=0, sticky=tk.W)
        vb = ttk.Combobox(rowe, textvariable=self.vcodec_var, values=["libx264", "h264_videotoolbox"], width=16, state="readonly")
        vb.grid(row=0, column=1, sticky=tk.W, padx=6)
        ttk.Label(rowe, text="CRF (x264)").grid(row=0, column=2, sticky=tk.W)
        ttk.Entry(rowe, textvariable=self.crf_var, width=6).grid(row=0, column=3, sticky=tk.W, padx=6)
        ttk.Label(rowe, text="preset (x264)").grid(row=0, column=4, sticky=tk.W)
        ttk.Entry(rowe, textvariable=self.preset_var, width=10).grid(row=0, column=5, sticky=tk.W, padx=6)
        ttk.Label(rowe, text="V bitrate (VT)").grid(row=0, column=6, sticky=tk.W)
        ttk.Entry(rowe, textvariable=self.vbitrate_var, width=8).grid(row=0, column=7, sticky=tk.W, padx=6)
        ttk.Label(rowe, text="A bitrate").grid(row=0, column=8, sticky=tk.W)
        ttk.Entry(rowe, textvariable=self.abitrate_var, width=8).grid(row=0, column=9, sticky=tk.W, padx=6)
        ttk.Checkbutton(enc, text="合併時完全 copy（可能 DTS 警告/長度漂移）", variable=self.concat_copy_var).pack(anchor=tk.W, padx=8)

        adv = ttk.LabelFrame(frm, text="全域搜尋參數")
        adv.pack(fill=tk.X, **pad)
        row2 = ttk.Frame(adv)
        row2.pack(fill=tk.X, **pad)
        ttk.Label(row2, text="TopK 候選").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(row2, textvariable=self.topk_var, width=8).grid(row=0, column=1, sticky=tk.W, padx=6)
        ttk.Label(row2, text="錨點數").grid(row=0, column=2, sticky=tk.W)
        ttk.Entry(row2, textvariable=self.anchors_var, width=8).grid(row=0, column=3, sticky=tk.W, padx=6)
        ttk.Label(row2, text="候選窗(秒)").grid(row=0, column=4, sticky=tk.W)
        ttk.Entry(row2, textvariable=self.candwin_var, width=8).grid(row=0, column=5, sticky=tk.W, padx=6)

        # Run controls
        run_bar = ttk.Frame(frm)
        run_bar.pack(fill=tk.X, **pad)
        self.run_btn = ttk.Button(run_bar, text="開始對齊與重剪", command=self._on_run)
        self.run_btn.pack(side=tk.LEFT)
        self.prog = ttk.Progressbar(run_bar, mode="determinate")
        self.prog.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        # Log
        self.log = tk.Text(frm, height=16)
        self.log.pack(fill=tk.BOTH, expand=True, **pad)
        self._log("就緒：請選擇參照與母帶影片，調整參數後開始。\n")

    def _row_path(self, parent: tk.Widget, label: str, var: tk.StringVar, cb):
        pad = {"padx": 8, "pady": 6}
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, **pad)
        ttk.Label(row, text=label, width=18).pack(side=tk.LEFT)
        ent = ttk.Entry(row, textvariable=var)
        ent.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="瀏覽…", command=cb).pack(side=tk.LEFT, padx=6)

    def _browse_ref(self):
        p = filedialog.askopenfilename(title="選擇參照影片")
        if p:
            self.ref_var.set(p)

    def _browse_src(self):
        p = filedialog.askopenfilename(title="選擇母帶影片")
        if p:
            self.src_var.set(p)

    def _browse_out(self):
        p = filedialog.askdirectory(title="選擇輸出資料夾")
        if p:
            self.out_var.set(p)

    def _log(self, msg: str):
        self.log.insert(tk.END, msg)
        self.log.see(tk.END)

    def _on_run(self):
        ref = Path(self.ref_var.get())
        src = Path(self.src_var.get())
        out = Path(self.out_var.get())
        if not ref.exists() or not src.exists():
            messagebox.showerror("路徑錯誤", "請確認參照與母帶影片路徑皆存在。")
            return

        self.run_btn.config(state=tk.DISABLED)
        self.prog.configure(value=0.0, maximum=1.0)
        if self.from_align_var.get():
            self._log("從現有 alignment.json 合成…\n")
        else:
            self._log("開始對齊…\n")

        def worker():
            try:
                if self.from_align_var.get():
                    align_path = out / "alignment.json"
                    if not align_path.exists():
                        raise FileNotFoundError(f"找不到 {align_path}")
                    alignment = json.load(open(align_path, "r", encoding="utf-8"))
                    out_path = render_from_alignment(
                        src,
                        alignment,
                        out,
                        run=True,
                        accurate_cut=not bool(self.fastcopy_var.get()),
                        vcodec=self.vcodec_var.get(),
                        crf=int(self.crf_var.get()),
                        preset=self.preset_var.get(),
                        vbitrate=self.vbitrate_var.get() if self.vbitrate_var.get() else None,
                        abitrate=self.abitrate_var.get(),
                        stabilize_audio=not bool(self.concat_copy_var.get()),
                    )
                    self._log(f"完成輸出：{out_path}\n")
                else:
                    cfg = PipelineConfig(
                        sample_step=float(self.step_var.get()),
                        feature_method=self.feature_var.get(),
                        search_margin=float(self.margin_var.get()),
                        dtw_window=int(self.dtw_var.get()) if int(self.dtw_var.get()) > 0 else None,
                        global_search=bool(self.global_var.get()),
                        topk_candidates=int(self.topk_var.get()),
                        anchor_count=int(self.anchors_var.get()),
                        candidate_window=float(self.candwin_var.get()),
                        cache_source_features=bool(self.cache_var.get()),
                        sc_threshold=float(self.sc_threshold_var.get()),
                        min_scene_len=float(self.min_shot_var.get()),
                        refine_boundaries=bool(self.refine_var.get()),
                        refine_window=float(self.refine_window_var.get()),
                        refine_metric=self.refine_metric_var.get(),
                    )

                    alignment = align(
                        ref,
                        src,
                        out,
                        cfg,
                        log=lambda m: self.after(0, self._log, m + "\n"),
                        progress=lambda v: self.after(0, self.prog.configure, {"value": v}),
                        resume=bool(self.resume_var.get()),
                    )
                    if self.render_var.get():
                        self._log("對齊完成，開始合成…\n")
                        out_path = render_from_alignment(
                            src,
                            alignment,
                            out,
                            run=True,
                            accurate_cut=not bool(self.fastcopy_var.get()),
                            vcodec=self.vcodec_var.get(),
                            crf=int(self.crf_var.get()),
                            preset=self.preset_var.get(),
                            vbitrate=self.vbitrate_var.get() if self.vbitrate_var.get() else None,
                            abitrate=self.abitrate_var.get(),
                            stabilize_audio=not bool(self.concat_copy_var.get()),
                        )
                        self._log(f"完成輸出：{out_path}\n")
                    else:
                        self._log(f"對齊完成。結果：{(out / 'alignment.json')}\n")
            except Exception as e:
                tb = traceback.format_exc()
                self.after(0, self._log, f"發生錯誤：{e}\n{tb}\n")
                messagebox.showerror("發生錯誤", str(e))
            finally:
                self.after(0, self.run_btn.config, {"state": tk.NORMAL})

        threading.Thread(target=worker, daemon=True).start()


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
