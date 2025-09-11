from __future__ import annotations

import threading
import traceback
import json
from pathlib import Path
import platform
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk  # type: ignore
import cv2  # type: ignore

from .pipeline import PipelineConfig, align, render_from_alignment


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Recut 對齊重剪")
        self.geometry("720x520")

        pad = {"padx": 8, "pady": 6}

        # 建立可滾動容器（Canvas + Scrollbar）
        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(container, highlightthickness=0)
        vscroll = ttk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        frm = ttk.Frame(canvas)
        frm_id = canvas.create_window((0, 0), window=frm, anchor="nw")

        def _on_configure(event=None):
            # 更新滾動區域
            canvas.configure(scrollregion=canvas.bbox("all"))
            # 自動調整內框寬度以填滿
            canvas.itemconfigure(frm_id, width=canvas.winfo_width())

        frm.bind("<Configure>", _on_configure)
        canvas.bind("<Configure>", _on_configure)

        # 滑鼠滾輪支援
        def _on_mousewheel(event):
            delta = 0
            if hasattr(event, 'delta') and event.delta:
                delta = int(-event.delta / 120)
            elif hasattr(event, 'num') and event.num in (4, 5):
                delta = -1 if event.num == 4 else 1
            if delta:
                canvas.yview_scroll(delta, "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)      # Windows/macOS
        canvas.bind_all("<Button-4>", _on_mousewheel)        # Linux
        canvas.bind_all("<Button-5>", _on_mousewheel)

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
        # Premiere 匯出
        self.export_xml_var = tk.BooleanVar(value=False)
        self.timeline_fps_var = tk.DoubleVar(value=30.0)
        self.ntsc_var = tk.BooleanVar(value=False)
        # 編碼選項
        sysname = platform.system().lower()
        default_vcodec = "h264_videotoolbox" if sysname == "darwin" else ("h264_nvenc" if sysname.startswith("win") else "libx264")
        self.vcodec_var = tk.StringVar(value=default_vcodec)
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
        feat = ttk.Combobox(grid, textvariable=self.feature_var, values=["clip"], width=8, state="readonly")
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

        # Premiere XML 匯出
        px = ttk.LabelFrame(frm, text="Premiere 匯出（FCP7 XML）")
        px.pack(fill=tk.X, **pad)
        ttk.Checkbutton(px, text="輸出 Premiere XML（recut_premiere.xml）", variable=self.export_xml_var).pack(anchor=tk.W, padx=8)
        rowpx = ttk.Frame(px)
        rowpx.pack(fill=tk.X, **pad)
        ttk.Label(rowpx, text="時間軸 FPS").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(rowpx, textvariable=self.timeline_fps_var, width=8).grid(row=0, column=1, sticky=tk.W, padx=6)
        ttk.Checkbutton(rowpx, text="NTSC (drop-frame)", variable=self.ntsc_var).grid(row=0, column=2, sticky=tk.W, padx=6)

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
        self.refine_metric_var.set("clip")
        ttk.Combobox(rowrf, textvariable=self.refine_metric_var, values=["clip"], width=8, state="readonly").grid(row=0, column=3, sticky=tk.W, padx=6)

        # 編碼參數區
        enc = ttk.LabelFrame(frm, text="輸出編碼")
        enc.pack(fill=tk.X, **pad)
        rowe = ttk.Frame(enc)
        rowe.pack(fill=tk.X, **pad)
        ttk.Label(rowe, text="視訊編碼").grid(row=0, column=0, sticky=tk.W)
        vb = ttk.Combobox(rowe, textvariable=self.vcodec_var, values=["h264_videotoolbox", "h264_nvenc"], width=16, state="readonly")
        vb.grid(row=0, column=1, sticky=tk.W, padx=6)
        ttk.Label(rowe, text="V bitrate (HW)").grid(row=0, column=2, sticky=tk.W)
        ttk.Entry(rowe, textvariable=self.vbitrate_var, width=8).grid(row=0, column=3, sticky=tk.W, padx=6)
        ttk.Label(rowe, text="A bitrate").grid(row=0, column=4, sticky=tk.W)
        ttk.Entry(rowe, textvariable=self.abitrate_var, width=8).grid(row=0, column=5, sticky=tk.W, padx=6)
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
        self.prev_btn = ttk.Button(run_bar, text="預覽對齊 (alignment)", command=self._open_preview)
        self.prev_btn.pack(side=tk.LEFT)

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
                    if self.export_xml_var.get():
                        from .export_premiere import export_fcp7_xml
                        xml_path = export_fcp7_xml(
                            alignment,
                            src,
                            out / "recut_premiere.xml",
                            timeline_fps=float(self.timeline_fps_var.get()),
                            ntsc=bool(self.ntsc_var.get()),
                            sequence_name="Recut",
                        )
                        self._log(f"已輸出 Premiere XML：{xml_path}\n")
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
                    if self.export_xml_var.get():
                        from .export_premiere import export_fcp7_xml
                        xml_path = export_fcp7_xml(
                            alignment,
                            src,
                            out / "recut_premiere.xml",
                            timeline_fps=float(self.timeline_fps_var.get()),
                            ntsc=bool(self.ntsc_var.get()),
                            sequence_name="Recut",
                        )
                        self._log(f"已輸出 Premiere XML：{xml_path}\n")
            except Exception as e:
                tb = traceback.format_exc()
                self.after(0, self._log, f"發生錯誤：{e}\n{tb}\n")
                messagebox.showerror("發生錯誤", str(e))
            finally:
                self.after(0, self.run_btn.config, {"state": tk.NORMAL})

        threading.Thread(target=worker, daemon=True).start()

    def _open_preview(self) -> None:
        out = Path(self.out_var.get())
        align_path = out / "alignment.json"
        if not align_path.exists():
            messagebox.showerror("預覽不可用", f"找不到 {align_path}")
            return
        try:
            alignment = json.load(open(align_path, "r", encoding="utf-8"))
        except Exception as e:
            messagebox.showerror("讀取失敗", str(e))
            return
        PreviewWindow(self, Path(self.ref_var.get()), Path(self.src_var.get()), alignment)


class PreviewWindow(tk.Toplevel):
    def __init__(self, parent: tk.Tk, ref_path: Path, src_path: Path, alignment: dict) -> None:
        super().__init__(parent)
        self.title("對齊預覽（參照 vs 母帶）")
        self.geometry("980x640")
        self.ref_path = ref_path
        self.src_path = src_path
        self.alignment = alignment
        self.matches = list(alignment.get("matches", []))
        pad = {"padx": 6, "pady": 6}

        top = ttk.Frame(self)
        top.pack(fill=tk.X, **pad)
        ttk.Label(top, text=str(ref_path)).pack(side=tk.LEFT)
        ttk.Label(top, text=" ⟷ ").pack(side=tk.LEFT)
        ttk.Label(top, text=str(src_path)).pack(side=tk.LEFT)

        mid = ttk.Frame(self)
        mid.pack(fill=tk.BOTH, expand=True, **pad)

        # 左：清單
        left = ttk.Frame(mid)
        left.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(left, text="鏡頭 (idx / 長度 / cost)").pack(anchor=tk.W)
        self.listbox = tk.Listbox(left, height=24, width=28)
        self.listbox.pack(fill=tk.Y, expand=False)
        self.listbox.bind("<<ListboxSelect>>", lambda _: self._on_select())

        # 右：畫面 + 滑桿
        right = ttk.Frame(mid)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas_ref = tk.Canvas(right, width=480, height=270, bg="black")
        self.canvas_ref.pack(side=tk.TOP, padx=4, pady=4)
        self.canvas_src = tk.Canvas(right, width=480, height=270, bg="black")
        self.canvas_src.pack(side=tk.TOP, padx=4, pady=4)

        ctrl = ttk.Frame(right)
        ctrl.pack(fill=tk.X, pady=4)
        ttk.Label(ctrl, text="段內偏移 (秒)").pack(side=tk.LEFT)
        self.offset_var = tk.DoubleVar(value=0.0)
        self.slider = ttk.Scale(ctrl, from_=0.0, to=1.0, variable=self.offset_var, command=lambda _=None: self._update_frames())
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self.pos_label = ttk.Label(ctrl, text="0.00s / 0.00s")
        self.pos_label.pack(side=tk.LEFT)

        # 準備清單
        for i, m in enumerate(self.matches, start=1):
            rs, re = float(m.get("ref_start", 0.0)), float(m.get("ref_end", 0.0))
            cost = float(m.get("cost", 0.0))
            dur = max(0.0, re - rs)
            self.listbox.insert(tk.END, f"{i:04d}  len={dur:.2f}s  cost={cost:.3f}")

        if self.matches:
            self.listbox.selection_set(0)
            self._on_select()

    def _on_select(self) -> None:
        sel = self.listbox.curselection()
        if not sel:
            return
        self.idx = sel[0]
        m = self.matches[self.idx]
        rs, re = float(m.get("ref_start", 0.0)), float(m.get("ref_end", 0.0))
        self.ref_len = max(0.0, re - rs)
        # 調整滑桿範圍 0..ref_len
        self.slider.configure(from_=0.0, to=max(0.01, self.ref_len))
        self.offset_var.set(min(self.offset_var.get(), self.ref_len))
        self._update_frames()

    def _update_frames(self) -> None:
        if not hasattr(self, "idx"):
            return
        m = self.matches[self.idx]
        rs = float(m.get("ref_start", 0.0))
        ss = float(m.get("src_start", 0.0))
        off = float(self.offset_var.get())
        # 載入兩端影格
        ref_t = max(0.0, rs + off)
        src_t = max(0.0, ss + off)
        ref_im = self._load_frame(self.ref_path, ref_t)
        src_im = self._load_frame(self.src_path, src_t)
        if ref_im is not None:
            self._draw_image(self.canvas_ref, ref_im)
        if src_im is not None:
            self._draw_image(self.canvas_src, src_im)
        self.pos_label.configure(text=f"{off:.2f}s / {self.ref_len:.2f}s")

    def _load_frame(self, path: Path, t: float):
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None
        # 等比縮放到寬 480
        h, w = frame.shape[:2]
        if w > 0 and h > 0:
            scale = 480.0 / float(w)
            nh = int(h * scale)
            frame = cv2.resize(frame, (480, nh))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def _draw_image(self, canvas: tk.Canvas, img: Image.Image) -> None:
        tkimg = ImageTk.PhotoImage(img)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=tkimg)
        # 防止被 GC 回收
        canvas.image = tkimg


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
