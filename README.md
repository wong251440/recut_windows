 背景：
有人拿了我的電影（90分鐘）去剪輯了一套解說（15分鐘，很多跳剪，加了一些水印，字幕，音訊沒有保留已經全部變成了他的解說），解說影片全部是從我的電影碎剪出來。
但問題是很低清，我有全高清的電影素材，打算用此來重剪一個一模一樣的，但手動再剪實在浪費太多時間。請幫我搭建一個專案能自動執行這個任務（我是用macbook air）

以下是我詢問了專業人士的建議流程，請根據此作建構：

下面給你一套「參考影片自動對齊重剪」的實作方案，重點是：用低清解說片當「參照」、自動在你的全高清母帶裡找到對應片段，產出一模一樣的剪接（跳剪節點一致），再用 ffmpeg 合成。整體不依賴音訊（因為已被替換），以視覺特徵為主。

# 建議技術堆疊（穩定、易上手）

* **鏡頭邊界偵測（找跳剪點）**：PySceneDetect。先在「解說片」上偵測出每個跳剪的時間戳，得到一份 EDL（Edit Decision List）。

* **視覺相似度與指紋（不靠聲音）**：
用 **CLIP** 對關鍵幀取向量，做語意層級的匹配，抗局部遮擋與輕度裁切。

* **時間序列對齊（處理掉幀/輕微速度差）**：對「解說片」每個片段與母帶的候選窗做序列比對，用 **DTW（Dynamic Time Warping）** 或其變體做單調遞增的最佳路徑搜尋，得到片段在母帶中的起訖時間。近年的影片對齊研究普遍採用「特徵 + DTW」。

* **疊加元素的魯棒性（字幕、水印、Logo）**：先做文字區域偵測（OpenCV 的 EAST / DB 系列），把偵測到的框當作遮罩，不讓這些區域影響特徵或雜湊比對。必要時可先做 inpainting 移除再取特徵。

* **重建與輸出**：用 ffmpeg 依照對齊出的時間碼批次裁切、再用 concat demuxer 合成，完全複製解說片的剪接節奏。

---

## 端到端流程（可直接照這個做）

1. **預處理**
   * 兩支影片統一幀率（例如 25 或 30fps）、分辨率與長寬比，避免特徵/雜湊的系統性偏差。

2. **在解說片上做鏡頭偵測**
   * `PySceneDetect detect-content` ，輸出每個跳剪的時間戳（T0、T1、…）。

3. **抽樣取特徵 / 雜湊**
   * 對「解說片」每個鏡頭，以固定間隔（例如每 0.5–1 秒）抽幀；同樣地，對「母帶」整段或滑動窗抽幀。
   * 對抽幀跑 **CLIP** 產向量。
   * 先跑「文字框偵測」把字幕/水印區域遮掉，再取特徵（顯著提升對疊加元素的魯棒性）。

4. **時間對齊與定位**
   * 對每個解說片鏡頭，在母帶上套一個**候選時間窗**（可用粗匹配先縮小範圍），然後以**序列相似度矩陣**（特徵/雜湊距離）跑 **DTW**，求得最優單調對齊路徑，得到（start, end）。

   * 若整部解說片維持原片順序（常見的碎剪），可在鏡頭間加入**全域單調約束**（Viterbi/DP），避免錯位跳回前面場景。文獻上也常見把 DTW 作為對齊核心。

5. **批次裁切與合成**
   * 生成一份帶時間碼的清單（或 EDL）。
   * 用 ffmpeg 批量 `-ss`/`-to` 裁切出所有段落，再用 concat demuxer 按順序合併，得到與解說片**一模一樣的剪點**、但畫質來自你的全高清母帶。

---

## 關鍵實作細節與小撇步（這段只作參考，請用你的專業判斷是否實行，或更好的方案）

* **鏡頭偵測參數**：PySceneDetect 的 `detect-content`（內容變化）對跳剪很靈

* **遮字幕/水印**：先用 EAST/DB 偵測文字框，建立 mask 後再算特徵

* **順序是否一致？** 若解說片偶爾穿插「倒序/交叉剪」，對那幾段放寬單調約束，改用全庫檢索取**最佳片段**再回到主線。

* **品質控管**：對齊完成後，對每段輸出**匹配分數**與**邊界重投票**（boundary refinement），避免誤切掉關鍵幀。

* **風險與對策**：TMK 在某些內容上可能有誤報，務必加上多種訊號交叉驗證（如 ORB/CLIP 與遮罩後的 SSIM）。


---

## 使用指南（CLI 與 GUI）

本 repo 已提供：

- CLI：依據流程以「參照影片 → 自動對齊 → 批次裁切 → 合成」完成重剪。
- 桌面 GUI（Tkinter）：可點選檔案與設定參數、觀察進度。

預設使用視覺特徵（優先 CLIP，缺少模型時退回 HSV 直方圖），並以 DTW 做時間序列對齊。場景偵測優先採用 PySceneDetect（若未安裝則退回內建 HSV 差異法）。

- 主要指令：`python -m recut.cli --ref 參照.mp4 --src 母帶.mp4 --out out --render`
 - 匯出 Premiere XML：在指令加上 `--export-xml --timeline-fps 30`，將生成 `out/recut_premiere.xml` 可直接在 Premiere 匯入

### 安裝依賴

1. 建議使用 Python 3.10+ 與虛擬環境。
2. 安裝必要依賴（OpenCV、NumPy、tqdm）：
   - `pip install -r requirements.txt`
3. 本專案僅支援 CLIP 特徵（必須）：
   - CPU：`pip install torch torchvision transformers`（僅做驗證，不建議大量運算）
   - Windows + NVIDIA GPU（建議）：安裝對應 CUDA 的 PyTorch（例 CUDA 12.x：`pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision`），再安裝 `transformers`。
   - macOS Apple Silicon：已支援 PyTorch MPS（Apple GPU）。請安裝官方 PyTorch 與 transformers；程式會自動偵測並使用 `mps` 裝置。
   首次執行會下載模型（需網路），缺失將直接報錯不退回。
4. 建議安裝 PySceneDetect（更準確場景偵測）：
   - `pip install scenedetect`

### 參數說明

- `--ref`: 參照（低清解說）影片路徑。
- `--src`: 高清母帶影片路徑。
- `--out`: 輸出資料夾（預設 `out/`）。
- `--step`: 取樣間隔秒數，越小越精準但越慢（預設 0.8）。
- `--feature`: 特徵方法，`auto` | `clip` | `hsv`。
- `--search-margin`: 每段搜尋範圍（秒）（預設 30）。
- `--dtw-window`: DTW 視窗大小（0 表示自動）。
- `--render`: 完成對齊後，直接用 ffmpeg 合成輸出。

執行後會輸出：

- `out/alignment.json`: 每段對齊的起迄時間與匹配分數。
- `out/segments/*.mp4`: 按對齊結果裁切出的片段。
- `out/concat.txt`: ffmpeg concat 列表。
- `out/recut_output.mp4`: 最終合成（若使用 `--render`）。

### 圖形介面（Tkinter）

- 啟動：`python -m recut.ui_tk`
- 視窗中選擇參照與母帶影片、輸出資料夾與參數（步長、特徵、搜尋範圍、DTW 視窗），按下「開始對齊與重剪」。
- 勾選「完成後直接合成輸出」會自動產生 `recut_output.mp4`。
 - 若勾選「輸出 Premiere XML（FCP7 XML）」會同時在 `out/` 產生 `recut_premiere.xml`，直接匯入 Premiere 使用。

### 實作備註

- 場景偵測僅支援 `PySceneDetect` 的 `ContentDetector`；未安裝或偵測不到分鏡將報錯，不退回 HSV。
- 文字/水印遮罩（EAST）需要額外的模型檔，程式已預留接點，未提供權重檔；一般情況下也可直接略過遮罩（用 CLIP 或 HSV 直方圖仍有一定魯棒性）。
- DTW 目前以單調遞增方式逐段搜尋，對一般碎剪（順序未被打亂）有良好效果；若遇到倒序/交叉剪，可手動放寬搜尋範圍或降低 `--step` 增加取樣密度。

---

## Windows 加速與安裝指引（RTX 50xx 支援）

本專案僅支援硬體加速編解碼：

- Windows：使用 `h264_nvenc`（NVENC + CUDA）
- macOS：使用 `h264_videotoolbox`（VideoToolbox）
- 其他平台目前不支援（將報錯）
同時，CLIP 特徵在 GPU 上自動啟用半精度（AMP）與批次推理。

### 必要條件（只需設定一次）

1. 安裝 NVIDIA 顯示卡驅動（支援 RTX 50xx）。
2. 安裝 CUDA 對應版 PyTorch（建議 CUDA 12.x）：
   - 於 PowerShell（或 CMD）執行 PyTorch 官網提供的 pip 指令（例如 `cu121`）。
   - 之後 `pip install transformers`。
3. 安裝 ffmpeg 並加入 PATH（必須，且需包含對應硬體編碼器）：
   - 方式 A（建議）：使用 winget：`winget install Gyan.FFmpeg` 或從 gyan.dev 下載 release zip，解壓後將 `bin` 資料夾加入 PATH。
   - 驗證：在新開的 PowerShell 執行 `ffmpeg -hide_banner`，能顯示版本資訊即可。

### 建議檢查

- 檢查 PyTorch 是否啟用 CUDA：在 Python 互動殼執行 `import torch; print(torch.cuda.is_available())` 應為 `True`。
- 檢查 ffmpeg NVENC/VTB 支援：
  - Windows：`ffmpeg -encoders | findstr nvenc` 應列出 `h264_nvenc`。
  - macOS：`ffmpeg -encoders | grep videotoolbox` 應列出 `h264_videotoolbox`。

### Windows 使用小叮嚀

- 本專案僅允許硬體編碼：Windows=`h264_nvenc`，macOS=`h264_videotoolbox`。若選其他將報錯。
- 若遇到「裁切速度慢」且 GPU 空轉，可確認：
  - `ffmpeg` 版本是否含 NVENC；
  - 顯卡驅動是否最新；
  - 源檔是否使用了少見編碼導致回退軟解（可嘗試加 `--fast-copy` 快速裁切測試）。
- 首次跑 CLIP 模型會下載權重，若在離線環境請事先在可連網機器下載後複製到 `~/.cache/huggingface/`（或設定 `HF_HOME`）。

### 你需要手動做的一次性步驟（Windows）

- 安裝 GPU 版 PyTorch（對應 CUDA 版本）與 transformers。
- 安裝 ffmpeg 並確保 PATH 正確。
- 更新 NVIDIA 驅動至支援你的 RTX 50xx 的最新版本。

完成以上後，直接使用：

`python -m recut.cli --ref ref.mp4 --src src.mp4 --out out --render`

或 GUI：`python -m recut.ui_tk`
