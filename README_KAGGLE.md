# Hướng dẫn train trên Kaggle

Tài liệu này hướng dẫn tạo từng cell trong Kaggle Notebook để train
`YOWOv2_tiny_generalized` với các fold 10-14 không trùng video nguồn giữa
train và test.

## 1. Chuẩn bị trước khi mở Notebook

1. Push phiên bản mới nhất của repo lên GitHub.
2. Tạo một Kaggle Dataset và upload file
   `lifeguard-training-data-kaggle.zip`.
3. Tạo Kaggle Notebook và thêm Dataset vừa tạo bằng nút **Add Input**.
4. Trong **Notebook options**, chọn GPU và bật Internet để clone GitHub cũng
   như tải pretrained weights lần đầu.

Không upload riêng thư mục `frames/`; file ZIP đã chứa thư mục này.

## 2. Cell 1 - Kiểm tra GPU

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

assert torch.cuda.is_available(), "Hãy bật GPU trong Notebook options trước."
```

Có thể xem thêm thông tin GPU bằng một cell riêng:

```bash
!nvidia-smi
```

## 3. Cell 2 - Clone hoặc cập nhật repo

```python
from pathlib import Path
import subprocess

WORKING_DIR = Path("/kaggle/working")
REPO_DIR = WORKING_DIR / "Drowing-Dection-System"
REPO_URL = "https://github.com/VuongQuocHung/Drowing-Dection-System.git"

if REPO_DIR.exists():
    subprocess.run(
        ["git", "-C", str(REPO_DIR), "pull", "--ff-only"],
        check=True,
    )
else:
    subprocess.run(
        ["git", "clone", REPO_URL, str(REPO_DIR)],
        check=True,
    )

subprocess.run(
    ["git", "-C", str(REPO_DIR), "log", "-1", "--oneline"],
    check=True,
)
```

Chuyển thư mục làm việc của Notebook vào repo:

```python
%cd /kaggle/working/Drowing-Dection-System
```

Nếu cell `git pull` không thấy các file `config/YOWOv2_tiny_generalized.py`
và `dataset_utils/splitlists/10_train.csv`, kiểm tra lại xem commit mới nhất đã
được push lên GitHub chưa.

## 4. Cell 3 - Cài các thư viện còn thiếu

Kaggle đã cài sẵn PyTorch, torchvision, NumPy, pandas và OpenCV. Chỉ cài thêm
các gói cần thiết để tránh tải lại PyTorch:

```python
!python -m pip install -q thop scikit-learn yt-dlp
```

Kiểm tra import:

```python
import cv2
import sklearn
import thop
import yt_dlp

print("Dependencies: OK")
```

## 5. Cell 4 - Tìm và giải nén dataset

Cell này tự tìm `lifeguard-training-data-kaggle.zip` trong mọi Kaggle Input,
nên không cần biết chính xác slug của Kaggle Dataset.

```python
from pathlib import Path
import subprocess

REPO_DIR = Path("/kaggle/working/Drowing-Dection-System")
DATASET_DIR = REPO_DIR / "dataset_lifeguard"
FRAMES_DIR = DATASET_DIR / "frames"

zip_files = list(
    Path("/kaggle/input").rglob("lifeguard-training-data-kaggle.zip")
)
assert zip_files, (
    "Không tìm thấy lifeguard-training-data-kaggle.zip. "
    "Hãy Add Input chứa file ZIP vào Notebook."
)

data_zip = zip_files[0]
print("Dataset ZIP:", data_zip)

if not FRAMES_DIR.exists() or not any(FRAMES_DIR.iterdir()):
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["unzip", "-q", str(data_zip), "-d", str(DATASET_DIR)],
        check=True,
    )
else:
    print("Frames đã tồn tại, bỏ qua giải nén.")
```

## 6. Cell 5 - Kiểm tra dataset và fold 10

```python
from pathlib import Path

REPO_DIR = Path("/kaggle/working/Drowing-Dection-System")
FRAMES_DIR = REPO_DIR / "dataset_lifeguard" / "frames"
LABELS_DIR = REPO_DIR / "dataset_lifeguard" / "labels"
SPLITS_DIR = REPO_DIR / "dataset_utils" / "splitlists"

frame_dirs = [path for path in FRAMES_DIR.iterdir() if path.is_dir()]
xml_files = list(LABELS_DIR.glob("*.xml"))

train_file = SPLITS_DIR / "10_train.csv"
test_file = SPLITS_DIR / "10_test.csv"

assert train_file.is_file(), f"Thiếu {train_file}"
assert test_file.is_file(), f"Thiếu {test_file}"

train_snippets = [line.strip() for line in train_file.read_text().splitlines()]
test_snippets = [line.split(",")[0].strip() for line in test_file.read_text().splitlines()]
missing_frames = [
    snippet
    for snippet in train_snippets + test_snippets
    if not (FRAMES_DIR / snippet).is_dir()
]

print("Frame folders:", len(frame_dirs))
print("XML labels:", len(xml_files))
print("Fold 10 train snippets:", len(train_snippets))
print("Fold 10 test snippets:", len(test_snippets))
print("Missing frame folders:", len(missing_frames))

assert not missing_frames, missing_frames[:10]
```

Kết quả hiện tại dự kiến:

```text
Fold 10 train snippets: 72
Fold 10 test snippets: 19
Missing frame folders: 0
```

## 7. Cell 6 - Xem cấu hình YOWOv2 sẽ train

```python
from importlib import import_module

cfg = import_module("config.YOWOv2_tiny_generalized").config
keys = [
    "mode",
    "version",
    "fold",
    "batch_size",
    "num_workers",
    "T_len",
    "T_distance",
    "train_size",
    "max_epoch",
    "lr_epoch",
    "base_lr",
    "zoom_out_prob",
    "zoom_out_min_scale",
]

for key in keys:
    print(f"{key}: {cfg[key]}")
```

Với YOWOv2, `mode` phải là `reference`. Model sử dụng 32 frame cách nhau 6
frame, input 224x224, train 15 epoch và áp dụng zoom-out với xác suất 30%.

## 8. Cell 7 - Train YOWOv2-tiny trên fold 10

Đây là cell train chính:

```python
!python -u train.py --config YOWOv2_tiny_generalized --fold 10 --cuda
```

Trong lần chạy đầu, chương trình sẽ tự tải hai pretrained weights:

- `yolo_free_tiny_coco.pth` cho 2D backbone.
- `kinetics_shufflenetv2_2.0x_RGB_16_best.pth` cho 3D backbone.

Đây là pretrained weights, không phải dataset. Không dừng cell khi đang tải.

Checkpoint và kết quả fold 10 được lưu tại:

```text
/kaggle/working/Drowing-Dection-System/temp/YOWOv2_tiny_generalized/10/
```

## 9. Cell thay thế - Train 2Dseq-tiny

Không chạy cell này cùng lúc với cell YOWOv2. Chỉ dùng nếu muốn train
2Dseq-tiny để so sánh:

```python
!python -u train.py --config 2Dseq_tiny_generalized --fold 10 --cuda
```

Kết quả được lưu tại:

```text
/kaggle/working/Drowing-Dection-System/temp/2Dseq_tiny_generalized/10/
```

## 10. Cell 8 - Kiểm tra checkpoint sau khi train

Cho YOWOv2:

```python
from pathlib import Path

RESULT_DIR = Path(
    "/kaggle/working/Drowing-Dection-System/"
    "temp/YOWOv2_tiny_generalized/10"
)

checkpoints = sorted(RESULT_DIR.glob("*.pth"))
print("Số checkpoint:", len(checkpoints))
for checkpoint in checkpoints:
    print(checkpoint.name, f"{checkpoint.stat().st_size / 1024**2:.1f} MB")

assert checkpoints, "Không tìm thấy checkpoint; kiểm tra log train phía trên."
```

## 11. Cell 9 - Nén kết quả để tải xuống

```python
from pathlib import Path
import shutil

RESULT_DIR = Path(
    "/kaggle/working/Drowing-Dection-System/"
    "temp/YOWOv2_tiny_generalized/10"
)
ARCHIVE_BASE = Path("/kaggle/working/YOWOv2_tiny_generalized_fold10")

archive_path = shutil.make_archive(
    str(ARCHIVE_BASE),
    "zip",
    root_dir=RESULT_DIR,
)
print("Đã tạo:", archive_path)
```

Hiện link tải ngay trong Notebook:

```python
from IPython.display import FileLink

FileLink("/kaggle/working/YOWOv2_tiny_generalized_fold10.zip")
```

Ngoài tải trực tiếp, nên chọn **Save Version** sau khi train để Kaggle lưu file
trong phần Output. Nếu session hết thời gian trước khi Save Version, dữ liệu ở
`/kaggle/working` có thể bị mất.

## 12. Train các fold còn lại

Sau khi fold 10 chạy ổn định, thay `--fold 10` lần lượt bằng `11`, `12`, `13`
và `14`:

```python
!python -u train.py --config YOWOv2_tiny_generalized --fold 11 --cuda
```

Nên chạy từng fold trong một Kaggle session/version riêng để tránh vượt giới
hạn thời gian. Kết quả cuối cùng nên báo cáo trung bình và độ lệch chuẩn mAP
của cả năm fold 10-14, không chỉ chọn fold có mAP cao nhất.

## Lỗi thường gặp

### `ModuleNotFoundError: No module named 'thop'`

Chạy lại Cell 3:

```python
!python -m pip install -q thop
```

### Không tìm thấy dataset ZIP

Kiểm tra tab **Input** bên phải Notebook. Dataset chứa
`lifeguard-training-data-kaggle.zip` phải được thêm bằng **Add Input**.

### `CUDA was requested` hoặc `torch.cuda.is_available() == False`

Mở **Notebook options**, chọn GPU rồi restart session.

### Log vẫn ghi `YOWO_V2_TINY`

Đây là đúng khi train YOWOv2. Với 2Dseq, `version` vẫn là
`yowo_v2_tiny`, nhưng dòng `mode` trong cấu hình phải là `2Dseq`.

### Mất kết quả sau khi Kaggle dừng session

Nén checkpoint bằng Cell 9 và chọn **Save Version** trước khi session kết thúc.
