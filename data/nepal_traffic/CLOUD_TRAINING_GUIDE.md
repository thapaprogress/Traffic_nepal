# Cloud Training Guide — Google Colab / Kaggle

Use this guide to run the YOLO-World fine-tuning process on a free or paid cloud GPU instance (Google Colab, Kaggle, RunPod, AWS, etc.).

---

## Google Colab Notebook Setup

Create a new Google Colab notebook, change the runtime type to **T4 GPU** (under *Runtime* -> *Change runtime type* -> *T4 GPU*), and run the following cells.

### Cell 1: Install Dependencies
```bash
!pip install ultralytics
!pip install -r https://raw.githubusercontent.com/thapaprogress/Traffic_nepal/main/requirements.txt
```

### Cell 2: Clone the Project Repository
```bash
!git clone https://github.com/thapaprogress/Traffic_nepal.git
%cd Traffic_nepal
```

### Cell 3: Mount Google Drive & Extract Dataset (If stored on Drive)
*If you uploaded your annotated dataset (`nepal_traffic.zip`) to Google Drive, mount Drive and extract it into the project folder:*
```python
from google.colab import drive
drive.mount('/content/drive')

# Extract dataset (adjust path to where you saved the zip)
!unzip /content/drive/MyDrive/nepal_traffic.zip -d data/
```
Ensure your directory structure looks like this:
```
Traffic_nepal/
└── data/
    └── nepal_traffic/
        ├── images/
        │   ├── train/
        │   └── val/
        └── labels/
            ├── train/
            └── val/
```

### Cell 4: Run Training Script
Run the custom training script we provided:
```bash
!python utils/train_yolo.py --epochs 80 --batch 16 --imgsz 640
```
*Note: If you run out of GPU VRAM (CUDA Out of Memory), reduce the batch size using `--batch 8`.*

### Cell 5: Download Training Outputs
Zip and download the trained weights (`best.pt`) and visual charts:
```python
import shutil
from google.colab import files

# Zip runs output folder containing results
shutil.make_archive('nepal_traffic_results', 'zip', 'runs/nepal_traffic/finetune_v1')

# Download the zip file
files.download('nepal_traffic_results.zip')
```

---

## Local Setup & Deployment of Trained Weights

Once training completes and you download `nepal_traffic_results.zip`:
1. Extract `best.pt` from the `weights/` folder inside the zip.
2. Place `best.pt` into your local `d:\traffic_nepal\Traffic_nepal\weights\` directory.
3. Update [settings.py](file:///d:/traffic_nepal/Traffic_nepal/config/settings.py):
   ```python
   YOLO_WEIGHTS = os.path.join(WEIGHTS_DIR, "best.pt")
   ```
4. Restart your dashboard.
