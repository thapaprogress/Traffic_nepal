# Traffic Eye Nepal — Remaining Tasks (GPU Required)
## Complete these after you get GPU access (RTX 4080)

**Last Updated:** June 24, 2026  
**Status:** 36 of 36 Python files built | Phase 1 ✅ | Phase 2 ✅ | Phase 3 ~70% (GPU tasks pending)

---

## Task 3.3 — YOLO-World Fine-Tuning on Nepal Traffic

### What
Fine-tune the YOLO-World model on Kathmandu-specific traffic data for higher accuracy on Nepal vehicles.

### Prerequisites
- [ ] RTX 4080 or equivalent GPU with CUDA
- [ ] Collect 5,000+ annotated images (see `data/nepal_traffic_classes.json`)
- [ ] Install: `pip install ultralytics`

### Steps

```bash
# Step 1: Organize dataset in YOLO format
# Place in: data/nepal_traffic/
#   data/nepal_traffic/images/train/
#   data/nepal_traffic/images/val/
#   data/nepal_traffic/labels/train/
#   data/nepal_traffic/labels/val/

# Step 2: Create dataset YAML
```

Create file `data/nepal_traffic.yaml`:
```yaml
path: ./data/nepal_traffic
train: images/train
val: images/val

names:
  0: motorcycle
  1: car
  2: bus
  3: truck
  4: microbus
  5: tempo
  6: electric rickshaw
  7: bicycle
  8: person
  9: helmet
  10: no helmet
  11: license plate
  12: traffic police
  13: overloaded vehicle
  14: school bus
```

```bash
# Step 3: Fine-tune
yolo detect train \
    model=yolov8s-worldv2.pt \
    data=data/nepal_traffic.yaml \
    epochs=80 \
    imgsz=640 \
    batch=16 \
    device=0 \
    project=runs/nepal_traffic \
    name=finetune_v1 \
    amp=True

# Step 4: Export the fine-tuned model
python deploy/export_engine.py --format all --half

# Step 5: Update config/settings.py
# Change YOLO_WEIGHTS to point to the new best.pt
```

### Validation
```bash
yolo detect val model=runs/nepal_traffic/finetune_v1/weights/best.pt data=data/nepal_traffic.yaml
```

### Expected Outcome
- mAP50 improvement from ~40% (zero-shot) to ~75%+ on Nepal classes
- Much better helmet/no-helmet discrimination
- Reliable microbus/tempo detection

---

## Task 3.4 — Real ByteTrack Integration

### What
Replace the custom IoU tracker with the official ByteTrack (Kalman filter + motion prediction).

### Prerequisites
- [ ] GPU (recommended for speed, works on CPU too)
- [ ] Install dependencies

### Steps

```bash
# Step 1: Install
pip install lapx cython_bbox

# Step 2: Clone ByteTrack
git clone https://github.com/ifzhang/ByteTrack.git
cd ByteTrack
pip install -r requirements.txt
python setup.py develop

# Step 3: Update tracking/bytetrack_wrapper.py
```

Replace the `_assign()` method in `tracking/bytetrack_wrapper.py` with:

```python
# At the top of the file, add:
from byte_tracker import BYTETracker

# Replace the ByteTracker class with:
class ByteTracker:
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8):
        self._tracker = BYTETracker(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
        )
        self.tracks = {}

    def update(self, detections, frame_shape):
        # Convert detections to numpy array [x1,y1,x2,y2,score]
        if not detections:
            return detections
        dets = np.array([[d.x1, d.y1, d.x2, d.y2, d.confidence] for d in detections])
        online_targets = self._tracker.update(dets, frame_shape, frame_shape)
        # Map back to Detection objects
        for t in online_targets:
            tlbr = t.tlbr
            idx = find_closest_detection(detections, tlbr)
            if idx >= 0:
                detections[idx].track_id = t.track_id
        return detections
```

### Expected Outcome
- 50% fewer ID switches at busy intersections
- Better tracking through occlusions (vehicles behind buses)
- Smoother speed estimation

---

## Task 3.8 — S3/MinIO Snapshot Storage

### What
Replace local disk snapshot storage with object storage for scalability.

### Prerequisites
- [ ] Docker (for MinIO) OR AWS S3 credentials
- [ ] `pip install boto3` or `pip install minio`

### Steps

```bash
# Step 1: Start MinIO (local S3-compatible storage)
docker run -d \
    -p 9000:9000 -p 9001:9001 \
    --name minio \
    -e MINIO_ROOT_USER=traffic \
    -e MINIO_ROOT_PASSWORD=traffic123 \
    minio/minio server /data --console-address ":9001"

# Step 2: Create bucket
# Open http://localhost:9001 → Create bucket "traffic-eye-snapshots"
```

Create file `alerts/snapshot_store.py`:
```python
import os
import io
import cv2
import numpy as np
from minio import Minio

MINIO_URL    = os.environ.get("MINIO_URL", "localhost:9000")
MINIO_KEY    = os.environ.get("MINIO_KEY", "traffic")
MINIO_SECRET = os.environ.get("MINIO_SECRET", "traffic123")
BUCKET       = "traffic-eye-snapshots"

client = Minio(MINIO_URL, access_key=MINIO_KEY,
               secret_key=MINIO_SECRET, secure=False)

# Ensure bucket exists
if not client.bucket_exists(BUCKET):
    client.make_bucket(BUCKET)

def upload_snapshot(frame: np.ndarray, filename: str) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    data = io.BytesIO(buf.tobytes())
    client.put_object(BUCKET, filename, data, len(buf),
                      content_type="image/jpeg")
    return f"http://{MINIO_URL}/{BUCKET}/{filename}"
```

Then update `alerts/alert_dispatcher.py`:
```python
# Replace cv2.imwrite(...) with:
from alerts.snapshot_store import upload_snapshot
url = upload_snapshot(snapshot_frame, fname)
alert.image_path = url
```

### Expected Outcome
- Snapshots stored reliably (no local disk fill-up)
- Accessible via URL from React dashboard
- Survives container restarts

---

## Task 3.9 — Leaflet Map View (React Component)

### What
Interactive Kathmandu map showing camera locations with live congestion coloring.

### Prerequisites
- [ ] `cd frontend && npm install leaflet react-leaflet @types/leaflet`
- [ ] Camera latitude/longitude in DB

### Steps

Create file `frontend/src/components/MapView.tsx`:
```tsx
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

interface Camera {
  camera_id: string;
  name: string;
  latitude: number;
  longitude: number;
  congestion?: string;
}

export default function MapView({ cameras }: { cameras: Camera[] }) {
  const center = [27.7172, 85.3240]; // Kathmandu

  return (
    <MapContainer center={center} zoom={13} className="h-[500px] rounded-xl">
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution="&copy; OpenStreetMap"
      />
      {cameras.map((cam) => (
        <Marker key={cam.camera_id} position={[cam.latitude, cam.longitude]}>
          <Popup>
            <strong>{cam.name}</strong><br/>
            Status: {cam.congestion || "Unknown"}
          </Popup>
        </Marker>
      ))}
    </MapContainer>
  );
}
```

Add to `frontend/src/pages/index.tsx`:
```tsx
import MapView from "../components/MapView";
// Inside the dashboard, add:
<section>
  <h2>📍 Camera Map</h2>
  <MapView cameras={cameras} />
</section>
```

---

## Task 3.12 — Multi-Intersection Scaling (Redis Pub/Sub)

### What
Handle 20-50 simultaneous cameras by distributing work via Redis.

### Prerequisites
- [ ] Redis running (see docker-compose.yml)
- [ ] `pip install redis`

### Steps

Create file `workers/redis_broker.py`:
```python
import redis
import json
import os

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
r = redis.from_url(REDIS_URL)

def publish_event(channel: str, data: dict):
    """Publish detection/violation events to Redis pub/sub."""
    r.publish(channel, json.dumps(data))

def subscribe_events(channel: str):
    """Subscribe to real-time events from all cameras."""
    pubsub = r.pubsub()
    pubsub.subscribe(channel)
    for message in pubsub.listen():
        if message["type"] == "message":
            yield json.loads(message["data"])

# Channels:
# "traffic:violations"  — all violation alerts
# "traffic:stats"       — congestion level changes
# "traffic:cam:{id}"    — per-camera frame events
```

Update `alerts/alert_dispatcher.py` — after DB write, add:
```python
from workers.redis_broker import publish_event
publish_event("traffic:violations", asdict(alert))
```

Update `workers/pipeline.py` — after congestion update:
```python
from workers.redis_broker import publish_event
publish_event("traffic:stats", {
    "camera_id": self.camera_id,
    "vehicle_count": cong_state.vehicle_count,
    "congestion": cong_state.congestion_level,
})
```

### Architecture After This Task
```
Camera 1 → Worker 1 → Redis pub/sub → API → WebSocket → React
Camera 2 → Worker 2 ↗
Camera 3 → Worker 3 ↗
...
Camera N → Worker N ↗
```

---

## Task 3.13 — Alembic Database Migrations

### What
Proper schema versioning so you can upgrade PostgreSQL without losing data.

### Prerequisites
- [ ] PostgreSQL running (see docker-compose.yml)
- [ ] `pip install alembic`

### Steps

```bash
# Step 1: Initialize Alembic
cd e:\yolo_new\YOLO_Projects\traffic_nepal
alembic init alembic

# Step 2: Edit alembic.ini
# Set: sqlalchemy.url = postgresql://traffic:traffic123@localhost:5432/traffic_eye

# Step 3: Edit alembic/env.py
# Import your models:
#   from api.database import Base
#   target_metadata = Base.metadata

# Step 4: Generate initial migration
alembic revision --autogenerate -m "Initial schema"

# Step 5: Apply migration
alembic upgrade head

# Future schema changes:
alembic revision --autogenerate -m "Add watchlist table"
alembic upgrade head
```

---

## Quick Reference — What's Already Built

```
traffic_nepal/
├── config/settings.py              ✅
├── detection/yoloworld_engine.py   ✅
├── ingest/stream_reader.py         ✅
├── ingest/camera_manager.py        ✅
├── tracking/bytetrack_wrapper.py   ✅ (IoU fallback + real BYTETracker if installed — Task 3.4 ✅)
├── intelligence/
│   ├── helmet_rule.py              ✅
│   ├── congestion_monitor.py       ✅
│   ├── speed_estimator.py          ✅
│   ├── plate_ocr.py               ✅
│   ├── wrong_lane.py              ✅
│   ├── night_enhance.py           ✅ (Phase 3)
│   └── watchlist.py               ✅ (Phase 3)
├── alerts/
│   ├── alert_dispatcher.py        ✅
│   ├── snapshot_store.py          ✅ (MinIO + local fallback — Task 3.8 ✅)
│   └── sms_alert.py               ✅
├── workers/
│   ├── pipeline.py                ✅
│   ├── redis_broker.py           ✅ (Redis pub/sub — Task 3.12 ✅)
│   └── celery_app.py             ✅ (async OCR queue)
├── api/
│   ├── main.py                    ✅ (13 endpoints)
│   ├── database.py                ✅
│   ├── models.py                  ✅
│   └── routers/ (6 files)         ✅
├── deploy/export_engine.py        ✅ (Phase 3)
├── frontend/ (React scaffold)     ✅
├── docker-compose.yml             ✅
├── Dockerfile.api                 ✅
├── Dockerfile.inference           ✅
├── nginx.conf                     ✅
├── alembic.ini                    ✅ (database migrations — Task 3.13 ✅)
├── data/nepal_traffic.yaml        ✅ (YOLO fine-tune config — Task 3.3, awaiting GPU)
├── data/nepal_traffic_classes.json ✅
└── app.py (Streamlit MVP)         ✅
```

---

## Checklist (print this and check off)

- [ ] **3.3** Fine-tune YOLO-World on Nepal data ← **BLOCKED: needs GPU + 5,000 annotated images**
- [x] **3.4** Real ByteTrack integration with IoU fallback ← ✅ DONE
- [x] **3.8** MinIO snapshot storage with local fallback ← ✅ DONE
- [x] **3.9** Leaflet MapView component (React dashboard) ← ✅ DONE
- [x] **3.12** Redis pub/sub event broker ← ✅ DONE
- [x] **3.13** Alembic database migrations ← ✅ DONE

---

## Estimated Time After GPU Access

| Task | Time |
|---|---|
| 3.3 Fine-tune (collect + annotate + train) | 1-2 weeks |
| 3.4 ByteTrack swap | 2 hours |
| 3.8 MinIO setup | 1 hour |
| 3.9 Map component | 3 hours |
| 3.12 Redis scaling | 4 hours |
| 3.13 Alembic | 1 hour |
| **Total** | **~2 weeks** (mostly dataset work) |

---

*Tasks 3.4, 3.8, 3.9, 3.12, 3.13 are COMPLETE. The only remaining task is 3.3 which requires a GPU and manual data collection.*

**Next Action:** Collect 5,000+ annotated images of Nepal traffic, then run fine-tuning once GPU is available. Use `data/nepal_traffic.yaml` for the dataset configuration.
