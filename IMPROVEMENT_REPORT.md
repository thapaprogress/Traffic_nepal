# Traffic Eye Nepal — System Audit & Improvement Report
**Date:** June 24, 2026  
**Auditor:** Kiro  
**Scope:** Full codebase review of `traffic_nepal/` — flaws, optimizations, and technology recommendations

---

## Part A — CRITICAL FLAWS FOUND

These are real bugs I found by reading the code. Ranked by severity.

### 🔴 A1. Speed estimation is fundamentally inaccurate (wall-clock vs video time)

**File:** `intelligence/speed_estimator.py`

**Problem:** Speed uses `time.time()` (wall-clock) between Line A and Line B crossings:
```python
self._enter_times[tid] = time.time()      # Line A
elapsed = time.time() - self._enter_times.pop(tid)  # Line B
speed_ms = self.real_distance_m / elapsed
```

The elapsed time includes **inference latency + frame-skip gaps**, not actual elapsed video time. On CPU, each frame takes 200-500ms to process, so a car that physically took 0.7s to cross will be measured as 2-3s → speed reported as **3-4x too slow**. On a video file this is completely wrong because processing speed ≠ playback speed.

**Fix:** Use frame numbers and the source FPS:
```python
elapsed = (frame_b - frame_a) / source_fps  # real video seconds
```

---

### 🔴 A2. Line-crossing detection misses fast vehicles

**File:** `intelligence/speed_estimator.py` (lines ~62-70)

**Problem:** Detection band is `line_y - 5 <= cy <= line_y + 5` (10px window). With `FRAME_SKIP=2` and a fast vehicle, the center point can jump 30-50px between processed frames, **completely skipping the 10px band** → no crossing registered → speed never calculated.

**Fix:** Detect line crossing by checking if the center moved *across* the line between two frames:
```python
if prev_cy < line_y <= curr_cy:  # crossed downward
```
Track previous position per track_id.

---

### 🔴 A3. Memory leak in speed estimator `_enter_times`

**File:** `intelligence/speed_estimator.py`

**Problem:** Vehicles that cross Line A but leave the frame (turn off, change lane) before reaching Line B stay in `self._enter_times` **forever**. Over hours of operation this dict grows unbounded.

**Fix:** Expire entries older than N seconds, or cap dict size with TTL cleanup.

---

### 🔴 A4. Track IDs are global across ALL cameras + never reset

**File:** `tracking/bytetrack_wrapper.py`

**Problem:** `Track._next_id` is a **class variable**:
```python
class Track:
    _next_id = 1  # SHARED across every ByteTracker instance
```
With multiple cameras, all share the same counter — IDs are not per-camera. Also it never resets, so restarting detection keeps incrementing from where it left off. On a 24/7 system this grows to millions.

**Fix:** Make `_next_id` an instance variable of `ByteTracker`, namespace IDs per-camera (e.g. `cam01_5`).

---

### 🔴 A5. Tracker matches across object classes

**File:** `tracking/bytetrack_wrapper.py` (`_assign`)

**Problem:** IoU matching ignores class. When a person walks in front of a parked car, the "car" track can get reassigned to the "person" detection if IoU is high enough → a car's track_id suddenly becomes a person, corrupting speed/plate association.

**Fix:** Add class-consistency constraint — only match track to detection of the **same class**.

---

### 🟠 A6. SQLite thread-safety race conditions

**File:** `alerts/alert_dispatcher.py`

**Problem:** Single shared connection with `check_same_thread=False`. Only `dispatch()` is under `self._lock`. But `log_traffic_stat()`, `query_violations()`, and `violation_counts()` are **NOT locked** and are called from both the pipeline thread and the Streamlit main thread → `sqlite3.OperationalError: database is locked` under load.

**Fix:** Put all DB access under the lock, or use a per-thread connection / a proper connection pool (or move to PostgreSQL as planned).

---

### 🟠 A7. Database grows unbounded — PLATE_READ spam

**File:** `workers/pipeline.py` + `alerts/alert_dispatcher.py`

**Problem:** Every new vehicle writes a `PLATE_READ` row. With the 30s cache TTL, a vehicle that re-enters after 30s creates a **duplicate** row. No deduplication, no retention policy. The `violations` table will reach millions of rows in days, and `violation_counts()` does a full `GROUP BY` scan every single frame for the dashboard.

**Fix:**
- Separate `plate_reads` into its own table (don't mix with violations)
- Add UNIQUE constraint or upsert on (camera_id, track_id, plate)
- Cache `violation_counts()` result, refresh every few seconds not every frame

---

### 🟠 A8. No database indexes

**File:** `alerts/alert_dispatcher.py`

**Problem:** Queries filter by `violation_type` and order by `detected_at`, but there are **no indexes**. Every query is a full table scan. Fine at 100 rows, catastrophic at 100,000.

**Fix:**
```sql
CREATE INDEX idx_viol_type ON violations(violation_type);
CREATE INDEX idx_viol_time ON violations(detected_at);
```

---

### 🟠 A9. Plate text normalization corrupts valid characters

**File:** `intelligence/plate_ocr.py` (`normalize_plate_text`)

**Problem:** It blindly replaces **every** O→0, I→1, S→5:
```python
text = text.replace("O", "0").replace("I", "1").replace("S", "5")
```
Nepal plates contain province letters. A plate like `KOSI` would become `K051`. This destroys legitimate letters.

**Fix:** Apply digit/letter substitution **positionally** — only in segments expected to be numeric, using the known Nepal plate pattern (province letters + number + letter + number).

---

### 🟡 A10. EasyOCR on CPU stalls the pipeline

**File:** `intelligence/plate_ocr.py`

**Problem:** EasyOCR on CPU is ~200-500ms per call. With `PLATE_MAX_OCR_PER_FRAME=5`, the first frame seeing 5 new vehicles blocks for **1-2.5 seconds**, freezing the live feed. The cache helps after the first sighting but the initial stall is jarring.

**Fix:** Offload OCR to the existing Celery worker (already scaffolded in `workers/celery_app.py`), or run OCR in a separate thread with a queue so the main pipeline never blocks.

---

### 🟡 A11. Helmet rule false positives

**File:** `intelligence/helmet_rule.py`

**Problem:** A pedestrian standing near a parked motorcycle is flagged as "no helmet". Association is pure spatial proximity with no motion or confidence check.

**Fix:** Require the person to be *on* the motorcycle (vertical overlap + person center above bike center), and confirm over multiple frames before flagging.

---

### 🟡 A12. Daemon thread exceptions die silently

**File:** `workers/pipeline.py`

**Problem:** `run()` sets `self.state["error"]` then re-raises, but it's a daemon thread — the exception vanishes. The Streamlit UI loop only checks `_stop_event`, never `state["error"]`, so a crashed pipeline shows a frozen last frame with no error shown to the user.

**Fix:** UI loop should check `state["error"]` each iteration and surface it; don't re-raise in the thread.

---

### 🟡 A13. Redundant full-frame copies per frame

**Files:** `yoloworld_engine.annotate`, `helmet_rule.draw_violations`, `congestion_monitor.draw_roi`

**Problem:** Each draw function does `frame.copy()`. That's 3-4 full 1080p copies per frame (~6MB each) = wasteful allocations hurting FPS.

**Fix:** Copy once, draw everything in place on the single annotated frame.

---

## Part B — PERFORMANCE OPTIMIZATIONS

### B1. GPU / device selection
The engine never sets a device — runs CPU even if a GPU exists. Add `device=0` auto-detection. **Single biggest win** (5-10x).

### B2. TensorRT FP16 export
Already scaffolded (`deploy/export_engine.py`). On RTX 4080: 3-6 FPS → 35-50 FPS.

### B3. Batch inference for multi-camera
Process N camera frames in one model call (batched tensor) instead of N separate calls.

### B4. Async OCR pipeline
Decouple OCR from detection (Celery/thread queue) so plate reading never blocks the live feed.

### B5. Cache dashboard aggregate queries
`violation_counts()` runs every frame. Cache it for 2-3 seconds.

### B6. Frame resolution tiering
Run detection at 640px, but crop plates from the **original** full-res frame for better OCR (currently OCR uses the already-processed frame, which is fine, but display downscaling should not feed OCR).

---

## Part C — NEW TECHNOLOGY RECOMMENDATIONS

### C1. Detection & Tracking
| Tech | Why adopt |
|---|---|
| **YOLO11 / YOLO12** | Newer Ultralytics models, better accuracy-speed tradeoff than YOLOv8 |
| **BoT-SORT** (over ByteTrack) | Adds camera-motion compensation + re-ID embeddings → far fewer ID switches |
| **RF-DETR / RT-DETR** | Transformer detectors, no NMS, strong on dense traffic scenes |

### C2. License Plate Recognition
| Tech | Why adopt |
|---|---|
| **PaddleOCR PP-OCRv4** | Faster + more accurate than EasyOCR, better on angled/small text |
| **Fast Plate OCR / YOLO-plate** | Purpose-built plate models vs generic OCR |
| **Dedicated plate detector** | Train a small YOLO just for plates → more reliable than YOLO-World prompts |

### C3. Inference Serving
| Tech | Why adopt |
|---|---|
| **NVIDIA Triton Inference Server** | Production model serving, dynamic batching, multi-model |
| **DeepStream SDK** | NVIDIA's full video-analytics pipeline — handles RTSP decode + inference + tracking on GPU end-to-end. Built exactly for this use case |
| **ONNX Runtime + TensorRT EP** | Portable acceleration |

### C4. Data & Backend
| Tech | Why adopt |
|---|---|
| **TimescaleDB** (Postgres extension) | Traffic stats are time-series — purpose-built, fast rollups |
| **MinIO / S3** | Snapshot object storage (already in TODO) |
| **Apache Kafka** | Event streaming for city-scale multi-intersection |
| **Redis Streams** | Lighter alternative for frame/event queues |

### C5. Edge Deployment
| Tech | Why adopt |
|---|---|
| **NVIDIA Jetson Orin** | Run inference at the camera, send only events — saves bandwidth |
| **DeepStream on Jetson** | Edge video analytics, multi-stream on one device |

### C6. MLOps
| Tech | Why adopt |
|---|---|
| **Roboflow / FiftyOne** | Dataset management + annotation for Nepal fine-tuning |
| **Weights & Biases** | Track fine-tuning experiments |
| **DVC** | Version datasets and model weights |

---

## Part D — PRIORITY ROADMAP

### Immediate (fix correctness bugs — 2-3 days)
1. **A1 + A2** — Fix speed calculation (frame-based timing + proper line crossing)
2. **A4 + A5** — Fix tracker (per-instance IDs + class consistency)
3. **A6 + A8** — DB locking + indexes
4. **A3** — Speed estimator memory leak

### Short-term (quality — 1 week)
5. **A7** — Separate plate_reads table + retention
6. **A9** — Positional plate normalization
7. **A11** — Helmet rule multi-frame confirmation
8. **A12 + A13** — Error surfacing + single-copy rendering

### Medium-term (performance — needs GPU)
9. **B1 + B2** — GPU + TensorRT (5-10x speedup)
10. **C1** — Swap to BoT-SORT tracker
11. **C2** — PaddleOCR + dedicated plate model
12. **B4** — Async OCR pipeline

### Long-term (scale — production)
13. **C3** — DeepStream / Triton serving
14. **C4** — TimescaleDB + Kafka
15. **C5** — Jetson edge deployment

---

## Summary

The system is a **solid MVP** with correct architecture, but has **5 critical correctness bugs** (speed timing, line crossing, tracker ID scope, class matching, DB races) that would produce wrong data in production. None are hard to fix.

The single highest-impact change is **GPU + TensorRT** (5-10x speed). The second is **fixing the speed estimator** — right now speed readings are not trustworthy.

For new tech, **NVIDIA DeepStream** is the most strategically aligned — it's purpose-built for exactly this multi-camera RTSP video-analytics pipeline and would replace much of the custom ingest/inference glue with battle-tested GPU code.

---

*Say "fix critical bugs" to implement Part D immediate fixes (A1-A6, A8), or pick specific items.*
