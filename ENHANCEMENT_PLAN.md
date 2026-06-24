# Traffic Eye Nepal — "Make It the Best" Enhancement Plan

## ✅ Completed (CPU-friendly, done now)

| # | Enhancement | Why it matters | Status |
|---|---|---|---|
| 1 | **Plate multi-read voting + confidence gate** | Same plate read across many frames; confidence-weighted consensus wins over a single noisy read. Big accuracy win. | ✅ Done |
| 3 | **Automated test suite (pytest)** | 19 tests covering tracker, speed, plate voting, helmet confirmation, wrong-lane. Locks in bug fixes. | ✅ Done |
| 4 | **Centralized logging** | `config/logging_config.py` — levels, console + file, quiets noisy libs. | ✅ Done |
| 5 | **Camera calibration profiles** | `config/camera_profiles.py` + sidebar save/load. No re-entering ROI/lines/limits. | ✅ Done |

## ⬜ Remaining (CPU-friendly)

| # | Enhancement | Why it matters | Status |
|---|---|---|---|
| 2 | **Perspective-corrected speed (homography)** | 4-point homography → accurate speed across the whole frame, not just one line. | ✅ Done |
| 6 | **Data retention / cleanup job** | Auto-purge old snapshots + DB rows so disk never fills. | ✅ Done |
| 7 | **Health & metrics endpoint** | `/health`, `/metrics`, `/maintenance/cleanup` for monitoring. | ✅ Done |

---

## 🚀 Top Performance Wins (GPU required)

| Win | Detail | Expected |
|---|---|---|
| **GPU + TensorRT FP16** | `python deploy/export_engine.py --format engine --half` — biggest single win | 3–6 FPS → **35–50 FPS** |
| **Async OCR pipeline** | ✅ Already implemented (background thread). Celery scaffold ready for multi-worker scale. | No feed freeze |
| **Cache dashboard count queries** | ✅ Already implemented (`violation_counts(cache_seconds=2.0)`) — was running every frame. | Lower DB load |

---

## 🧠 New Tech to Adopt (strategic roadmap)

| Tech | Why adopt | Phase |
|---|---|---|
| **NVIDIA DeepStream** | Purpose-built for multi-camera RTSP analytics on GPU — decode + infer + track end-to-end. Most strategically aligned for production. | Production |
| **BoT-SORT** (over ByteTrack) | Camera-motion compensation + re-ID embeddings → far fewer ID switches at busy intersections. | GPU phase |
| **PaddleOCR PP-OCRv4** | Faster + more accurate than EasyOCR, better on angled/small Nepali plates. | Can prototype on CPU |
| **TimescaleDB** | Traffic stats are time-series — hypertables + continuous aggregates for fast rollups. | Scale phase |
| **Jetson Orin** | Edge inference at the camera — send only events, save bandwidth. | Deployment |

---

## Execution status
Accuracy + quality items (1, 3, 4, 5) — **done on CPU**.
Performance wins (TensorRT) + new tech (DeepStream, BoT-SORT, TimescaleDB, Jetson) — **GPU phase**, see `TODO_GPU_TASKS.md`.
Remaining CPU items (2, 6, 7) — optional polish.
