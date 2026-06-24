# -*- coding: utf-8 -*-
"""
Traffic Eye Nepal — Phase 1 MVP Dashboard
Streamlit app with memplace-style live UI.
Run: streamlit run traffic_nepal/app.py
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import os
import sys
import threading

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Eye Nepal",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background: #0b0f1a; }
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1220 0%, #111827 100%);
    border-right: 1px solid #1e3a5f;
  }
  [data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 14px;
  }
  [data-testid="metric-container"] label { color: #38bdf8 !important; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f1f5f9 !important; font-size: 2rem !important;
  }
  .violation-badge {
    display:inline-block; padding:4px 12px; border-radius:20px;
    font-weight:700; font-size:0.85rem; margin:2px;
  }
  .badge-helmet { background:#7f1d1d; color:#fca5a5; }
  .badge-speed  { background:#78350f; color:#fcd34d; }
  .badge-low    { background:#14532d; color:#86efac; }
  .badge-medium { background:#78350f; color:#fcd34d; }
  .badge-high   { background:#7f1d1d; color:#fca5a5; }
  h1,h2,h3 { color: #38bdf8 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(90deg,#0d1220,#1e3a5f,#0d1220);
     border-radius:14px; padding:22px 36px; margin-bottom:18px;
     border:1px solid #38bdf844; text-align:center;">
  <h1 style="margin:0; font-size:2.4rem; letter-spacing:2px; color:#38bdf8;">
    🚦 Traffic Eye Nepal
  </h1>
  <p style="color:#64748b; margin:6px 0 0 0;">
    AI Traffic Intelligence · YOLO-World Zero-Shot · Real-time Violation Detection
  </p>
</div>
""", unsafe_allow_html=True)

# ─── Session state ────────────────────────────────────────────────────────────
if "pipeline"    not in st.session_state: st.session_state.pipeline    = None
if "pipe_thread" not in st.session_state: st.session_state.pipe_thread = None
if "running"     not in st.session_state: st.session_state.running     = False

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:8px 0 14px;">
      <span style="font-size:2.5rem;">🚦</span>
      <h2 style="color:#38bdf8; margin:4px 0 0 0;">Control Panel</h2>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("#### 📷 Camera Source")
    source_type = st.radio("Input", ["Webcam", "Video File", "RTSP URL"], horizontal=False)

    video_source = "0"
    uploaded_video = None

    if source_type == "Webcam":
        video_source = "0"
    elif source_type == "Video File":
        uploaded_video = st.file_uploader("Upload traffic video",
                                          type=["mp4","avi","mov","mkv"])
        if uploaded_video:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(delete=False,
                      suffix=os.path.splitext(uploaded_video.name)[-1])
            tmp.write(uploaded_video.read()); tmp.close()
            video_source = tmp.name
    elif source_type == "RTSP URL":
        video_source = st.text_input("RTSP URL",
                                     "rtsp://admin:password@192.168.1.10/live")

    st.markdown("---")
    st.markdown("#### 🎯 Detection Settings")
    conf_val  = st.slider("Confidence", 0.10, 0.90, 0.30, 0.05)
    imgsz_val = st.select_slider("Image Size", [320, 480, 640, 800], value=640)
    skip_val  = st.slider("Frame Skip", 1, 6, 2)

    st.markdown("---")
    st.markdown("#### 🛣️ Speed Detection")
    line_a = st.slider("Speed Line A (y-px)", 50, 600, 280)
    line_b = st.slider("Speed Line B (y-px)", 100, 700, 430)
    speed_limit = st.number_input("Speed Limit (km/h)", 20, 120, 60)

    st.markdown("---")
    st.markdown("#### 🧠 Intelligence Modules")
    en_wrong_lane = st.checkbox("↔️ Wrong-Lane Detection", value=True)
    allowed_dir = st.select_slider(
        "Allowed traffic direction",
        options=["Up", "Down", "Left", "Right"],
        value="Down",
        help="Direction vehicles should normally travel",
    )
    en_night = st.checkbox("🌙 Night Enhancement (auto)", value=True)
    en_watchlist = st.checkbox("🚨 Stolen-Vehicle Watchlist", value=True)

    st.markdown("---")
    st.markdown("#### 📍 Camera Info")
    cam_name = st.text_input("Camera Name", "Kalanki Chowk")
    cam_loc  = st.text_input("Location", "Kathmandu, Nepal")

    st.markdown("---")
    col_start, col_stop = st.columns(2)
    with col_start:
        start_btn = st.button("▶ Start", type="primary", use_container_width=True)
    with col_stop:
        stop_btn  = st.button("⏹ Stop",  use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#334155; font-size:0.75rem;">
      🚦 Traffic Eye Nepal v1.0<br>
      <span style="color:#38bdf8;">Phase 1 MVP</span>
    </div>""", unsafe_allow_html=True)

# ─── Start / Stop pipeline ────────────────────────────────────────────────────
if start_btn and not st.session_state.running:
    from config.settings import YOLO_CONF, YOLO_IMGSZ
    from config import settings
    settings.YOLO_CONF  = conf_val
    settings.YOLO_IMGSZ = imgsz_val
    settings.FRAME_SKIP = skip_val

    from workers.pipeline import CameraPipeline
    from detection.yoloworld_engine import YOLOWorldEngine
    YOLOWorldEngine._instance = None   # reset so new conf takes effect

    _DIR_MAP = {"Up": 0.0, "Right": 90.0, "Down": 180.0, "Left": 270.0}
    pipe = CameraPipeline(
        camera_id   = "cam_01",
        camera_name = cam_name,
        source      = video_source,
        line_y_a    = line_a,
        line_y_b    = line_b,
        speed_limit = speed_limit,
        allowed_direction_deg = _DIR_MAP.get(allowed_dir, 180.0),
        enable_wrong_lane     = en_wrong_lane,
        enable_night_enhance  = en_night,
        enable_watchlist      = en_watchlist,
    )
    t = pipe.start_thread()
    st.session_state.pipeline    = pipe
    st.session_state.pipe_thread = t
    st.session_state.running     = True
    st.rerun()

if stop_btn and st.session_state.running:
    if st.session_state.pipeline:
        st.session_state.pipeline.stop()
    st.session_state.running     = False
    st.session_state.pipeline    = None
    st.session_state.pipe_thread = None
    st.rerun()

# ─── Main layout ──────────────────────────────────────────────────────────────
tab_live, tab_violations, tab_stats = st.tabs([
    "📹 Live Feed", "🚨 Violations", "📊 Analytics"
])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — LIVE FEED
# ══════════════════════════════════════════════════════════════════
with tab_live:
    if not st.session_state.running:
        st.markdown("""
        <div style="text-align:center; padding:70px 20px;
             border:2px dashed #1e3a5f; border-radius:14px; margin-top:10px;">
          <div style="font-size:4rem;">🚦</div>
          <h3 style="color:#334155;">Ready to Monitor</h3>
          <p style="color:#475569;">
            Configure camera settings in the sidebar and click
            <strong style="color:#38bdf8;">▶ Start</strong> to begin.
          </p>
        </div>""", unsafe_allow_html=True)
    else:
        pipe  = st.session_state.pipeline
        state = pipe.state if pipe else {}

        col_feed, col_stats = st.columns([3, 1])

        with col_feed:
            st.markdown("#### 📹 Live Camera Feed")
            frame_ph = st.empty()

        with col_stats:
            st.markdown("#### 📊 Live Stats")
            m_fps    = st.empty()
            m_veh    = st.empty()
            m_cong   = st.empty()
            m_helmet = st.empty()
            m_speed  = st.empty()

        alert_ph = st.empty()

        def get_recent_alerts(pipe):
            try:
                from alerts.alert_dispatcher import get_dispatcher
                return get_dispatcher().recent_alerts
            except Exception:
                return []

        # ── Memplace-style live loop ──────────────────────────────────────
        while st.session_state.running and pipe and not pipe._stop_event.is_set():
            s = pipe.state

            # Surface pipeline errors (fix A12)
            if s.get("error"):
                frame_ph.error(f"Pipeline error: {s['error']}")
                st.session_state.running = False
                break

            # Frame
            if s.get("frame_rgb") is not None:
                frame_ph.image(s["frame_rgb"], channels="RGB")

            # Metrics
            m_fps.metric("FPS",       s.get("fps", 0))
            m_veh.metric("Vehicles",  s.get("vehicle_count", 0))

            level = s.get("congestion", "LOW")
            badge_cls = f"badge-{level.lower()}"
            m_cong.markdown(
                f'<span class="violation-badge {badge_cls}">🚗 {level}</span>',
                unsafe_allow_html=True)

            hv = s.get("helmet_violations", 0)
            sv = s.get("speed_violations", 0)
            m_helmet.metric("Helmet ⚠️",  hv,
                            delta=None if hv == 0 else f"+{hv}",
                            delta_color="inverse")
            m_speed.metric("Speed ⚠️",   sv,
                           delta=None if sv == 0 else f"+{sv}",
                           delta_color="inverse")

            # Plates read count
            pr_count = s.get("plates_read", 0)
            m_speed.metric("🔢 Plates Read", pr_count)

            # Wrong-lane + watchlist (high-priority)
            wl = s.get("wrong_lane_violations", 0)
            wh = s.get("watchlist_hits", 0)
            m_cong.markdown(
                f'<span class="violation-badge {badge_cls}">🚗 {level}</span>'
                + (f'<br><span class="violation-badge badge-helmet">↔️ Wrong-lane: {wl}</span>' if wl else '')
                + (f'<br><span class="violation-badge badge-helmet">🚨 WATCHLIST: {wh}</span>' if wh else ''),
                unsafe_allow_html=True)

            # Recent alerts ticker
            if pipe:
                recent = get_recent_alerts(pipe)
                if recent:
                    rows = "".join(
                        f'<tr><td style="color:#94a3b8">{a["violation_type"]}</td>'
                        f'<td style="color:#f1f5f9">#{a["track_id"]}</td>'
                        f'<td style="color:#64748b">{a["location"]}</td></tr>'
                        for a in recent[:8]
                    )
                    alert_ph.markdown(
                        f'<table style="width:100%;font-size:0.8rem;">'
                        f'<tr><th style="color:#38bdf8">Type</th>'
                        f'<th style="color:#38bdf8">ID</th>'
                        f'<th style="color:#38bdf8">Location</th></tr>'
                        f'{rows}</table>',
                        unsafe_allow_html=True)

            time.sleep(0.04)   # ~25 UI updates/sec

# ══════════════════════════════════════════════════════════════════
# TAB 2 — VIOLATIONS
# ══════════════════════════════════════════════════════════════════
with tab_violations:
    st.markdown("#### 🚨 Violation Log")

    try:
        from alerts.alert_dispatcher import get_dispatcher
        dispatcher = get_dispatcher()
        counts = dispatcher.violation_counts()

        c1, c2, c3 = st.columns(3)
        c1.metric("🪖 Helmet Violations",  counts.get("HELMET", 0))
        c2.metric("💨 Speed Violations",   counts.get("SPEED",  0))
        c3.metric("↔️ Wrong-Lane",         counts.get("WRONG_LANE", 0))

        # ── Watchlist management ───────────────────────────────────────────
        with st.expander("🚨 Stolen-Vehicle Watchlist", expanded=False):
            from intelligence.watchlist import get_watchlist
            wl_sys = get_watchlist()
            wc1, wc2 = st.columns([2, 1])
            with wc1:
                new_plate = st.text_input("Add plate to watchlist", "", key="wl_add")
                reason = st.selectbox("Reason", ["STOLEN", "WANTED", "BLACKLISTED"])
                if st.button("➕ Add to Watchlist") and new_plate.strip():
                    wl_sys.add_plate(new_plate.strip(), reason)
                    st.success(f"Added {new_plate} ({reason})")
            with wc2:
                st.metric("Watchlist size", wl_sys.count)
            entries = wl_sys.get_all()
            if entries:
                st.dataframe(pd.DataFrame(entries), use_container_width=True, height=160)

        st.markdown("---")
        vtype = st.selectbox("Filter by type", ["ALL", "HELMET", "SPEED", "WRONG_LANE"])
        violations = dispatcher.query_violations(
            limit=200,
            violation_type=None if vtype == "ALL" else vtype
        )

        if violations:
            df = pd.DataFrame(violations)
            df["detected_at"] = pd.to_datetime(df["detected_at"], unit="s")
            df = df[["detected_at", "violation_type", "track_id",
                     "speed_kmh", "location", "vehicle_number", "image_path"]]
            df.columns = ["Time", "Type", "Track ID", "Speed (km/h)",
                          "Location", "Plate", "Snapshot"]
            st.dataframe(
                df.style.map(
                    lambda v: "background-color:#7f1d1d; color:#fca5a5"
                              if v == "HELMET"
                              else ("background-color:#78350f; color:#fcd34d"
                                    if v == "SPEED" else ""),
                    subset=["Type"]
                ),
                use_container_width=True, height=400
            )
            st.download_button(
                "⬇️ Export Violations CSV",
                df.to_csv(index=False),
                "violations_traffic_eye.csv", "text/csv",
                use_container_width=True
            )

            # Snapshot preview
            st.markdown("#### 📸 Recent Snapshots")
            snap_cols = st.columns(4)
            shown = 0
            for row in violations[:12]:
                if row.get("image_path") and os.path.isfile(row["image_path"]):
                    with snap_cols[shown % 4]:
                        import cv2 as cv
                        img = cv.imread(row["image_path"])
                        if img is not None:
                            st.image(cv.cvtColor(img, cv.COLOR_BGR2RGB),
                                     caption=f"{row['violation_type']} #{row['track_id']}")
                    shown += 1
        else:
            st.info("No violations recorded yet. Start detection to begin monitoring.")

    except Exception as e:
        st.warning(f"Violation log not available: {e}")

# ══════════════════════════════════════════════════════════════════
# TAB 3 — ANALYTICS
# ══════════════════════════════════════════════════════════════════
with tab_stats:
    st.markdown("#### 📊 Traffic Analytics")

    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from alerts.alert_dispatcher import get_dispatcher
        dispatcher = get_dispatcher()

        counts = dispatcher.violation_counts()

        if counts:
            col_a, col_b = st.columns(2)

            with col_a:
                fig_pie = px.pie(
                    names=list(counts.keys()),
                    values=list(counts.values()),
                    title="Violations by Type",
                    template="plotly_dark",
                    color_discrete_sequence=["#ef4444", "#f97316", "#3b82f6"],
                )
                fig_pie.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#94a3b8",
                    title_font_color="#38bdf8",
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_b:
                violations = dispatcher.query_violations(limit=500)
                if violations:
                    df = pd.DataFrame(violations)
                    df["hour"] = pd.to_datetime(df["detected_at"], unit="s").dt.hour
                    hourly = df.groupby("hour").size().reset_index(name="count")
                    fig_bar = px.bar(
                        hourly, x="hour", y="count",
                        title="Violations by Hour",
                        template="plotly_dark",
                        color="count",
                        color_continuous_scale="Reds",
                    )
                    fig_bar.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(11,15,26,0.7)",
                        font_color="#94a3b8",
                        title_font_color="#38bdf8",
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

            # Traffic stats timeline
            import sqlite3
            from config.settings import DB_PATH
            conn = sqlite3.connect(DB_PATH)
            stats_df = pd.read_sql(
                "SELECT * FROM traffic_stats ORDER BY recorded_at DESC LIMIT 200",
                conn
            )
            conn.close()
            if not stats_df.empty:
                stats_df["time"] = pd.to_datetime(stats_df["recorded_at"], unit="s")
                fig_line = px.line(
                    stats_df, x="time", y="vehicle_count",
                    color="congestion_level",
                    title="Vehicle Count Timeline",
                    template="plotly_dark",
                    color_discrete_map={"LOW":"#22c55e","MEDIUM":"#f97316","HIGH":"#ef4444"},
                )
                fig_line.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(11,15,26,0.7)",
                    font_color="#94a3b8",
                    title_font_color="#38bdf8",
                )
                st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No analytics data yet. Run detection to populate charts.")

    except Exception as e:
        st.warning(f"Analytics not available: {e}")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#1e3a5f; padding:10px 0; font-size:0.82rem;">
  🚦 <strong style="color:#38bdf8;">Traffic Eye Nepal</strong> &nbsp;·&nbsp;
  YOLO-World · ByteTrack · OpenCV · Streamlit &nbsp;·&nbsp;
  Phase 1 MVP
</div>
""", unsafe_allow_html=True)
