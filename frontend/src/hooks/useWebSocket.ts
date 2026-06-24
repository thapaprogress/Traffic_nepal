import { useState, useEffect, useRef, useCallback } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "ws://localhost:8000";

interface FrameData {
  frame: string; // base64 JPEG
  stats: {
    fps: number;
    vehicle_count: number;
    congestion: string;
    helmet_violations: number;
    speed_violations: number;
    frame_num: number;
  };
}

export function useLiveStream(cameraId: string) {
  const [data, setData] = useState<FrameData | null>(null);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const ws = new WebSocket(`${API_BASE}/ws/live/${cameraId}`);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);
    ws.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data) as FrameData;
        setData(parsed);
      } catch {}
    };

    return () => {
      ws.close();
    };
  }, [cameraId]);

  return { data, connected };
}

interface Alert {
  violation_type: string;
  track_id: number;
  camera_id: string;
  location: string;
  speed_kmh: number;
  timestamp: number;
}

export function useAlertStream() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const ws = new WebSocket(`${API_BASE}/ws/alerts`);

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onmessage = (event) => {
      try {
        const alert = JSON.parse(event.data) as Alert;
        setAlerts((prev) => [alert, ...prev].slice(0, 100));
      } catch {}
    };

    return () => ws.close();
  }, []);

  return { alerts, connected };
}
