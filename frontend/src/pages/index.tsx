import React, { useEffect, useState } from "react";
import axios from "axios";
import CameraGrid from "../components/CameraGrid";
import ViolationTable from "../components/ViolationTable";
import { useAlertStream } from "../hooks/useWebSocket";
import dynamic from "next/dynamic";

const MapView = dynamic(() => import("../components/MapView"), { ssr: false });

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Dashboard() {
  const [cameras, setCameras] = useState([]);
  const [stats, setStats] = useState<any>(null);
  const { alerts } = useAlertStream();

  useEffect(() => {
    axios.get(`${API}/cameras/`).then((r) => setCameras(r.data)).catch(() => {});
    axios.get(`${API}/stats/summary`).then((r) => setStats(r.data)).catch(() => {});
    const interval = setInterval(() => {
      axios.get(`${API}/stats/summary`).then((r) => setStats(r.data)).catch(() => {});
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-[#0b0f1a] text-white">
      {/* Header */}
      <header className="border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between max-w-7xl mx-auto">
          <h1 className="text-2xl font-bold text-cyan-400">
            🚦 Traffic Eye Nepal
          </h1>
          <div className="flex gap-4 text-sm text-gray-400">
            <span>{cameras.length} cameras</span>
            <span className="text-green-400">● Live</span>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-6 space-y-8">
        {/* Stats Cards */}
        {stats && (
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <StatCard label="Total Violations" value={stats.total_violations} color="text-white" />
            <StatCard label="Helmet" value={stats.helmet_count} color="text-red-400" />
            <StatCard label="Speed" value={stats.speed_count} color="text-orange-400" />
            <StatCard label="Wrong Lane" value={stats.wrong_lane_count} color="text-purple-400" />
            <StatCard label="Active Cams" value={stats.active_cameras} color="text-cyan-400" />
          </div>
        )}

        {/* Camera Grid & Map View */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <h2 className="text-lg font-semibold text-gray-200 mb-3">📹 Live Cameras</h2>
            <CameraGrid cameras={cameras} />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-gray-200 mb-3">📍 Camera Map View</h2>
            <MapView cameras={cameras} />
          </div>
        </div>

        {/* Recent Alerts Ticker */}
        {alerts.length > 0 && (
          <section>
            <h2 className="text-lg font-semibold text-gray-200 mb-3">🚨 Live Alerts</h2>
            <div className="flex gap-3 overflow-x-auto pb-2">
              {alerts.slice(0, 10).map((a, i) => (
                <div key={i} className="flex-shrink-0 bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-xs">
                  <span className="text-red-400 font-bold">{a.violation_type}</span>
                  <span className="text-gray-400 ml-2">#{a.track_id}</span>
                  <span className="text-gray-500 ml-2">{a.location}</span>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Violations */}
        <section>
          <h2 className="text-lg font-semibold text-gray-200 mb-3">📋 Violation Log</h2>
          <ViolationTable />
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-800 text-center py-4 text-gray-600 text-xs">
        🚦 Traffic Eye Nepal v2.0 · YOLO-World · FastAPI · Next.js
      </footer>
    </div>
  );
}

function StatCard({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="bg-gray-900 border border-gray-700 rounded-xl p-4 text-center">
      <p className={`text-2xl font-bold ${color}`}>{value}</p>
      <p className="text-xs text-gray-500 mt-1">{label}</p>
    </div>
  );
}
