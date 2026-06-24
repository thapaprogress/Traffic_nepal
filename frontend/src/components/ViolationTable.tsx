import React, { useEffect, useState } from "react";
import axios from "axios";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Violation {
  id: number;
  violation_type: string;
  track_id: number;
  camera_id: string;
  speed_kmh: number;
  vehicle_number: string;
  location: string;
  detected_at: number;
  image_path: string;
}

export default function ViolationTable() {
  const [violations, setViolations] = useState<Violation[]>([]);
  const [filter, setFilter] = useState("ALL");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchViolations();
    const interval = setInterval(fetchViolations, 5000);
    return () => clearInterval(interval);
  }, [filter]);

  async function fetchViolations() {
    try {
      const params = filter !== "ALL" ? { violation_type: filter } : {};
      const res = await axios.get(`${API}/violations/`, { params });
      setViolations(res.data);
    } catch (e) {}
    setLoading(false);
  }

  const badgeColor: Record<string, string> = {
    HELMET: "bg-red-900 text-red-300",
    SPEED: "bg-orange-900 text-orange-300",
    WRONG_LANE: "bg-purple-900 text-purple-300",
  };

  return (
    <div className="space-y-4">
      {/* Filter */}
      <div className="flex gap-2">
        {["ALL", "HELMET", "SPEED", "WRONG_LANE"].map((t) => (
          <button
            key={t}
            onClick={() => setFilter(t)}
            className={`px-3 py-1 rounded-lg text-xs font-semibold transition ${
              filter === t
                ? "bg-cyan-600 text-white"
                : "bg-gray-800 text-gray-400 hover:bg-gray-700"
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {/* Table */}
      <div className="overflow-x-auto rounded-lg border border-gray-700">
        <table className="w-full text-sm text-left text-gray-300">
          <thead className="bg-gray-800 text-gray-400 uppercase text-xs">
            <tr>
              <th className="px-4 py-3">Type</th>
              <th className="px-4 py-3">Track ID</th>
              <th className="px-4 py-3">Speed</th>
              <th className="px-4 py-3">Plate</th>
              <th className="px-4 py-3">Location</th>
              <th className="px-4 py-3">Time</th>
            </tr>
          </thead>
          <tbody>
            {violations.map((v) => (
              <tr key={v.id} className="border-b border-gray-800 hover:bg-gray-800/50">
                <td className="px-4 py-2">
                  <span className={`px-2 py-0.5 rounded text-xs font-bold ${badgeColor[v.violation_type] || "bg-gray-700 text-gray-300"}`}>
                    {v.violation_type}
                  </span>
                </td>
                <td className="px-4 py-2">#{v.track_id}</td>
                <td className="px-4 py-2">{v.speed_kmh > 0 ? `${v.speed_kmh} km/h` : "—"}</td>
                <td className="px-4 py-2 font-mono">{v.vehicle_number || "—"}</td>
                <td className="px-4 py-2">{v.location}</td>
                <td className="px-4 py-2 text-gray-500">
                  {new Date(v.detected_at * 1000).toLocaleTimeString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {violations.length === 0 && !loading && (
          <p className="text-center py-8 text-gray-500">No violations found.</p>
        )}
      </div>
    </div>
  );
}
