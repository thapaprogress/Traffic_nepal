import React from "react";
import { useLiveStream } from "../hooks/useWebSocket";

interface Props {
  cameraId: string;
  cameraName: string;
}

export default function CameraCard({ cameraId, cameraName }: Props) {
  const { data, connected } = useLiveStream(cameraId);
  const stats = data?.stats;

  const congestionColor: Record<string, string> = {
    LOW: "bg-green-500",
    MEDIUM: "bg-yellow-500",
    HIGH: "bg-red-500",
  };

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-xl overflow-hidden shadow-lg">
      {/* Video Frame */}
      <div className="relative aspect-video bg-black">
        {data?.frame ? (
          <img
            src={`data:image/jpeg;base64,${data.frame}`}
            alt={cameraName}
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="flex items-center justify-center h-full text-gray-500">
            {connected ? "Waiting for frames..." : "Disconnected"}
          </div>
        )}
        {/* Connection badge */}
        <div className={`absolute top-2 right-2 w-3 h-3 rounded-full ${connected ? "bg-green-400" : "bg-red-400"}`} />
      </div>

      {/* Stats Bar */}
      <div className="p-3 space-y-2">
        <div className="flex justify-between items-center">
          <h3 className="text-white font-semibold text-sm">{cameraName}</h3>
          <span className={`text-xs px-2 py-0.5 rounded-full text-white ${congestionColor[stats?.congestion || "LOW"]}`}>
            {stats?.congestion || "—"}
          </span>
        </div>
        <div className="grid grid-cols-4 gap-2 text-xs text-gray-400">
          <div>
            <span className="text-cyan-400 font-bold">{stats?.fps || 0}</span> FPS
          </div>
          <div>
            <span className="text-blue-400 font-bold">{stats?.vehicle_count || 0}</span> Veh
          </div>
          <div>
            <span className="text-red-400 font-bold">{stats?.helmet_violations || 0}</span> Helm
          </div>
          <div>
            <span className="text-orange-400 font-bold">{stats?.speed_violations || 0}</span> Spd
          </div>
        </div>
      </div>
    </div>
  );
}
