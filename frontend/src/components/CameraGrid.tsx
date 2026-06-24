import React from "react";
import CameraCard from "./CameraCard";

interface Camera {
  camera_id: string;
  name: string;
}

interface Props {
  cameras: Camera[];
}

export default function CameraGrid({ cameras }: Props) {
  if (cameras.length === 0) {
    return (
      <div className="text-center py-20 text-gray-500">
        <p className="text-4xl mb-4">📷</p>
        <p>No cameras registered. Add one via the API.</p>
      </div>
    );
  }

  const gridCols =
    cameras.length <= 1 ? "grid-cols-1" :
    cameras.length <= 4 ? "grid-cols-2" :
    "grid-cols-3";

  return (
    <div className={`grid ${gridCols} gap-4`}>
      {cameras.map((cam) => (
        <CameraCard
          key={cam.camera_id}
          cameraId={cam.camera_id}
          cameraName={cam.name}
        />
      ))}
    </div>
  );
}
