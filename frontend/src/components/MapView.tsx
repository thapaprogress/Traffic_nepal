import React from 'react';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// Fix Leaflet default icon markers in Next.js
// @ts-ignore
if (typeof window !== 'undefined') {
  delete L.Icon.Default.prototype._getIconUrl;
  L.Icon.Default.mergeOptions({
    iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
    iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
  });
}

interface Camera {
  camera_id: string;
  name: string;
  latitude: number;
  longitude: number;
  congestion?: string;
  active?: boolean;
}

export default function MapView({ cameras }: { cameras: Camera[] }) {
  const center = [27.7172, 85.3240]; // Kathmandu coordinates

  // Dynamic colors for markers
  const getMarkerIcon = (congestion?: string) => {
    let color = '#22c55e'; // Green for LOW
    if (congestion === 'MEDIUM') color = '#f97316'; // Orange
    if (congestion === 'HIGH') color = '#ef4444'; // Red

    return L.divIcon({
      html: `<span style="background-color: ${color}; width: 14px; height: 14px; border: 2px solid white; border-radius: 50%; display: inline-block; box-shadow: 0 0 8px rgba(0,0,0,0.5);"></span>`,
      className: 'custom-leaflet-icon',
      iconSize: [14, 14],
      iconAnchor: [7, 7]
    });
  };

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden p-4">
      <MapContainer 
        center={center as [number, number]} 
        zoom={13} 
        style={{ height: '350px', width: '100%', borderRadius: '8px' }}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        />
        {cameras.map((cam) => {
          if (!cam.latitude || !cam.longitude) return null;
          return (
            <Marker 
              key={cam.camera_id} 
              position={[cam.latitude, cam.longitude]}
              icon={getMarkerIcon(cam.congestion)}
            >
              <Popup>
                <div className="text-gray-900 font-sans text-xs">
                  <h4 className="font-bold text-sm text-cyan-700">{cam.name}</h4>
                  <p className="mt-1">ID: <code className="bg-gray-100 px-1 rounded">{cam.camera_id}</code></p>
                  <p>Congestion: <span className="font-semibold">{cam.congestion || 'LOW'}</span></p>
                  <p>Status: <span className={cam.active ? "text-green-600" : "text-gray-500"}>{cam.active ? "Online" : "Offline"}</span></p>
                </div>
              </Popup>
            </Marker>
          );
        })}
      </MapContainer>
    </div>
  );
}
