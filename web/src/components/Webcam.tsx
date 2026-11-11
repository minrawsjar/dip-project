import { useEffect, useRef, useState } from "react";

type Props = {
  onPhoto: (b: Blob) => void;
  onClip: (b: Blob) => void;
  onClose: () => void;
  clipSeconds?: number;
};

/** Webcam capture: still photo (canvas -> JPEG) and a short clip (MediaRecorder -> webm) for liveness. */
export function Webcam({ onPhoto, onClip, onClose, clipSeconds = 4 }: Props) {
  const video = useRef<HTMLVideoElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [recording, setRecording] = useState(0);

  useEffect(() => {
    let cancelled = false;
    navigator.mediaDevices
      ?.getUserMedia({ video: { width: 1280, height: 720, facingMode: "user" }, audio: false })
      .then((s) => {
        if (cancelled) return s.getTracks().forEach((t) => t.stop());
        stream.current = s;
        if (video.current) video.current.srcObject = s;
      })
      .catch((e: Error) => setError(e.message || "camera unavailable"));
    return () => {
      cancelled = true;
      stream.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const photo = () => {
    const v = video.current;
    if (!v || !v.videoWidth) return;
    const c = document.createElement("canvas");
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext("2d")!.drawImage(v, 0, 0);
    c.toBlob((b) => b && onPhoto(b), "image/jpeg", 0.92);
  };

  const record = () => {
    if (!stream.current) return;
    const mime = ["video/webm;codecs=vp8", "video/webm", "video/mp4"].find((m) => MediaRecorder.isTypeSupported(m));
    const rec = new MediaRecorder(stream.current, mime ? { mimeType: mime } : undefined);
    const chunks: Blob[] = [];
    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    rec.onstop = () => {
      setRecording(0);
      onClip(new Blob(chunks, { type: rec.mimeType }));
    };
    rec.start();
    let left = clipSeconds;
    setRecording(left);
    const t = setInterval(() => {
      left -= 1;
      setRecording(left);
      if (left <= 0) {
        clearInterval(t);
        rec.stop();
      }
    }, 1000);
  };

  return (
    <div className="card stack webcam">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h3>Webcam</h3>
        <button className="ghost" onClick={onClose}>close</button>
      </div>
      {error ? (
        <div className="error">Camera: {error}</div>
      ) : (
        <video ref={video} autoPlay muted playsInline />
      )}
      <div className="row">
        <button className="primary" onClick={photo} disabled={!!error || recording > 0}>
          Take selfie
        </button>
        <button className="ghost" onClick={record} disabled={!!error || recording > 0}>
          {recording ? `Recording… ${recording}s` : `Record ${clipSeconds}s clip (blink!)`}
        </button>
      </div>
      <p className="muted small" style={{ margin: 0 }}>
        Frames stay in this browser until you press Verify; the server keeps nothing after responding.
      </p>
    </div>
  );
}
