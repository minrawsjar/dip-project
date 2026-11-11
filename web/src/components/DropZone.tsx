import { useEffect, useRef, useState } from "react";

type Props = {
  label: string;
  hint?: string;
  accept: string;
  value: Blob | null;
  onChange: (b: Blob | null) => void;
};

/** Click-or-drop file input with an image/video preview. */
export function DropZone({ label, hint, accept, value, onChange }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const [url, setUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!value) return setUrl(null);
    const u = URL.createObjectURL(value);
    setUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [value]);

  const isVideo = value?.type.startsWith("video");
  return (
    <div className="stack" style={{ gap: 6 }}>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <strong>{label}</strong>
        {value && (
          <button className="ghost small" onClick={() => onChange(null)} style={{ padding: "2px 8px" }}>
            clear
          </button>
        )}
      </div>
      <div
        className={`drop${over ? " over" : ""}`}
        role="button"
        tabIndex={0}
        aria-label={`Choose ${label}`}
        onClick={() => input.current?.click()}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          const f = e.dataTransfer.files[0];
          if (f) onChange(f);
        }}
      >
        {url ? (
          isVideo ? <video src={url} muted autoPlay loop playsInline /> : <img src={url} alt={label} />
        ) : (
          <span className="muted small">{hint ?? "Drop a file or click to browse"}</span>
        )}
        <input
          ref={input}
          type="file"
          accept={accept}
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) onChange(f);
            e.target.value = "";
          }}
        />
      </div>
    </div>
  );
}
