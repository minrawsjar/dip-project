/** Small SVG/CSS charts: grey-level histogram, score bar with threshold marker, stage timing bars. */

export function Histogram({ counts, label }: { counts: number[]; label?: string }) {
  const max = Math.max(1, ...counts);
  return (
    <svg className="hist" viewBox="0 0 256 64" preserveAspectRatio="none" role="img" aria-label={label ?? "histogram"}>
      {counts.map((c, i) => (
        <rect key={i} x={i} y={64 - (c / max) * 64} width={1} height={(c / max) * 64} />
      ))}
    </svg>
  );
}

type ScoreProps = {
  label: string;
  value: number | null | undefined;
  threshold?: number;
  min?: number;
  max?: number;
  format?: (v: number) => string;
  higherIsBetter?: boolean;
};

export function ScoreBar({ label, value, threshold, min = 0, max = 1, format = (v) => v.toFixed(3), higherIsBetter = true }: ScoreProps) {
  const pct = (v: number) => `${Math.max(0, Math.min(1, (v - min) / (max - min))) * 100}%`;
  const pass = value == null || threshold == null ? null : higherIsBetter ? value >= threshold : value <= threshold;
  const colour = pass == null ? "var(--muted)" : pass ? "var(--ok)" : "var(--bad)";
  return (
    <div className="score">
      <span className="small">{label}</span>
      <span className="mono small" style={{ color: colour }}>
        {value == null ? "n/a" : format(value)}
        {threshold != null && <span className="muted"> / {format(threshold)}</span>}
      </span>
      <div className="track">
        {value != null && <div className="fill" style={{ width: pct(value), background: colour }} />}
        {threshold != null && <div className="thr" style={{ left: pct(threshold) }} title={`threshold ${threshold}`} />}
      </div>
    </div>
  );
}

export function Timings({ ms }: { ms: Record<string, number> }) {
  const max = Math.max(1, ...Object.values(ms));
  const total = Object.values(ms).reduce((a, b) => a + b, 0);
  return (
    <div className="timing">
      {Object.entries(ms).map(([k, v]) => (
        <div key={k} style={{ display: "contents" }}>
          <span className="muted">{k}</span>
          <div className="bar" style={{ width: `${(v / max) * 100}%` }} />
          <span className="mono" style={{ textAlign: "right" }}>{v.toFixed(0)} ms</span>
        </div>
      ))}
      <span className="muted">total</span>
      <span />
      <strong className="mono" style={{ textAlign: "right" }}>{total.toFixed(0)} ms</strong>
    </div>
  );
}
