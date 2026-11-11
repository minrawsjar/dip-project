import { useEffect, useMemo, useState } from "react";
import { api, type Quality, type Sample, type VerifyResult } from "../api";
import { ScoreBar, Timings } from "../components/Charts";
import { DataTable, KeyValues } from "../components/DataTable";
import { DropZone } from "../components/DropZone";
import { Gallery } from "../components/Gallery";
import { Webcam } from "../components/Webcam";

const STAGE_LABEL: Record<string, string> = {
  document: "Document",
  face: "Face (improved)",
  face_baseline: "Face (baseline)",
  liveness: "Liveness",
  quality: "Quality gate",
};

export function VerifyPage() {
  const [selfie, setSelfie] = useState<Blob | null>(null);
  const [idImg, setIdImg] = useState<Blob | null>(null);
  const [clip, setClip] = useState<Blob | null>(null);
  const [cam, setCam] = useState(false);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [res, setRes] = useState<VerifyResult | null>(null);

  useEffect(() => {
    api.samples().then(setSamples).catch(() => setSamples([]));
  }, []);

  const loadSample = async (s: Sample) => {
    setError(null);
    try {
      const [a, b] = await Promise.all([api.blob(s.selfie), api.blob(s.id)]);
      setSelfie(a);
      setIdImg(b);
      setClip(null);
      setRes(null);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const run = async () => {
    if (!selfie || !idImg) return;
    setBusy(true);
    setError(null);
    try {
      setRes(await api.verify(selfie, idImg, clip));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="grid cols-2">
      <aside className="stack">
        <div className="card stack">
          <h2>Inputs</h2>
          <DropZone label="Selfie" accept="image/*" value={selfie} onChange={setSelfie} />
          <DropZone label="ID document photo" accept="image/*" value={idImg} onChange={setIdImg} />
          <DropZone
            label="Selfie clip (optional)"
            accept="video/*"
            value={clip}
            onChange={setClip}
            hint="A 3–10 s clip enables motion + blink liveness"
          />
          <div className="row">
            <button className="ghost" onClick={() => setCam((c) => !c)}>{cam ? "Hide webcam" : "Use webcam"}</button>
          </div>
          <button className="primary" disabled={!selfie || !idImg || busy} onClick={run}>
            {busy ? "Verifying…" : "Verify identity"}
          </button>
        </div>
        {cam && <Webcam onPhoto={setSelfie} onClip={setClip} onClose={() => setCam(false)} />}
        {samples.length > 0 && (
          <div className="card stack">
            <h3>Sample pairs</h3>
            <p className="muted small" style={{ margin: 0 }}>
              Synthetic: an LFW face printed on a mock ID card, then photographed with perspective, glare, blur and JPEG.
            </p>
            <div className="chips">
              {samples.map((s) => (
                <button key={s.pair_id} onClick={() => loadSample(s)} title={s.name}>
                  {s.label} #{s.pair_id}
                </button>
              ))}
            </div>
          </div>
        )}
      </aside>

      <section className="stack">
        {error && <div className="error">{error}</div>}
        {!res && !busy && (
          <div className="card empty">
            <h2>Upload a selfie and an ID photo</h2>
            <p>
              The pipeline runs a quality gate, rectifies the card, crops the portrait, reads the fields, matches faces and
              checks liveness. Every intermediate image is shown below the decision.
            </p>
          </div>
        )}
        {busy && <div className="card empty">Running the pipeline…</div>}
        {res && <Result res={res} />}
      </section>
    </div>
  );
}

function QualityBlock({ title, q }: { title: string; q: Quality }) {
  const s = q.scores;
  return (
    <div className="stack" style={{ gap: 6 }}>
      <div className="row">
        <strong>{title}</strong>
        <span className={`badge ${q.passed ? "ok" : "warn"}`}>{q.passed ? "pass" : "retake advised"}</span>
      </div>
      <KeyValues
        data={{
          "blur (var Laplacian)": s.blur,
          "mean grey": s.mean,
          "dynamic range": s.dynamic_range,
          "glare ratio": s.glare_ratio,
          ...(s.face_area != null ? { "face area": s.face_area } : {}),
        }}
      />
      {q.reasons.map((r) => (
        <span key={r} className="small" style={{ color: "var(--warn)" }}>• {r}</span>
      ))}
    </div>
  );
}

function Result({ res }: { res: VerifyResult }) {
  const stages = useMemo(() => [...new Set(res.images.map((i) => i.stage))], [res]);
  const [stage, setStage] = useState(stages[0] ?? "document");
  const f = res.fusion;
  const d = res.document;
  const lv = res.liveness;

  return (
    <>
      <div className={`decision ${res.decision}`}>
        <span className="big">{res.decision}</span>
        <div className="stack" style={{ gap: 4 }}>
          <span className="small">
            {f.method === "logistic_regression"
              ? `Logistic-regression fusion, P(genuine) = ${f.prob?.toFixed(3)} (baseline AND rule: ${f.baseline_decision})`
              : "Fixed per-module thresholds with AND logic (train `ekyc eval fusion` for logistic regression)"}
          </span>
          <div className="row">
            {Object.entries(f.checks).map(([k, ok]) => (
              <span key={k} className={`badge ${ok ? "ok" : "bad"}`}>{ok ? "✓" : "✗"} {k}</span>
            ))}
          </div>
        </div>
      </div>

      <div className="grid cols-auto">
        <div className="card stack">
          <h3>Face match</h3>
          <ScoreBar label={`Improved · ${res.face.method} (cosine)`} value={res.face.score} threshold={res.face.threshold} min={-0.2} max={1} />
          {res.face_baseline && (
            <ScoreBar
              label="Baseline · Haar + landmark distances (−L2)"
              value={res.face_baseline.score}
              threshold={res.face_baseline.threshold}
              min={-0.6}
              max={0}
            />
          )}
          <div className="row small">
            <span className={`badge ${res.face.selfie_found ? "ok" : "bad"}`}>selfie face</span>
            <span className={`badge ${res.face.portrait_found ? "ok" : "bad"}`}>ID portrait</span>
          </div>
        </div>

        <div className="card stack">
          <h3>Liveness</h3>
          <ScoreBar label={`Combined (${lv.method})`} value={lv.liveness_score} threshold={0.5} />
          {Object.entries(lv.scores).map(([k, v]) => (
            <ScoreBar key={k} label={k} value={v} />
          ))}
          <span className="muted small">
            {res.inputs.frames > 2
              ? `${res.inputs.frames} frames · ${lv.features.blink_count ?? 0} blink(s)`
              : "Single image: only man-made cues apply. Add a clip for motion + blink."}
          </span>
          {lv.reasons.map((r) => (
            <span key={r} className="small" style={{ color: "var(--warn)" }}>• {r}</span>
          ))}
        </div>

        <div className="card stack">
          <h3>Document</h3>
          <div className="row">
            <span className={`badge ${d.found ? "ok" : "bad"}`}>{d.found ? "card located" : "no card"}</span>
            <span className={`badge ${d.validity.valid ? "ok" : "warn"}`}>{d.validity.valid ? "shape valid" : "shape rules failed"}</span>
            <span className="badge neutral">{d.corner_method}</span>
          </div>
          <KeyValues data={d.validity.features} />
          {d.validity.reasons.map((r) => (
            <span key={r} className="small" style={{ color: "var(--warn)" }}>• {r}</span>
          ))}
        </div>

        <div className="card stack">
          <h3>Quality gate</h3>
          <QualityBlock title="Selfie" q={res.quality.selfie} />
          <QualityBlock title="ID photo" q={res.quality.id} />
        </div>
      </div>

      <div className="grid" style={{ gridTemplateColumns: "minmax(0,1.4fr) minmax(0,1fr)" }}>
        <div className="card">
          <h3>OCR fields</h3>
          <DataTable
            columns={["field", "text", "confidence"]}
            rows={Object.entries(d.fields).map(([k, v]) => ({ field: k, text: v.text || "—", confidence: v.conf }))}
          />
        </div>
        <div className="card">
          <h3>Per-stage runtime</h3>
          <Timings ms={res.timings_ms} />
        </div>
      </div>

      <div className="card stack">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h3>Intermediate images</h3>
          <div className="chips">
            {stages.map((s) => (
              <button key={s} aria-pressed={s === stage} onClick={() => setStage(s)}>
                {STAGE_LABEL[s] ?? s}
              </button>
            ))}
          </div>
        </div>
        <Gallery
          items={res.images
            .filter((i) => i.stage === stage)
            .map((i) => ({ label: i.key, src: i.src, caption: [i.group, i.label].filter(Boolean).join(" / ") }))}
        />
      </div>
    </>
  );
}
