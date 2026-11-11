import { useEffect, useRef, useState } from "react";
import { api, type LabResult, type Sample } from "../api";
import { Histogram } from "../components/Charts";
import { KeyValues } from "../components/DataTable";
import { DropZone } from "../components/DropZone";
import { Gallery } from "../components/Gallery";

type Param =
  | { name: string; kind: "select"; options: string[]; value: string }
  | { name: string; kind: "range"; min: number; max: number; step: number; value: number };

type Op = { group: string; label: string; api: string; fixed?: Record<string, unknown>; concept: string; matlab: string; params: Param[] };

const sel = (name: string, options: string[], value = options[0]): Param => ({ name, kind: "select", options, value });
const rng = (name: string, min: number, max: number, step: number, value: number): Param => ({ name, kind: "range", min, max, step, value });
const INTERPS = ["nearest", "bilinear", "bicubic", "lanczos"];

// Every operation maps to one course concept and its MATLAB equivalent (CLAUDE.md §10).
const OPS: Record<string, Op> = {
  enhance: { group: "Point & histogram", label: "Intensity transform / equalisation", api: "enhance",
    concept: "Contrast stretch, log, gamma, global histeq, CLAHE. Mode hsv_v enhances only V and preserves hue; rgb equalises each channel and shifts hue.",
    matlab: "imadjust, stretchlim, histeq, adapthisteq, rgb2hsv", params: [sel("method", ["histeq", "clahe", "stretch", "gamma", "log", "none"]), sel("mode", ["hsv_v", "rgb", "gray"])] },
  quantize: { group: "Point & histogram", label: "Quantization (bit depth)", api: "degrade", fixed: { op: "quantize_bits" },
    concept: "Fewer grey levels → false contouring.", matlab: "bitshift / uint8 reduction", params: [rng("bits", 1, 8, 1, 3)] },
  lowlight: { group: "Point & histogram", label: "Brightness / contrast (low light)", api: "degrade", fixed: { op: "brightness_contrast" },
    concept: "out = α·in + β.", matlab: "imadjust / arithmetic", params: [rng("alpha", 0.1, 2, 0.05, 0.4), rng("beta", -100, 100, 1, -20)] },
  filter: { group: "Spatial", label: "Spatial filter", api: "filter",
    concept: "Mean/Gaussian smooth, median/mode remove impulse noise, Laplacian/LoG respond to edges, unsharp sharpens.",
    matlab: "imfilter, fspecial, medfilt2", params: [sel("name", ["median", "mean", "gaussian", "mode", "laplacian", "log", "unsharp"]), rng("k", 3, 15, 2, 3)] },
  noise: { group: "Spatial", label: "Add noise", api: "degrade", fixed: { op: "add_noise" },
    concept: "Gaussian vs salt-and-pepper noise; try a median filter afterwards.", matlab: "imnoise", params: [sel("kind", ["salt_pepper", "gaussian"]), rng("level", 0, 0.3, 0.01, 0.05)] },
  resize: { group: "Spatial", label: "Resolution + interpolation", api: "degrade", fixed: { op: "downsample" },
    concept: "Downsample, then restore with each interpolation kernel.", matlab: "imresize(..., 'nearest'|'bilinear'|'bicubic')", params: [rng("scale", 0.05, 1, 0.05, 0.2), sel("interp", INTERPS, "bicubic")] },
  edges: { group: "Segmentation", label: "Edge detection", api: "edges",
    concept: "Sobel/Prewitt gradient magnitude with MATLAB's automatic cutoff vs Canny (NMS + hysteresis).", matlab: "edge(I, 'sobel'|'prewitt'|'canny')", params: [sel("method", ["canny", "sobel", "prewitt"])] },
  threshold: { group: "Segmentation", label: "Thresholding", api: "threshold",
    concept: "Global Otsu vs local adaptive mean/Gaussian; adaptive wins under uneven light.", matlab: "graythresh, imbinarize(I, 'adaptive')",
    params: [sel("method", ["otsu", "adaptive_mean", "adaptive_gaussian"]), rng("block", 3, 101, 2, 31), rng("c", -20, 30, 1, 10)] },
  morphology: { group: "Segmentation", label: "Morphology on Otsu mask", api: "morphology",
    concept: "Erosion, dilation, opening, closing, gradient, top-hat.", matlab: "imerode, imdilate, imopen, imclose", params: [sel("op", ["open", "close", "erode", "dilate", "gradient", "tophat"]), rng("k", 1, 21, 1, 3)] },
  jpeg: { group: "Degradation", label: "JPEG compression", api: "degrade", fixed: { op: "jpeg" }, concept: "8×8 DCT quantisation → blocking.", matlab: "imwrite(..., 'Quality', q)", params: [rng("quality", 1, 100, 1, 15)] },
  blur: { group: "Degradation", label: "Gaussian blur", api: "degrade", fixed: { op: "gaussian_blur" }, concept: "Defocus; watch the Laplacian variance drop in the quality gate.", matlab: "imgaussfilt", params: [rng("sigma", 0, 10, 0.5, 2)] },
  motion: { group: "Degradation", label: "Motion blur", api: "degrade", fixed: { op: "motion_blur" }, concept: "Linear PSF.", matlab: "fspecial('motion', len, theta)", params: [rng("length", 1, 41, 2, 15), rng("angle", 0, 180, 5, 30)] },
  glare: { group: "Degradation", label: "Glare", api: "degrade", fixed: { op: "glare" }, concept: "Additive specular highlight.", matlab: "arithmetic", params: [rng("radius", 0.05, 0.6, 0.05, 0.25), rng("strength", 0, 1.5, 0.05, 0.7), rng("seed", 0, 50, 1, 0)] },
  halftone: { group: "Degradation", label: "Halftone (print)", api: "degrade", fixed: { op: "halftone" }, concept: "Clustered-dot screen of a printed photo.", matlab: "ordered dither", params: [rng("cell", 2, 12, 1, 4)] },
  perspective: { group: "Degradation", label: "Perspective warp", api: "degrade", fixed: { op: "perspective" }, concept: "Random homography.", matlab: "fitgeotrans('projective'), imwarp", params: [rng("max_shift", 0, 0.25, 0.01, 0.08), rng("seed", 0, 50, 1, 0)] },
  quality: { group: "Pipeline stages", label: "Quality gate", api: "quality", concept: "Blur (var of Laplacian), exposure histogram, glare = HSV slicing + cleanup.", matlab: "fspecial('laplacian'), imhist, rgb2hsv, bwareaopen, imfill", params: [] },
  hough: { group: "Pipeline stages", label: "Card corners + rectification", api: "hough",
    concept: "Edges → probabilistic Hough → 4 dominant lines → corners → homography to ID-1.", matlab: "hough, houghpeaks, houghlines, detectHarrisFeatures, imwarp",
    params: [sel("edge", ["canny", "sobel", "prewitt"]), sel("corner", ["hough", "contour", "harris"])] },
  document: { group: "Pipeline stages", label: "Full document + OCR", api: "document", concept: "Locate, validate, rectify, portrait, per-field enhance → denoise → threshold → Tesseract.", matlab: "regionprops, imbinarize, ocr", params: [] },
  face: { group: "Pipeline stages", label: "Face detection + alignment", api: "face", concept: "Baseline: Haar + landmark distances. Improved: YuNet + 5-point similarity + HSV-V contrast stretch.", matlab: "vision.CascadeObjectDetector, fitgeotrans", params: [sel("variant", ["improved", "baseline"])] },
};

const GROUPS = [...new Set(Object.values(OPS).map((o) => o.group))];

export function LabPage() {
  const [img, setImg] = useState<Blob | null>(null);
  const [opKey, setOpKey] = useState("enhance");
  const [vals, setVals] = useState<Record<string, string | number>>({});
  const [res, setRes] = useState<LabResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [samples, setSamples] = useState<Sample[]>([]);
  const abort = useRef<AbortController | null>(null);
  const op = OPS[opKey];

  useEffect(() => {
    api.samples().then(setSamples).catch(() => setSamples([]));
  }, []);

  useEffect(() => {
    setVals(Object.fromEntries(op.params.map((p) => [p.name, p.value])));
  }, [opKey]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!img) return;
    const t = setTimeout(async () => {
      abort.current?.abort();
      const ctl = new AbortController();
      abort.current = ctl;
      setBusy(true);
      setError(null);
      try {
        setRes(await api.lab(img, op.api, { ...op.fixed, ...vals }, ctl.signal));
      } catch (e) {
        if ((e as Error).name !== "AbortError") setError((e as Error).message);
      } finally {
        if (abort.current === ctl) setBusy(false);
      }
    }, 250);
    return () => clearTimeout(t);
  }, [img, vals]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="grid cols-2">
      <aside className="stack">
        <div className="card stack">
          <h2>DIP Lab</h2>
          <DropZone label="Image" accept="image/*" value={img} onChange={setImg} />
          {samples.length > 0 && (
            <div className="chips">
              {samples.slice(0, 3).flatMap((s) => [
                <button key={s.pair_id + "id"} onClick={async () => setImg(await api.blob(s.id))}>ID #{s.pair_id}</button>,
                <button key={s.pair_id + "s"} onClick={async () => setImg(await api.blob(s.selfie))}>selfie #{s.pair_id}</button>,
              ])}
            </div>
          )}
          <label className="field">
            Operation
            <select value={opKey} onChange={(e) => setOpKey(e.target.value)}>
              {GROUPS.map((g) => (
                <optgroup key={g} label={g}>
                  {Object.entries(OPS)
                    .filter(([, o]) => o.group === g)
                    .map(([k, o]) => <option key={k} value={k}>{o.label}</option>)}
                </optgroup>
              ))}
            </select>
          </label>
          {op.params.map((p) =>
            p.kind === "select" ? (
              <label key={p.name} className="field">
                {p.name}
                <select value={String(vals[p.name] ?? p.value)} onChange={(e) => setVals({ ...vals, [p.name]: e.target.value })}>
                  {p.options.map((o) => <option key={o}>{o}</option>)}
                </select>
              </label>
            ) : (
              <label key={p.name} className="field">
                {p.name}: <span className="mono">{vals[p.name] ?? p.value}</span>
                <input type="range" min={p.min} max={p.max} step={p.step} value={Number(vals[p.name] ?? p.value)}
                  onChange={(e) => setVals({ ...vals, [p.name]: Number(e.target.value) })} />
              </label>
            ),
          )}
        </div>
        <div className="card stack small">
          <strong>Concept</strong>
          <span>{op.concept}</span>
          <strong>MATLAB equivalent</strong>
          <code>{op.matlab}</code>
        </div>
      </aside>

      <section className="stack">
        {error && <div className="error">{error}</div>}
        {!img && <div className="card empty">Upload an image (or pick a sample) and choose an operation. Results update as you move the sliders.</div>}
        {res && (
          <>
            <div className="card stack" style={{ opacity: busy ? 0.6 : 1 }}>
              <div className="pair">
                <div className="stack" style={{ gap: 6 }}>
                  <strong>Input</strong>
                  <img src={res.input} alt="input" />
                  <Histogram counts={res.hist_in} label="input histogram" />
                </div>
                <div className="stack" style={{ gap: 6 }}>
                  <strong>Output · {op.label}</strong>
                  <img src={res.output} alt="output" />
                  <Histogram counts={res.hist_out} label="output histogram" />
                </div>
              </div>
            </div>
            {Object.keys(res.info).length > 0 && (
              <div className="card">
                <h3>Measurements</h3>
                <KeyValues data={res.info} />
              </div>
            )}
            {res.extras.length > 0 && (
              <div className="card">
                <h3>Intermediate images</h3>
                <Gallery items={res.extras} />
              </div>
            )}
          </>
        )}
      </section>
    </div>
  );
}
