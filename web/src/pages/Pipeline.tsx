import { DataTable } from "../components/DataTable";

type Box = { x: number; y: number; w: number; label: string; kind?: "dip" | "learned" };
const H = 34;

// Block diagram (CLAUDE.md §2). Blue = classical DIP stage, amber = learned model, grey = I/O / logic.
const lanes: { y: number; name: string; boxes: Box[] }[] = [
  { y: 30, name: "SELFIE", boxes: [
    { x: 10, y: 30, w: 95, label: "Selfie / clip" },
    { x: 125, y: 30, w: 100, label: "Quality gate", kind: "dip" },
    { x: 245, y: 30, w: 110, label: "Detect (YuNet)", kind: "learned" },
    { x: 375, y: 30, w: 115, label: "5-pt similarity", kind: "dip" },
    { x: 510, y: 30, w: 110, label: "HSV-V stretch", kind: "dip" },
    { x: 640, y: 30, w: 115, label: "SFace embed", kind: "learned" },
  ] },
  { y: 110, name: "LIVENESS", boxes: [
    { x: 125, y: 110, w: 130, label: "Frame diff + morph", kind: "dip" },
    { x: 275, y: 110, w: 95, label: "EAR blink", kind: "dip" },
    { x: 390, y: 110, w: 165, label: "Man-made cues (Hough/FFT)", kind: "dip" },
    { x: 595, y: 110, w: 95, label: "CNN (opt.)", kind: "learned" },
  ] },
  { y: 190, name: "ID", boxes: [
    { x: 10, y: 190, w: 95, label: "ID photo" },
    { x: 125, y: 190, w: 100, label: "Quality gate", kind: "dip" },
    { x: 245, y: 190, w: 100, label: "Colour edges", kind: "dip" },
    { x: 365, y: 190, w: 100, label: "Hough lines", kind: "dip" },
    { x: 485, y: 190, w: 100, label: "Corners", kind: "dip" },
    { x: 605, y: 190, w: 110, label: "Homography", kind: "dip" },
    { x: 735, y: 190, w: 95, label: "Validity rules", kind: "dip" },
  ] },
  { y: 270, name: "OCR", boxes: [
    { x: 365, y: 270, w: 90, label: "Field ROIs" },
    { x: 475, y: 270, w: 90, label: "Enhance", kind: "dip" },
    { x: 585, y: 270, w: 90, label: "Denoise", kind: "dip" },
    { x: 695, y: 270, w: 90, label: "Threshold", kind: "dip" },
    { x: 805, y: 270, w: 90, label: "Morphology", kind: "dip" },
    { x: 915, y: 270, w: 90, label: "Tesseract", kind: "learned" },
  ] },
];

const right = (b: Box) => [b.x + b.w, b.y + H / 2] as const;
const left = (b: Box) => [b.x, b.y + H / 2] as const;

export function PipelinePage() {
  const all = lanes.flatMap((l) => l.boxes);
  const chain = lanes.flatMap((l) => l.boxes.slice(0, -1).map((b, i) => [b, l.boxes[i + 1]] as const));
  const find = (label: string, y: number) => all.find((b) => b.label === label && b.y === y)!;
  const homog = find("Homography", 190);
  const fieldRoi = find("Field ROIs", 270);
  const align = find("5-pt similarity", 30);
  const fusion: Box = { x: 880, y: 100, w: 165, label: "Fusion (LR)" };
  const decision: Box = { x: 880, y: 175, w: 165, label: "ACCEPT · REVIEW · REJECT" };

  return (
    <div className="stack">
      <div className="card diagram">
        <h2>Pipeline</h2>
        <p className="muted small">
          <span className="badge" style={{ background: "var(--accent-soft)", color: "var(--accent)" }}>classical DIP</span>{" "}
          <span className="badge warn">learned model</span>{" "}
          <span className="badge neutral">I/O · logic</span> — every stage dumps its intermediate image (CLI <code>--debug-dir</code>, web Verify gallery).
        </p>
        <svg viewBox="0 0 1060 320" role="img" aria-label="eKYC pipeline block diagram">
          <defs>
            <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
              <path d="M0,0 L10,5 L0,10 z" fill="currentColor" style={{ color: "var(--muted)" }} />
            </marker>
          </defs>
          {chain.map(([a, b], i) => (
            <path key={i} className="edge" d={`M${right(a)[0]},${right(a)[1]} L${left(b)[0]},${left(b)[1]}`} />
          ))}
          {/* selfie quality gate -> liveness, homography -> portrait -> align, homography -> field ROIs */}
          <path className="edge" d={`M175,64 L175,110`} />
          <path className="edge" d={`M${homog.x + 30},${homog.y} C${homog.x + 30},165 575,170 575,150 L575,90 C575,76 ${align.x + 60},78 ${align.x + 64},64`} strokeDasharray="4 3" />
          <text x={homog.x + 36} y={170}>portrait crop</text>
          <path className="edge" d={`M${homog.x + 55},${homog.y + H} C${homog.x + 55},250 ${fieldRoi.x + 45},250 ${fieldRoi.x + 45},270`} />
          <path className="edge" d="M755,47 C840,47 860,90 880,112" />
          <path className="edge" d="M690,127 L880,120" />
          <path className="edge" d="M1005,270 C1050,230 1050,160 1030,134" />
          <path className="edge" d="M830,207 C855,200 860,150 880,128" />
          <path className="edge" d={`M962,${fusion.y + H} L962,${decision.y}`} />
          {lanes.map((l) => (
            <text key={l.name} x={0} y={l.y - 6} className="lane">{l.name}</text>
          ))}
          {[...all, fusion, decision].map((b) => (
            <g key={b.label + b.y}>
              <rect x={b.x} y={b.y} width={b.w} height={H} rx={7} className={b.kind ?? "box"} />
              <text x={b.x + b.w / 2} y={b.y + H / 2 + 4} textAnchor="middle">{b.label}</text>
            </g>
          ))}
        </svg>
      </div>

      <div className="card">
        <h3>Course concept → code</h3>
        <DataTable columns={["concept", "module", "MATLAB"]} rows={CONCEPTS.map(([concept, module, matlab]) => ({ concept, module, MATLAB: matlab }))} />
      </div>
    </div>
  );
}

const CONCEPTS = [
  ["Quantization, bit depth", "degrade.quantize_bits", "bitshift / uint8 reduction"],
  ["Interpolation", "degrade.downsample, face.align.upscale", "imresize(..., 'nearest'/'bilinear'/'bicubic')"],
  ["Image subtraction", "liveness.motion.change_masks", "imabsdiff"],
  ["Threshold + cleanup", "liveness.motion, quality.glare_mask", "imbinarize, bwareaopen, imfill"],
  ["Masking, logical ops", "liveness.motion.region_masks, quality.box_mask", ".*, &"],
  ["Edge detection", "document.edges", "edge(I,'sobel'/'prewitt'/'canny')"],
  ["Hough transform", "document.hough", "hough, houghpeaks, houghlines"],
  ["Harris corners", "document.corners.harris_refine", "detectHarrisFeatures"],
  ["Projective rectification", "document.corners.warp", "fitgeotrans('projective'), imwarp"],
  ["Region properties", "document.validity", "regionprops, bwconvhull"],
  ["Rule-based classification", "document.validity, liveness.manmade.rule_classify", "Lab 5 shape rules"],
  ["Contrast stretch, log, gamma", "enhance.contrast_stretch / log_transform / gamma", "imadjust, stretchlim"],
  ["Histogram equalisation, CLAHE", "enhance.histeq (from scratch), enhance.clahe", "histeq, adapthisteq"],
  ["HSV colour preservation", "enhance.apply(mode='hsv_v')", "rgb2hsv, hsv2rgb"],
  ["Spatial filters", "filters.*", "imfilter, fspecial, medfilt2"],
  ["Laplacian / LoG", "quality.blur_score, filters.log", "fspecial('laplacian'/'log')"],
  ["Thresholding (global/adaptive)", "document.ocr.threshold", "graythresh, imbinarize(...,'adaptive')"],
  ["Morphology", "document.ocr.morph, document.corners.region_mask", "imopen, imclose"],
  ["Landmark distances", "face.geometric", "face line diagram lecture"],
  ["Man-made vs natural features", "liveness.manmade", "17 Sep lecture"],
  ["Fixed kernels → CNN", "face.embed (SFace), liveness.cnn", "Aug–Sep CNN lectures"],
];
