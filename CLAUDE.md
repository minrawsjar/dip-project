# CLAUDE.md — eKYC Identity Verification (DIP Course Project, Topic 029)

Handover for Claude Code. Read this file fully before writing any code. It defines the goal, grading constraints, architecture, module specs, data, experiments and conventions. When something here conflicts with a later instruction from the user, the user wins; update this file to match.

---

## 1. Context

- **Course:** Digital Image Processing (DIP), B.Tech IT, RGIPT, 2026. Two-student team.
- **Topic:** HSR-AIML-029, *Multimodal eKYC Identity Verification with Face, Document and Liveness Signals*.
- **Language:** Python 3.11 (managed with `uv`). The course labs were in MATLAB; the report names the MATLAB equivalent of each operation (see Section 10), so keep function boundaries aligned with lab concepts.
- **Final demo date:** TBD (ask the user if it affects scheduling).

### What the graders require (from the course brief; non-negotiable)

1. Image/video data in, with a defined problem and dataset source.
2. At least one meaningful DIP stage. This project uses many, deliberately.
3. **Baseline vs improved/alternative approach**, with quantitative metrics.
4. **Intermediate images** and final outputs shown as figures.
5. Success and failure case analysis.
6. Reproducible code + README with run instructions + dataset description.
7. Block diagram of the pipeline showing where DIP operations sit.
8. Both team members must understand the whole pipeline.

Explicitly **not** sufficient: a general app, a one-call model demo. Image processing and evaluation must stay central.

**Web UI (user decision, 2026-11-12):** the user asked for a proper frontend + backend. There is now a FastAPI backend (`src/ekyc/api.py`) and a React/Vite/TypeScript frontend (`web/`). It is a demo and inspection layer over the same pure pipeline functions: it shows every intermediate image, an interactive DIP lab and the experiment results. It must never contain DIP logic of its own, and no cloud, database or auth is added.

### Core thesis

> Classical DIP stages taught in class measurably improve identity verification under real capture conditions (glare, blur, perspective, low light, print/compression degradation), and learned models improve further on top of them.

Every module therefore has a **classical baseline built from lab operations** and a **learned or enhanced improvement**, plus ablations that isolate each DIP stage's contribution.

---

## 2. Pipeline

```
Selfie video/image ─► Quality gate ─► Face detect + align ─► Photometric norm (HSV-V) ─► Embedding ─┐
                         │                                                                          ├─► face_match_score
                         └────────► Liveness: motion (frame diff, EAR blink) + man-made cues + CNN ─┼─► liveness_score
ID image ─► Quality gate ─► Edges ─► Hough lines ─► Corners ─► Homography warp ─► Card validity     │
                                                                  │                                 │
                                                                  ├─► Portrait crop ────────────────┘
                                                                  └─► Field ROIs ─► Enhance ─► Threshold ─► Morphology ─► OCR ─► ocr_score
                                                                                                    │
                                              Fusion (logistic regression) ─► ACCEPT / REJECT / REVIEW
```

Every stage dumps its intermediate image through `io.Debug` (CLI `--debug-dir`, web Verify gallery). These dumps become report figures. Rendered diagram: `report/block_diagram.svg`.

---

## 3. Repository layout (as built)

```
CLAUDE.md  README.md  pyproject.toml  uv.lock  .gitignore
configs/default.yaml        # all thresholds, kernel sizes, paths, model URLs
scripts/
  download_lfw.py           # sklearn downloader -> data/raw/sklearn/lfw_home
  download_midv2020.py      # FTP + MD5 verify -> data/raw/midv2020/{templates,photo}
  prepare_liveness.py       # own-capture index + subject-disjoint split
  make_degraded_pairs.py    # synthetic printed-card captures + doc-portrait/selfie pairs
src/ekyc/
  io.py                     # load_config, with_overrides, imread/imwrite, read_video, dtype helpers, Debug, write_results
  models.py                 # model_path(): download-on-first-use into models/
  quality.py  enhance.py  filters.py  degrade.py  metrics.py  viz.py  fusion.py
  document/                 # edges, hough, corners (+warp), validity, ocr, template (MIDV loader + synthetic cards), __init__ (locate/process/process_baseline)
  face/                     # detect (haar, yunet), landmarks (MediaPipe Tasks), align, geometric, embed (SFace), __init__ (represent/score)
  liveness/                 # motion, blink (EAR), manmade, cnn (optional torch), __init__ (analyze)
  eval/                     # datasets, eval_face, eval_document, eval_liveness, eval_fusion, ablations (E1-E14)
  pipeline.py               # verify(): timed stages, shared Debug
  api.py                    # FastAPI: /api/health, /api/config, /api/verify, /api/lab, /api/experiments, /api/samples
  cli.py                    # ekyc verify | eval | ablate | serve | models
web/                        # Vite + React + TS; pages: Verify, DIP Lab, Experiments, Pipeline
report/block_diagram.svg
tests/                      # metrics, degradation determinism, enhance, geometry, API smoke
data/ models/ results/      # gitignored (results/final/ is versioned)
```

Small modules are flat files; `document/`, `face/`, `liveness/` are packages because they have several stages each. Eval code lives inside the package so the CLI can import it.

---

## 4. Module specifications

Each module exposes a pure function per stage (`np.ndarray -> np.ndarray` or `-> dict`), a baseline variant and an improved variant selected by config. No hidden global state (model objects are cached by path only).

### 4.1 Quality gate (`quality.py`)

| Check | Method | Course link |
|---|---|---|
| Blur | Variance of Laplacian on grayscale | Lab 7 (Laplacian) |
| Exposure | Histogram: mean, percent of pixels in bottom/top 2% bins, dynamic range | Lab 6 (histograms) |
| Glare | HSV intensity slicing: V > t_v AND S < t_s, then Gaussian smoothing, `remove_small_objects`, fill holes; glare ratio inside ROI | Lab 3 cleanup, 17 Sep HSV |
| Face size | Detected face box area / image area | — |

Output: `{passed: bool, scores: {...}, reasons: [...]}`. Glare ROI = face box (selfie) or detected card quad (ID); whole-frame glare falsely fires on white walls/shirts. Glare = V > 250 (clipped), not 230 (pale card stock). Thresholds calibrated on val folds 0–6 so ~95% of undegraded captures pass each check (`blur_min` 15, glare `ratio_max` 0.30). The first guesses (60 / 0.08) rejected 71% of ordinary captures. Calibrated E13: 88% of ordinary captures pass, 100% of blurred/underexposed and 65% of heavy-glare captures are caught.

### 4.2 Document (`document/`)

**Baseline:** grayscale → Canny → largest external contour → bounding box crop (no perspective correction) → Otsu threshold → Tesseract.

**Improved:**
1. Gaussian blur.
2. Edge detection: Sobel, Prewitt (MATLAB auto cutoff `mag² > 4·mean(mag²)`), Canny. **Colour edges** (`document.color_edges`): OR of per-BGR-channel edge maps, because grey conversion can erase chromatic card/background boundaries (found on synthetic data: grey header bar on a red desk).
3. Probabilistic Hough lines → near-horizontal / near-vertical groups (±30°; diagonal text strokes dropped) → length-weighted 1-D 2-means per group → outermost line among those ≥ 0.4 × the longest (inner header bars/text rows are long too; 0.7 picked the header bar's lower edge on 30% of cards) → least-squares refit of each side on edge pixels within 3 px (Hough is only bin-accurate) → intersect to 4 corners. **Region veto:** if the closed-edge blob passes the validity rules and the Hough quad's IoU with it is < 0.9 (or its side aspect is implausible), use the contour quad instead. Tuned on every third synthetic card: IoU>0.9 went 65% → 92%, mean corner error 22 → 9 canonical px. Contour corners alone are more precise when right (median 3 px) but fail badly on ~20% of cards. ρθ accumulator figure in E7.
4. Alternative corner path: contour `approxPolyDP` (+ Harris snap). Corner error is measured in canonical card pixels through the GT homography.
5. Homography warp to canonical ID-1 (856 × 540 px, aspect 1.586) or to the MIDV template size.
6. Card validity rules on the largest closed-edge blob: aspect, rectangularity, solidity, perimeter²/area, area fraction.
7. Field ROIs: MIDV template annotations, synthetic-card ground truth, or the generic layout in config.
8–11. Per-ROI enhance → denoise → threshold → morphology. Defaults (stretch, median, Otsu, none) picked on validation seeds; E4/E8/E9 show when the alternatives help.
12. Tesseract `--psm 7` per field.
13. Portrait: YuNet on the rectified card (layout slot as fallback).

**MIDV-2020 facts (verified 2026-11-12):** `ftp://smartengines.com/midv-2020/dataset/` (anonymous), CC BY-SA 2.5. VIA v2 JSON. `templates/annotations/<code>.json` = field rects with `field_name` + `value`; `photo/annotations/<code>.json` = `doc_quad` polygon + `face` rect; same filename ⇒ same document. templates.tar 0.86 GB, photo.tar 4.0 GB; MD5s in `scripts/download_midv2020.py`.

### 4.3 Face match (`face/`)

**Baseline:** Haar cascade + normalised landmark distances (7 distances / face width from the MediaPipe mesh), score = −L2.

**Improved:** YuNet (`cv2.FaceDetectorYN`) → upscale small portraits (interp is config, ablated in E2) → 5-point similarity to the ArcFace 112×112 template → HSV-V contrast stretch → **SFace** (`cv2.FaceRecognizerSF`, OpenCV Zoo; margin-based hypersphere loss in the ArcFace family), cosine similarity. SFace replaces insightface `buffalo_l` because it runs through OpenCV DNN with no extra runtime or 280 MB download. Swap in ArcFace via onnxruntime only if a stronger embedding is needed.

**Descreening fallback** (`face.descreen_sigma: 1.5`): if no face is found, Gaussian low-pass the image and retry. Every synthetic false reject in the failure gallery was a halftoned print the detector missed. σ=1.5 recovered 17/17 and broke none of 150 working portraits. It is off in `face.BASELINE`; E12 has a "no descreen" row.

Photometric choice (val folds 0–6 EER): none / stretch-V / gamma-V 5.7%, **CLAHE-V 7.1%**. CLAHE hurts a learned embedding trained on natural image statistics. Stretch-V is the default because it ties on normal exposure and helps under low light (E4). E12 shows **alignment** is the big DIP win (LFW EER 6.5% → 1.0%).

Thresholds `face.match_threshold` / `geometric_threshold` / `fusion.fixed.face` = EER points on synthetic folds 0–6 (val); folds 7–9 are test. `ekyc eval face` writes them to `results/face_thresholds.csv` and logs any mismatch with the config; copy them into `configs/default.yaml` after changing anything in the face path.

### 4.4 Liveness (`liveness/`)

1. **Motion (Lab 3):** grey, blur, `absdiff`, threshold (E10 sweeps 30/50/70/100), `remove_small_objects`, fill holes; energy in eye+mouth vs rest-of-face vs background masks built from mesh landmarks; ratio + face/background rigidity.
2. **EAR blink:** relative threshold `ratio × median(EAR)` for k frames; continuous score = depth of deepest dip.
3. **Man-made cues:** on the band around the face box: Hough line count/length, edge density, entropy, Laplacian variance, gradient mean; on the face crop: high-frequency FFT energy, HSV stats. Rule classifier baseline, LR/SVM in eval.
4. **CNN:** MobileNetV3-Small fine-tuned on face crops (`--extra cnn`); fused model = CNN embedding + handcrafted → LR.

`liveness.analyze` uses `models/liveness_lr.joblib` when it exists (clips only), else the rule mean. Report a cross-dataset test if a second dataset is available.

### 4.5 Fusion (`fusion.py`)

Logistic regression on `[face_score, liveness_score, ocr_conf, selfie_blur, id_blur, id_glare]`. Baseline: fixed thresholds with AND logic (face or liveness fail ⇒ REJECT, only OCR/quality fail ⇒ REVIEW). Two probability thresholds give ACCEPT / REVIEW / REJECT. With no joint dataset, `eval_fusion` assembles attempts assuming module independence; without own-capture liveness it fixes liveness = 1 and says so in the output.

---

## 5. Data

Do not commit data. Download scripts write to `data/raw/`.

| Purpose | Dataset | Notes |
|---|---|---|
| ID documents | **MIDV-2020** | See §4.2 facts. Citation: Bulatov et al., Computer Optics 46(2), 2022, arXiv:2107.00396. |
| Face verification | **LFW** | `scripts/download_lfw.py` (sklearn downloader); pairs parsed from the official `pairs.txt` 10-fold file; balanced subsets spread across folds. |
| Doc-portrait vs selfie pairs + document benchmark | **Synthetic**, from LFW | `scripts/make_degraded_pairs.py`: LFW face A on a rendered mock ID-1 card → halftone print → random homography onto a textured desk → glare → gamma → blur → JPEG; portrait cropped back via GT quad; paired with B. Manifest has GT quad, field values/rects and every degradation parameter. Seeded. |
| Liveness (video) | **Own consented capture**, primary | `data/own_capture/<subject_id>/{live,print,replay}/*.mp4`, ≥10 subjects, signed consent, subject-disjoint splits (`scripts/prepare_liveness.py`). |
| Liveness (public) | Replay-Attack or MSU-MFSD (license required) | Requested through the course instructor; optional until access is granted. |

---

## 6. Degradation generator (`degrade.py`)

Every function returns `(image, params)`; seeded ops (`add_noise`, `glare`, `perspective`) are bit-exact reproducible (tested). `chain()` applies a list with seed + i per step; `print_capture()` is the CLAUDE.md chain (halftone → perspective → glare → JPEG).

---

## 7. Metrics (`metrics.py`)

From scratch with unit tests (AUC cross-checked against sklearn): ROC, AUC, EER (interpolated), FAR/FRR, TAR@FAR, ISO 30107-3 APCER (per attack type, worst case)/BPCER/ACER, Levenshtein CER/WER, quad IoU (`cv2.intersectConvexConvex`), corner error, bootstrap CI. Failure-to-acquire faces are scored as rejects and reported as FTA.

---

## 8. Experiments (`src/ekyc/eval/ablations.py`, `ekyc ablate --exp E1 ...`)

E1 bit depth · E2 resolution × interpolation · E3 JPEG (EER + CER) · E4 enhancement under low light (EER + CER) · E5 RGB vs HSV-V equalisation (hue shift + EER) · E6 edge operator (quad IoU) · E7 Hough vs contour vs Harris (corner error + accumulator) · E8 denoise vs noise (CER) · E9 global vs adaptive threshold, clean and uneven light (CER) · E10 motion threshold sweep · E11 liveness signals · E12 face headline · E13 quality gate on/off · E14 fusion. Face experiments degrade only the document side. OCR experiments rectify with the GT quad so only the studied stage varies. Each writes `results/<E>.csv` (+ commit hash) and `results/figures/<E>*.png`. E10/E11 need own-capture clips. `ekyc ablate --exp FAILURES` writes the failure galleries (worst face pairs, worst document localisations with Hough/region/warp intermediates).

---

## 9. Conventions

- Images in memory: `np.uint8` BGR; float32 [0,1] for arithmetic, clip on the way back (`io.to_float` / `io.to_uint8`).
- Every stochastic step takes a `seed`. Global seed in config.
- All thresholds, kernel sizes, paths and model URLs come from `configs/default.yaml`; ablations use `io.with_overrides(cfg, {"dotted.key": value})`.
- Logging via `logging`. Results as CSV with the git commit hash.
- Type hints and docstrings on public functions; each docstring names the course concept.
- `ruff` + `black` (line length 110); `pytest`.
- CLI: `ekyc verify --selfie p --id p [--video p] [--debug-dir out/]`, `ekyc eval {document,face,liveness,fusion}`, `ekyc ablate --exp E1 [E2 ...] | --all`, `ekyc serve [--reload]`, `ekyc models`.
- Web: `cd web && npm run build` then `ekyc serve` (serves `web/dist`); dev: `ekyc serve --reload` + `npm run dev` (Vite proxies `/api`, `/results`).
- Never commit raw data, model weights or personal recordings. The API keeps no uploads (images return as data URLs; clip temp files are deleted).

### Environment gotchas (verified)

- OpenCV is **5.x** (`opencv-contrib-python`, required by MediaPipe; do not also install `opencv-python`). OpenCV 5 wheels **do not ship Haar XMLs**: `models.py` downloads `haarcascade_frontalface_default.xml`.
- MediaPipe ≥ 1.0 has **no `mp.solutions`**; use the Tasks API (`FaceLandmarker` + `face_landmarker.task`).
- scikit-image 0.26: `remove_small_objects(..., max_size=n)` (the old `min_size` is deprecated).
- Tesseract is a system binary (`brew install tesseract`); OCR degrades to empty text with a warning if missing.

---

## 10. Course concept → code map (for the report and viva)

| Course concept | Module / function | MATLAB equivalent used in labs |
|---|---|---|
| Quantization, bit depth | `degrade.quantize_bits` | bit-shift / `uint8` reduction |
| Interpolation | `degrade.downsample`, `face.align.upscale` | `imresize(..., 'nearest'/'bilinear'/'bicubic')` |
| Image subtraction | `liveness.motion.change_masks` | `imabsdiff` |
| Threshold + cleanup | `liveness.motion`, `quality.glare_mask` | `imbinarize`, `bwareaopen`, `imfill` |
| Masking, logical ops | `liveness.motion.region_masks`, `quality.box_mask/quad_mask` | element-wise `.*`, `&` |
| Edge detection | `document.edges` | `edge(I,'sobel'/'prewitt'/'canny')` |
| Hough transform | `document.hough` | `hough`, `houghpeaks`, `houghlines` |
| Harris corners | `document.corners.harris_refine` | `detectHarrisFeatures` |
| Projective rectification | `document.corners.warp` | `fitgeotrans('projective')`, `imwarp` |
| Region properties | `document.validity` | `regionprops`, `bwconvhull` |
| Rule-based classification | `document.validity`, `liveness.manmade.rule_classify` | Lab 5 shape rules |
| Contrast stretching, log, gamma | `enhance.contrast_stretch/log_transform/gamma` | `imadjust`, `stretchlim` |
| Histogram equalization, CLAHE | `enhance.histeq` (from scratch), `enhance.clahe` | `histeq`, `adapthisteq` |
| HSV colour preservation | `enhance.apply(mode="hsv_v")` | `rgb2hsv`, `hsv2rgb` |
| Spatial filters | `filters.*` | `imfilter`, `fspecial`, `medfilt2` |
| Laplacian / LoG | `quality.blur_score`, `filters.log` | `fspecial('laplacian'/'log')` |
| Global / adaptive threshold | `document.ocr.threshold` | `graythresh`, `imbinarize(...,'adaptive')` |
| Morphology | `document.ocr.morph`, `document.corners.region_mask` | `imopen`, `imclose` |
| Landmark distances | `face.geometric` | face line diagram lecture |
| Man-made vs natural features | `liveness.manmade` | 17 Sep lecture |
| Fixed kernels → CNN → ViT | `face.embed` (SFace), `liveness.cnn` | Aug–Sep CNN/ViT lectures |

---

## 11. Milestones

- **Phase 0 (done):** scaffold, io, metrics + tests, degrade + determinism tests, LFW download, MIDV script, synthetic pairs, baseline + improved face eval (`ekyc eval face` → `results/face_summary.csv` + `results/figures/face_intermediates.png`).
- **Phase 1 (done):** document module baseline + improved, E6–E9.
- **Phase 2 (done):** improved face module, E1–E5, E12.
- **Phase 3 (code done, blocked on data):** liveness; E10/E11 run once `data/own_capture/` has clips.
- **Phase 4 (done; E14 without liveness until Phase 3 data):** quality gate, fusion, E13–E14.
- **Phase 5 (done):** block diagram, README (headline block regenerated by `scripts/summarize_results.py`), web UI, failure galleries, curated `results/final/`. Remaining: report write-up; MIDV-2020 download + re-run of `ekyc eval document`, E6, E7 for real-capture numbers; re-tune card-validity bounds (pass only ~70% of genuine synthetic cards).

Split: Student A owns quality gate + document; Student B owns face + liveness; both share degradation, metrics, fusion, report, web UI. Each must be able to run and explain the other's code.

## 12. Out of scope

Deepfake detection, active challenge-response liveness, mobile apps, cloud deployment, databases, auth, real identity documents or real PII of any kind.

## 13. Open items for the user

- Final demo and report deadlines.
- Partner name and roll number for README/report.
- Status of public liveness dataset license requests.
- Own-capture liveness recordings (blocks E10/E11 and full E14).
- Whether the instructor wants any MATLAB code in the submission.
