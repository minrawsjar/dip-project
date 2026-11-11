// Typed client for the FastAPI backend (src/ekyc/api.py). Shapes mirror the Python responses.

export type Decision = "ACCEPT" | "REJECT" | "REVIEW";
export type StageImage = { key: string; stage: string; group: string; label: string; src: string };
export type Quality = { passed: boolean; scores: Record<string, number>; reasons: string[] };

export type VerifyResult = {
  decision: Decision;
  scores: {
    face_score: number | null;
    liveness_score: number;
    ocr_conf: number;
    selfie_blur: number;
    id_blur: number;
    id_glare: number;
    quality_passed: boolean;
  };
  quality: { selfie: Quality; id: Quality };
  document: {
    found: boolean;
    quad: number[][] | null;
    corner_method: string;
    validity: { valid: boolean; features: Record<string, number>; reasons: string[] };
    fields: Record<string, { text: string; conf: number }>;
    ocr_conf: number;
  };
  face: { score: number | null; threshold: number; selfie_found: boolean; portrait_found: boolean; method: string };
  face_baseline?: { score: number | null; threshold: number; method: string };
  liveness: {
    features: Record<string, number>;
    scores: Record<string, number>;
    liveness_score: number;
    method: string;
    reasons: string[];
  };
  fusion: { method: string; decision: Decision; prob: number | null; checks: Record<string, boolean>; baseline_decision?: Decision };
  timings_ms: Record<string, number>;
  images: StageImage[];
  inputs: { selfie: string; id: string; frames: number };
};

export type LabResult = {
  input: string;
  output: string;
  extras: { label: string; src: string }[];
  info: Record<string, unknown>;
  hist_in: number[];
  hist_out: number[];
};

export type Experiment = {
  id: string;
  title: string;
  description: string;
  columns: string[];
  rows: Record<string, string | number | boolean | null>[];
  figures: string[];
  available: boolean;
};

export type Health = { status: string; tesseract: boolean; models: Record<string, boolean>; trained: Record<string, boolean> };
export type Sample = { pair_id: string; label: "genuine" | "impostor"; name: string; selfie: string; id: string };

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) {
    let detail = `${r.status} ${r.statusText}`;
    try {
      detail = (await r.json()).detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return r.json() as Promise<T>;
}

const fileName = (b: Blob, fallback: string) => (b instanceof File ? b.name : fallback);

export const api = {
  health: () => request<Health>("/api/health"),
  samples: () => request<Sample[]>("/api/samples"),
  experiments: () => request<Experiment[]>("/api/experiments"),
  verify(selfie: Blob, id: Blob, video?: Blob | null) {
    const fd = new FormData();
    fd.append("selfie", selfie, fileName(selfie, "selfie.jpg"));
    fd.append("id_image", id, fileName(id, "id.jpg"));
    if (video) fd.append("video", video, fileName(video, "clip.webm"));
    return request<VerifyResult>("/api/verify", { method: "POST", body: fd });
  },
  lab(image: Blob, op: string, params: Record<string, unknown>, signal?: AbortSignal) {
    const fd = new FormData();
    fd.append("image", image, fileName(image, "image.jpg"));
    fd.append("op", op);
    fd.append("params", JSON.stringify(params));
    return request<LabResult>("/api/lab", { method: "POST", body: fd, signal });
  },
  async blob(url: string) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url}: ${r.status}`);
    return r.blob();
  },
};
