"""Index own-capture liveness clips and assign subject-disjoint train/val/test splits.

Expected layout (anonymised subject IDs only, consent forms kept offline):
    data/own_capture/S01/live/clip1.mp4
    data/own_capture/S01/print/clip1.mp4
    data/own_capture/S01/replay/clip1.mp4

Writes data/processed/liveness_manifest.csv and prints per-split counts.
"""

from ekyc.eval.datasets import own_capture, subject_split
from ekyc.io import load_config, path

cfg = load_config()
df = own_capture(cfg)
if df.empty:
    raise SystemExit(f"no clips under {path(cfg, 'own_capture')}/<subject>/{{live,print,replay}}/")
df = subject_split(df, cfg["seed"])
out = path(cfg, "data") / "processed" / "liveness_manifest.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(df.groupby(["split", "kind"]).size().unstack(fill_value=0))
print("subjects per split:", df.groupby("split")["subject"].nunique().to_dict())
print("wrote", out)
