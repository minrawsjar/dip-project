"""Download LFW (funneled) + the official pairs files via scikit-learn into data/raw/sklearn/lfw_home.

scikit-learn fetches the archive and pairsDevTrain/pairsDevTest/pairs.txt (10-fold protocol). We only
use it as a downloader; images are read lazily from disk by ekyc.eval.datasets.lfw_pairs.

Citation: G. B. Huang et al., "Labeled Faces in the Wild", UMass Amherst TR 07-49, 2007.
"""

import logging

from sklearn.datasets import fetch_lfw_pairs

from ekyc.io import load_config, path

logging.basicConfig(level=logging.INFO)
cfg = load_config()
home = path(cfg, "raw") / "sklearn"
fetch_lfw_pairs(subset="test", data_home=home, funneled=True, download_if_missing=True)
lfw = path(cfg, "lfw_home")
print("LFW ready:", lfw, sorted(p.name for p in lfw.iterdir()))
