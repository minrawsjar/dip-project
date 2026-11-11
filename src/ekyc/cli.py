"""Command line: `ekyc verify | eval | ablate | serve | models`."""

from __future__ import annotations

import argparse
import json
import logging
import sys


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="ekyc", description="eKYC identity verification with classical DIP stages"
    )
    ap.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify", help="verify one selfie (+ optional clip) against one ID image")
    v.add_argument("--selfie", required=True)
    v.add_argument("--id", required=True, dest="id_image")
    v.add_argument("--video", default=None, help="selfie clip for motion/blink liveness")
    v.add_argument("--debug-dir", default=None, help="dump every intermediate image here")

    e = sub.add_parser("eval", help="evaluate a module (baseline vs improved)")
    e.add_argument("module", choices=["document", "face", "liveness", "fusion"])

    a = sub.add_parser("ablate", help="run experiments E1-E14")
    a.add_argument("--exp", nargs="+", default=None, help="e.g. E1 E6; omit with --all")
    a.add_argument("--all", action="store_true")

    s = sub.add_parser("serve", help="web UI + REST API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")

    sub.add_parser("models", help="download all model weights")

    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    from .io import load_config

    cfg = load_config(args.config)

    if args.cmd == "verify":
        from .io import Debug, imread, read_video
        from .pipeline import verify

        frames = (
            read_video(args.video, cfg["liveness"]["max_frames"], cfg["liveness"]["frame_stride"])
            if args.video
            else None
        )
        res = verify(imread(args.selfie), imread(args.id_image), cfg, frames, Debug(args.debug_dir))
        json.dump(res, sys.stdout, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        print()
    elif args.cmd == "eval":
        from importlib import import_module

        import_module(f"ekyc.eval.eval_{args.module}").main(cfg)
    elif args.cmd == "ablate":
        from .eval.ablations import EXPERIMENTS, run

        for exp in EXPERIMENTS if args.all else (args.exp or ap.error("give --exp or --all")):
            print(f"== {exp} ==")
            run(exp, cfg)
    elif args.cmd == "serve":
        import uvicorn

        uvicorn.run("ekyc.api:app", host=args.host, port=args.port, reload=args.reload)
    elif args.cmd == "models":
        from .models import download_all

        download_all(cfg)


if __name__ == "__main__":
    main()
