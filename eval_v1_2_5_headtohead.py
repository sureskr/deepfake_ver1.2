"""Head-to-head eval: score a held-out test set with the v1.2.4 head vs the v1.2.5 head.

Reports EER + accuracy overall and split by fake source (mlaad vs asvspoof), so you
can see whether v1.2.5 improved on MODERN TTS specifically (the mlaad rows) rather
than just on ASVspoof-family attacks the model already handled.

Usage:
  python eval_v1_2_5_headtohead.py \
    --config ml_engine/config/training_v1_2_5.json \
    --heads v1_2_4=ml_engine/models/v1_2_4_proper/best_model.pth \
            v1_2_5=ml_engine/models/v1_2_5_proper/best_model.pth
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent / "ml_engine"))
from training_config import load_training_config  # noqa: E402
from train_v1_2_4_proper import (  # noqa: E402
    FrozenW2V2Linear,
    LinearHeadLarge,
    load_audio_mono,
    load_v13_head_weights,
)
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model  # noqa: E402

AUDIO_EXT = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def eer(y: np.ndarray, s: np.ndarray) -> float:
    if len(set(y.tolist())) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1.0 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float(fpr[i])


def list_test(real_dir: Path, fake_dir: Path) -> list[tuple[Path, int, str]]:
    rows: list[tuple[Path, int, str]] = []
    for label, d in ((0, real_dir), (1, fake_dir)):
        for p in sorted(Path(d).rglob("*")):
            if p.is_file() and p.suffix.lower() in AUDIO_EXT:
                source = p.name.split("_", 1)[0]  # asvspoof / mlaad
                rows.append((p, label, source))
    return rows


def score_all(model, processor, rows, target_samples, device, bs=32) -> np.ndarray:
    scores: list[float] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            waves = []
            for p, _, _ in rows[i : i + bs]:
                w = load_audio_mono(p)
                if len(w) > target_samples:
                    w = w[:target_samples]
                elif len(w) < target_samples:
                    w = np.pad(w, (0, target_samples - len(w)), mode="constant")
                waves.append(w.astype(np.float32))
            inp = processor(list(waves), sampling_rate=16000, return_tensors="pt", padding=False)
            logits = model(inp.input_values.to(device))
            scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return np.array(scores)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="Training config JSON (for backbone + test dirs).")
    ap.add_argument("--heads", nargs="+", required=True,
                    help="name=path pairs, e.g. v1_2_4=.../best_model.pth v1_2_5=.../best_model.pth")
    ap.add_argument("--limit", type=int, default=0, help="Cap files per class (quick smoke test).")
    ap.add_argument("--out", default="eval_headtohead.json")
    args = ap.parse_args()

    cfg = load_training_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_samples = cfg.target_samples

    rows = list_test(cfg.held_out_test_real_dir, cfg.held_out_test_fake_dir)
    if not rows:
        raise SystemExit(f"No test files under {cfg.held_out_test_real_dir} / {cfg.held_out_test_fake_dir}")
    if args.limit:
        real = [r for r in rows if r[1] == 0][: args.limit]
        fake = [r for r in rows if r[1] == 1][: args.limit]
        rows = real + fake
    y = np.array([r[1] for r in rows])
    src = np.array([r[2] for r in rows])
    print(f"test files: {len(rows)}  real={int((y == 0).sum())}  fake={int((y == 1).sum())}  "
          f"fake sources={sorted(set(src[y == 1]))}", flush=True)

    processor = Wav2Vec2FeatureExtractor.from_pretrained(str(cfg.wav2vec2_dir), local_files_only=True)
    backbone = Wav2Vec2Model.from_pretrained(str(cfg.wav2vec2_dir), local_files_only=True)
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()

    results: dict[str, dict] = {}
    for spec in args.heads:
        if "=" not in spec:
            raise SystemExit(f"--heads entries must be name=path, got {spec!r}")
        name, path = spec.split("=", 1)
        head = LinearHeadLarge(in_dim=backbone.config.hidden_size)
        load_v13_head_weights(head, Path(path))
        model = FrozenW2V2Linear(backbone, head).to(device)
        s = score_all(model, processor, rows, target_samples, device)

        pred50 = (s >= 0.50).astype(int)
        pred55 = (s >= 0.55).astype(int)
        r = {
            "n": int(len(rows)),
            "eer_overall": eer(y, s),
            "acc@0.50": float((pred50 == y).mean()),
            "acc@0.55": float((pred55 == y).mean()),
        }
        # per fake source: real (all) vs that source's fake only
        for source in sorted(set(src[y == 1])):
            mask = (y == 0) | ((y == 1) & (src == source))
            r[f"eer[{source}]"] = eer(y[mask], s[mask])
            # recall on that source's fakes at production threshold 0.55
            fake_mask = (y == 1) & (src == source)
            r[f"recall@0.55[{source}]"] = float((s[fake_mask] >= 0.55).mean())
        results[name] = r
        print(f"\n=== {name}  ({path}) ===")
        for k, v in r.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
