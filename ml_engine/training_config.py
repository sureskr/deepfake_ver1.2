"""
Load and resolve JSON training configs for ml_engine scripts.

Paths may use ``{repo_root}`` (auto-detected as parent of ``ml_engine/``) or
``{auto}`` for repo_root itself.
"""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_config_path() -> Path:
    return Path(__file__).resolve().parent / "config" / "training_v1_2_4_proper.json"


def _resolve_placeholders(value: Any, root: Path) -> Any:
    if isinstance(value, str):
        if value == "{auto}":
            return str(root)
        return value.replace("{repo_root}", str(root).replace("\\", "/"))
    if isinstance(value, list):
        return [_resolve_placeholders(v, root) for v in value]
    if isinstance(value, dict):
        return {k: _resolve_placeholders(v, root) for k, v in value.items()}
    return value


@dataclass
class TrainingSettings:
    run_name: str
    description: str

    wav2vec2_dir: Path
    head_init_checkpoint: Path
    head_init_fallbacks: list[Path]
    train_real_dir: Path
    train_fake_dir: Path
    val_real_dir: Path
    val_fake_dir: Path
    output_dir: Path
    log_dir: Path
    manifest_path: Path | None

    sample_rate: int
    target_length_seconds: float
    audio_extensions: set[str]

    seed: int
    learning_rate: float
    batch_size: int
    max_epochs: int
    patience: int
    early_stopping_metric: str
    pos_weight: float | str
    optimizer: str
    augmentation: bool
    backbone_frozen: bool

    locked_test_real_dir: Path
    locked_test_fake_dir: Path
    held_out_test_real_dir: Path
    held_out_test_fake_dir: Path
    default_threshold: float

    raw: dict = field(repr=False)

    @property
    def output_model(self) -> Path:
        return self.output_dir / "best_model.pth"

    @property
    def log_jsonl(self) -> Path:
        return self.log_dir / f"train_{self.run_name}.jsonl"

    @property
    def target_samples(self) -> int:
        return int(self.sample_rate * self.target_length_seconds)


def load_training_config(path: Path | str | None = None) -> TrainingSettings:
    cfg_path = Path(path) if path else default_config_path()
    root = repo_root()
    with open(cfg_path, encoding="utf-8") as f:
        raw = json.load(f)
    resolved = _resolve_placeholders(deepcopy(raw), root)

    paths = resolved["paths"]
    audio = resolved["audio"]
    training = resolved["training"]
    evaluation = resolved["evaluation"]

    manifest_raw = paths.get("manifest_path")
    manifest_path = Path(manifest_raw) if manifest_raw else None

    return TrainingSettings(
        run_name=resolved.get("run_name", "run"),
        description=resolved.get("description", ""),
        wav2vec2_dir=Path(paths["wav2vec2_dir"]),
        head_init_checkpoint=Path(paths["head_init_checkpoint"]),
        head_init_fallbacks=[Path(p) for p in paths.get("head_init_fallbacks", [])],
        train_real_dir=Path(paths["train_real_dir"]),
        train_fake_dir=Path(paths["train_fake_dir"]),
        val_real_dir=Path(paths["val_real_dir"]),
        val_fake_dir=Path(paths["val_fake_dir"]),
        output_dir=Path(paths["output_dir"]),
        log_dir=Path(paths["log_dir"]),
        manifest_path=manifest_path,
        sample_rate=int(audio["sample_rate"]),
        target_length_seconds=float(audio["target_length_seconds"]),
        audio_extensions={ext.lower() for ext in audio.get("extensions", [".wav"])},
        seed=int(training["seed"]),
        learning_rate=float(training["learning_rate"]),
        batch_size=int(training["batch_size"]),
        max_epochs=int(training["max_epochs"]),
        patience=int(training["patience"]),
        early_stopping_metric=str(training.get("early_stopping_metric", "val_eer")),
        pos_weight=training.get("pos_weight", "auto"),
        optimizer=str(training.get("optimizer", "Adam")),
        augmentation=bool(training.get("augmentation", False)),
        backbone_frozen=bool(training.get("backbone_frozen", True)),
        locked_test_real_dir=Path(evaluation["locked_test_real_dir"]),
        locked_test_fake_dir=Path(evaluation["locked_test_fake_dir"]),
        held_out_test_real_dir=Path(evaluation["held_out_test_real_dir"]),
        held_out_test_fake_dir=Path(evaluation["held_out_test_fake_dir"]),
        default_threshold=float(evaluation.get("default_threshold", 0.55)),
        raw=resolved,
    )


def compute_pos_weight(n_real: int, n_fake: int) -> float:
    if n_fake <= 0:
        raise ValueError("Cannot compute pos_weight: no fake (positive) training samples.")
    return float(n_real) / float(n_fake)
