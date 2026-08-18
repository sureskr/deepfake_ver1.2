"""Download a curated set of MLAAD English TTS engines from Hugging Face.

The full `hf download mueller91/MLAAD --include "fake/en/*"` stalls enumerating
the whole multi-language repo, so this fetches a fixed engine list per-engine in
parallel (fast, reproducible). 8 modern engines are reserved for the held-out
TEST split; the rest are training engines. --n-per-engine caps files per engine
(MLAAD has up to 1000 each).

Usage:
  python scripts/download_mlaad_en.py --dest data/raw/mlaad --n-per-engine 1000
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import HfApi, hf_hub_download

REPO = "mueller91/MLAAD"

# Reserved for the held-out TEST split — never used in training (see prep_v1_2_5).
HOLDOUT_ENGINES = [
    "ElevenLabs-v3", "OpenAI TTS-1 HD", "Gemini-3.1-Flash-TTS", "Cartesia.ai (Sonic-3)",
    "f5-tts", "kokoro", "sesame_csm", "MegaTTS3",
]
# Training engines — a broad spread of modern + classic architectures.
TRAIN_ENGINES = [
    "ChatTTS", "Edge-TTS", "MeloTTS", "tts_models_multilingual_multi-dataset_xtts_v2",
    "suno_bark", "Higgs-Audio-V2", "Index-TTS-2.0", "Llasa-3B", "Spark-TTS-0.5B",
    "VoxCPM-0.5B", "Qwen2.5-Omni", "WhisperSpeech", "parler_tts_mini_v1",
    "microsoft_speecht5_tts", "MiniMax-Speech-2.8-Turbo", "orpheus-tts-0.1-finetune",
    "Step-Audio-EditX", "DeepGram", "FishTTS", "Marvis-TTS", "Chatterbox",
    "ElevenLabs-Turbo-v2.5", "tts_models_en_ljspeech_vits",
    "tts_models_en_ljspeech_tacotron2-DDC",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default="data/raw/mlaad")
    ap.add_argument("--n-per-engine", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    api = HfApi()
    engines = HOLDOUT_ENGINES + TRAIN_ENGINES
    tasks: list[str] = []
    for eng in engines:
        try:
            items = list(api.list_repo_tree(REPO, path_in_repo=f"fake/en/{eng}", repo_type="dataset"))
        except Exception as e:  # noqa: BLE001
            print("SKIP", eng, repr(e)[:100])
            continue
        wavs = [it.path for it in items if it.path.endswith(".wav")][: args.n_per_engine]
        meta = [it.path for it in items if it.path.endswith("meta.csv")]
        tasks += wavs + meta
    print(f"engines={len(engines)} (holdout={len(HOLDOUT_ENGINES)}, train={len(TRAIN_ENGINES)})  "
          f"files_to_fetch={len(tasks)}", flush=True)

    def dl(rel: str) -> bool:
        try:
            hf_hub_download(REPO, rel, repo_type="dataset", local_dir=args.dest)
            return True
        except Exception as e:  # noqa: BLE001
            print("err", rel, repr(e)[:120])
            return False

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        ok = sum(ex.map(dl, tasks))
    print(f"DONE downloaded {ok}/{len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
