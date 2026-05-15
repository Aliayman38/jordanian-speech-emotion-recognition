"""
wav2vec.py — Feature extraction using facebook/wav2vec2-large-xlsr-53

Strategy:
  - Mean-pool the last hidden state → fixed 1024-dim vector per audio file
  - Cache extracted features as .npy files so expensive inference runs only once
  - Re-running on already-processed files is skipped automatically (unless force=True)

Why these choices for Jordanian dialect + 2700 samples:
  - XLSR-53 was pretrained on 53 languages including Arabic → strong dialect coverage
  - Mean pooling is robust to variable-length audio and well-suited for SER classifiers
  - Caching is critical: wav2vec2-large is ~300M params; re-extracting every run is wasteful
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass, field

import torch
import librosa
import soundfile as sf
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Config ───────────────────────────────────────────────────────────────────

@dataclass
class Wav2VecConfig:
    model_name: str = "facebook/wav2vec2-large-xlsr-53"
    target_sr: int = 16_000          # wav2vec2 always expects 16 kHz
    max_duration_sec: float = 10.0   # clip/pad audio to this length
    batch_size: int = 8              # how many files to process per GPU/CPU batch
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    cache_dir: Optional[str] = None  # where to store the HuggingFace model weights


# ── Core Extractor ────────────────────────────────────────────────────────────

class Wav2VecExtractor:
    """
    Load wav2vec2-large-xlsr-53 once and extract mean-pooled embeddings.

    Usage
    -----
    extractor = Wav2VecExtractor()

    # Single file
    feat = extractor.extract_file("path/to/audio.wav")   # np.ndarray (1024,)

    # Whole dataset → save .npy per file
    extractor.extract_dataset(
        audio_dir="data/Dataset",
        output_dir="data/features/wav2vec",
    )

    # Load cached features back for training
    features, labels, paths = Wav2VecExtractor.load_cached(
        feature_dir="data/features/wav2vec",
        label_from_parent=True,   # infer label from parent folder name
    )
    """

    def __init__(self, config: Optional[Wav2VecConfig] = None):
        self.cfg = config or Wav2VecConfig()
        self._model: Optional[Wav2Vec2Model] = None
        self._feature_extractor: Optional[Wav2Vec2FeatureExtractor] = None

    # ── Lazy model loading ────────────────────────────────────────────────────

    def _load_model(self):
        if self._model is not None:
            return
        log.info("Loading %s …", self.cfg.model_name)
        # Wav2Vec2FeatureExtractor is used instead of Wav2Vec2Processor because
        # xlsr-53 ships without a tokenizer, which causes a TypeError in newer
        # transformers versions when Wav2Vec2Processor tries to load one.
        self._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.cfg.model_name,
            cache_dir=self.cfg.cache_dir,
        )
        self._model = Wav2Vec2Model.from_pretrained(
            self.cfg.model_name,
            cache_dir=self.cfg.cache_dir,
        )
        self._model.eval()
        self._model.to(self.cfg.device)
        log.info("Model loaded on %s", self.cfg.device)

    # ── Audio loading ─────────────────────────────────────────────────────────

    def _load_audio(self, path: Union[str, Path]) -> np.ndarray:
        """
        Load an audio file, resample to 16 kHz, convert to mono,
        and clip/pad to max_duration_sec.
        """
        path = str(path)
        try:
            audio, sr = librosa.load(path, sr=self.cfg.target_sr, mono=True)
        except Exception:
            # fallback via soundfile (handles more codecs)
            audio, sr = sf.read(path, always_2d=False)
            if sr != self.cfg.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.cfg.target_sr)

        max_samples = int(self.cfg.max_duration_sec * self.cfg.target_sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        elif len(audio) < max_samples:
            audio = np.pad(audio, (0, max_samples - len(audio)))

        return audio.astype(np.float32)

    # ── Single-file extraction ────────────────────────────────────────────────

    def extract_file(self, audio_path: Union[str, Path]) -> np.ndarray:
        """
        Extract a single 1024-dim embedding from one audio file.

        Returns
        -------
        np.ndarray of shape (1024,)
        """
        self._load_model()
        audio = self._load_audio(audio_path)
        return self._embed_batch([audio])[0]

    # ── Batch embedding ───────────────────────────────────────────────────────

    def _embed_batch(self, waveforms: list[np.ndarray]) -> np.ndarray:
        """
        Run inference on a list of waveforms (all same length after padding).
        Returns np.ndarray of shape (N, 1024).
        """
        inputs = self._feature_extractor(
            waveforms,
            sampling_rate=self.cfg.target_sr,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self.cfg.device)

        with torch.no_grad():
            outputs = self._model(input_values)

        # last_hidden_state: (batch, time_steps, 1024) → mean over time
        embeddings = outputs.last_hidden_state.mean(dim=1)   # (batch, 1024)
        return embeddings.cpu().numpy()

    # ── Dataset-level extraction ──────────────────────────────────────────────

    def extract_dataset(
        self,
        audio_dir: Union[str, Path],
        output_dir: Union[str, Path],
        extensions: tuple[str, ...] = (".wav", ".mp3", ".flac", ".ogg"),
        force: bool = False,
    ) -> dict[str, np.ndarray]:
        """
        Walk audio_dir recursively, extract features for every audio file,
        and save each as a .npy file under output_dir (mirroring the folder tree).

        Parameters
        ----------
        audio_dir   : root folder that contains your audio files
        output_dir  : where to write cached .npy feature files
        extensions  : audio formats to process
        force       : if True, re-extract even if .npy already exists

        Returns
        -------
        dict mapping original audio path → embedding (1024,)
        """
        self._load_model()
        audio_dir = Path(audio_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── collect all audio paths ───────────────────────────────────────────
        all_paths = sorted([
            p for p in audio_dir.rglob("*")
            if p.suffix.lower() in extensions
        ])
        if not all_paths:
            raise FileNotFoundError(f"No audio files found under: {audio_dir}")
        log.info("Found %d audio files in %s", len(all_paths), audio_dir)

        # ── skip already-cached files ─────────────────────────────────────────
        todo, skip = [], []
        for p in all_paths:
            out_path = self._npy_path(p, audio_dir, output_dir)
            if out_path.exists() and not force:
                skip.append(p)
            else:
                todo.append(p)

        if skip:
            log.info("Skipping %d already-cached files (use force=True to re-extract)", len(skip))
        log.info("Extracting features for %d files …", len(todo))

        # ── batch processing with progress bar ────────────────────────────────
        results: dict[str, np.ndarray] = {}
        failed: list[str] = []

        for i in tqdm(range(0, len(todo), self.cfg.batch_size), desc="wav2vec batches"):
            batch_paths = todo[i : i + self.cfg.batch_size]
            waveforms, valid_paths = [], []

            for p in batch_paths:
                try:
                    waveforms.append(self._load_audio(p))
                    valid_paths.append(p)
                except Exception as e:
                    log.warning("Could not load %s: %s", p.name, e)
                    failed.append(str(p))

            if not waveforms:
                continue

            embeddings = self._embed_batch(waveforms)   # (B, 1024)

            for path, emb in zip(valid_paths, embeddings):
                out_path = self._npy_path(path, audio_dir, output_dir)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(out_path, emb)
                results[str(path)] = emb

        # also add already-cached files to results
        for p in skip:
            out_path = self._npy_path(p, audio_dir, output_dir)
            results[str(p)] = np.load(out_path)

        if failed:
            log.warning("%d files failed to process: %s", len(failed), failed)

        log.info(
            "Done. %d features saved to %s",
            len(results),
            output_dir,
        )
        return results

    # ── Load cached features for training ────────────────────────────────────

    @staticmethod
    def load_cached(
        feature_dir: Union[str, Path],
        label_from_parent: bool = True,
    ) -> tuple[np.ndarray, list[str], list[str]]:
        """
        Load all cached .npy files from feature_dir into a matrix.

        Expects folder structure like:
            feature_dir/
                angry/   file1.npy  file2.npy …
                happy/   file3.npy  …
                sad/     …

        Parameters
        ----------
        feature_dir       : directory produced by extract_dataset()
        label_from_parent : if True, use the parent folder name as the label

        Returns
        -------
        features  : np.ndarray of shape (N, 1024)
        labels    : list of N label strings  (empty string if label_from_parent=False)
        file_paths: list of N .npy file paths
        """
        feature_dir = Path(feature_dir)
        npy_files = sorted(feature_dir.rglob("*.npy"))
        if not npy_files:
            raise FileNotFoundError(f"No .npy files found under: {feature_dir}")

        features, labels, paths = [], [], []
        for fp in npy_files:
            features.append(np.load(fp))
            labels.append(fp.parent.name if label_from_parent else "")
            paths.append(str(fp))

        log.info(
            "Loaded %d features from %s | classes: %s",
            len(features),
            feature_dir,
            sorted(set(labels)),
        )
        return np.stack(features), labels, paths

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _npy_path(
        audio_path: Path,
        audio_root: Path,
        output_root: Path,
    ) -> Path:
        """Mirror the audio file's relative path under output_root, with .npy extension."""
        rel = audio_path.relative_to(audio_root)
        return (output_root / rel).with_suffix(".npy")


# ── CLI / quick-test entry point ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract wav2vec2-xlsr-53 features")
    parser.add_argument("--audio_dir",  required=True,  help="Root folder of audio files")
    parser.add_argument("--output_dir", required=True,  help="Where to save .npy features")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_sec",    type=float, default=10.0, help="Max audio duration (seconds)")
    parser.add_argument("--force",      action="store_true", help="Re-extract even if cached")
    parser.add_argument("--device",     default=None, help="cuda | cpu (auto-detected if omitted)")
    args = parser.parse_args()

    cfg = Wav2VecConfig(
        batch_size=args.batch_size,
        max_duration_sec=args.max_sec,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )

    extractor = Wav2VecExtractor(cfg)
    extractor.extract_dataset(
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        force=args.force,
    )

    # Quick sanity check: load features back
    X, y, paths = Wav2VecExtractor.load_cached(args.output_dir)
    print(f"\n✓ Feature matrix shape : {X.shape}")
    print(f"✓ Label distribution   : { {l: y.count(l) for l in sorted(set(y))} }")