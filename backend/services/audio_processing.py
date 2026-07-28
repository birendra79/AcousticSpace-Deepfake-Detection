"""
AcousticSpace Backend — Audio Processing Service
=================================================
Week 1 deliverable  (TASK-07): load_and_validate_audio
Week 1 deliverable  (TASK-08): generate_mel_spectrogram, generate_waveform_preview

This module handles all audio I/O and low-level DSP operations:

  - load_and_validate_audio   — decode raw bytes → mono float32 ndarray at 16 kHz
  - generate_mel_spectrogram  — compute a dB-scaled log-Mel spectrogram
  - generate_waveform_preview — downsample a waveform to a fixed number of points
                                for frontend rendering

Design notes:
  - Two-loader strategy: SoundFile (fast, lossless) with Librosa fallback
    (permissive, supports MP3/M4A via audioread).          [FR-2.1, NFR-4.3]
  - All audio is normalised to mono float32 at TARGET_SAMPLE_RATE before
    any downstream processing.                             [FR-2.2, FR-2.3]
  - TARGET_SAMPLE_RATE is read from the SAMPLE_RATE environment variable
    so it can be overridden without code changes.          [NFR-6.1]
"""

import io
import os

import librosa
import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Configuration  [NFR-6.1, FR-2.3, DR-3]
# ---------------------------------------------------------------------------
TARGET_SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", 16000))
MAX_ANALYZE_SECONDS: int = int(os.getenv("MAX_ANALYZE_SECONDS", 30))


def load_and_validate_audio(
    raw_bytes: bytes,
    filename: str,
) -> tuple[np.ndarray, int]:
    """
    Decode raw audio bytes into a mono float32 NumPy array at TARGET_SAMPLE_RATE.

    Implements the two-loader strategy:
      1. SoundFile is tried first — fast, supports WAV and FLAC natively.  [FR-2.1]
      2. On failure, Librosa is used as a fallback — handles MP3, M4A, OGG
         via the audioread backend.                                         [NFR-4.3]

    After loading, the following normalisation steps are applied in order:
      - Empty-signal guard: raises immediately if the decoded array is empty. [FR-1.5]
      - Stereo / multi-channel → mono by averaging across the channel axis.  [FR-2.2]
      - Resample to TARGET_SAMPLE_RATE if the source rate differs.           [FR-2.3]
      - Cast to float32.

    Args:
        raw_bytes: Raw bytes of the uploaded audio file.
        filename:  Original filename, used only in error messages.

    Returns:
        Tuple of (audio_array, sample_rate) where:
          - audio_array: shape (N,), dtype float32, mono at TARGET_SAMPLE_RATE.
          - sample_rate: always TARGET_SAMPLE_RATE (integer).

    Raises:
        ValueError: If the decoded audio signal is empty.
        Exception:  Re-raised from Librosa if both loaders fail.
    """
    # -- Loader 1: SoundFile (primary) --------------------------------------
    try:
        audio_array, sample_rate = sf.read(io.BytesIO(raw_bytes), always_2d=False)
    except Exception:
        # -- Loader 2: Librosa fallback (MP3, M4A, OGG via audioread) -------
        # mono=False so we receive the raw channel layout and handle mixing
        # ourselves below, consistent with the SoundFile path.
        audio_array, sample_rate = librosa.load(
            io.BytesIO(raw_bytes), sr=None, mono=False
        )

    # -- Empty-signal guard  [FR-1.5] ---------------------------------------
    # Check immediately after loading, before any transformation, so the
    # error message is clear and no obscure downstream error is raised.
    if audio_array.size == 0:
        raise ValueError(
            f"'{filename}' produced an empty audio signal after decoding."
        )

    # -- Stereo / multi-channel → mono  [FR-2.2] ----------------------------
    # SoundFile returns shape (N,) for mono or (N, C) for multi-channel.
    # Librosa with mono=False returns shape (C, N) for multi-channel or (N,) for mono.
    # Normalise to (N,) by averaging across the channel axis in both cases.
    if audio_array.ndim > 1:
        # SoundFile layout: (N, C) — average across axis 1
        # Librosa layout:   (C, N) — average across axis 0
        # np.mean(..., axis=-1) handles SoundFile (C last); for Librosa (C first)
        # we need axis=0. Detect layout by comparing which axis is smaller.
        if audio_array.shape[0] < audio_array.shape[-1]:
            # Librosa layout (C, N): C is the smaller dimension
            audio_array = np.mean(audio_array, axis=0)
        else:
            # SoundFile layout (N, C): C is the smaller dimension
            audio_array = np.mean(audio_array, axis=-1)

    # -- Resample to TARGET_SAMPLE_RATE  [FR-2.3] ---------------------------
    if sample_rate != TARGET_SAMPLE_RATE:
        audio_array = librosa.resample(
            y=audio_array.astype(np.float32),
            orig_sr=sample_rate,
            target_sr=TARGET_SAMPLE_RATE,
        )
        sample_rate = TARGET_SAMPLE_RATE

    # -- Cap duration to keep memory/CPU bounded on low-resource hosts ------
    # Long recordings blow up memory during feature extraction + CNN inference
    # (which is what was crashing the service on larger uploads). The first
    # MAX_ANALYZE_SECONDS is more than enough for deepfake detection.
    max_samples = MAX_ANALYZE_SECONDS * TARGET_SAMPLE_RATE
    if audio_array.shape[0] > max_samples:
        audio_array = audio_array[:max_samples]

    return audio_array.astype(np.float32), sample_rate


def generate_mel_spectrogram(
    audio_array: np.ndarray,
    sample_rate: int,
    n_mels: int = 128,
) -> np.ndarray:
    """
    Compute a dB-scaled log-Mel spectrogram from a mono audio array.  [FR-2.4]

    Args:
        audio_array: Mono float32 audio array at any sample rate.
        sample_rate: Sample rate of audio_array in Hz.
        n_mels:      Number of Mel filter banks (default 128).

    Returns:
        2D NumPy array of shape (n_mels, T) in dB scale.
        Values are non-positive floats; max value is 0.0 dB (relative to max power).
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio_array,
        sr=sample_rate,
        n_mels=n_mels,
        fmax=sample_rate // 2,
    )
    return librosa.power_to_db(mel_spec, ref=np.max)


def generate_waveform_preview(
    audio_array: np.ndarray,
    num_points: int = 800,
) -> list[float]:
    """
    Downsample the audio waveform to a fixed number of points for frontend rendering.

    Uses strided indexing (O(1)) — no FFT or smoothing.    [FR-2.5, NFR-1.3]

    Args:
        audio_array: Mono float32 audio array of any length.
        num_points:  Target number of output samples (default 800).

    Returns:
        List of float values with length min(len(audio_array), num_points).
        For inputs shorter than num_points, all samples are returned as-is.
    """
    if len(audio_array) <= num_points:
        return audio_array.tolist()
    step = len(audio_array) // num_points
    return audio_array[::step][:num_points].tolist()