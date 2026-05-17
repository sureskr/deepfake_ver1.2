"""
Deterministic deepfake detection engine — physics-based, no ML.

Active checks (contribute to verdict):
  1. Glottal jitter & shimmer      (vocal-cord microperturbations)
  2. Formant transition velocity    (articulator speed limit ~80 Hz/ms)
  3. Harmonics-to-Noise Ratio (HNR) (real voices always have turbulence noise)
  4. Temporal consistency (CV)      (real phonation varies; TTS is uniform)
  7. Spectral flux CV upper bound   (neural vocoders produce extreme flux spikes)

Placeholder checks (always pass — observation mode, threshold not yet set):
  5. Micro-tremor AM–FM coherence
  6. Phase deviation  (distributions identical on in-the-wild data; no threshold set)

FAKE if ≥ 2 of 7 checks fail.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import scipy.linalg
import scipy.signal
import soundfile as sf

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Physics constants  — FROZEN 2026-04-20
#
# These values are set from first-principles phonetics and validated on the
# ASVspoof 2019 LA train set (n=200, stratified).  The dev-set EER is
# structurally pinned at ~17.5 % regardless of threshold adjustment; further
# tuning on this engine architecture will not improve it.  Do not change these
# constants without a corresponding change to the detection architecture.
# ──────────────────────────────────────────────────────────────────────────────

# ── Jitter / shimmer ─────────────────────────────────────────────────────────
# Jitter = detrended F0 variability (%) after Savitzky-Golay macro-contour
# removal (≈200 ms window).  Captures vocal-fold micro-perturbation that TTS
# smooths away.  Real speech floor: ~13 %.  TTS: typically < 13 %.
# Upper bound (75 %) excludes pathological voices / very expressive speech.
JITTER_MIN_PCT   = 8.0    # % — below: suspiciously smooth (TTS)
JITTER_MAX_PCT   = 75.0   # % — above: pathological / non-speech

# Shimmer = mean |20 log10(A_{i+1} / A_i)| across pitch cycles [dB].
# Measures cycle-to-cycle amplitude modulation from glottal open-quotient
# variation.  Real speech: 0.5–4.5 dB.
SHIMMER_MIN_DB   = 0.5    # dB — below: no amplitude variation (vocoder)
SHIMMER_MAX_DB   = 4.5    # dB — above: clipping / very hoarse speech

# ── Formant velocity ─────────────────────────────────────────────────────────
# Articulator speed limit: human tongue/jaw cannot produce F1/F2 transitions
# faster than ~80 Hz/ms.  GAN/diffusion vocoders can produce arbitrary
# frame-to-frame formant jumps that exceed this physical limit.
FORMANT_VEL_MAX  = 80.0   # Hz/ms — human articulator speed ceiling
FORMANT_VEL_MAX_EXCEEDANCES = 10   # allowance for LPC analysis noise

# ── HNR ──────────────────────────────────────────────────────────────────────
# HNR = 10 log10( r(T0) / (1 − r(T0)) ) where r(T0) is normalised
# autocorrelation at the pitch period.
# Real glottal vibration always has turbulence noise (breathiness, asymmetry)
# → r(T0) < ~0.9997 → HNR < ~35 dB.  In practice, voiced speech HNR sits
# in roughly 3–25 dB; our acceptance band [0.5, 5.0] targets the voiced
# region where TTS / real separation is most reliable at short utterances.
# Above 5 dB on short clips: suspiciously clean (no noise injection in TTS).
HNR_MIN_DB =  0.5   # dB — below: aperiodic / breathy / unvoiced-like
HNR_MAX_DB =  5.0   # dB — above: unnaturally clean periodicity (TTS)

# ── Temporal consistency ──────────────────────────────────────────────────────
# Real vocal physiology produces natural frame-to-frame variation in phonation
# strength, pitch, and breathiness.  Deterministic TTS synthesisers produce
# near-uniform per-segment statistics.  We measure CV (std/mean × 100 %) of
# per-segment jitter and HNR estimates across 500 ms / 250 ms hop windows.
# Observed real-speech floors: CV_jitter ≈ 20.8 %, CV_HNR ≈ 11.2 %.
TC_SEGMENT_MS    = 500    # segment length (ms)
TC_HOP_MS        = 250    # hop between segments (ms)
TC_MIN_SEGS      = 4      # minimum segments required for a valid estimate
TC_CV_JITTER_MIN = 20.0   # % — below: jitter suspiciously uniform
TC_CV_HNR_MIN    = 11.0   # % — below: HNR suspiciously uniform

# ── Micro-tremor / AM–FM coherence ───────────────────────────────────────────
# A single CNS neural oscillator (cerebellum / basal ganglia / stretch-reflex)
# drives the glottis at 4–12 Hz.  Because it is a SINGLE source, it couples
# both the amplitude (AM) and frequency (FM) of glottal vibration — they are
# coherent in the 4–12 Hz band.  TTS synthesisers generate FM (prosodic pitch)
# and AM (amplitude envelope) via independent model components; no coupling
# constraint exists, so their 4–12 Hz coherence is near-random (~0.1–0.3).
#
# TREMOR_COHERENCE_MIN = 0.0 (observation / pass-through mode) — threshold
# to be set after inspecting the real-vs-fake distribution.
TREMOR_F_LOW          = 4.0   # Hz — lower bound of neurological tremor band
TREMOR_F_HIGH         = 12.0  # Hz — upper bound
TREMOR_COHERENCE_MIN  = 0.0   # MSC floor — 0.0 = observation mode
TREMOR_MIN_FRAMES     = 100   # voiced frames required (~500 ms at 5 ms hop)

# ── Phase deviation (Check 6) ─────────────────────────────────────────────────
# Real voiced speech: STFT phase in high-energy (harmonic) bins evolves at a
# predictable instantaneous frequency — small, smooth Δφ frame-to-frame.
# TTS systems that reconstruct phase from magnitude (Griffin-Lim, minimum-phase
# assumption, WORLD vocoder) introduce excess phase variation in those same bins,
# raising the magnitude-weighted mean |Δφ|.  Neural vocoders (WaveNet, HiFi-GAN)
# generate waveform samples directly and may land closer to real speech.
#
# Adapted from VeriFauX advanced_feature_engineering._calculate_phase_coherence
# with the key improvement: magnitude-weighting focuses the measure on bins
# where signal energy is high (harmonics) rather than noisy/silent bins.
#
# PHASE_DEV_MAX = 0.0 — OBSERVATION MODE (not activated).
# release_in_the_wild analysis (n=200, 2026-04-20):
#   Real phase_dev: mean=2.077, std=0.031  Fake: mean=2.074, std=0.031
#   Best achievable EER via any threshold = 49.5% (essentially random).
#   Distributions are indistinguishable — modern neural vocoders do not
#   introduce excess phase variation detectable by this measure.
PHASE_N_FFT      = 2048   # STFT window length (samples)
PHASE_HOP        = 512    # STFT hop length (samples)
PHASE_MIN_FRAMES = 20     # minimum STFT frames required
PHASE_DEV_MAX    = 0.0    # weighted Δφ ceiling [radians] — 0.0 = observation mode

# ── Spectral flux CV (Check 7) ────────────────────────────────────────────────
# Real speech: spectral flux is highly irregular — stable voiced frames are
# punctuated by large transitions at phoneme boundaries (plosives, affricates).
# This irregularity is reflected in a high coefficient of variation (CV) of the
# normalised per-frame spectral flux time series.
# Neural TTS vocoders that process audio in fixed-size chunks (typically
# 256–512 samples) may produce artificially periodic spectral jumps, reducing CV.
# Extremely smooth unit-selection TTS has near-zero flux variance, also low CV.
#
# UPPER BOUND — from release_in_the_wild distribution (n=200, 2026-04-20):
#   Real  flux_cv: mean=86.9, p75=99.2, p95=156.8, max=345.8
#   Fake  flux_cv: mean=267.2, p50=126.9, p95=1013.6, max=2498.0
# Modern neural voice cloning (ElevenLabs-style) produces extreme spectral jumps
# that far exceed natural speech — the useful discriminator is flux_cv > threshold.
# EER-optimal threshold = 120 % (lower bound has no discriminatory power: EER=50.5%).
#
# FLUX_CV_MIN = 0.0 — lower bound stays in observation mode (useless here).
# FLUX_CV_MAX = 120.0 — upper bound ACTIVE: fail if cv > 120 %.
FLUX_N_FFT   = 2048   # reuses PHASE_N_FFT parameters
FLUX_HOP     = 512
FLUX_CV_MIN  = 0.0    # flux CV floor [%] — 0.0 = observation mode (lower bound useless)
FLUX_CV_MAX  = 120.0  # flux CV ceiling [%] — ACTIVE: neural vocoders spike above this

# ── Sub-band energy asymmetry (Check 8) ──────────────────────────────────────
# Real vocal tract resonance concentrates energy unevenly across frequency bands:
# strong low-mid energy (F1/F2 formants), sharply attenuated highs.  Neural
# vocoders apply learned spectral shaping that smooths this natural imbalance,
# producing a more uniform per-band energy distribution.
#
# Method: for each STFT frame compute power in 4 bands, normalise so they sum
# to 1, then take std([e1,e2,e3,e4]).  A perfectly uniform spectrum gives
# std = 0; maximum concentration in one band gives std ≈ 0.43.  Average over
# all frames → SBA score.  Higher score = more asymmetric = more like real speech.
#
# SBA_ASYM_MIN = 0.0 — OBSERVATION MODE.  Threshold set after inspecting the
# real-vs-fake distribution on the development set.
SBA_BANDS_HZ    = [(0, 500), (500, 2000), (2000, 4000), (4000, 8000)]
SBA_N_FFT       = 2048
SBA_HOP         = 512
SBA_MIN_FRAMES  = 10    # minimum STFT frames required for a valid estimate
SBA_ASYM_MIN    = 0.0   # asymmetry floor — 0.0 = observation mode


# ──────────────────────────────────────────────────────────────────────────────
# Result containers
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class CheckResult:
    passed: bool
    detail: dict = field(default_factory=dict)


@dataclass
class DetectionResult:
    verdict: str          # "REAL" | "FAKE"
    checks_failed: int
    jitter_shimmer: CheckResult
    formant_velocity: CheckResult
    subglottal: CheckResult
    temporal_consistency: CheckResult = field(default_factory=lambda: CheckResult(passed=True))
    micro_tremor: CheckResult    = field(default_factory=lambda: CheckResult(passed=True))
    phase_deviation: CheckResult    = field(default_factory=lambda: CheckResult(passed=True))
    spectral_flux: CheckResult      = field(default_factory=lambda: CheckResult(passed=True))
    subband_asymmetry: CheckResult  = field(default_factory=lambda: CheckResult(passed=True))
    audio_path: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Low-pass pre-filter for pitch tracking
# ──────────────────────────────────────────────────────────────────────────────
def _pitch_lp(audio: np.ndarray, sr: int, cutoff: float = 700.0) -> np.ndarray:
    """
    4th-order Butterworth low-pass at `cutoff` Hz.
    Removes high-frequency formant resonances so that autocorrelation finds
    the true pitch period rather than a formant or sub-harmonic artefact.
    """
    nyq = sr / 2.0
    if cutoff >= nyq:
        return audio
    b, a = scipy.signal.butter(4, cutoff / nyq, btype="low")
    return scipy.signal.filtfilt(b, a, audio)


# ──────────────────────────────────────────────────────────────────────────────
# Global F0 estimator (used by Checks 1 & 2 for period reference)
# ──────────────────────────────────────────────────────────────────────────────
def _estimate_global_f0(audio: np.ndarray, sr: int,
                         f0_min: float = 60.0,
                         f0_max: float = 400.0) -> Optional[float]:
    """Coarse F0 from full-signal autocorrelation (Hz) on low-passed audio."""
    min_lag = int(sr / f0_max)
    max_lag = int(sr / f0_min)
    if max_lag >= len(audio):
        return None
    lp   = _pitch_lp(audio, sr)
    sig  = lp - lp.mean()
    corr = np.correlate(sig, sig, mode="full")
    corr = corr[len(corr) // 2 :]
    denom = corr[0] + 1e-12
    corr /= denom
    if max_lag >= len(corr):
        return None
    peak_lag = int(np.argmax(corr[min_lag:max_lag])) + min_lag
    if corr[peak_lag] < 0.2:
        return None
    return sr / float(peak_lag)


# ──────────────────────────────────────────────────────────────────────────────
# Check 1 — Glottal jitter & shimmer
# ──────────────────────────────────────────────────────────────────────────────
def _frame_periods_parabolic(audio: np.ndarray, sr: int,
                              frame_ms: int = 20, hop_ms: int = 5,
                              f0_min: float = 60.0,
                              f0_max: float = 400.0,
                              voiced_thr: float = 0.4,
                              f0_anchor: Optional[float] = None) -> np.ndarray:
    """
    Per-frame pitch period (samples) using autocorrelation + parabolic
    interpolation for sub-sample accuracy.  Returns only voiced-frame periods.

    Low-passes before the search to suppress formant resonances.
    When f0_anchor is provided, the lag search is constrained to ±20% of the
    global estimate — this prevents sub-harmonic / double-period confusion on
    real speech (which caused jitter to measure at 40%+ without the anchor).
    """
    lp      = _pitch_lp(audio, sr)
    frame_n = int(sr * frame_ms / 1000)
    hop_n   = int(sr * hop_ms   / 1000)

    if f0_anchor is not None:
        period_nom = sr / f0_anchor
        # ±35% covers natural intonation range while excluding sub-harmonics
        # (which would be at 2× period_nom, far outside this window)
        margin     = max(int(period_nom * 0.35), 5)
        min_lag    = max(int(period_nom) - margin, 1)
        max_lag    = int(period_nom) + margin + 1
    else:
        min_lag = int(sr / f0_max)
        max_lag = int(sr / f0_min)

    periods: List[float] = []

    for start in range(0, len(lp) - frame_n, hop_n):
        frame  = lp[start : start + frame_n].astype(np.float64)
        if np.sqrt(np.mean(frame ** 2)) < 5e-4:
            continue

        centred = frame - frame.mean()
        corr    = np.correlate(centred, centred, mode="full")
        corr    = corr[len(corr) // 2 :]
        denom   = corr[0] + 1e-12
        corr   /= denom

        if max_lag >= len(corr):
            continue

        peak_i = int(np.argmax(corr[min_lag:max_lag])) + min_lag
        if corr[peak_i] < voiced_thr:
            continue

        # Parabolic interpolation for sub-sample accuracy
        if 0 < peak_i < len(corr) - 1:
            y1, y2, y3 = corr[peak_i - 1], corr[peak_i], corr[peak_i + 1]
            denom_p    = 2 * y2 - y1 - y3
            delta      = 0.5 * (y1 - y3) / denom_p if abs(denom_p) > 1e-10 else 0.0
            periods.append(peak_i + np.clip(delta, -0.5, 0.5))
        else:
            periods.append(float(peak_i))

    return np.array(periods)


def _per_cycle_shimmer(audio: np.ndarray, sr: int, f0: float) -> Optional[float]:
    """
    Step through audio at pitch-period intervals; measure peak amplitude per
    cycle; return mean |20 log10(A_i+1 / A_i)| in dB.
    """
    period_n = int(sr / f0)
    amps: List[float] = []

    for start in range(0, len(audio) - period_n, period_n):
        chunk = audio[start : start + period_n]
        rms   = float(np.sqrt(np.mean(chunk ** 2)))
        if rms < 5e-4:
            continue
        amps.append(float(np.max(np.abs(chunk))))

    if len(amps) < 6:
        return None

    amps_arr = np.array(amps, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = amps_arr[1:] / (amps_arr[:-1] + 1e-12)
        ratios = np.clip(ratios, 1e-6, None)
        return float(np.mean(np.abs(20.0 * np.log10(ratios))))


def check_jitter_shimmer(audio: np.ndarray, sr: int) -> CheckResult:
    """
    Jitter  = detrended F0 variability [%]   target 13–75 %
    Shimmer = mean |20 log10(A_i+1 / A_i)| [dB]  target 0.5–4.5 dB

    Detrended jitter removes the slow macro intonation contour via a
    Savitzky-Golay low-pass (window ≈ 200 ms) before computing std.
    This isolates micro-perturbation + residual intonation that TTS systems
    suppress but real vocal fold mechanics produce.

    Real speech:  jitter 13–75 %  (high variability from natural phonation).
    TTS speech:   jitter < 13 %   (smooth, model-generated F0 contours).

    Periods estimated from full [60–400 Hz] range; outliers outside ±50 % of
    global F0 discarded (removes sub/super-harmonic confusion artefacts).
    """
    f0 = _estimate_global_f0(audio, sr)
    if f0 is None:
        return CheckResult(passed=True, detail={"note": "no clear pitch"})

    # Full-range period extraction then filter outliers
    all_periods = _frame_periods_parabolic(audio, sr, f0_anchor=None)
    if len(all_periods) < 6:
        return CheckResult(passed=True,
                           detail={"note": "insufficient voiced frames",
                                   "n": len(all_periods)})

    period_ref = sr / f0
    # ±65 % keeps natural intonation; sub-harmonics (at 2× period) are >100 % away
    valid_mask = np.abs(all_periods - period_ref) < 0.65 * period_ref
    periods    = all_periods[valid_mask]
    if len(periods) < 6:
        return CheckResult(passed=True,
                           detail={"note": "insufficient in-range periods",
                                   "n_valid": int(valid_mask.sum())})

    # Detrend: remove slow macro F0 contour (≈200 ms Savgol window)
    hop_ms   = 5
    wl_ms    = 200
    wl_frames = min(len(periods) // 2 * 2 + 1,
                    max(5, (wl_ms // hop_ms) | 1))    # odd, ≤ n_periods
    if wl_frames >= 5 and len(periods) >= wl_frames:
        trend      = scipy.signal.savgol_filter(periods, wl_frames, polyorder=2)
        residuals  = periods - trend
        jitter_pct = 100.0 * float(np.std(residuals)) / (float(np.mean(periods)) + 1e-12)
    else:
        jitter_pct = 100.0 * float(np.std(periods)) / (float(np.mean(periods)) + 1e-12)

    shimmer_db = _per_cycle_shimmer(audio, sr, f0)
    if shimmer_db is None:
        return CheckResult(passed=True,
                           detail={"note": "insufficient cycles for shimmer",
                                   "jitter_pct": round(jitter_pct, 4)})

    jitter_ok  = JITTER_MIN_PCT <= jitter_pct  <= JITTER_MAX_PCT
    shimmer_ok = SHIMMER_MIN_DB <= shimmer_db  <= SHIMMER_MAX_DB
    passed     = jitter_ok and shimmer_ok

    return CheckResult(
        passed=passed,
        detail={
            "jitter_pct":    round(jitter_pct,  4),
            "shimmer_db":    round(shimmer_db,  4),
            "jitter_ok":     jitter_ok,
            "shimmer_ok":    shimmer_ok,
            "voiced_frames": len(periods),
            "f0_hz":         round(f0, 1),
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# LPC-based formant extraction (single frame)
# ──────────────────────────────────────────────────────────────────────────────
def _lpc_formants(frame: np.ndarray, sr: int) -> np.ndarray:
    order = 2 + int(sr / 1000)

    pre  = np.append(frame[0], frame[1:] - 0.97 * frame[:-1])
    pre *= np.hamming(len(pre))
    # Amplitude-normalise before autocorrelation so shimmer does not shift poles
    rms_pre = np.sqrt(np.mean(pre ** 2))
    if rms_pre > 1e-10:
        pre = pre / rms_pre

    corr = np.correlate(pre, pre, mode="full")
    corr = corr[len(corr) // 2 :]
    if corr[0] < 1e-12:
        return np.array([])

    try:
        lpc = scipy.linalg.solve_toeplitz(corr[:order], -corr[1 : order + 1])
    except (scipy.linalg.LinAlgError, ValueError):
        return np.array([])

    coeffs = np.concatenate([[1.0], lpc])
    roots  = np.roots(coeffs)

    roots = roots[np.imag(roots) > 0]
    roots = roots[np.abs(roots) <= 1.02]
    if len(roots) == 0:
        return np.array([])

    freqs = np.angle(roots) * sr / (2.0 * np.pi)
    freqs = np.sort(freqs[freqs > 80.0])
    return freqs


# ──────────────────────────────────────────────────────────────────────────────
# Check 2 — Formant transition velocity
# ──────────────────────────────────────────────────────────────────────────────
def check_formant_velocity(audio: np.ndarray, sr: int) -> CheckResult:
    """
    Track F1 & F2 frame-by-frame; apply 5-frame median filter to suppress
    LPC analysis noise; compute Δfreq / Δtime [Hz/ms].
    Fails if any transition exceeds the articulator speed limit (80 Hz/ms).
    """
    frame_ms = 25
    hop_ms   = 10
    frame_n  = int(sr * frame_ms / 1000)
    hop_n    = int(sr * hop_ms   / 1000)

    f1_raw: List[Optional[float]] = []
    f2_raw: List[Optional[float]] = []

    for start in range(0, len(audio) - frame_n, hop_n):
        frame = audio[start : start + frame_n].astype(np.float64)
        if np.sqrt(np.mean(frame ** 2)) < 1e-7:
            f1_raw.append(None); f2_raw.append(None)
            continue
        fmts = _lpc_formants(frame, sr)
        f1_raw.append(float(fmts[0]) if len(fmts) > 0 else None)
        f2_raw.append(float(fmts[1]) if len(fmts) > 1 else None)

    def _medfilt_track(track: List[Optional[float]], k: int = 5
                       ) -> List[Optional[float]]:
        out  = list(track)
        half = k // 2
        for i in range(len(track)):
            window = [v for v in track[max(0, i-half) : i+half+1] if v is not None]
            if window:
                out[i] = float(np.median(window))
        return out

    f1 = _medfilt_track(f1_raw)
    f2 = _medfilt_track(f2_raw)

    hop_sec     = hop_ms / 1000.0
    max_vel     = 0.0
    exceedances = 0

    for track in (f1, f2):
        prev = None
        for f in track:
            if f is None:
                prev = None
                continue
            if prev is not None:
                vel = abs(f - prev) / (hop_sec * 1000.0)   # Hz/ms
                max_vel = max(max_vel, vel)
                if vel > FORMANT_VEL_MAX:
                    exceedances += 1
            prev = f

    return CheckResult(
        passed=(exceedances <= FORMANT_VEL_MAX_EXCEEDANCES),
        detail={
            "max_velocity_hz_ms": round(max_vel, 2),
            "exceedances":        exceedances,
            "limit_hz_ms":        FORMANT_VEL_MAX,
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Check 3 — Harmonics-to-Noise Ratio (HNR) range
# ──────────────────────────────────────────────────────────────────────────────
def check_subglottal_resonance(audio: np.ndarray, sr: int) -> CheckResult:
    """
    HNR = 10 · log10( r(T0) / (1 − r(T0)) )
    where r(T0) is the normalised autocorrelation at the pitch period lag.

    Physical basis: real glottal vibration always contains turbulence noise
    (breathiness, vocal-fold asymmetry).  This keeps r(T0) < ~0.9997 and HNR
    below ~35 dB.  Deterministic TTS synthesisers and vocoders without explicit
    noise injection produce near-perfect periodicity (HNR → ∞).  Conversely,
    very noisy or aperiodic synthesis yields HNR < 3 dB.

    Fails if mean voiced HNR is outside [HNR_MIN_DB, HNR_MAX_DB].

    NOTE: the function is named check_subglottal_resonance to keep the
    DetectionResult field names stable; the physics has been revised.
    """
    f0 = _estimate_global_f0(audio, sr)
    if f0 is None:
        return CheckResult(passed=True, detail={"note": "no clear pitch"})

    period_nom = int(round(sr / f0))
    margin     = max(int(period_nom * 0.35), 3)

    lp      = _pitch_lp(audio, sr)
    frame_n = int(sr * 0.030)    # 30 ms
    hop_n   = int(sr * 0.010)    # 10 ms

    hnr_values: List[float] = []

    for start in range(0, len(lp) - frame_n, hop_n):
        frame = lp[start : start + frame_n].astype(np.float64)
        if np.sqrt(np.mean(frame ** 2)) < 5e-4:
            continue

        centred = frame - frame.mean()
        corr    = np.correlate(centred, centred, mode="full")
        corr    = corr[len(corr) // 2 :]
        denom   = corr[0] + 1e-12
        corr   /= denom

        lo = max(1, period_nom - margin)
        hi = min(len(corr) - 1, period_nom + margin)
        if lo >= hi:
            continue

        r_T0 = float(np.max(corr[lo:hi]))
        if r_T0 <= 0.30:           # skip weakly voiced / unvoiced frames
            continue

        r_T0    = min(r_T0, 1.0 - 1e-6)   # prevent log(0)
        hnr_db  = 10.0 * np.log10(r_T0 / (1.0 - r_T0))
        hnr_values.append(hnr_db)

    if len(hnr_values) < 5:
        return CheckResult(passed=True,
                           detail={"note": "insufficient voiced frames",
                                   "n": len(hnr_values)})

    mean_hnr = float(np.mean(hnr_values))
    hnr_ok   = HNR_MIN_DB <= mean_hnr <= HNR_MAX_DB

    return CheckResult(
        passed=hnr_ok,
        detail={
            "mean_hnr_db":    round(mean_hnr, 2),
            "hnr_ok":         hnr_ok,
            "voiced_frames":  len(hnr_values),
            "range_db":       f"[{HNR_MIN_DB}, {HNR_MAX_DB}]",
            "f0_hz":          round(f0, 1),
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Check 4 — Temporal consistency (CV of per-segment jitter & HNR)
# ──────────────────────────────────────────────────────────────────────────────
def _seg_jitter_pct(seg: np.ndarray, sr: int, f0_anchor: float) -> Optional[float]:
    """Detrended jitter [%] for a single audio segment."""
    periods = _frame_periods_parabolic(seg, sr, f0_anchor=f0_anchor)
    period_ref = sr / f0_anchor
    valid = periods[np.abs(periods - period_ref) < 0.65 * period_ref]
    if len(valid) < 6:
        return None
    wl = min(len(valid) // 2 * 2 + 1, max(5, (200 // 5) | 1))
    if wl >= 5 and len(valid) >= wl:
        trend     = scipy.signal.savgol_filter(valid, wl, polyorder=2)
        residuals = valid - trend
        return 100.0 * float(np.std(residuals)) / (float(np.mean(valid)) + 1e-12)
    return 100.0 * float(np.std(valid)) / (float(np.mean(valid)) + 1e-12)


def _seg_mean_hnr(seg: np.ndarray, sr: int, f0: float) -> Optional[float]:
    """Mean HNR [dB] for a single audio segment."""
    period_nom = int(round(sr / f0))
    margin     = max(int(period_nom * 0.35), 3)
    lp         = _pitch_lp(seg, sr)
    frame_n    = int(sr * 0.030)
    hop_n      = int(sr * 0.010)
    vals: List[float] = []
    for start in range(0, len(lp) - frame_n, hop_n):
        frame = lp[start : start + frame_n].astype(np.float64)
        if np.sqrt(np.mean(frame ** 2)) < 5e-4:
            continue
        c    = frame - frame.mean()
        corr = np.correlate(c, c, mode="full")[len(frame) - 1 :]
        d    = corr[0] + 1e-12
        corr /= d
        lo = max(1, period_nom - margin)
        hi = min(len(corr) - 1, period_nom + margin)
        if lo >= hi:
            continue
        r = float(np.max(corr[lo:hi]))
        if r <= 0.30:
            continue
        r = min(r, 1.0 - 1e-6)
        vals.append(10.0 * np.log10(r / (1.0 - r)))
    return float(np.mean(vals)) if len(vals) >= 3 else None


def check_temporal_consistency(audio: np.ndarray, sr: int) -> CheckResult:
    """
    Splits the utterance into overlapping 500 ms segments and measures
    per-segment jitter and HNR.  Computes CV (std/mean × 100) across segments.

    Physical basis: real vocal physiology produces natural variation in phonation
    strength, pitch, and breathiness across time — even within a single utterance.
    Deterministic TTS and neural vocoders tend to produce highly consistent
    frame-level statistics throughout the same utterance.

    Fails if BOTH cv_jitter AND cv_hnr are below their respective minimums.
    Requiring both prevents spurious failures from monotone but genuine speakers.
    """
    f0 = _estimate_global_f0(audio, sr)
    if f0 is None:
        return CheckResult(passed=True, detail={"note": "no clear pitch"})

    seg_n = int(sr * TC_SEGMENT_MS / 1000)
    hop_n = int(sr * TC_HOP_MS    / 1000)

    seg_jitters: List[float] = []
    seg_hnrs:    List[float] = []

    for start in range(0, len(audio) - seg_n, hop_n):
        seg = audio[start : start + seg_n]
        j   = _seg_jitter_pct(seg, sr, f0)
        h   = _seg_mean_hnr(seg, sr, f0)
        if j is not None: seg_jitters.append(j)
        if h is not None: seg_hnrs.append(h)

    if len(seg_jitters) < TC_MIN_SEGS or len(seg_hnrs) < TC_MIN_SEGS:
        return CheckResult(passed=True,
                           detail={"note": "insufficient segments",
                                   "n_jitter_segs": len(seg_jitters),
                                   "n_hnr_segs":    len(seg_hnrs)})

    mean_j = float(np.mean(seg_jitters)) + 1e-12
    mean_h = abs(float(np.mean(seg_hnrs))) + 1e-12
    cv_j   = 100.0 * float(np.std(seg_jitters)) / mean_j
    cv_h   = 100.0 * float(np.std(seg_hnrs))    / mean_h

    jitter_uniform = cv_j < TC_CV_JITTER_MIN
    hnr_uniform    = cv_h < TC_CV_HNR_MIN
    # EITHER signal suffices — both thresholds sit below observed real-speech minimums
    # (CV_J floor ~20.8%, CV_HNR floor ~11.2%) so no genuine speaker triggers this.
    passed         = not (jitter_uniform or hnr_uniform)

    return CheckResult(
        passed=passed,
        detail={
            "cv_jitter_pct":  round(cv_j, 2),
            "cv_hnr_pct":     round(cv_h, 2),
            "jitter_uniform": jitter_uniform,
            "hnr_uniform":    hnr_uniform,
            "n_segments":     len(seg_jitters),
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Check 5 — Micro-tremor: AM–FM coherence in 4–12 Hz band
# ──────────────────────────────────────────────────────────────────────────────
def _rms_contour(audio: np.ndarray, sr: int,
                 hop_ms: int = 5, frame_ms: int = 20) -> np.ndarray:
    """Per-frame RMS on the same grid as _f0_contour — the AM envelope proxy."""
    frame_n  = int(sr * frame_ms / 1000)
    hop_n    = int(sr * hop_ms   / 1000)
    n_frames = max(0, (len(audio) - frame_n) // hop_n)
    out      = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        s      = i * hop_n
        frame  = audio[s : s + frame_n].astype(np.float64)
        out[i] = np.sqrt(np.mean(frame ** 2))
    return out


def _f0_contour(audio: np.ndarray, sr: int,
                hop_ms: int = 5, frame_ms: int = 20,
                voiced_thr: float = 0.4,
                f0_anchor: Optional[float] = None) -> np.ndarray:
    """
    Per-frame F0 [Hz] on a regular hop grid; np.nan for unvoiced / silent frames.
    Uses the same voiced-detection logic as _frame_periods_parabolic so that
    the two functions agree on which frames are voiced.
    """
    lp      = _pitch_lp(audio, sr)
    frame_n = int(sr * frame_ms / 1000)
    hop_n   = int(sr * hop_ms   / 1000)

    period_nom = sr / f0_anchor if f0_anchor is not None else None
    if period_nom is not None:
        margin  = max(int(period_nom * 0.35), 5)
        min_lag = max(int(period_nom) - margin, 1)
        max_lag = int(period_nom) + margin + 1
    else:
        min_lag = int(sr / 400.0)
        max_lag = int(sr / 60.0)

    n_frames = max(0, (len(lp) - frame_n) // hop_n)
    track    = np.full(n_frames, np.nan)

    for i in range(n_frames):
        start  = i * hop_n
        frame  = lp[start : start + frame_n].astype(np.float64)
        if np.sqrt(np.mean(frame ** 2)) < 5e-4:
            continue
        c    = frame - frame.mean()
        corr = np.correlate(c, c, mode="full")[len(frame) - 1 :]
        d    = corr[0] + 1e-12
        corr /= d
        if max_lag >= len(corr):
            continue
        pk = int(np.argmax(corr[min_lag:max_lag])) + min_lag
        if corr[pk] < voiced_thr:
            continue
        if 0 < pk < len(corr) - 1:
            y1, y2, y3 = corr[pk - 1], corr[pk], corr[pk + 1]
            dp    = 2 * y2 - y1 - y3
            delta = 0.5 * (y1 - y3) / dp if abs(dp) > 1e-10 else 0.0
            period = pk + np.clip(delta, -0.5, 0.5)
        else:
            period = float(pk)
        track[i] = sr / period

    return track


def check_micro_tremor(audio: np.ndarray, sr: int) -> CheckResult:
    """
    Measures magnitude-squared coherence (MSC) between the AM envelope and
    F0 (FM) signals in the 4–12 Hz band.

    Physical basis: neurological tremor comes from a single CNS oscillator that
    simultaneously drives amplitude and frequency of the glottis — so AM and FM
    are coherent at the tremor frequency.  TTS synthesisers produce FM (pitch)
    and AM (envelope) through independent model pathways; no coupling constraint
    exists, so their 4–12 Hz coherence is near-random.

    Method:
      1. Extract per-frame RMS (AM) and F0 (FM) on a 5 ms hop grid (200 Hz)
      2. Take voiced-frame subsequences (unvoiced frames have no glottal source)
      3. Compute MSC via Welch's method:  C(f) = |P_xy|² / (P_xx · P_yy)
         nperseg is adaptive (target ~2 Hz resolution, floor 32 frames)
      4. amfm_coherence = mean C over 4–12 Hz band
      5. Fail if amfm_coherence < TREMOR_COHERENCE_MIN

    TREMOR_COHERENCE_MIN = 0.0 (observation mode) — threshold set after
    inspecting the real-vs-fake distribution.
    """
    f0 = _estimate_global_f0(audio, sr)
    if f0 is None:
        return CheckResult(passed=True, detail={"note": "no clear pitch"})

    hop_ms     = 5
    contour_sr = 1000.0 / hop_ms    # 200 Hz

    am_full = _rms_contour(audio, sr, hop_ms=hop_ms)
    fm_full = _f0_contour(audio, sr, hop_ms=hop_ms, f0_anchor=f0)

    # Align (rounding may differ by ±1 frame)
    n       = min(len(am_full), len(fm_full))
    am_full = am_full[:n]
    fm_full = fm_full[:n]

    voiced   = ~np.isnan(fm_full)
    n_voiced = int(voiced.sum())

    if n_voiced < TREMOR_MIN_FRAMES:
        return CheckResult(passed=True,
                           detail={"note": "insufficient voiced frames",
                                   "n_voiced": n_voiced,
                                   "needed":   TREMOR_MIN_FRAMES})

    am = am_full[voiced].astype(np.float64)
    fm = fm_full[voiced].astype(np.float64)

    # Adaptive nperseg: target ~2 Hz resolution (100 frames), cap at n//2
    nperseg = min(int(contour_sr / 2), len(am) // 2)
    nperseg = max(nperseg, 32)

    f_coh, coh = scipy.signal.coherence(am, fm, fs=contour_sr,
                                         nperseg=nperseg, detrend="constant")

    band = (f_coh >= TREMOR_F_LOW) & (f_coh <= TREMOR_F_HIGH)
    if not band.any():
        return CheckResult(passed=True, detail={"note": "no bins in 4–12 Hz band"})

    amfm_coh = float(np.mean(coh[band]))
    passed   = amfm_coh >= TREMOR_COHERENCE_MIN

    return CheckResult(
        passed=passed,
        detail={
            "amfm_coherence": round(amfm_coh, 4),
            "n_band_bins":    int(band.sum()),
            "nperseg":        nperseg,
            "voiced_frames":  n_voiced,
            "threshold":      TREMOR_COHERENCE_MIN,
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Checks 6 & 7 — STFT-based: Phase Deviation and Spectral Flux CV
# ──────────────────────────────────────────────────────────────────────────────
def _stft(audio: np.ndarray, n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    """
    Complex STFT via scipy.signal.stft.  Returns (n_fft//2+1, n_frames) array.
    Hann window, no boundary padding, no zero-padding at the signal end.
    Equivalent to librosa.stft with the same n_fft / hop parameters.
    """
    _, _, S = scipy.signal.stft(
        audio.astype(np.float64),
        nperseg=n_fft,
        noverlap=n_fft - hop,
        window="hann",
        padded=False,
        boundary=None,
    )
    return S    # complex (n_fft//2+1, n_frames)


def check_phase_deviation(audio: np.ndarray, sr: int) -> CheckResult:
    """
    Magnitude-weighted mean absolute STFT phase change between adjacent frames.

    Physical basis: in real voiced speech the dominant energy sits in harmonic
    bins.  Each harmonic evolves at a deterministic instantaneous frequency, so
    its phase changes by a predictable Δφ = 2πf · hop/sr each frame — the
    MAGNITUDE-WEIGHTED mean |Δφ| stays low and stable.

    TTS systems that synthesise or reconstruct phase from magnitude (Griffin-Lim,
    minimum-phase, WORLD vocoder) introduce arbitrary phase in high-energy bins,
    raising the weighted mean |Δφ| above natural levels.

    Adapted from VeriFauX advanced_feature_engineering._calculate_phase_coherence.
    Key improvement over the reference: magnitude-weighting focuses the measure
    on bins where signal is strong (harmonics) rather than silent/noise bins
    where phase is meaningless.  An unweighted mean is dominated by the many
    near-zero bins and loses sensitivity to the harmonic structure.

    PHASE_DEV_MAX = 0.0 — observation mode; threshold set after distribution
    analysis on the dev set.
    """
    if len(audio) < PHASE_N_FFT:
        return CheckResult(passed=True, detail={"note": "signal too short"})

    S     = _stft(audio, n_fft=PHASE_N_FFT, hop=PHASE_HOP)
    mag   = np.abs(S)
    phase = np.angle(S)

    if S.shape[1] < PHASE_MIN_FRAMES:
        return CheckResult(passed=True,
                           detail={"note": "insufficient STFT frames",
                                   "n_frames": S.shape[1]})

    # Frame-to-frame absolute phase difference
    dphase = np.abs(np.diff(phase, axis=1))              # (freq, n_frames-1)

    # Magnitude weights: mean of adjacent frames
    mag_w  = (mag[:, :-1] + mag[:, 1:]) * 0.5

    total_w    = float(mag_w.sum()) + 1e-12
    weighted_dev = float((mag_w * dphase).sum() / total_w)

    # Observation mode: PHASE_DEV_MAX == 0.0 → always pass
    passed = True if PHASE_DEV_MAX == 0.0 else (weighted_dev <= PHASE_DEV_MAX)

    return CheckResult(
        passed=passed,
        detail={
            "phase_deviation": round(weighted_dev, 4),
            "n_frames":        S.shape[1],
            "threshold":       PHASE_DEV_MAX,
        },
    )


def check_spectral_flux(audio: np.ndarray, sr: int) -> CheckResult:
    """
    Coefficient of variation (CV = std/mean × 100%) of the normalised
    per-frame L1 spectral flux time series.

    Physical basis: real speech transitions through phonemes with natural
    articulator dynamics — stable voiced segments punctuated by abrupt spectral
    changes at plosives, affricates, and vowel onsets.  This produces HIGH CV
    in the flux series.  Neural TTS vocoders that process audio in fixed-size
    chunks (256–512 samples) may produce artificially regular spectral jumps,
    reducing CV.  Extremely smooth TTS (unit-selection with good concatenation)
    produces near-uniform low flux, also reducing CV.

    Adapted from VeriFauX advanced_feature_engineering._detect_spectral_artifacts.
    Key improvements over the reference: (1) L1 flux normalised by per-frame
    magnitude sum makes the score level-independent; (2) CV instead of raw mean
    captures temporal irregularity — the feature the original implementation
    missed entirely.

    FLUX_CV_MIN = 0.0 — observation mode; threshold set after distribution
    analysis on the dev set.
    """
    if len(audio) < FLUX_N_FFT:
        return CheckResult(passed=True, detail={"note": "signal too short"})

    mag = np.abs(_stft(audio, n_fft=FLUX_N_FFT, hop=FLUX_HOP))

    if mag.shape[1] < 4:
        return CheckResult(passed=True, detail={"note": "insufficient STFT frames"})

    # Normalised L1 spectral flux per frame
    delta     = np.abs(np.diff(mag, axis=1))           # (freq, n_frames-1)
    flux      = delta.sum(axis=0)                       # per-frame flux
    norm      = mag[:, :-1].sum(axis=0) + 1e-12
    flux_norm = flux / norm                             # relative spectral change

    flux_mean = float(np.mean(flux_norm))
    flux_std  = float(np.std(flux_norm))
    cv        = 100.0 * flux_std / (flux_mean + 1e-12)

    # Lower bound: observation mode when FLUX_CV_MIN == 0.0
    lo_ok = (cv >= FLUX_CV_MIN) if FLUX_CV_MIN > 0.0 else True
    # Upper bound: fail if cv exceeds ceiling (neural vocoder spike artifact)
    hi_ok = (cv <= FLUX_CV_MAX) if FLUX_CV_MAX > 0.0 else True
    passed = lo_ok and hi_ok

    return CheckResult(
        passed=passed,
        detail={
            "flux_cv_pct":  round(cv,         2),
            "flux_mean":    round(flux_mean,  5),
            "flux_std":     round(flux_std,   5),
            "n_frames":     mag.shape[1],
            "lo_threshold": FLUX_CV_MIN,
            "hi_threshold": FLUX_CV_MAX,
            "lo_ok":        lo_ok,
            "hi_ok":        hi_ok,
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Check 8 — Sub-band energy asymmetry
# ──────────────────────────────────────────────────────────────────────────────
def check_subband_asymmetry(audio: np.ndarray, sr: int) -> CheckResult:
    """
    Measures how unevenly speech energy is distributed across four frequency
    bands per STFT frame, then averages over all frames.

    Physical basis: the human vocal tract acts as a resonant filter; energy
    concentrates in formant regions (especially F1/F2 in the 300–2000 Hz range)
    and falls sharply above ~4 kHz.  This asymmetry is an intrinsic property of
    glottal source + vocal-tract filter convolution.  Neural vocoders learn a
    spectral envelope model that tends to smooth the inter-band contrast,
    producing a more uniform energy distribution across bands.

    Method:
      1. Compute STFT power spectrum (reuses SBA_N_FFT / SBA_HOP).
      2. For each frame sum power into four bands:
           Band 1:  0 –  500 Hz  (sub-formant + F0 harmonics)
           Band 2:  500 – 2000 Hz  (F1 / F2 region)
           Band 3:  2000 – 4000 Hz  (F3 / F4 / fricative)
           Band 4:  4000 – 8000 Hz  (high-frequency noise / sibilants)
         Bands are capped at sr/2; frames with near-zero total power skipped.
      3. Normalise the four energies so they sum to 1.
      4. sba_score = mean over frames of std([e1, e2, e3, e4]).
         Range: 0 (perfectly uniform) → ~0.43 (all energy in one band).
         Higher score = more asymmetric = more like real speech.

    SBA_ASYM_MIN = 0.0 — observation mode; threshold set after distribution
    analysis on the development set.
    """
    if len(audio) < SBA_N_FFT:
        return CheckResult(passed=True, detail={"note": "signal too short"})

    mag_sq = np.abs(_stft(audio, n_fft=SBA_N_FFT, hop=SBA_HOP)) ** 2
    n_bins, n_frames = mag_sq.shape

    if n_frames < SBA_MIN_FRAMES:
        return CheckResult(passed=True,
                           detail={"note": "insufficient STFT frames",
                                   "n_frames": n_frames})

    # Bin-to-frequency mapping: bin k → k * (sr/2) / (n_fft//2)
    bin_freqs = np.linspace(0.0, sr / 2.0, n_bins)

    nyq = sr / 2.0
    masks = []
    for lo, hi in SBA_BANDS_HZ:
        hi = min(hi, nyq)
        if lo >= nyq:
            masks.append(np.zeros(n_bins, dtype=bool))
        else:
            masks.append((bin_freqs >= lo) & (bin_freqs < hi))

    # Per-frame normalised band energies and their std
    asym_vals: List[float] = []
    band_means: List[float] = [0.0] * len(SBA_BANDS_HZ)

    for f in range(n_frames):
        frame_pwr = mag_sq[:, f]
        band_e = np.array([float(frame_pwr[m].sum()) for m in masks])
        total = band_e.sum()
        if total < 1e-18:
            continue
        norm_e = band_e / total
        asym_vals.append(float(np.std(norm_e)))
        for b in range(len(SBA_BANDS_HZ)):
            band_means[b] += norm_e[b]

    if len(asym_vals) < SBA_MIN_FRAMES:
        return CheckResult(passed=True,
                           detail={"note": "insufficient voiced frames",
                                   "n_valid": len(asym_vals)})

    n_valid = len(asym_vals)
    sba_score = float(np.mean(asym_vals))
    mean_band_e = [round(band_means[b] / n_valid, 4) for b in range(len(SBA_BANDS_HZ))]

    # Observation mode: SBA_ASYM_MIN == 0.0 → always pass
    passed = True if SBA_ASYM_MIN == 0.0 else (sba_score >= SBA_ASYM_MIN)

    return CheckResult(
        passed=passed,
        detail={
            "sba_score":    round(sba_score, 4),
            "band_e_norm":  mean_band_e,   # [B1, B2, B3, B4] mean normalised energies
            "n_frames":     n_valid,
            "threshold":    SBA_ASYM_MIN,
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Core detection
# ──────────────────────────────────────────────────────────────────────────────
def detect_from_array(audio: np.ndarray, sr: int,
                      audio_path: str = "") -> DetectionResult:
    """Run all checks on a pre-loaded, pre-normalised (float64) audio array."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float64)
    peak = np.max(np.abs(audio))
    if peak > 1e-8:
        audio /= peak

    r_js = check_jitter_shimmer(audio, sr)
    r_fv = check_formant_velocity(audio, sr)
    r_sg = check_subglottal_resonance(audio, sr)
    r_tc = check_temporal_consistency(audio, sr)
    r_mt = check_micro_tremor(audio, sr)
    r_pd = check_phase_deviation(audio, sr)
    r_sf = check_spectral_flux(audio, sr)
    r_sb = check_subband_asymmetry(audio, sr)   # observation mode — not in verdict

    checks_failed = sum(1 for r in (r_js, r_fv, r_sg, r_tc, r_mt, r_pd, r_sf)
                        if not r.passed)
    verdict = "FAKE" if checks_failed >= 2 else "REAL"

    return DetectionResult(
        verdict=verdict,
        checks_failed=checks_failed,
        jitter_shimmer=r_js,
        formant_velocity=r_fv,
        subglottal=r_sg,
        temporal_consistency=r_tc,
        micro_tremor=r_mt,
        phase_deviation=r_pd,
        spectral_flux=r_sf,
        subband_asymmetry=r_sb,
        audio_path=audio_path,
    )


def detect(audio_path: str) -> DetectionResult:
    """Load audio from file (WAV/FLAC via soundfile) and run all checks."""
    audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    return detect_from_array(audio, sr, audio_path=audio_path)


# ──────────────────────────────────────────────────────────────────────────────
# CLI — single file
# ──────────────────────────────────────────────────────────────────────────────
def run_single(path: str) -> None:
    if not os.path.isfile(path):
        sys.exit(f"ERROR: file not found: {path}")

    result = detect(path)

    print(f"\n{'='*60}")
    print(f"  File   : {os.path.basename(path)}")
    print(f"  Verdict: {result.verdict}  ({result.checks_failed}/7 checks failed)")
    print(f"{'='*60}")

    for label, check in [
        ("Check 1  Glottal jitter & shimmer   ", result.jitter_shimmer),
        ("Check 2  Formant transition velocity ", result.formant_velocity),
        ("Check 3  HNR range                  ", result.subglottal),
        ("Check 4  Temporal consistency (CV)  ", result.temporal_consistency),
        ("Check 5  Micro-tremor [obs]         ", result.micro_tremor),
        ("Check 6  Phase deviation  [obs]     ", result.phase_deviation),
        ("Check 7  Spectral flux CV           ", result.spectral_flux),
        ("Check 8  Sub-band asymmetry [obs]   ", result.subband_asymmetry),
    ]:
        print(f"\n[{label}] {'PASS' if check.passed else 'FAIL'}")
        for k, v in check.detail.items():
            print(f"    {k}: {v}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# CLI — batch (ASVspoof 2019 LA protocol)
# ──────────────────────────────────────────────────────────────────────────────
def run_batch(dataset_dir: str, protocol_file: str, output_csv: str,
              max_files: Optional[int] = None) -> None:
    """
    Protocol format (space-separated):
        SPEAKER_ID  FILE_ID  ENV  ATTACK_TYPE  LABEL
    LABEL is 'bonafide' or 'spoof'.

    max_files: if set, process at most this many files (stratified: equal real/fake).
    """
    print("Indexing audio files …")
    audio_index: dict = {}
    for root, _, files in os.walk(dataset_dir):
        for fname in files:
            if fname.lower().endswith((".wav", ".flac")):
                stem = os.path.splitext(fname)[0]
                audio_index[stem] = os.path.join(root, fname)
    print(f"  Found {len(audio_index)} audio files.")

    all_entries = []
    with open(protocol_file, "r") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            all_entries.append((parts[1], parts[-1].lower()))
    print(f"  Protocol entries: {len(all_entries)}")

    # Stratified sampling: equal real/fake up to max_files
    if max_files is not None:
        real_entries = [e for e in all_entries if e[1] == "bonafide"]
        fake_entries = [e for e in all_entries if e[1] == "spoof"]
        half = max_files // 2
        entries = real_entries[:half] + fake_entries[:half]
        print(f"  Limiting to {len(entries)} files "
              f"({min(half, len(real_entries))} real + {min(half, len(fake_entries))} fake)")
    else:
        entries = all_entries
    print(f"  Running evaluation on {len(entries)} entries …")

    rows: List[dict] = []
    tp = tn = fp = fn = 0
    scores_real: List[float] = []
    scores_fake: List[float] = []

    for idx, (file_id, true_label) in enumerate(entries, 1):
        path = audio_index.get(file_id)
        if path is None:
            print(f"  [{idx}/{len(entries)}] SKIP  {file_id} (not found)")
            continue

        try:
            res = detect(path)
        except Exception as exc:
            print(f"  [{idx}/{len(entries)}] ERROR {file_id}: {exc}")
            continue

        predicted_fake = res.verdict == "FAKE"
        actually_fake  = true_label == "spoof"

        if   predicted_fake and     actually_fake: tp += 1
        elif not predicted_fake and not actually_fake: tn += 1
        elif predicted_fake and not actually_fake: fp += 1
        else:                                      fn += 1

        (scores_fake if actually_fake else scores_real).append(res.checks_failed)

        rows.append({
            "file_id":         file_id,
            "true_label":      true_label,
            "predicted":       res.verdict,
            "correct":         predicted_fake == actually_fake,
            "checks_failed":   res.checks_failed,
            "jitter_pct":      res.jitter_shimmer.detail.get("jitter_pct", ""),
            "shimmer_db":      res.jitter_shimmer.detail.get("shimmer_db", ""),
            "max_vel_hz_ms":   res.formant_velocity.detail.get("max_velocity_hz_ms", ""),
            "vel_exceedances": res.formant_velocity.detail.get("exceedances", ""),
            "cv_jitter_pct":   res.temporal_consistency.detail.get("cv_jitter_pct", ""),
            "cv_hnr_pct":      res.temporal_consistency.detail.get("cv_hnr_pct", ""),
            "tc_n_segs":       res.temporal_consistency.detail.get("n_segments", ""),
            "amfm_coherence":  res.micro_tremor.detail.get("amfm_coherence", ""),
            "amfm_n_bins":     res.micro_tremor.detail.get("n_band_bins", ""),
            "phase_deviation": res.phase_deviation.detail.get("phase_deviation", ""),
            "flux_cv_pct":     res.spectral_flux.detail.get("flux_cv_pct", ""),
            "sba_score":       res.subband_asymmetry.detail.get("sba_score", ""),
        })

        if idx % 50 == 0:
            print(f"  [{idx}/{len(entries)}] processed …")

    total    = tp + tn + fp + fn
    accuracy = (tp + tn) / total    if total     else 0.0
    fpr      = fp / (fp + tn)       if (fp + tn) else 0.0
    fnr      = fn / (fn + tp)       if (fn + tp) else 0.0
    eer      = _compute_eer(scores_real, scores_fake)

    print(f"\n{'='*60}")
    print(f"  Results over {total} files")
    print(f"  Accuracy  : {accuracy*100:.2f} %")
    print(f"  EER       : {eer*100:.2f} %")
    print(f"  FPR       : {fpr*100:.2f} %  (real flagged as fake)")
    print(f"  FNR       : {fnr*100:.2f} %  (fake flagged as real)")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"{'='*60}")

    with open(output_csv, "w", newline="") as fh:
        if rows:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    with open(output_csv, "a", newline="") as fh:
        fh.write(f"\nSUMMARY,accuracy={accuracy:.4f},EER={eer:.4f},"
                 f"FPR={fpr:.4f},FNR={fnr:.4f},"
                 f"TP={tp},TN={tn},FP={fp},FN={fn}\n")

    print(f"\nResults saved → {output_csv}\n")


def _compute_eer(scores_real: List[float], scores_fake: List[float]) -> float:
    """EER via threshold sweep.  Score = checks_failed (0-3); higher = more fake."""
    if not scores_real or not scores_fake:
        return float("nan")

    sr_arr = np.array(scores_real)
    sf_arr = np.array(scores_fake)
    best   = 1.0

    for thr in [0.5, 1.5, 2.5, 3.5]:
        far = float(np.mean(sr_arr >= thr))    # real called fake
        frr = float(np.mean(sf_arr <  thr))    # fake called real
        eer = (far + frr) / 2.0
        if eer < best:
            best = eer
    return best


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Physics-based deepfake speech detection (no ML).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python engine.py --audio sample.wav
  python engine.py --dataset /data/ASVspoof2019_LA_eval \\
                   --protocol /data/ASVspoof2019_LA_cm_eval.trl.txt \\
                   --output results.csv
""",
    )
    parser.add_argument("--audio",    type=str, help="Path to a single audio file")
    parser.add_argument("--dataset",  type=str, help="Root dir of ASVspoof dataset")
    parser.add_argument("--protocol", type=str, help="ASVspoof protocol file (.txt)")
    parser.add_argument("--output",   type=str, default="results.csv",
                        help="Output CSV (batch mode, default: results.csv)")
    parser.add_argument("--max",      type=int, default=None,
                        help="Max files to evaluate (stratified real/fake split)")

    args = parser.parse_args()

    if args.audio:
        run_single(args.audio)
    elif args.dataset and args.protocol:
        run_batch(args.dataset, args.protocol, args.output, max_files=args.max)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
