
from pathlib import Path
from importlib import import_module
import math
from typing import Any

import librosa
import numpy as np
import soundfile as sf

try:
    stoi: Any = import_module("pystoi.stoi").stoi
except ImportError:  # STOI is unavailable until the optional package is installed.
    stoi = None

try:
    pesq: Any = import_module("pesq").pesq
except ImportError:  # PESQ uses a native extension that may not build on Windows.
    pesq = None


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
ORIGINAL_PATH = OUTPUT_DIR / "carrier_audio_before_embedding.wav"
STEGO_PATH = OUTPUT_DIR / "stego_audio.wav"
PAYLOAD_PATH = OUTPUT_DIR / "encrypted_payload.bin"

REPETITIONS = 8
HEADER_BYTES = 41


def format_value(value, digits=8):
    if value is None:
        return "N/A"
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return "N/A"
        if math.isinf(float(value)):
            return "Infinity" if value > 0 else "-Infinity"
        return f"{value:.{digits}f}"
    return str(value)


def print_table(title, headers, rows):
    values = [[str(value) for value in row] for row in rows]
    widths = [
        max(len(str(header)), *(len(row[index]) for row in values))
        for index, header in enumerate(headers)
    ]
    border = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    print(f"\n{title}")
    print(border)
    print("| " + " | ".join(
        str(header).ljust(widths[index])
        for index, header in enumerate(headers)
    ) + " |")
    print(border)
    for row in values:
        print("| " + " | ".join(
            row[index].ljust(widths[index])
            for index in range(len(headers))
        ) + " |")
    print(border)


def calculate_vector_metrics(reference, comparison):
    reference = np.asarray(reference, dtype=np.float64).ravel()
    comparison = np.asarray(comparison, dtype=np.float64).ravel()
    if reference.shape != comparison.shape:
        raise ValueError(
            f"Compared arrays must have the same shape: "
            f"{reference.shape} != {comparison.shape}"
        )

    error = reference - comparison
    mse = float(np.mean(error ** 2))
    denominator = float(np.linalg.norm(reference) * np.linalg.norm(comparison))
    correlation = (
        float(np.corrcoef(reference, comparison)[0, 1])
        if np.std(reference) > 0 and np.std(comparison) > 0
        else float("nan")
    )

    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(error))),
        "correlation": correlation,
        "cosine_similarity": (
            float(np.dot(reference, comparison) / denominator)
            if denominator > 0 else float("nan")
        ),
    }


def calculate_perceptual_metrics(reference, comparison, sample_rate):
    reference = np.asarray(reference, dtype=np.float32)
    comparison = np.asarray(comparison, dtype=np.float32)
    if reference.ndim == 2:
        reference = np.asarray(reference.mean(axis=1), dtype=np.float32)
    if comparison.ndim == 2:
        comparison = np.asarray(comparison.mean(axis=1), dtype=np.float32)
    if reference.ndim != 1 or comparison.ndim != 1:
        raise ValueError(
            "PESQ and STOI require mono audio or frames-by-channels arrays."
        )
    if reference.shape != comparison.shape:
        raise ValueError(
            "PESQ and STOI require equal-length signals: "
            f"{reference.shape} != {comparison.shape}"
        )
    if reference.size == 0 or sample_rate <= 0:
        raise ValueError("Audio and sample rate must be non-empty and positive.")
    if not np.isfinite(reference).all() or not np.isfinite(comparison).all():
        raise ValueError("Audio contains non-finite values.")

    perceptual_score = None
    if pesq is not None:
        pesq_sample_rate = 8000 if sample_rate <= 8000 else 16000
        pesq_reference = librosa.resample(
            reference,
            orig_sr=sample_rate,
            target_sr=pesq_sample_rate,
        )
        pesq_comparison = librosa.resample(
            comparison,
            orig_sr=sample_rate,
            target_sr=pesq_sample_rate,
        )
        perceptual_score = float(
            pesq(
                pesq_sample_rate,
                pesq_reference,
                pesq_comparison,
                "nb" if pesq_sample_rate == 8000 else "wb",
            )
        )
    intelligibility_score = (
        float(stoi(reference, comparison, sample_rate, extended=False))
        if stoi is not None
        else None
    )
    return {
        "pesq": perceptual_score,
        "stoi": intelligibility_score,
    }


def calculate_feature_rows(original, stego, sample_rate):
    original_mono = np.mean(original, axis=1)
    stego_mono = np.mean(stego, axis=1)

    def get_features(audio):
        return librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=13,
            n_fft=1024,
            hop_length=256,
        )

    original_features = get_features(original_mono)
    stego_features = get_features(stego_mono)
    feature_sets = (
        ("All MFCC coefficients", original_features, stego_features),
        ("Content proxy (MFCC 0-7)", original_features[:8], stego_features[:8]),
        ("Style proxy (MFCC 8-12)", original_features[8:], stego_features[8:]),
    )

    rows = []
    for name, reference, comparison in feature_sets:
        metrics = calculate_vector_metrics(reference, comparison)
        rows.append([
            name,
            f"{reference.shape[0]} x {reference.shape[1]}",
            format_value(metrics["mse"], 10),
            format_value(metrics["rmse"], 10),
            format_value(metrics["mae"], 10),
            format_value(metrics["cosine_similarity"], 8),
        ])
    return rows


def main():
    for audio_path in (ORIGINAL_PATH, STEGO_PATH):
        if not audio_path.exists():
            raise FileNotFoundError(f"Required audio file not found: {audio_path}")

    original, original_sr = sf.read(
        ORIGINAL_PATH, dtype="float64", always_2d=True
    )
    stego, stego_sr = sf.read(
        STEGO_PATH, dtype="float64", always_2d=True
    )

    if original_sr != stego_sr:
        raise ValueError(
            f"Sample rates differ: original={original_sr}, stego={stego_sr}"
        )
    if original.shape != stego.shape:
        raise ValueError(
            f"Audio shapes differ: original={original.shape}, "
            f"stego={stego.shape}. Metrics require identical lengths/channels."
        )

    metrics = calculate_vector_metrics(original, stego)
    perceptual_metrics = calculate_perceptual_metrics(
        original,
        stego,
        original_sr,
    )
    signal_power = float(np.mean(original ** 2))
    noise_power = metrics["mse"]
    snr_db = (
        float("inf") if noise_power == 0
        else 10.0 * math.log10(signal_power / noise_power)
        if signal_power > 0 else float("-inf")
    )
    sample_count = original.shape[0]
    duration = sample_count / original_sr

    print("=" * 78)
    print("AUDIO QUALITY EVALUATION")
    print("=" * 78)
    print_table(
        "Waveform quality metrics",
        ("Metric", "Measured value"),
        (
            ("Original carrier", str(ORIGINAL_PATH)),
            ("Stego audio", str(STEGO_PATH)),
            ("Sample rate (Hz)", original_sr),
            ("Channels", original.shape[1]),
            ("Samples per channel", sample_count),
            ("Duration (seconds)", format_value(duration, 6)),
            ("SNR (dB)", format_value(snr_db, 8)),
            ("MSE", format_value(metrics["mse"], 12)),
            ("RMSE", format_value(metrics["rmse"], 12)),
            ("MAE", format_value(metrics["mae"], 12)),
            ("Pearson correlation", format_value(metrics["correlation"], 10)),
            ("Cosine similarity", format_value(metrics["cosine_similarity"], 10)),
            ("PESQ", format_value(perceptual_metrics["pesq"], 6)),
            ("STOI", format_value(perceptual_metrics["stoi"], 6)),
        ),
    )

    print_table(
        "MFCC feature comparison (diagnostic proxies, not HIFI latent vectors)",
        ("Feature set", "Shape", "MSE", "RMSE", "MAE", "Cosine similarity"),
        calculate_feature_rows(original, stego, original_sr),
    )

    pair_count = sample_count // 2
    capacity_bits = pair_count // REPETITIONS
    capacity_bytes = capacity_bits // 8
    payload_bytes = PAYLOAD_PATH.stat().st_size if PAYLOAD_PATH.exists() else 0
    frame_bytes = HEADER_BYTES + payload_bytes
    frame_bits = frame_bytes * 8
    utilization = (
        100.0 * frame_bytes / capacity_bytes if capacity_bytes else float("nan")
    )
    print_table(
        "Embedding capacity (theoretical for this sample-pair implementation)",
        ("Metric", "Value"),
        (
            ("Available sample pairs", f"{pair_count:,}"),
            ("Estimated capacity (bytes)", f"{capacity_bytes:,}"),
            ("Encrypted payload (bytes)", f"{payload_bytes:,}"),
            ("Frame including header (bytes)", f"{frame_bytes:,}"),
            ("Frame bitrate (bps)", format_value(frame_bits / duration, 4)),
            ("Capacity utilization (%)", format_value(utilization, 4)),
        ),
    )


if __name__ == "__main__":
    main()
