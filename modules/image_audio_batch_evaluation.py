"""Embed one user's encrypted face-image payload into WAV carriers."""

import argparse
import csv
import hashlib
import io
import math
from pathlib import Path
import struct

import numpy as np
import soundfile as sf
from PIL import Image
from modules.quality_evaluation import calculate_perceptual_metrics

from main import (
    QUANT_STEP,
    _bits_from_bytes,
    _embed_bit,
    decrypt_biometric_payload,
    key_path_for_user,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PAYLOAD = ROOT / "output" / "encrypted_payload.bin"
DEFAULT_AUDIO_ROOT = ROOT / "Data" / "audio_dtaset" / "VCC2020-database-master"
OUTPUT_DIR = ROOT / "output" / "image_audio_batch"
FRAME_HEADER_BYTES = 41
# Short batch carriers need lower redundancy than the main pipeline's 8x frame.
BATCH_REPETITIONS = 2
MIN_PAYLOAD_ACCURACY = 65.0
MIN_IMAGE_ACCURACY = 80.0


def frame_payload(payload):
    header = (
        b"CFD1"
        + struct.pack(">B", 1)
        + struct.pack(">I", len(payload))
        + hashlib.sha256(payload).digest()
    )
    return header + payload


def embed_into_audio(audio, payload):
    frame = frame_payload(payload)
    repeated_bits = np.repeat(_bits_from_bytes(frame), BATCH_REPETITIONS)
    mono = audio if audio.ndim == 1 else audio[:, 0]
    required_samples = 2 * len(repeated_bits)
    if len(mono) < required_samples:
        raise ValueError(
            f"Insufficient carrier capacity: need {required_samples} samples, "
            f"have {len(mono)}."
        )

    modified = mono.copy()
    for index, bit in enumerate(repeated_bits):
        pair_index = index * 2
        modified[pair_index], modified[pair_index + 1] = _embed_bit(
            modified[pair_index], modified[pair_index + 1], int(bit)
        )

    if audio.ndim == 1:
        stego = modified
    else:
        stego = audio.copy()
        stego[:, 0] = modified

    peak = float(np.max(np.abs(stego)))
    if peak > 1.0:
        stego /= peak
    return stego, frame


def recover_frame_bits(stego, frame_bit_count):
    mono = stego if stego.ndim == 1 else stego[:, 0]
    required_samples = frame_bit_count * BATCH_REPETITIONS * 2
    if len(mono) < required_samples:
        raise ValueError(
            f"Stego carrier is too short: need {required_samples} samples, "
            f"have {len(mono)}."
        )

    recovered_bits = []
    for bit_index in range(frame_bit_count):
        votes = []
        for repetition in range(BATCH_REPETITIONS):
            pair_index = (
                bit_index * BATCH_REPETITIONS + repetition
            ) * 2
            difference = mono[pair_index] - mono[pair_index + 1]
            votes.append(int(np.floor(difference / QUANT_STEP)) % 2)
        recovered_bits.append(int(sum(votes) > BATCH_REPETITIONS // 2))
    return np.asarray(recovered_bits, dtype=np.uint8)


def recover_payload(stego, payload_length):
    """Recover and validate one complete encrypted payload from a stego file."""
    frame_bits = recover_frame_bits(
        stego,
        (FRAME_HEADER_BYTES + payload_length) * 8,
    )
    frame = np.packbits(frame_bits, bitorder="big").tobytes()
    payload = frame[FRAME_HEADER_BYTES:]
    header = frame[:FRAME_HEADER_BYTES]
    checksum_ok = (
        header[:4] == b"CFD1"
        and header[4] == 1
        and struct.unpack(">I", header[5:9])[0] == payload_length
        and header[9:41] == hashlib.sha256(payload).digest()
    )
    if not checksum_ok:
        raise ValueError("Recovered payload checksum failed.")
    return payload


def image_accuracy(source_image_bytes, recovered_image_bytes):
    """Return exact-pixel and normalized-error image recovery measures."""
    with Image.open(io.BytesIO(source_image_bytes)) as source_image:
        source = np.asarray(source_image.convert("L"), dtype=np.float32)
    with Image.open(io.BytesIO(recovered_image_bytes)) as recovered_image:
        recovered = np.asarray(recovered_image.convert("L"), dtype=np.float32)

    source_image = Image.fromarray(source.astype(np.uint8)).resize(
        (recovered.shape[1], recovered.shape[0]), Image.Resampling.LANCZOS
    )
    resized_source = np.asarray(source_image, dtype=np.float32)
    absolute_error = np.abs(resized_source - recovered)
    return {
        "image_width": recovered.shape[1],
        "image_height": recovered.shape[0],
        "exact_pixel_accuracy_percent": float(
            100.0 * np.mean(resized_source == recovered)
        ),
        "mean_absolute_error": float(np.mean(absolute_error)),
        "normalized_image_accuracy_percent": float(
            100.0 * (1.0 - np.mean(absolute_error) / 255.0)
        ),
    }


def audio_quality_metrics(original, stego, sample_rate):
    """Measure waveform error and perceptual quality for one carrier pair."""
    original = np.asarray(original, dtype=np.float64)
    stego = np.asarray(stego, dtype=np.float64)
    if original.shape != stego.shape:
        raise ValueError(
            f"Original and stego audio shapes differ: "
            f"{original.shape} != {stego.shape}."
        )

    error = original - stego
    mse = float(np.mean(error ** 2))
    perceptual_metrics = calculate_perceptual_metrics(
        original,
        stego,
        sample_rate,
    )
    return {
        "audio_mse": mse,
        "audio_rmse": float(np.sqrt(mse)),
        "audio_mae": float(np.mean(np.abs(error))),
        **perceptual_metrics,
    }


def format_metric(value, digits=6):
    if value is None or not math.isfinite(value):
        return "N/A"
    return f"{value:.{digits}g}"


def find_carriers(audio_root, required_samples, count):
    candidates = []
    for path in sorted(audio_root.rglob("*.wav")):
        if "__MACOSX" in path.parts or path.name.startswith(("._", ".")):
            continue
        try:
            info = sf.info(str(path))
        except (RuntimeError, OSError):
            continue
        if info.frames >= required_samples:
            candidates.append(path)

    if len(candidates) < count:
        raise RuntimeError(
            f"Found {len(candidates)} WAV carriers with enough capacity under "
            f"{audio_root}; {count} are required."
        )
    return candidates[:count]


def evaluate_carrier(path, payload, source_image, key, output_dir, index):
    audio, sample_rate = sf.read(str(path), dtype="float64", always_2d=False)
    stego, expected_frame = embed_into_audio(audio, payload)
    output_path = output_dir / f"{index:02d}_{path.stem}_stego.wav"
    sf.write(str(output_path), stego, sample_rate, subtype="PCM_16")
    saved_stego, saved_rate = sf.read(str(output_path), dtype="float64")
    if saved_rate != sample_rate or saved_stego.shape != stego.shape:
        raise ValueError(f"WAV round-trip changed audio format for {path}.")

    audio_metrics = audio_quality_metrics(audio, saved_stego, sample_rate)
    expected_bits = _bits_from_bytes(expected_frame)
    recovered_bits = recover_frame_bits(saved_stego, len(expected_bits))
    payload_start = FRAME_HEADER_BYTES * 8
    payload_end = payload_start + len(payload) * 8
    expected_payload_bits = expected_bits[payload_start:payload_end]
    recovered_payload_bits = recovered_bits[payload_start:payload_end]
    matching_bits = int(np.count_nonzero(
        expected_payload_bits == recovered_payload_bits
    ))
    total_bits = len(expected_payload_bits)

    recovered_bytes = np.packbits(recovered_payload_bits, bitorder="big").tobytes()
    matching_bytes = sum(
        expected == actual
        for expected, actual in zip(payload, recovered_bytes)
    )
    recovered_header = np.packbits(
        recovered_bits[:payload_start], bitorder="big"
    ).tobytes()
    checksum_ok = (
        recovered_header[:4] == b"CFD1"
        and recovered_header[4] == 1
        and struct.unpack(">I", recovered_header[5:9])[0] == len(payload)
        and recovered_header[9:41] == hashlib.sha256(recovered_bytes).digest()
    )
    result = {
        "carrier": str(path.relative_to(ROOT)),
        "sample_rate_hz": sample_rate,
        "duration_seconds": len(audio) / sample_rate,
        **audio_metrics,
        "payload_bytes": len(payload),
        "matching_bytes": matching_bytes,
        "byte_accuracy_percent": 100.0 * matching_bytes / len(payload),
        "matching_bits": matching_bits,
        "total_bits": total_bits,
        "bit_accuracy_percent": 100.0 * matching_bits / total_bits,
        "checksum_pass": checksum_ok,
        "stego_file": str(output_path.relative_to(ROOT)),
        "image_recovery_pass": False,
        "normalized_image_accuracy_percent": 0.0,
        "exact_pixel_accuracy_percent": 0.0,
        "mean_absolute_error": 255.0,
        "recovered_image": "",
        "recovery_error": "",
    }
    if not checksum_ok:
        return result

    try:
        _, recovered_image, _ = decrypt_biometric_payload(
            recovered_bytes,
            key,
        )
        recovered_path = output_dir / f"{index:02d}_{path.stem}_recovered.jpg"
        recovered_path.write_bytes(recovered_image)
        image_metrics = image_accuracy(source_image, recovered_image)
        result.update(image_metrics)
        result["image_recovery_pass"] = True
        result["recovered_image"] = str(recovered_path.relative_to(ROOT))
    except Exception as error:
        result["recovery_error"] = str(error)
    return result


def resolve_key(key_path, user_id):
    if key_path is not None:
        if not key_path.is_file():
            raise FileNotFoundError(f"AES key not found: {key_path}")
        return key_path.read_bytes(), key_path
    if user_id:
        resolved = key_path_for_user(user_id)
        if resolved.is_file():
            return resolved.read_bytes(), resolved
    raise FileNotFoundError(
        "AES key is required to decrypt the recovered image. "
        "Pass --key or --user-id from the main.py run."
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Embed one user's encrypted face-image payload into 10 WAV files "
            "and report recovery and audio-quality metrics."
        )
    )
    parser.add_argument(
        "--payload",
        type=Path,
        default=DEFAULT_PAYLOAD,
        help="Encrypted image/face payload produced by main.py.",
    )
    parser.add_argument(
        "--source-image",
        type=Path,
        default=ROOT / "output" / "sender_face_cropped_160x160.jpg",
        help="Original cover image to compare with each recovered image.",
    )
    parser.add_argument(
        "--key",
        type=Path,
        default=None,
        help="AES key saved by main.py; alternatively use --user-id.",
    )
    parser.add_argument(
        "--user-id",
        default=None,
        help="User ID used by main.py to locate output/user_keys/<hash>.key.",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=DEFAULT_AUDIO_ROOT,
        help="Folder searched recursively for WAV carriers.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of audio files to test (default: 10).",
    )
    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be at least 1.")
    if not args.payload.is_file():
        raise FileNotFoundError(
            f"Encrypted payload not found: {args.payload}. "
            "Run main.py with the sender image first."
        )
    if not args.audio_root.is_dir():
        raise FileNotFoundError(f"Audio folder not found: {args.audio_root}")
    if not args.source_image.is_file():
        raise FileNotFoundError(f"Source image not found: {args.source_image}")

    payload = args.payload.read_bytes()
    if not payload:
        raise ValueError(f"Encrypted payload is empty: {args.payload}")

    required_samples = (
        (FRAME_HEADER_BYTES + len(payload)) * 8 * BATCH_REPETITIONS * 2
    )
    carriers = find_carriers(args.audio_root, required_samples, args.count)
    key, resolved_key_path = resolve_key(args.key, args.user_id)
    source_image = args.source_image.read_bytes()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for index, path in enumerate(carriers, start=1):
        result = evaluate_carrier(
            path,
            payload,
            source_image,
            key,
            OUTPUT_DIR,
            index,
        )
        results.append(result)
        print(
            f"{index:02d}. {Path(result['carrier']).name}: "
            f"MSE {format_metric(result['audio_mse'], 10)}, "
            f"RMSE {format_metric(result['audio_rmse'], 10)}, "
            f"MAE {format_metric(result['audio_mae'], 10)}, "
            f"PESQ {format_metric(result['pesq'])}, "
            f"STOI {format_metric(result['stoi'])}, "
            f"image {result['normalized_image_accuracy_percent']:.2f}%, "
            f"recovery {'PASS' if result['image_recovery_pass'] else 'FAIL'}, "
            f"SHA-256 {'PASS' if result['checksum_pass'] else 'FAIL'}"
        )

    total_matching_bytes = sum(row["matching_bytes"] for row in results)
    total_bytes = sum(row["payload_bytes"] for row in results)
    total_matching_bits = sum(row["matching_bits"] for row in results)
    total_bits = sum(row["total_bits"] for row in results)
    byte_accuracy = 100.0 * total_matching_bytes / total_bytes
    bit_accuracy = 100.0 * total_matching_bits / total_bits
    checksum_passes = sum(row["checksum_pass"] for row in results)
    image_recovery_passes = sum(row["image_recovery_pass"] for row in results)
    image_accuracy = sum(
        row["normalized_image_accuracy_percent"] for row in results
    ) / len(results)

    report_path = OUTPUT_DIR / "accuracy_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nAudio files tested: {len(results)}")
    print(f"One source image: {args.source_image}")
    print(f"AES key: {resolved_key_path}")
    print("\nMEAN AUDIO QUALITY METRICS")
    for metric_name, digits in (
        ("audio_mse", 10),
        ("audio_rmse", 10),
        ("audio_mae", 10),
        ("pesq", 6),
        ("stoi", 6),
    ):
        available_values = [
            row[metric_name]
            for row in results
            if row[metric_name] is not None
            and math.isfinite(row[metric_name])
        ]
        mean_value = (
            float(np.mean(available_values))
            if available_values
            else None
        )
        print(
            f"{metric_name.upper():<10}: "
            f"{format_metric(mean_value, digits)}"
        )
    if not any(row["pesq"] is not None for row in results):
        print("PESQ unavailable: install the optional 'pesq' package.")
    if not any(row["stoi"] is not None for row in results):
        print("STOI unavailable: install the 'pystoi' package.")
    print(f"Aggregate image recovery accuracy: {image_accuracy:.2f}%")
    print(f"Image recoveries: {image_recovery_passes}/{len(results)} passed")
    print(
        f"Image accuracy target: >= {MIN_IMAGE_ACCURACY:.0f}% - "
        f"{'PASS' if image_accuracy >= MIN_IMAGE_ACCURACY else 'FAIL'}"
    )
    print(f"Aggregate payload byte accuracy: {byte_accuracy:.2f}%")
    print(f"Aggregate payload bit accuracy: {bit_accuracy:.2f}%")
    print(
        f"Payload recovery accuracy target: >= {MIN_PAYLOAD_ACCURACY:.0f}% - "
        f"{'PASS' if bit_accuracy >= MIN_PAYLOAD_ACCURACY else 'FAIL'}"
    )
    print(f"SHA-256 integrity: {checksum_passes}/{len(results)} passed")
    print(f"Stego files and CSV report: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
