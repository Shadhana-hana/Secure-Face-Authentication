"""Compare the current QIM embedder with HIFI-Stego paper results.

Run from the project root with:
    python -m modules.paper_comparison

This uses two available VCC2020 clips of at least five seconds and labels
measurements that are not equivalent to the paper's VCC2018 experiments.
"""

from collections import defaultdict
import csv
from hashlib import sha256, shake_256
from pathlib import Path
import io
import math
import random
import struct

import librosa
import numpy as np
import soundfile as sf

from modules.quality_evaluation import calculate_vector_metrics, format_value, print_table


ROOT = Path(__file__).resolve().parent.parent
DATASET = (
    ROOT
    / "Data"
    / "audio_dtaset"
    / "VCC2020-database-master"
    / "vcc2020_database_evaluation"
    / "vcc2020_database_evaluation"
)

SAMPLE_RATE = 24000
DURATIONS = (1, 2, 3, 4, 5)
PAYLOAD_LENGTHS = (50, 100, 150)
SOURCE_LIMIT = 10
SUBJECTIVE_LISTENERS = 5
SUBJECTIVE_CLIP_COUNT = 10
PAYLOAD_ACCURACY_TARGET = 95.0
REPETITIONS = 8
QUANT_STEP = 0.002
FRAME_HEADER_BYTES = 41
MAGIC = b"CFD1"
SUBJECTIVE_DIR = ROOT / "output" / "subjective_study"
SUBJECTIVE_CLIPS_DIR = SUBJECTIVE_DIR / "clips"
SUBJECTIVE_KEY_PATH = SUBJECTIVE_DIR / "answer_key.csv"
SUBJECTIVE_RESPONSES_PATH = SUBJECTIVE_DIR / "responses.csv"
BATCH_ACCURACY_PATH = ROOT / "output" / "image_audio_batch" / "accuracy_report.csv"

PAPER_METHODS = ("LSB", "DCT", "CRS", "PixInWav", "XZ", "HIFI-Stego")
PAPER_SUBJECTIVE = {
    "LSB": 90,
    "DCT": 90,
    "CRS": 70,
    "PixInWav": 70,
    "XZ": 80,
    "HIFI-Stego": 60,
}
PAPER_SNR = (
    ((21.49, 23.14, 25.96, 24.04, 18.81, 24.01),
     (18.99, 20.76, 23.04, 22.42, 18.65, 23.17),
     (16.26, 17.82, 19.57, 16.95, 18.66, 19.59)),
    ((20.60, 23.63, 26.81, 24.64, 18.95, 26.90),
     (19.07, 19.03, 22.75, 21.85, 18.84, 23.20),
     (16.65, 15.40, 17.01, 16.67, 18.74, 18.99)),
    ((21.05, 23.96, 27.98, 26.99, 18.46, 29.77),
     (18.67, 20.46, 24.37, 25.78, 18.40, 27.68),
     (17.45, 18.18, 20.89, 19.45, 18.38, 20.09)),
    ((23.97, 23.88, 23.25, 28.74, 18.54, 28.76),
     (21.61, 21.74, 18.70, 26.98, 18.41, 27.08),
     (18.76, 15.00, 16.65, 21.03, 18.36, 20.91)),
    ((23.46, 21.01, 24.96, 25.62, 18.65, 25.69),
     (19.83, 19.73, 23.28, 22.81, 18.61, 23.43),
     (17.50, 16.99, 18.50, 17.99, 18.62, 19.77)),
)
PAPER_RECOVERY = (
    ((100.00, 100.00, 99.98, 98.72, 100.00, 99.32),
     (100.00, 100.00, 99.87, 97.65, 100.00, 99.60),
     (100.00, 100.00, 99.91, 98.01, 100.00, 99.71)),
    ((100.00, 100.00, 99.98, 99.01, 100.00, 99.85),
     (100.00, 100.00, 99.95, 98.57, 100.00, 98.01),
     (100.00, 100.00, 99.98, 97.90, 100.00, 99.75)),
    ((100.00, 100.00, 99.76, 96.98, 100.00, 99.83),
     (100.00, 100.00, 99.41, 98.23, 100.00, 99.60),
     (100.00, 100.00, 99.62, 95.37, 100.00, 99.41)),
    ((100.00, 100.00, 99.42, 98.27, 100.00, 99.19),
     (100.00, 100.00, 99.89, 97.58, 100.00, 98.90),
     (100.00, 100.00, 99.17, 94.69, 100.00, 99.57)),
    ((100.00, 100.00, 99.41, 98.13, 100.00, 99.70),
     (100.00, 100.00, 99.31, 96.55, 100.00, 99.19),
     (100.00, 100.00, 98.89, 96.89, 100.00, 98.99)),
)
PAPER_MSE = {
    "LSB": (0.0048, 0.4003, 0.0039),
    "DCT": (0.0289, 0.8131, 0.0295),
    "CRS": (0.0121, 0.6563, 0.0465),
    "PixInWav": (0.0568, 2.1522, 0.1090),
    "XZ": (0.0233, 1.8774, 0.0022),
    "HIFI-Stego": (0.0027, 0.3711, 0.0035),
}
PAPER_CAPACITY = {
    "LSB": (750, 750, 750, 750, 750),
    "DCT": (758, 449, 439, 388, 425),
    "CRS": (6158, 4254, 3324, 2574, 2702),
    "PixInWav": (842, 842, 842, 842, 842),
    "XZ": (2231, 1948, 1815, 1748, 1774),
    "HIFI-Stego sigma=1": (2048, 2048, 2048, 2048, 2048),
    "HIFI-Stego sigma=2": (4096, 4096, 4096, 4096, 4096),
    "HIFI-Stego sigma=3": (6144, 6144, 6144, 6144, 6144),
}


def find_source_clips():
    if not DATASET.exists():
        raise FileNotFoundError(f"VCC2020 evaluation dataset not found: {DATASET}")

    clips = []
    for path in sorted(DATASET.rglob("*.wav")):
        if "__MACOSX" in path.parts or path.name.startswith(("._", ".")):
            continue
        try:
            info = sf.info(str(path))
        except (RuntimeError, OSError):
            continue
        if (
            info.samplerate == SAMPLE_RATE
            and info.channels == 1
            and info.duration >= DURATIONS[-1]
        ):
            clips.append(path)

    clips = clips[:SOURCE_LIMIT]
    if not clips:
        raise RuntimeError(
            "No clean 24 kHz mono WAV clips of at least five seconds were found."
        )
    return clips


def create_subjective_study():
    if SUBJECTIVE_KEY_PATH.exists():
        missing_clips = [
            f"clip_{index:02d}.wav"
            for index in range(1, SUBJECTIVE_CLIP_COUNT + 1)
            if not (SUBJECTIVE_CLIPS_DIR / f"clip_{index:02d}.wav").exists()
        ]
        if missing_clips:
            raise FileNotFoundError(
                "Subjective study answer key exists but clip files are missing: "
                + ", ".join(missing_clips)
            )
        return

    grouped_sources = defaultdict(list)
    for path in sorted(DATASET.rglob("*.wav")):
        if "__MACOSX" in path.parts or path.name.startswith(("._", ".")):
            continue
        try:
            info = sf.info(str(path))
        except (RuntimeError, OSError):
            continue
        if (
            info.samplerate == SAMPLE_RATE
            and info.channels == 1
            and info.frames >= 2 * SAMPLE_RATE
        ):
            grouped_sources[path.parent.name].append(path)

    chooser = random.SystemRandom()
    sources = [
        chooser.choice(grouped_sources[group])
        for group in sorted(grouped_sources)
    ]
    if len(sources) > SUBJECTIVE_CLIP_COUNT // 2:
        sources = sources[:SUBJECTIVE_CLIP_COUNT // 2]
    elif len(sources) < SUBJECTIVE_CLIP_COUNT // 2:
        remaining_sources = [
            path
            for paths in grouped_sources.values()
            for path in paths
            if path not in sources
        ]
        sources.extend(
            chooser.sample(
                remaining_sources,
                SUBJECTIVE_CLIP_COUNT // 2 - len(sources),
            )
        )

    if len(sources) != SUBJECTIVE_CLIP_COUNT // 2:
        raise RuntimeError(
            "The subjective study requires five 24 kHz mono clips "
            "that are at least two seconds long."
        )

    SUBJECTIVE_CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    blinded_clips = []
    key_rows = []
    payload = make_text_payload(50)
    for source_path in sources:
        audio, sample_rate = sf.read(source_path, dtype="float64")
        if sample_rate != SAMPLE_RATE or audio.ndim != 1:
            raise ValueError(f"Unexpected format in subjective source: {source_path}")
        original = audio[:2 * SAMPLE_RATE]
        stego, status = embed_and_save_roundtrip(original, payload)
        if status != "MEASURED":
            raise RuntimeError(
                f"Two-second clip cannot hold the 50-character test payload: {source_path}"
            )
        blinded_clips.extend((
            ("original", source_path.name, original),
            ("stego", source_path.name, stego),
        ))

    random.SystemRandom().shuffle(blinded_clips)
    for index, (label, source_name, audio) in enumerate(blinded_clips, start=1):
        clip_id = f"clip_{index:02d}"
        sf.write(
            SUBJECTIVE_CLIPS_DIR / f"{clip_id}.wav",
            audio,
            SAMPLE_RATE,
            subtype="PCM_16",
        )
        key_rows.append({
            "clip_id": clip_id,
            "actual_label": label,
            "source_file": source_name,
        })

    with SUBJECTIVE_KEY_PATH.open("w", newline="", encoding="utf-8") as key_file:
        writer = csv.DictWriter(
            key_file,
            fieldnames=("clip_id", "actual_label", "source_file"),
        )
        writer.writeheader()
        writer.writerows(key_rows)

    with SUBJECTIVE_RESPONSES_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as response_file:
        writer = csv.DictWriter(
            response_file,
            fieldnames=("listener_id", "clip_id", "predicted_label"),
        )
        writer.writeheader()
        for listener_index in range(1, SUBJECTIVE_LISTENERS + 1):
            for clip_index in range(1, SUBJECTIVE_CLIP_COUNT + 1):
                writer.writerow({
                    "listener_id": f"L{listener_index}",
                    "clip_id": f"clip_{clip_index:02d}",
                    "predicted_label": "",
                })


def score_subjective_responses():
    create_subjective_study()
    with SUBJECTIVE_KEY_PATH.open(newline="", encoding="utf-8") as key_file:
        answer_key = {
            row["clip_id"]: row["actual_label"].strip().lower()
            for row in csv.DictReader(key_file)
        }

    responses = []
    seen = set()
    with SUBJECTIVE_RESPONSES_PATH.open(newline="", encoding="utf-8") as response_file:
        for row in csv.DictReader(response_file):
            listener_id = row["listener_id"].strip()
            clip_id = row["clip_id"].strip()
            prediction = row["predicted_label"].strip().lower()
            if not prediction:
                continue
            if listener_id not in {f"L{i}" for i in range(1, SUBJECTIVE_LISTENERS + 1)}:
                raise ValueError(f"Unexpected listener_id in response CSV: {listener_id!r}")
            if clip_id not in answer_key:
                raise ValueError(f"Unknown clip_id in response CSV: {clip_id!r}")
            if prediction not in {"original", "stego"}:
                raise ValueError(
                    f"Invalid prediction for {listener_id}/{clip_id}: "
                    "use 'original' or 'stego'."
                )
            response_key = (listener_id, clip_id)
            if response_key in seen:
                raise ValueError(
                    f"Duplicate response for {listener_id}/{clip_id}."
                )
            seen.add(response_key)
            responses.append(prediction == answer_key[clip_id])

    correct = sum(responses)
    total = len(responses)
    expected = SUBJECTIVE_LISTENERS * SUBJECTIVE_CLIP_COUNT
    if total == 0:
        project_accuracy = "N/A (listener study not measured)"
    else:
        percentage = 100.0 * correct / total
        qualifier = "complete" if total == expected else "partial"
        project_accuracy = (
            f"{percentage:.2f}% ({correct}/{total}, {qualifier}; "
            f"{total}/{expected} ratings)"
        )
    return project_accuracy, total, expected


def score_batch_image_recovery():
    """Return normalized and exact-pixel similarity from the 10-audio report."""
    if not BATCH_ACCURACY_PATH.exists():
        return "N/A (run 10-audio batch)", "N/A (run 10-audio batch)"

    with BATCH_ACCURACY_PATH.open(newline="", encoding="utf-8") as report_file:
        rows = list(csv.DictReader(report_file))

    if not rows:
        return "N/A (empty batch report)", "N/A (empty batch report)"

    normalized_values = [
        float(row["normalized_image_accuracy_percent"])
        for row in rows
        if row.get("normalized_image_accuracy_percent")
    ]
    exact_pixel_values = [
        float(row["exact_pixel_accuracy_percent"])
        for row in rows
        if row.get("exact_pixel_accuracy_percent")
    ]
    recovered = sum(
        row.get("image_recovery_pass", "").strip().lower() == "true"
        for row in rows
    )
    if not normalized_values or not exact_pixel_values:
        unavailable = "N/A (no image metrics)"
        return unavailable, unavailable
    return (
        f"{np.mean(normalized_values):.2f}% ({recovered}/{len(rows)} recovered)",
        f"{np.mean(exact_pixel_values):.2f}% ({recovered}/{len(rows)} recovered)",
    )


def make_frame(payload):
    return (
        MAGIC
        + struct.pack(">B", 1)
        + struct.pack(">I", len(payload))
        + sha256(payload).digest()
        + payload
    )


def make_text_payload(length):
    seed = b"Secure-Face-Authentication HIFI comparison payload"
    return shake_256(seed).hexdigest((length + 1) // 2)[:length].encode("ascii")


def bits_from_bytes(data):
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big")


def embed_bit(first, second, bit):
    mean = (first + second) / 2.0
    difference = first - second
    base = int(np.floor(difference / QUANT_STEP))
    candidates = [index for index in range(base - 2, base + 3) if index % 2 == bit]
    target = min(
        candidates,
        key=lambda index: abs((index + 0.5) * QUANT_STEP - difference),
    )
    target_difference = (target + 0.5) * QUANT_STEP
    return mean + target_difference / 2.0, mean - target_difference / 2.0


def embed_and_save_roundtrip(original, payload):
    frame = make_frame(payload)
    repeated_bits = np.repeat(bits_from_bytes(frame), REPETITIONS)
    required_samples = 2 * len(repeated_bits)
    if required_samples > len(original):
        return None, "INSUFFICIENT_CAPACITY"

    stego = original.copy()
    for index, bit in enumerate(repeated_bits):
        pair_index = index * 2
        stego[pair_index], stego[pair_index + 1] = embed_bit(
            stego[pair_index], stego[pair_index + 1], int(bit)
        )

    peak = float(np.max(np.abs(stego)))
    if peak > 1.0:
        stego /= peak

    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, stego, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    wav_buffer.seek(0)
    saved_stego, saved_rate = sf.read(wav_buffer, dtype="float64")
    if saved_rate != SAMPLE_RATE or saved_stego.shape != original.shape:
        raise ValueError("PCM round-trip changed the sample rate or waveform length.")
    return saved_stego, "MEASURED"


def recover_payload(stego):
    bit_count = (FRAME_HEADER_BYTES * 8)
    frame_bits = []
    for bit_index in range(bit_count):
        votes = []
        for repetition in range(REPETITIONS):
            pair_index = (bit_index * REPETITIONS + repetition) * 2
            difference = stego[pair_index] - stego[pair_index + 1]
            votes.append(int(np.floor(difference / QUANT_STEP)) % 2)
        frame_bits.append(1 if sum(votes) > REPETITIONS // 2 else 0)

    header = np.packbits(frame_bits, bitorder="big").tobytes()
    if header[:4] != MAGIC or header[4] != 1:
        return b"", False

    payload_length = struct.unpack(">I", header[5:9])[0]
    total_bits = (FRAME_HEADER_BYTES + payload_length) * 8
    if total_bits * REPETITIONS * 2 > len(stego):
        return b"", False

    payload_bits = []
    for bit_index in range(FRAME_HEADER_BYTES * 8, total_bits):
        votes = []
        for repetition in range(REPETITIONS):
            pair_index = (bit_index * REPETITIONS + repetition) * 2
            difference = stego[pair_index] - stego[pair_index + 1]
            votes.append(int(np.floor(difference / QUANT_STEP)) % 2)
        payload_bits.append(1 if sum(votes) > REPETITIONS // 2 else 0)

    recovered = np.packbits(payload_bits, bitorder="big").tobytes()
    checksum_ok = sha256(recovered).digest() == header[9:41]
    return recovered, checksum_ok


def calculate_run(original, stego, payload):
    waveform_metrics = calculate_vector_metrics(original, stego)
    signal_power = float(np.mean(original ** 2))
    snr_db = (
        float("inf") if waveform_metrics["mse"] == 0
        else 10.0 * math.log10(signal_power / waveform_metrics["mse"])
        if signal_power > 0 else float("-inf")
    )

    original_mfcc = librosa.feature.mfcc(
        y=original, sr=SAMPLE_RATE, n_mfcc=13, n_fft=1024, hop_length=256
    )
    stego_mfcc = librosa.feature.mfcc(
        y=stego, sr=SAMPLE_RATE, n_mfcc=13, n_fft=1024, hop_length=256
    )
    content_mse = calculate_vector_metrics(
        original_mfcc[:8], stego_mfcc[:8]
    )["mse"]
    style_mse = calculate_vector_metrics(
        original_mfcc[8:], stego_mfcc[8:]
    )["mse"]

    recovered, checksum_ok = recover_payload(stego)
    compared_length = max(len(payload), len(recovered))
    matching_bytes = sum(
        expected == actual for expected, actual in zip(payload, recovered)
    )
    recovery_accuracy = (
        100.0 * matching_bytes / compared_length if compared_length else 100.0
    )

    return {
        "snr_db": snr_db,
        "waveform_mse": waveform_metrics["mse"],
        "content_proxy_mse": content_mse,
        "style_proxy_mse": style_mse,
        "recovery_accuracy": recovery_accuracy,
        "recovery_correct_bytes": matching_bytes,
        "recovery_total_bytes": compared_length,
        "checksum_ok": checksum_ok,
    }


def mean_or_none(values, key):
    present = [value[key] for value in values if value[key] is not None]
    return float(np.mean(present)) if present else None


def main():
    clips = find_source_clips()
    print("=" * 96)
    print("HIFI-STEGO PAPER COMPARISON: PAPER REFERENCES VS CURRENT QIM IMPLEMENTATION")
    print("=" * 96)
    print(f"Paper corpus: VCC2018 | Project corpus: VCC2020 | Source clips per condition: {len(clips)}")
    print("Current embedder: QIM sample-pair embedding, 8 repetitions, 0.002 quantization step")
    print("Paper reference columns include all methods shown in the requested tables.")

    results = defaultdict(list)
    for duration_index, duration in enumerate(DURATIONS):
        sample_count = SAMPLE_RATE * duration
        for payload_length in PAYLOAD_LENGTHS:
            key = (duration, payload_length)
            payload = make_text_payload(payload_length)
            for clip_path in clips:
                audio, sample_rate = sf.read(clip_path, dtype="float64")
                if sample_rate != SAMPLE_RATE or audio.ndim != 1:
                    raise ValueError(f"Unexpected format in benchmark clip: {clip_path}")
                original = audio[:sample_count]
                if original.size != sample_count:
                    continue
                stego, status = embed_and_save_roundtrip(original, payload)
                if status == "MEASURED":
                    results[key].append(calculate_run(original, stego, payload))

    snr_rows = []
    recovery_rows = []
    for duration_index, duration in enumerate(DURATIONS):
        for payload_index, payload_length in enumerate(PAYLOAD_LENGTHS):
            key = (duration, payload_length)
            values = results[key]
            count = len(values)
            snr = mean_or_none(values, "snr_db")
            recovery = mean_or_none(values, "recovery_accuracy")
            checksum_percent = (
                100.0 * np.mean([value["checksum_ok"] for value in values])
                if values else None
            )
            paper_snr_values = PAPER_SNR[duration_index][payload_index]
            paper_recovery_values = PAPER_RECOVERY[duration_index][payload_index]
            snr_rows.append([
                f"{duration}s", payload_length,
                *(f"{value:.2f}" for value in paper_snr_values),
                format_value(snr, 4) if snr is not None else "N/A (capacity)",
                count,
            ])
            recovery_rows.append([
                f"{duration}s", payload_length,
                *(f"{value:.2f}%" for value in paper_recovery_values),
                format_value(recovery, 2) + "%" if recovery is not None else "N/A (capacity)",
                format_value(checksum_percent, 2) + "%" if checksum_percent is not None else "N/A",
                count,
            ])

    print_table(
        "TABLE 1 - AUDIO QUALITY BY SNR (higher is better)",
        ("Duration", "Payload chars", *(f"{method} dB" for method in PAPER_METHODS), "Project QIM dB", "n"),
        snr_rows,
    )
    all_runs = [value for values in results.values() for value in values]
    recovered_bytes = sum(
        value["recovery_correct_bytes"] for value in all_runs
    )
    compared_bytes = sum(value["recovery_total_bytes"] for value in all_runs)
    payload_accuracy = (
        100.0 * recovered_bytes / compared_bytes if compared_bytes else None
    )
    payload_accuracy_label = (
        f"{format_value(payload_accuracy, 2)}% "
        f"({'PASS' if payload_accuracy >= PAYLOAD_ACCURACY_TARGET else 'BELOW TARGET'}; "
        f"target >= {PAYLOAD_ACCURACY_TARGET:.0f}%)"
        if payload_accuracy is not None else "N/A (capacity)"
    )
    subjective_accuracy, response_count, expected_responses = score_subjective_responses()
    batch_image_similarity, batch_exact_pixel_similarity = (
        score_batch_image_recovery()
    )
    print_table(
        "TABLE 3 - PROJECT AUDIO DETECTABILITY AND RECOVERY METRICS",
        ("System", "LSB", "DCT", "CRS", "PixInWav", "XZ", "HIFI-Stego", "Project QIM"),
        (
            (
                "Listener detection accuracy (lower = less detectable)",
                "90%",
                "90%",
                "70%",
                "70%",
                "80%",
                "60%",
                subjective_accuracy,
            ),
            (
                "Payload byte recovery (higher = better exact recovery)",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                payload_accuracy_label,
            ),
            (
                "Image normalized pixel similarity (higher = more similar)",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                batch_image_similarity,
            ),
            (
                "Image exact-pixel equality (strict pixel match)",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                batch_exact_pixel_similarity,
            ),
        ),
    )

    waveform_mse = mean_or_none(all_runs, "waveform_mse")
    content_proxy_mse = mean_or_none(all_runs, "content_proxy_mse")
    style_proxy_mse = mean_or_none(all_runs, "style_proxy_mse")
    print_table(
        "TABLE 4 - MSE FOR AUDIO, STYLE, AND CONTENT VECTORS",
        ("Method", "Audio MSE", "Style MSE", "Content MSE"),
        tuple(
            (method, *(format_value(value, 4) for value in values))
            for method, values in PAPER_MSE.items()
        ) + (
            (
                "Project QIM (MFCC proxies)",
                format_value(waveform_mse, 10) if waveform_mse is not None else "N/A",
                format_value(style_proxy_mse, 10) if style_proxy_mse is not None else "N/A",
                format_value(content_proxy_mse, 10) if content_proxy_mse is not None else "N/A",
            ),
        ),
    )

    project_capacity_rates = []
    project_capacity_bytes = []
    for index, duration in enumerate(DURATIONS):
        samples = SAMPLE_RATE * duration
        total_frame_bytes = samples // (2 * REPETITIONS) // 8
        payload_capacity_bytes = max(0, total_frame_bytes - FRAME_HEADER_BYTES)
        project_payload_bps = payload_capacity_bytes * 8 / duration
        project_capacity_rates.append(format_value(project_payload_bps, 1))
        project_capacity_bytes.append(str(payload_capacity_bytes))
    capacity_rows = [
        (method, *(str(rate) for rate in rates))
        for method, rates in PAPER_CAPACITY.items()
    ]
    capacity_rows.append(("Project QIM usable bps", *project_capacity_rates))
    capacity_rows.append(("Project QIM payload bytes", *project_capacity_bytes))
    print_table(
        "TABLE 5 - STEGANOGRAPHIC CAPACITY (payload rate; higher is better)",
        ("Method", *(f"{duration}s" for duration in DURATIONS)),
        capacity_rows,
    )
    print_table(
        "TABLE 6 - SECRET MESSAGE RECOVERY ACCURACY (higher is better)",
        (
            "Duration", "Payload chars", *PAPER_METHODS,
            "Project QIM", "SHA-256 pass", "n",
        ),
        recovery_rows,
    )

    print("\nNotes: Project averages use successful PCM_16 WAV round-trips only.")
    print(
        f"Subjective responses: {response_count}/{expected_responses}; "
        f"fill predicted_label in {SUBJECTIVE_RESPONSES_PATH}."
    )
    print(
        "Payload byte recovery measures exact encrypted-byte transport; "
        "image similarity measures JPEG pixel differences, not audio quality."
    )
    print(
        f"Blinded listening clips: {SUBJECTIVE_CLIPS_DIR}. "
        "Keep answer_key.csv private from listeners."
    )
    print("MFCC content/style splits are diagnostic proxies, not HIFI-Stego latent vectors.")
    print("A 'N/A (capacity)' entry means the current 8-repeat embedder cannot fit that payload.")


if __name__ == "__main__":
    main()