"""
=============================================================================
  SECURE FACE AUTHENTICATION — AES-256-GCM + AUDIO STEGANOGRAPHY PIPELINE
=============================================================================

  Full pipeline:
    [1] Take user face image as input
    [2] Display the face image in the console
    [3] Detect & extract 512-D facial embedding
    [4] Display raw (pre-encryption) feature values
    [5] Encrypt features with AES-256-GCM
    [6] Display encrypted (ciphertext) feature bytes
    [7] Encode encrypted payload
    [8] Embed into carrier WAV audio
    [9] Display before / after waveform comparison
   [10] Save stego-audio and report summary

  Requirements:
      pip install facenet-pytorch torch Pillow numpy soundfile
                  cryptography pillow-avif-plugin climage
=============================================================================
"""

import os
import sys
import struct
import hashlib
import argparse
import textwrap
import shutil
import io

try:
    import winsound
except ImportError:
    winsound = None

import cv2
import numpy as np
import soundfile as sf
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ─── Optional: rich-terminal image display ───────────────────────────────────
try:
    import climage                  # renders images as ANSI art in terminal
    _HAS_CLIMAGE = True
except ImportError:
    _HAS_CLIMAGE = False

# ─── Deep-learning models ─────────────────────────────────────────────────────
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# ─── AES-256-GCM ─────────────────────────────────────────────────────────────
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from modules.cancelable_biometrics import CancelableBiometrics
from modules.pipeline_storage import PipelineStorage

# =============================================================================
# GLOBAL PATHS
# =============================================================================

BASE_DIR   = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

STEGO_OUTPUT   = OUTPUT_DIR / "stego_audio.wav"
CARRIER_COPY_OUTPUT = OUTPUT_DIR / "carrier_audio_before_embedding.wav"
KEY_OUTPUT     = OUTPUT_DIR / "aes_key.bin"
KEYS_DIR       = OUTPUT_DIR / "user_keys"
PAYLOAD_OUTPUT = OUTPUT_DIR / "encrypted_payload.bin"

# Fallback audio path – first WAV found in Data/
AUDIO_SEARCH_ROOT = BASE_DIR / "Data"


def key_path_for_user(user_id: str) -> Path:
    user_key_name = hashlib.sha256(user_id.encode("utf-8")).hexdigest() + ".key"
    return KEYS_DIR / user_key_name

# =============================================================================
# ANSI COLOUR HELPERS
# =============================================================================

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
MAGENTA= "\033[95m"
BLUE   = "\033[94m"
DIM    = "\033[2m"

def banner(title, colour=CYAN):
    width = 70
    line  = "═" * width
    pad   = (width - len(title) - 2) // 2
    print(f"\n{colour}{BOLD}╔{line}╗")
    print(f"║{' ' * pad} {title} {' ' * (width - pad - len(title) - 1)}║")
    print(f"╚{line}╝{RESET}")

def step(n, text, colour=GREEN):
    print(f"\n{colour}{BOLD}[STEP {n}]{RESET} {BOLD}{text}{RESET}")
    print(f"{DIM}{'─' * 60}{RESET}")

def info(label, value, colour=YELLOW):
    print(f"  {colour}► {RESET}{label:<35} {BOLD}{value}{RESET}")

def ok(msg):
    print(f"  {GREEN}✔  {msg}{RESET}")

def warn(msg):
    print(f"  {YELLOW}⚠  {msg}{RESET}")

def err(msg):
    print(f"  {RED}✘  {msg}{RESET}")

# =============================================================================
# IMAGE DISPLAY IN CONSOLE
# =============================================================================

def show_image_in_console(image_path: Path, label: str = "", width: int = 80):
    """
    Render the image directly inside the terminal window.

    Strategy:
      1. If `climage` is installed → full ANSI-colour block art.
      2. Fallback → ASCII-art using grayscale brightness chars.
    """
    print(f"\n{MAGENTA}{BOLD}  ┌── {label} {'─' * (60 - len(label))}┐{RESET}")

    if not image_path.exists():
        warn(f"Image file not found: {image_path}")
        return

    try:
        if _HAS_CLIMAGE:
            # climage renders coloured ANSI block characters
            output = climage.convert(str(image_path), width=width, is_unicode=True)
            # Indent every line
            for line in output.splitlines():
                print("  " + line)
        else:
            # ── Fallback pure-PIL ASCII art ──────────────────────────────────
            _ascii_image(image_path, width=width)
    except Exception as e:
        warn(f"Could not render image: {e}")

    print(f"{MAGENTA}{BOLD}  └{'─' * (62)}┘{RESET}")


def _ascii_image(image_path: Path, width: int = 80):
    """Pure-Python ASCII art renderer (no external libs)."""
    CHARS = "@#S%?*+;:,. "   # dense → light
    img   = Image.open(image_path).convert("L")   # grayscale
    # Keep aspect ratio (terminal chars are ~2× taller than wide)
    aspect  = img.height / img.width
    height  = int(aspect * width * 0.45)
    img     = img.resize((width, height), Image.Resampling.LANCZOS)
    pixels  = np.array(img)
    for row in pixels:
        line = "".join(CHARS[int(v / 255 * (len(CHARS) - 1))] for v in row)
        print("  " + line)


def display_face_annotated(image_path: Path, boxes, output_path: Path):
    """
    Draw bounding-box(es) on the original image and save it.
    Returns the path to the annotated image.
    """
    img  = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    if boxes is not None:
        coordinates = np.asarray(boxes, dtype=float)
        if coordinates.ndim == 1:
            coordinates = coordinates.reshape(1, -1)
        for x1, y1, x2, y2 in coordinates:
            draw.rectangle(
                (int(x1), int(y1), int(x2), int(y2)),
                outline=(0, 255, 0),
                width=3,
            )
    img.save(output_path)
    return output_path


# =============================================================================
# WAVEFORM ASCII VISUALIZER
# =============================================================================

def waveform_ascii(samples: np.ndarray, label: str, width: int = 68, height: int = 10):
    """Print a simple ASCII waveform plot in the terminal."""
    print(f"\n  {BLUE}{BOLD}{label}{RESET}")
    print(f"  {DIM}{'─' * (width + 4)}{RESET}")

    # Downsample to `width` columns
    idx    = np.linspace(0, len(samples) - 1, width).astype(int)
    chunk  = samples[idx]
    # Normalise
    peak   = np.max(np.abs(chunk))
    if peak > 0:
        chunk = chunk / peak

    # Build rows top → bottom
    half = height // 2
    for row in range(height, -1, -1):
        level   = (row - half) / half          # +1 → top, -1 → bottom
        bar_row = []
        for v in chunk:
            if (level >= 0 and v >= level) or (level < 0 and v <= level):
                bar_row.append("█")
            elif abs(level) < 0.05:
                bar_row.append("─")
            else:
                bar_row.append(" ")
        print("  │" + "".join(bar_row) + "│")

    print(f"  {DIM}{'─' * (width + 4)}{RESET}")
    info("Samples shown", f"{len(samples):,}")


# =============================================================================
# DEVICE SETUP
# =============================================================================

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info("Compute device", str(device).upper())
    return device


# =============================================================================
# STEP 1 — LOAD FACE IMAGE
# =============================================================================

def load_face_image(image_path: Path) -> Image.Image:
    step(1, "Loading Face Image")

    if not image_path.exists():
        err(f"Image not found: {image_path}")
        sys.exit(1)

    img = Image.open(image_path).convert("RGB")
    info("Image path",  str(image_path))
    info("Image size",  f"{img.width} × {img.height} px")
    info("Image mode",  img.mode)
    ok("Image loaded successfully.")

    # ── Display the original image ────────────────────────────────────────────
    show_image_in_console(image_path, label="INPUT FACE IMAGE")

    return img


def capture_face_from_camera(
    camera_index: int = 0,
    capture_path: Path | None = None,
    capture_title: str = "Capture Face Image with OpenCV",
) -> Path:
    """Capture and save only the largest face detected by OpenCV."""
    step(1, capture_title)
    cascade_path = (
        Path(cv2.__file__).resolve().parent
        / "data"
        / "haarcascade_frontalface_default.xml"
    )
    face_detector = cv2.CascadeClassifier(str(cascade_path))
    if face_detector.empty():
        raise RuntimeError(f"Could not load OpenCV face detector: {cascade_path}")

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open camera {camera_index}. Check that a webcam is "
            "connected and not in use by another application."
        )

    capture_path = capture_path or OUTPUT_DIR / "captured_face.jpg"
    captured = False
    capture_requested = False
    current_face_crop = None

    def handle_capture_click(event, x, y, flags, parameter):
        nonlocal capture_requested
        if event == cv2.EVENT_LBUTTONDOWN and 20 <= x <= 280 and 48 <= y <= 96:
            capture_requested = True

    window_name = "Face capture"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, handle_capture_click)
    print("Live preview: click Capture, press C/Space/Enter to capture, or Q to quit.")
    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a frame from the webcam.")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80),
            )
            preview = cv2.flip(frame, 1)
            current_face_crop = None
            if len(faces):
                x, y, width, height = max(faces, key=lambda box: box[2] * box[3])
                padding_x = int(width * 0.15)
                padding_y = int(height * 0.15)
                left = max(0, x - padding_x)
                top = max(0, y - padding_y)
                right = min(frame.shape[1], x + width + padding_x)
                bottom = min(frame.shape[0], y + height + padding_y)
                current_face_crop = frame[top:bottom, left:right]

                mirrored_x = frame.shape[1] - x - width
                cv2.rectangle(
                    preview,
                    (mirrored_x, y),
                    (mirrored_x + width, y + height),
                    (40, 220, 40),
                    2,
                )
                button_color = (30, 130, 30)
                status = "Face detected - ready"
            else:
                button_color = (50, 50, 170)
                status = "No face detected - move into view"

            cv2.rectangle(preview, (20, 48), (280, 96), button_color, -1)
            cv2.putText(
                preview,
                "CAPTURE (C / SPACE)",
                (31, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                preview,
                status,
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (40, 220, 40) if current_face_crop is not None else (40, 40, 230),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                preview,
                "Q: quit",
                (20, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF

            if capture_requested or key in (ord("c"), ord("C"), ord(" "), 13):
                capture_requested = False
                if current_face_crop is None:
                    print("No face is detected yet. Center your face in the camera and try again.")
                    continue
                if not cv2.imwrite(str(capture_path), current_face_crop):
                    raise RuntimeError("Could not save the detected face crop.")
                captured = True
                break
            if key == ord("q") or key == 27:
                raise RuntimeError("Face capture cancelled.")
    finally:
        camera.release()
        cv2.destroyAllWindows()

    if not captured:
        raise RuntimeError("No face image was captured.")

    ok(f"Captured image saved to {capture_path}")
    return capture_path


# =============================================================================
# STEP 2 — FACE DETECTION + FEATURE EXTRACTION
# =============================================================================

def extract_features(image_path: Path, device, artifact_prefix: str = "sender"):
    step(2, "Face Detection & Feature Extraction")

    # ── Build models ──────────────────────────────────────────────────────────
    mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device)
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    img = Image.open(image_path).convert("RGB")

    # Detect with boxes so we can draw them
    detection = mtcnn.detect(img)
    boxes, probs = detection[0], detection[1]
    ok(f"Face detected  (confidence: {float(probs[0]) * 100:.1f}%)" if boxes is not None
       else "WARNING: No face detected.")

    if boxes is None:
        err("Cannot continue — no face in image.")
        sys.exit(1)

    # Save annotated image and display it
    annotated_path = OUTPUT_DIR / f"{artifact_prefix}_face_detected.jpg"
    display_face_annotated(image_path, boxes[0], annotated_path)
    show_image_in_console(annotated_path, label="DETECTED FACE  (bounding box)")

    # ── Crop & embed ──────────────────────────────────────────────────────────
    face_tensor = mtcnn(img)
    if face_tensor is None:
        err("MTCNN crop returned None.")
        sys.exit(1)

    # Save the 160×160 cropped face thumbnail
    cropped_np   = face_tensor.permute(1, 2, 0).cpu().numpy()
    cropped_np   = ((cropped_np - cropped_np.min()) /
                    (cropped_np.max() - cropped_np.min()) * 255).astype(np.uint8)
    cropped_pil  = Image.fromarray(cropped_np, "RGB")
    cropped_path = OUTPUT_DIR / f"{artifact_prefix}_face_cropped_160x160.jpg"
    cropped_pil.save(cropped_path)
    show_image_in_console(cropped_path, label="CROPPED FACE  (160 × 160 px — fed to model)")

    face_tensor = face_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(face_tensor)

    embedding = embedding.cpu().numpy()[0]
    norm      = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    info("Embedding shape",    str(embedding.shape))
    info("Embedding dtype",    str(embedding.dtype))
    info("Embedding L2-norm",  f"{np.linalg.norm(embedding):.6f}")

    return embedding


# =============================================================================
# STEP 3 — DISPLAY RAW FEATURES (BEFORE ENCRYPTION)
# =============================================================================

def display_raw_features(embedding: np.ndarray):
    step(3, "Facial Feature Vector  [BEFORE ENCRYPTION]")

    print(f"\n  {YELLOW}{BOLD}  ┌── RAW 512-D FACE EMBEDDING (first 64 values) ─────────────────┐{RESET}")
    rows = embedding[:64].reshape(8, 8)
    for row in rows:
        vals = "  ".join(f"{v:+.4f}" for v in row)
        print(f"  {DIM}│{RESET}  {vals}  {DIM}│{RESET}")
    print(f"  {YELLOW}{BOLD}  └───────────────────────────────────────────────────────────────┘{RESET}")

    info("Min value",  f"{embedding.min():.6f}")
    info("Max value",  f"{embedding.max():.6f}")
    info("Mean value", f"{embedding.mean():.6f}")
    info("Std value",  f"{embedding.std():.6f}")
    ok("Raw feature display complete.")


def make_cancelable_template(
    embedding: np.ndarray,
    user_id: str,
    stage_title: str = "Cancelable Biometric Template",
) -> np.ndarray:
    step(4, stage_title)
    print("\n  Raw face embedding before transform (first 16 values):")
    print(" ", np.array2string(embedding[:16], precision=5))

    transform = CancelableBiometrics(dimension=embedding.size)
    template = transform.transform(embedding, user_id)

    print("\n  Cancelable template after transform (first 16 values):")
    print(" ", np.array2string(template[:16], precision=5))
    info("User key identifier", user_id)
    info("Template shape", str(template.shape))
    info("Raw template L2 norm", f"{np.linalg.norm(embedding):.6f}")
    info("Cancelable template L2 norm", f"{np.linalg.norm(template):.6f}")
    ok("Cancelable biometric transform complete.")
    return template


# =============================================================================
# STEP 4 — AES-256-GCM ENCRYPTION
# =============================================================================

def compress_face_image(image_path: Path, max_bytes: int = 600) -> tuple[bytes, tuple[int, int]]:
    image = Image.open(image_path).convert("L")
    for size in (96, 80, 72, 64, 56, 48, 40, 32):
        resized = image.resize((size, size), Image.Resampling.LANCZOS)
        for quality in (40, 35, 30, 25, 20):
            buffer = io.BytesIO()
            resized.save(buffer, format="JPEG", quality=quality, optimize=True)
            image_data = buffer.getvalue()
            if len(image_data) <= max_bytes:
                return image_data, (size, quality)
    raise ValueError("Could not compress face crop to fit the audio payload.")


def encrypt_features(embedding: np.ndarray, user_id: str, face_image: bytes):
    step(4, "AES-256-GCM Encryption of Facial Features")

    # ── Quantise float32 → int8 to shrink payload ────────────────────────────
    max_abs = np.max(np.abs(embedding))
    scale   = np.float32(max_abs / 127.0)
    quant   = np.round(embedding / scale).astype(np.int8)

    info("Original size (float32)",  f"{embedding.nbytes} bytes")
    info("Quantised size (int8)",    f"{quant.nbytes} bytes")
    info("Quantisation scale",       f"{float(scale):.8f}")

    # ── Build compact plaintext with header ──────────────────────────────────
    MAGIC   = b"CBQ2"
    VERSION = 1
    header  = (MAGIC
                + struct.pack(">B", VERSION)
                + struct.pack(">I", len(quant))
                + struct.pack(">f", float(scale))
                + struct.pack(">I", len(face_image)))
    plaintext = header + quant.tobytes() + face_image

    info("Header size",    f"{len(header)} bytes")
    info("Plaintext size", f"{len(plaintext)} bytes")
    info("Embedded face image", f"{len(face_image)} encrypted bytes")

    # ── Generate AES-256 key ──────────────────────────────────────────────────
    key   = AESGCM.generate_key(bit_length=256)
    nonce = os.urandom(12)          # 96-bit nonce for GCM

    user_key_path = key_path_for_user(user_id)
    user_key_path.parent.mkdir(parents=True, exist_ok=True)
    user_key_path.write_bytes(key)
    info("AES-256 key",   f"32 bytes  (saved locally → {user_key_path})")
    info("GCM nonce",     f"12 bytes  ({nonce.hex()[:24]}...)")

    # ── Encrypt ───────────────────────────────────────────────────────────────
    aesgcm        = AESGCM(key)
    ciphertext    = aesgcm.encrypt(nonce, plaintext, None)  # includes 16-byte GCM tag
    encrypted_pkg = nonce + ciphertext                       # nonce || ciphertext+tag

    PAYLOAD_OUTPUT.write_bytes(encrypted_pkg)
    info("Encrypted payload", f"{len(encrypted_pkg)} bytes  (saved → output/encrypted_payload.bin)")
    ok("AES-256-GCM encryption complete.")

    return key, nonce, encrypted_pkg, quant, scale


# =============================================================================
# STEP 5 — DISPLAY ENCRYPTED FEATURES (AFTER ENCRYPTION)
# =============================================================================

def display_encrypted_features(encrypted_pkg: bytes):
    step(5, "Encrypted Payload  [AFTER ENCRYPTION]")

    ciphertext_body = encrypted_pkg[12:]   # skip 12-byte nonce

    print(f"\n  {RED}{BOLD}  ┌── CIPHERTEXT HEX DUMP (first 128 bytes) ───────────────────────┐{RESET}")
    chunk = ciphertext_body[:128]
    lines = textwrap.wrap(chunk.hex(), 32)
    for i, line in enumerate(lines):
        pairs = " ".join(line[j:j+2] for j in range(0, len(line), 2))
        offset = f"0x{i*16:04X}"
        print(f"  {DIM}│{RESET}  {CYAN}{offset}{RESET}  {pairs}  {DIM}│{RESET}")
    print(f"  {RED}{BOLD}  └───────────────────────────────────────────────────────────────┘{RESET}")

    info("Total encrypted bytes",  len(encrypted_pkg))
    info("Nonce (GCM IV)",         "12 bytes")
    info("Ciphertext + GCM tag",   f"{len(ciphertext_body)} bytes")

    print(f"\n  {DIM}  NOTE: The plaintext face features are completely unrecognisable")
    print(f"        in the ciphertext. AES-256-GCM guarantees confidentiality,")
    print(f"        integrity and authenticity.{RESET}")
    ok("Encrypted feature display complete.")


# =============================================================================
# STEP 6 — FIND CARRIER AUDIO
# =============================================================================

def find_carrier_audio(user_id: str, encrypted_payload_size: int) -> Path:
    step(6, "Selecting Carrier WAV Audio")

    # Require capacity for the frame header, encrypted payload, and repetition.
    required_samples = (
        (41 + encrypted_payload_size)
        * 8
        * REPETITIONS
        * 2
    )

    dataset_root = AUDIO_SEARCH_ROOT / "audio_dtaset"
    if not dataset_root.exists():
        raise FileNotFoundError(f"Audio dataset folder not found: {dataset_root}")

    candidates = sorted(
        path for path in dataset_root.rglob("*.wav")
        if "__MACOSX" not in path.parts and not path.name.startswith(("._", "."))
    )
    if not candidates:
        err(f"No WAV files found under {dataset_root}")
        sys.exit(1)

    valid_candidates = []
    for candidate in candidates:
        try:
            audio_info = sf.info(str(candidate))
            if audio_info.frames >= required_samples and audio_info.samplerate > 0:
                valid_candidates.append((audio_info.frames, candidate))
        except (RuntimeError, OSError):
            continue

    if not valid_candidates:
        raise ValueError(
            f"No dataset WAV has enough capacity: need {required_samples:,} samples."
        )

    user_hash = hashlib.sha256(user_id.encode("utf-8")).digest()
    selected_index = int.from_bytes(user_hash[:8], "big") % len(valid_candidates)
    _, chosen = valid_candidates[selected_index]
    shutil.copy2(chosen, CARRIER_COPY_OUTPUT)
    audio, sr = sf.read(str(chosen), dtype="float64")

    info("Carrier file",    str(chosen.name))
    info("Carrier selected for username", user_id)
    info("Sample rate",     f"{sr} Hz")
    info("Duration",        f"{len(audio) / sr:.2f} s")
    info("Channels",        "mono" if audio.ndim == 1 else f"{audio.shape[1]}")
    info("File size",       f"{chosen.stat().st_size // 1024} KB")
    info("Before-embedding copy", str(CARRIER_COPY_OUTPUT))
    ok("Carrier audio selected.")

    return chosen


# =============================================================================
# STEP 7 — DISPLAY CARRIER WAVEFORM (BEFORE EMBEDDING)
# =============================================================================

def display_waveform_before(carrier_path: Path):
    step(7, "Carrier Audio Waveform  [BEFORE EMBEDDING]")
    audio, sr = sf.read(str(carrier_path), dtype="float64")
    mono = audio if audio.ndim == 1 else audio[:, 0]
    waveform_ascii(mono, label="CARRIER AUDIO — Original Waveform")
    info("Peak amplitude", f"{np.max(np.abs(mono)):.6f}")
    ok("Pre-embedding waveform displayed.")
    return audio, sr, mono


def play_audio_file(audio_path: Path, stage_label: str) -> None:
    """Play a WAV synchronously on Windows so each audio stage is audible."""
    if winsound is None:
        warn("Automatic WAV playback is only available on Windows.")
        return
    info("Playing audio", f"{stage_label}: {audio_path}")
    winsound.PlaySound(str(audio_path), winsound.SND_FILENAME)
    ok(f"Finished playing {stage_label}.")


# =============================================================================
# STEP 8 — BUILD FRAMED PAYLOAD & EMBED INTO AUDIO (QIM steganography)
# =============================================================================

REPETITIONS = 8        # each bit repeated 8× for robustness
QUANT_STEP  = 0.002    # QIM step size


def _bits_from_bytes(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big")


def _embed_bit(a: float, b: float, bit: int):
    """Quantisation-index modulation on a sample pair."""
    mean       = (a + b) / 2.0
    difference = a - b
    base       = int(np.floor(difference / QUANT_STEP))
    candidates = [k for k in range(base - 2, base + 3) if k % 2 == bit]
    target_k   = min(candidates,
                     key=lambda k: abs((k + 0.5) * QUANT_STEP - difference))
    target_diff = (target_k + 0.5) * QUANT_STEP
    return mean + target_diff / 2.0, mean - target_diff / 2.0


def embed_payload(
    carrier_audio: np.ndarray,
    sample_rate:   int,
    encrypted_pkg: bytes,
    carrier_mono:  np.ndarray
):
    step(8, "Steganographic Embedding into Audio")

    # ── Wrap payload in a frame with integrity header ─────────────────────────
    frame_magic   = b"CFD1"
    frame_version = 1
    frame_header  = (frame_magic
                     + struct.pack(">B", frame_version)
                     + struct.pack(">I", len(encrypted_pkg))
                     + hashlib.sha256(encrypted_pkg).digest())
    frame         = frame_header + encrypted_pkg

    bits          = _bits_from_bytes(frame)
    repeated_bits = np.repeat(bits, REPETITIONS)

    info("Frame header size",    f"{len(frame_header)} bytes")
    info("Total frame size",     f"{len(frame)} bytes")
    info("Bits to embed",        f"{len(bits):,}")
    info("Repeated bits",        f"{len(repeated_bits):,} ({REPETITIONS}× repetition)")

    required_samples = 2 * len(repeated_bits)
    available        = len(carrier_mono)
    info("Required samples",     f"{required_samples:,}")
    info("Available samples",    f"{available:,}")

    if required_samples > available:
        err(f"Carrier too short!  Need {required_samples}, have {available}.")
        sys.exit(1)

    working = carrier_audio.copy() if carrier_audio.ndim == 1 else carrier_audio[:, 0].copy()

    for i, bit in enumerate(repeated_bits):
        j = 2 * i
        working[j], working[j + 1] = _embed_bit(working[j], working[j + 1], int(bit))

    if carrier_audio.ndim == 1:
        stego = working
    else:
        stego = carrier_audio.copy()
        stego[:, 0] = working

    peak = np.max(np.abs(stego if carrier_audio.ndim == 1 else stego[:, 0]))
    if peak > 1.0:
        warn(f"Peak {peak:.4f} > 1.0 — normalising to prevent clipping.")
        stego = stego / peak

    sf.write(str(STEGO_OUTPUT), stego, sample_rate, subtype="PCM_16")
    info("Output file", str(STEGO_OUTPUT))
    ok("Steganographic embedding complete.")

    stego_mono = stego if stego.ndim == 1 else stego[:, 0]
    return stego, stego_mono


# =============================================================================
# STEP 9 — DISPLAY STEGO WAVEFORM (AFTER EMBEDDING) + DIFFERENCE
# =============================================================================

def display_waveform_after(
    carrier_mono: np.ndarray,
    stego_mono:   np.ndarray,
    sample_rate:  int
):
    step(9, "Stego Audio Waveform  [AFTER EMBEDDING]")

    waveform_ascii(stego_mono, label="STEGO AUDIO — After Embedding")
    info("Peak amplitude",      f"{np.max(np.abs(stego_mono)):.6f}")

    diff   = stego_mono - carrier_mono
    snr_db = (20 * np.log10(np.linalg.norm(carrier_mono) /
              (np.linalg.norm(diff) + 1e-12)))

    print(f"\n  {MAGENTA}{BOLD}  ┌── DIFFERENCE (carrier − stego) ────────────────────────────────┐{RESET}")
    waveform_ascii(diff[:8000], label="PERTURBATION SIGNAL (magnified view — first 8000 samples)")
    print(f"  {MAGENTA}{BOLD}  └───────────────────────────────────────────────────────────────┘{RESET}")

    info("Max perturbation",  f"{np.max(np.abs(diff)):.8f}")
    info("SNR",               f"{snr_db:.2f} dB  (higher = more transparent)")
    metrics = {
        "mse": float(np.mean(diff ** 2)),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "mae": float(np.mean(np.abs(diff))),
        "snr_db": float(snr_db),
        "duration_seconds": len(carrier_mono) / sample_rate,
        "sample_rate": sample_rate,
    }
    info("MSE", f"{metrics['mse']:.10g}")
    info("RMSE", f"{metrics['rmse']:.10g}")
    info("MAE", f"{metrics['mae']:.10g}")
    ok("Post-embedding waveform displayed.")
    return metrics


def extract_encrypted_package(stego_mono: np.ndarray) -> bytes:
    """Recover and checksum-validate the encrypted payload from audio samples."""
    header_bit_count = 41 * 8
    header_bits = []
    for bit_index in range(header_bit_count):
        votes = []
        for repetition in range(REPETITIONS):
            pair_index = (bit_index * REPETITIONS + repetition) * 2
            if pair_index + 1 >= len(stego_mono):
                raise ValueError("Stego audio is too short to contain its frame header.")
            difference = stego_mono[pair_index] - stego_mono[pair_index + 1]
            votes.append(int(np.floor(difference / QUANT_STEP)) % 2)
        header_bits.append(1 if sum(votes) > REPETITIONS // 2 else 0)

    frame_header = np.packbits(header_bits, bitorder="big").tobytes()
    if frame_header[:4] != b"CFD1" or frame_header[4] != 1:
        raise ValueError("Stego frame header is invalid or unsupported.")

    payload_length = struct.unpack(">I", frame_header[5:9])[0]
    total_bits = (41 + payload_length) * 8
    if total_bits * REPETITIONS * 2 > len(stego_mono):
        raise ValueError("Stego audio does not have enough samples for its declared payload.")

    payload_bits = []
    for bit_index in range(header_bit_count, total_bits):
        votes = []
        for repetition in range(REPETITIONS):
            pair_index = (bit_index * REPETITIONS + repetition) * 2
            difference = stego_mono[pair_index] - stego_mono[pair_index + 1]
            votes.append(int(np.floor(difference / QUANT_STEP)) % 2)
        payload_bits.append(1 if sum(votes) > REPETITIONS // 2 else 0)

    encrypted_package = np.packbits(payload_bits, bitorder="big").tobytes()
    if hashlib.sha256(encrypted_package).digest() != frame_header[9:41]:
        raise ValueError("Stego payload checksum failed.")
    return encrypted_package


def decrypt_biometric_payload(
    encrypted_package: bytes,
    key: bytes,
) -> tuple[np.ndarray, bytes]:
    nonce = encrypted_package[:12]
    plaintext = AESGCM(key).decrypt(nonce, encrypted_package[12:], None)
    if plaintext[:4] == b"CBQ1":
        template_length = struct.unpack(">I", plaintext[5:9])[0]
        scale = struct.unpack(">f", plaintext[9:13])[0]
        quantized = np.frombuffer(plaintext[13:13 + template_length], dtype=np.int8)
        if quantized.size != template_length:
            raise ValueError("Decrypted biometric template has an invalid size.")
        return quantized.astype(np.float32) * scale, b""

    if plaintext[:4] != b"CBQ2" or plaintext[4] != 1:
        raise ValueError("Encrypted biometric template header is invalid.")

    template_length = struct.unpack(">I", plaintext[5:9])[0]
    scale = struct.unpack(">f", plaintext[9:13])[0]
    image_length = struct.unpack(">I", plaintext[13:17])[0]
    template_start = 17
    image_start = template_start + template_length
    quantized = np.frombuffer(
        plaintext[template_start:image_start],
        dtype=np.int8,
    )
    if quantized.size != template_length:
        raise ValueError("Decrypted biometric template has an invalid size.")
    face_image = plaintext[image_start:image_start + image_length]
    if len(face_image) != image_length:
        raise ValueError("Recovered face image data is incomplete.")
    return quantized.astype(np.float32) * scale, face_image


def decrypt_cancelable_template(encrypted_package: bytes, key: bytes) -> np.ndarray:
    template, _ = decrypt_biometric_payload(encrypted_package, key)
    return template


def reject_duplicate_face(
    embedding: np.ndarray,
    user_id: str,
    storage: PipelineStorage,
    threshold: float = 0.90,
) -> None:
    step(3, "Checking Face Against Registered Users")
    registered_users = [name for name in storage.list_user_ids() if name != user_id]
    if not registered_users:
        ok("No other registered users to compare.")
        return

    checked = 0
    skipped = []
    for registered_user in registered_users:
        key_file = key_path_for_user(registered_user)
        if not key_file.exists() and not KEY_OUTPUT.exists():
            skipped.append(registered_user)
            continue

        audio_bytes = storage.get_stego_audio(registered_user)
        if audio_bytes is None:
            skipped.append(registered_user)
            continue

        try:
            audio, _ = sf.read(io.BytesIO(audio_bytes), dtype="float64")
            audio_mono = audio if audio.ndim == 1 else audio[:, 0]
            encrypted_package = extract_encrypted_package(audio_mono)
            registered_template = decrypt_cancelable_template(
                encrypted_package,
                key_file.read_bytes() if key_file.exists() else KEY_OUTPUT.read_bytes(),
            )
        except Exception as error:
            warn(f"Could not compare registered user {registered_user!r}: {error}")
            skipped.append(registered_user)
            continue

        query_template = CancelableBiometrics(embedding.size).transform(
            embedding,
            registered_user,
        )
        similarity = float(
            np.dot(query_template, registered_template)
            / (np.linalg.norm(query_template) * np.linalg.norm(registered_template) + 1e-12)
        )
        checked += 1
        info(f"Similarity with {registered_user}", f"{similarity:.6f}")
        if similarity >= threshold:
            err(
                f"Duplicate face detected: it matches registered user "
                f"{registered_user!r} (cosine {similarity:.4f})."
            )
            raise ValueError("Registration stopped because this face is already registered.")

    info("Registered faces checked", str(checked))
    if skipped:
        raise RuntimeError(
            "Duplicate check was incomplete. Could not validate stored users: "
            + ", ".join(skipped)
            + ". Restore each user's local AES key or remove obsolete records "
            "before registering another user."
        )
    ok("No duplicate was found among decryptable registered faces.")


# =============================================================================
# STEP 10 — VERIFY ROUND-TRIP DECRYPTION
# =============================================================================

def verify_roundtrip(
    stego_mono: np.ndarray,
    key: bytes,
    expected_template: np.ndarray,
    sender_payload: bytes,
):
    step(10, "Receiver Extraction, Decryption & Authentication")

    # ── Extract bits from stego ───────────────────────────────────────────────
    frame_header_bits = (4 + 1 + 4 + 32) * 8          # 41 bytes
    raw_bits    = []
    for i in range(frame_header_bits * REPETITIONS):
        j   = 2 * i
        difference = stego_mono[j] - stego_mono[j + 1]
        bit = int(np.floor(difference / QUANT_STEP)) % 2
        raw_bits.append(bit)

    # Majority-vote demodulation
    demod = []
    for i in range(frame_header_bits):
        group = raw_bits[i * REPETITIONS: (i + 1) * REPETITIONS]
        demod.append(1 if sum(group) > REPETITIONS // 2 else 0)

    frame_header_bytes = np.packbits(demod, bitorder="big").tobytes()

    # Parse header
    magic_r   = frame_header_bytes[:4]
    version_r = struct.unpack(">B", frame_header_bytes[4:5])[0]
    plen_r    = struct.unpack(">I", frame_header_bytes[5:9])[0]

    info("Recovered magic",    magic_r.decode(errors="replace"))
    info("Recovered version",  str(version_r))
    info("Recovered plen",     f"{plen_r} bytes")

    if magic_r != b"CFD1":
        warn("Magic mismatch — embedding may be corrupted.")
        return {"aes_status": False, "sender_match": False, "authenticated": False}

    # Extract payload bits
    total_bits   = (41 + plen_r) * 8
    payload_bits = []
    for i in range(frame_header_bits * REPETITIONS, total_bits * REPETITIONS):
        j   = 2 * i
        difference = stego_mono[j] - stego_mono[j + 1]
        bit = int(np.floor(difference / QUANT_STEP)) % 2
        payload_bits.append(bit)

    # Demod payload
    demod_payload = []
    for i in range(plen_r * 8):
        group = payload_bits[i * REPETITIONS: (i + 1) * REPETITIONS]
        demod_payload.append(1 if sum(group) > REPETITIONS // 2 else 0)

    recovered_encrypted = np.packbits(demod_payload, bitorder="big").tobytes()
    info("Recovered encrypted bytes", len(recovered_encrypted))
    sender_match = recovered_encrypted == sender_payload
    info("Sender payload matches receiver", "YES" if sender_match else "NO")

    expected_digest = frame_header_bytes[9:41]
    actual_digest = hashlib.sha256(recovered_encrypted).digest()
    if actual_digest != expected_digest:
        warn("Payload checksum mismatch — recovered audio data is corrupted.")
        return {"aes_status": False, "sender_match": sender_match, "authenticated": False}
    ok("Payload SHA-256 checksum verified.")

    # Decrypt
    try:
        recovered_emb, _ = decrypt_biometric_payload(recovered_encrypted, key)
        ok("AES-256-GCM decryption successful — GCM tag verified ✔")
    except Exception as e:
        warn(f"Decryption failed: {e}")
        return {"aes_status": False, "sender_match": sender_match, "authenticated": False}

    cosine_sim = float(
        np.dot(expected_template, recovered_emb) /
        (np.linalg.norm(expected_template) * np.linalg.norm(recovered_emb) + 1e-12)
    )

    threshold = 0.90
    authenticated = cosine_sim >= threshold
    info("Cancelable template cosine similarity", f"{cosine_sim:.6f}")
    info("Authentication threshold", f"{threshold:.2f}")
    if authenticated:
        ok("AUTHENTICATION SUCCESS — receiver template matches sender.")
    else:
        warn("AUTHENTICATION FAILURE — receiver template does not match sender.")

    return {
        "aes_status": True,
        "sender_match": sender_match,
        "authenticated": authenticated,
        "cosine_similarity": cosine_sim,
    }


# =============================================================================
# FINAL SUMMARY
# =============================================================================

def print_summary(
    image_path: Path,
    carrier_path: Path,
    metrics: dict,
    verification: dict,
    user_key_path: Path,
    database_saved: bool,
):
    banner("PIPELINE SUMMARY", GREEN)
    print()
    rows = [
        ("Input image",          str(image_path)),
        ("Sender detected face", str(OUTPUT_DIR / "sender_face_detected.jpg")),
        ("Sender face crop", str(OUTPUT_DIR / "sender_face_cropped_160x160.jpg")),
        ("Receiver face", str(OUTPUT_DIR / "receiver_face.jpg")),
        ("Receiver detected face", str(OUTPUT_DIR / "receiver_face_detected.jpg")),
        ("Recovered embedded face image", str(OUTPUT_DIR / "recovered_face.jpg")),
        ("Carrier WAV", str(carrier_path)),
        ("Carrier duration / sample rate", "{:.2f} s / {} Hz".format(
            metrics["duration_seconds"], metrics["sample_rate"]
        )),
        ("Local AES-256 key", str(user_key_path)),
        ("Encrypted payload",    str(PAYLOAD_OUTPUT)),
        ("Stego audio",          str(STEGO_OUTPUT)),
        ("Carrier before embedding", str(CARRIER_COPY_OUTPUT)),
        ("AES-GCM status", "PASS" if verification.get("aes_status") else "FAIL"),
        ("Sender payload match", "YES" if verification.get("sender_match") else "NO"),
        ("Receiver authentication", "PASS" if verification.get("authenticated") else "FAIL"),
        ("Database status", "SAVED" if database_saved else "NOT SAVED"),
        ("Cosine similarity", f"{verification.get('cosine_similarity', 0.0):.6f}"),
        ("Audio MSE / RMSE / MAE", "{:.6g} / {:.6g} / {:.6g}".format(
            metrics["mse"], metrics["rmse"], metrics["mae"]
        )),
        ("Audio SNR", f"{metrics['snr_db']:.2f} dB"),
    ]
    for label, value in rows:
        info(label, value)

    print()
    print(f"  {CYAN}{BOLD}  ╔══════════════════════════════════════════════════════════════╗")
    print(f"  ║  Pipeline stages completed; see verification results above.    ║")
    print(f"  ║                                                                ║")
    print(f"  ║  Face features ─► AES-256-GCM encrypted ─► embedded in audio  ║")
    print(f"  ╚══════════════════════════════════════════════════════════════╝{RESET}")
    print()


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Secure Face Authentication — AES-256-GCM + Audio Steganography"
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Path to an input face image (JPG / PNG). "
             "If omitted, capture an image from the webcam."
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index to use when capturing (default: 0)."
    )
    parser.add_argument(
        "--user-id",
        "--username",
        dest="user_id",
        default=None,
        help="Username key for the cancelable template and MySQL audio record; prompted if omitted."
    )
    parser.add_argument(
        "--auth-image",
        type=Path,
        default=None,
        help="Optional receiver-side face image; if omitted, the webcam captures a second face."
    )
    args = parser.parse_args()

    # ── Enable ANSI colours on Windows ───────────────────────────────────────
    if sys.platform == "win32":
        os.system("color")          # activates ANSI escape sequences
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleMode(
                ctypes.windll.kernel32.GetStdHandle(-11), 7
            )
        except Exception:
            pass

    banner("SECURE FACE AUTHENTICATION PIPELINE", CYAN)
    print(f"  {DIM}AES-256-GCM Encryption  +  Audio Steganography{RESET}")
    print(f"  {DIM}{'─' * 60}{RESET}")

    user_id = args.user_id
    while not user_id:
        user_id = input("Enter username for this face/audio record: ").strip()
    if len(user_id) > 191:
        parser.error("Username must be 191 characters or fewer.")
    info("User record", user_id)

    # ── Resolve image path ────────────────────────────────────────────────────
    if args.image:
        image_path = Path(args.image)
    else:
        image_path = capture_face_from_camera(args.camera_index)

    # ── Run pipeline ──────────────────────────────────────────────────────────
    device = get_device()

    load_face_image(image_path)                                          # Step 1
    embedding = extract_features(image_path, device, "sender")           # Step 2
    display_raw_features(embedding)                                      # Step 3
    storage = PipelineStorage()
    try:
        reject_duplicate_face(embedding, user_id, storage)
    except (ValueError, RuntimeError) as error:
        err(str(error))
        return
    cancelable_template = make_cancelable_template(embedding, user_id)
    sender_face_crop = OUTPUT_DIR / "sender_face_cropped_160x160.jpg"
    face_image_bytes, compressed_size = compress_face_image(sender_face_crop)
    info("Image embedded in audio", f"{compressed_size[0]}x{compressed_size[0]} grayscale JPEG")
    key, _, encrypted_pkg, _, _ = encrypt_features(
        cancelable_template,
        user_id,
        face_image_bytes,
    )
    user_key_path = key_path_for_user(user_id)
    display_encrypted_features(encrypted_pkg)                            # Step 5
    carrier_path = find_carrier_audio(user_id, len(encrypted_pkg))       # Step 6
    carrier_audio, sr, carrier_mono = display_waveform_before(carrier_path)  # Step 7
    play_audio_file(CARRIER_COPY_OUTPUT, "Carrier audio before embedding")
    _, _ = embed_payload(                                                 # Step 8
        carrier_audio, sr, encrypted_pkg, carrier_mono
    )
    play_audio_file(STEGO_OUTPUT, "Stego audio after embedding")
    received_audio, _ = sf.read(str(STEGO_OUTPUT), dtype="float64")
    received_mono = received_audio if received_audio.ndim == 1 else received_audio[:, 0]
    metrics = display_waveform_after(carrier_mono, received_mono, sr)      # Step 9
    storage.save_stego_audio(user_id, STEGO_OUTPUT)
    database_receiver_audio = storage.retrieve_stego_audio(
        user_id,
        OUTPUT_DIR / "receiver_stego_audio.wav",
    )
    info("MySQL database", f"{storage.database} on {storage.host}:{storage.port}")
    info("Database receiver audio", str(database_receiver_audio))
    received_audio, received_rate = sf.read(str(database_receiver_audio), dtype="float64")
    if received_rate != sr:
        raise ValueError("Stored receiver audio sample rate does not match the carrier.")
    received_mono = received_audio if received_audio.ndim == 1 else received_audio[:, 0]

    if args.auth_image is not None:
        receiver_image_path = args.auth_image
    else:
        receiver_image_path = capture_face_from_camera(
            args.camera_index,
            OUTPUT_DIR / "receiver_face.jpg",
            "Capture Receiver Authentication Face",
        )
    load_face_image(receiver_image_path)
    receiver_embedding = extract_features(receiver_image_path, device, "receiver")
    receiver_template = make_cancelable_template(
        receiver_embedding,
        user_id,
        "Receiver Query Cancelable Template",
    )

    verification = verify_roundtrip(                                     # Step 10
        received_mono,
        key,
        receiver_template,
        encrypted_pkg,
    )
    recovered_package = extract_encrypted_package(received_mono)
    _, recovered_face_image = decrypt_biometric_payload(recovered_package, key)
    if recovered_face_image:
        recovered_image_path = OUTPUT_DIR / "recovered_face.jpg"
        recovered_image_path.write_bytes(recovered_face_image)
        info("Recovered embedded image", str(recovered_image_path))
        show_image_in_console(
            recovered_image_path,
            label="FACE IMAGE RECOVERED FROM STEGO AUDIO",
        )
    else:
        warn("This stego payload contains no recoverable image.")
    print_summary(image_path, carrier_path, metrics, verification, user_key_path, True)


if __name__ == "__main__":
    main()
