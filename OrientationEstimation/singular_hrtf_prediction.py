
from pathlib import Path

import h5py
import numpy as np
import soundfile as sf

from scipy.signal import fftconvolve, resample_poly, stft
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# --------------------------------------------------
# 1. File paths and settings
# --------------------------------------------------

SOFA_PATH = Path("your_listener.sofa")
SPEECH_PATH = Path("speech.wav")

# Keep measurements close to the horizontal plane.
MAX_ELEVATION = 10.0

# Number of synthetic examples generated per direction.
EXAMPLES_PER_DIRECTION = 12

# Duration of each speech excerpt, in seconds.
CLIP_SECONDS = 2.0

RANDOM_SEED = 42


# --------------------------------------------------
# 2. Read a SimpleFreeFieldHRIR SOFA file
# --------------------------------------------------

def read_sofa(path):
    with h5py.File(path, "r") as f:
        convention = f.attrs.get("SOFAConventions", "")
        if isinstance(convention, bytes):
            convention = convention.decode()

        print("SOFA convention:", convention)

        if convention != "SimpleFreeFieldHRIR":
            raise ValueError(
                "This first script expects SimpleFreeFieldHRIR. "
                "Inspect the convention before adapting the loader."
            )

        ir = np.asarray(f["Data.IR"], dtype=np.float64)
        positions = np.asarray(
            f["SourcePosition"], dtype=np.float64
        )

        sr = float(np.asarray(f["Data.SamplingRate"]).ravel()[0])

        pos_type = f["SourcePosition"].attrs.get("Type", "")
        pos_units = f["SourcePosition"].attrs.get("Units", "")

        if isinstance(pos_type, bytes):
            pos_type = pos_type.decode()
        if isinstance(pos_units, bytes):
            pos_units = pos_units.decode()

        # Data.IR should be measurement x receiver x sample.
        if ir.ndim != 3 or ir.shape[1] != 2:
            raise ValueError(
                f"Unexpected Data.IR shape: {ir.shape}"
            )

        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(
                f"Unexpected SourcePosition shape: {positions.shape}"
            )

        if len(positions) != len(ir):
            raise ValueError("Measurement counts do not match.")

        # This script uses azimuth/elevation in degrees.
        if pos_type.lower() != "spherical":
            raise ValueError(
                f"Expected spherical SourcePosition, got {pos_type!r}. "
                "Convert Cartesian positions to angles first."
            )

        if pos_units.lower() not in ("degree", "degrees"):
            raise ValueError(
                f"Expected degree units, got {pos_units!r}. "
                "Convert the coordinates before proceeding."
            )

        # First column: azimuth; second: elevation.
        azimuth = positions[:, 0]
        elevation = positions[:, 1]

    return ir, azimuth, elevation, sr


# --------------------------------------------------
# 3. Load and resample mono speech
# --------------------------------------------------

def load_speech(path, target_sr):
    audio, sr = sf.read(path)

    # Convert stereo speech to mono if necessary.
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        from math import gcd
        g = gcd(int(sr), int(target_sr))
        audio = resample_poly(
            audio, int(target_sr) // g, int(sr) // g
        )

    audio = audio.astype(np.float64)
    audio -= audio.mean()

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak

    return audio


# --------------------------------------------------
# 4. Create binaural examples and extract features
# --------------------------------------------------

def extract_features(left, right, sr):
    # Average short-time power spectra.
    # The window is about 32 ms, with a 50% overlap.
    nperseg = min(512, len(left))
    noverlap = nperseg // 2

    _, _, zl = stft(
        left, fs=sr, nperseg=nperseg, noverlap=noverlap
    )
    _, _, zr = stft(
        right, fs=sr, nperseg=nperseg, noverlap=noverlap
    )

    pl = np.mean(np.abs(zl) ** 2, axis=1)
    pr = np.mean(np.abs(zr) ** 2, axis=1)

    eps = 1e-12
    log_l = 10 * np.log10(pl + eps)
    log_r = 10 * np.log10(pr + eps)

    # Features: left spectrum, right spectrum, and
    # frequency-dependent interaural level difference.
    return np.concatenate([
        log_l,
        log_r,
        log_l - log_r
    ])


def make_examples(ir, azimuth, elevation, sr, speech):
    rng = np.random.default_rng(RANDOM_SEED)

    # Use approximately horizontal measurements.
    valid = np.abs(elevation) <= MAX_ELEVATION
    ir = ir[valid]
    azimuth = azimuth[valid]

    if len(ir) == 0:
        raise ValueError("No measurements in the selected elevation range.")

    clip_len = int(CLIP_SECONDS * sr)

    if len(speech) < clip_len:
        raise ValueError("Speech file is shorter than the requested clip.")

    features = []
    labels = []

    for m in range(len(ir)):
        h_left = ir[m, 0]
        h_right = ir[m, 1]

        for _ in range(EXAMPLES_PER_DIRECTION):
            start = rng.integers(0, len(speech) - clip_len + 1)
            segment = speech[start:start + clip_len]

            left = fftconvolve(segment, h_left, mode="full")
            right = fftconvolve(segment, h_right, mode="full")

            # Remove the filter tail to retain a fixed duration.
            left = left[:clip_len]
            right = right[:clip_len]

            features.append(extract_features(left, right, sr))
            labels.append(azimuth[m])

    return np.asarray(features), np.asarray(labels)


# --------------------------------------------------
# 5. Train and evaluate a first baseline
# --------------------------------------------------

def angular_error_deg(predicted, actual):
    # Circular angular difference, in [-180, 180).
    return (predicted - actual + 180) % 360 - 180


def main():
    ir, azimuth, elevation, sofa_sr = read_sofa(SOFA_PATH)

    print("HRIR array shape:", ir.shape)
    print("SOFA sampling rate:", sofa_sr)
    print("Available measurements:", len(azimuth))

    speech = load_speech(SPEECH_PATH, int(sofa_sr))

    X, y = make_examples(
        ir, azimuth, elevation, int(sofa_sr), speech
    )

    # Split synthetic examples into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    errors = angular_error_deg(predictions, y_test)

    print("\n--- Baseline results ---")
    print(f"Training examples: {len(X_train)}")
    print(f"Testing examples:  {len(X_test)}")
    print(f"Exact-angle accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(f"Mean absolute angular error: {np.mean(np.abs(errors)):.2f} degrees")
    print(f"Median absolute angular error: {np.median(np.abs(errors)):.2f} degrees")

    print("\n--- Example predictions ---")
    for actual, predicted in zip(y_test[:20], predictions[:20]):
        error = angular_error_deg(predicted, actual)
        print(
            f"Actual: {actual:7.2f} deg | "
            f"Predicted: {predicted:7.2f} deg | "
            f"Error: {error:+7.2f} deg"
        )


if __name__ == "__main__":
    main()
