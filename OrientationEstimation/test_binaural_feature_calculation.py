import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d
from helper_scripts import features_takawale_roy

dataset = fr"D:\Python Projects\P9-Joint-Orientation-Estimation\DataGeneration\Datasets\combined_dataset_commands.h5"

with h5py.File(dataset, "r") as f:
    # Load the first entry of a known dataset
    binaural_signal = f["/train/signals"][1]
    print(binaural_signal)

def compare_energy_thresholds(signal, fs):

    left = np.asarray(signal[0], dtype=np.float64)
    right = np.asarray(signal[1], dtype=np.float64)

    left = features_takawale_roy._ensure_one_second(left, fs)
    right = features_takawale_roy._ensure_one_second(right, fs)

    left_spectrum = np.fft.rfft(left, norm="forward")
    right_spectrum = np.fft.rfft(right, norm="forward")

    binaural_power = (
        np.abs(left_spectrum)**2 +
        np.abs(right_spectrum)**2
    ) / 2

    max_power = np.max(binaural_power)

    print("\nFEATURE COMPARISON")
    print("-" * 60)

    thresholds_db = [-60, -80, -100, -120]

    reference_features = None

    for threshold_db in thresholds_db:

        threshold = 10 ** (threshold_db / 10)
        energy_mask = binaural_power > max_power * threshold

        left_masked = left_spectrum.copy()
        right_masked = right_spectrum.copy()

        left_masked[~energy_mask] = 0
        right_masked[~energy_mask] = 0

        # Extract features using the masked spectra.
        # We will add this part once we have isolated the
        # feature calculations from your existing function.

        print(
            f"Threshold: {threshold_db} dB | "
            f"Retained bins: {np.sum(energy_mask)}"
        )

def compare_energy_thresholds_with_pefac(signal, fs):
    print('this is function: compare_energy_thresholds_with_pefac()')

    # Prepare binaural signal
    left = features_takawale_roy._ensure_one_second(
        signal[0], fs
    )
    right = features_takawale_roy._ensure_one_second(
        signal[1], fs
    )

    # Apply the same PEFAC voicing procedure
    _, _, voiced_sample_mask = (
        features_takawale_roy._get_pefac_voicing(
            left, right, fs, voicing_threshold=0.5
        )
    )

    left_voiced = left * voiced_sample_mask
    right_voiced = right * voiced_sample_mask

    # Calculate FFT
    left_spectrum = np.fft.rfft(
        left_voiced, norm="forward"
    )
    right_spectrum = np.fft.rfft(
        right_voiced, norm="forward"
    )

    frequencies = np.fft.rfftfreq(
        left_voiced.size, d=1.0 / fs
    )

    # Calculate binaural power
    binaural_power = (
        np.abs(left_spectrum)**2 +
        np.abs(right_spectrum)**2
    ) / 2

    max_power = np.max(binaural_power)

    if max_power == 0:
        raise ValueError("Signal contains no usable spectral energy.")

    relative_power_db = 10 * np.log10(
        binaural_power / max_power + 1e-30
    )

    print("\nSPECTRAL ENERGY AFTER PEFAC")
    print("-" * 60)

    for threshold_db in [-40, -60, -80, -100, -120]:
        retained = relative_power_db >= threshold_db

        print(
            f"{threshold_db:4d} dB: "
            f"{np.sum(retained):5d} / {len(retained)} bins "
            f"({100 * np.mean(retained):6.2f}%)"
        )

    # Extract reference features without an additional energy mask
    reference_features = (
        features_takawale_roy._extract_binaural_features(
            left_spectrum,
            right_spectrum,
            frequencies,
        )
    )

    feature_names = [
        "ILD low",
        "ILD high",
        "ITD low",
        "ITD high",
        "Inverse convolution",
    ]

    thresholds_db = [-60, -80, -100, -120]

    print("\nFEATURE COMPARISON")
    print("-" * 75)

    print("Reference: No additional energy mask")

    for threshold_db in thresholds_db:

        # Convert dB power threshold to linear power
        threshold = 10 ** (threshold_db / 10)

        energy_mask = (
            binaural_power > max_power * threshold
        )

        # Apply the same mask to both ears
        left_masked = left_spectrum.copy()
        right_masked = right_spectrum.copy()

        left_masked[~energy_mask] = 0
        right_masked[~energy_mask] = 0

        # Extract features while excluding invalid bins
        masked_features = (
            features_takawale_roy._extract_binaural_features(
                left_masked,
                right_masked,
                frequencies,
                valid_frequency_mask=energy_mask,
            )
        )

        difference = masked_features - reference_features

        print(
            f"\nThreshold: {threshold_db} dB "
            f"({np.sum(energy_mask)} retained bins)"
        )

        for i, name in enumerate(feature_names):

            mae = np.mean(np.abs(difference[i]))
            max_diff = np.max(np.abs(difference[i]))

            reference_rms = np.sqrt(
                np.mean(reference_features[i]**2)
            )

            difference_rms = np.sqrt(
                np.mean(difference[i]**2)
            )

            relative_rms = (
                difference_rms / reference_rms
                if reference_rms > 1e-12
                else np.nan
            )

            print(
                f"{name:22s} | "
                f"MAE: {mae:.6e} | "
                f"Max: {max_diff:.6e} | "
                f"Relative RMS: {relative_rms:.4%}"
            )

def compare_pefac_spectral_spreading(signal, fs):

    left = features_takawale_roy._ensure_one_second(
        signal[0], fs
    )
    right = features_takawale_roy._ensure_one_second(
        signal[1], fs
    )

    pefac_times, pefac_probability, voiced_mask = (
        features_takawale_roy._get_pefac_voicing(
            left, right, fs, 0.5
        )
    )

    # Create a smoothed version of the binary mask.
    from scipy.ndimage import gaussian_filter1d

    smooth_mask = gaussian_filter1d(
        voiced_mask.astype(np.float64),
        sigma=fs * 0.002
    )

    smooth_mask = np.clip(smooth_mask, 0.0, 1.0)

    signals = {
        "Original": (left, right),
        "Binary PEFAC": (
            left * voiced_mask,
            right * voiced_mask
        ),
        "Smoothed PEFAC": (
            left * smooth_mask,
            right * smooth_mask
        )
    }

    print("\nPEFAC SPECTRAL SPREADING")
    print("-" * 70)

    for name, (left_signal, right_signal) in signals.items():

        left_fft = np.fft.rfft(left_signal, norm="forward")
        right_fft = np.fft.rfft(right_signal, norm="forward")

        power = (
            np.abs(left_fft)**2 +
            np.abs(right_fft)**2
        ) / 2

        max_power = np.max(power)

        relative_power_db = 10 * np.log10(
            power / max_power + 1e-30
        )

        print(f"\n{name}")

        for threshold_db in [-40, -60, -80, -100]:
            retained = relative_power_db >= threshold_db

            print(
                f"{threshold_db:4d} dB: "
                f"{np.sum(retained):5d} / {len(power)} "
                f"({100 * np.mean(retained):6.2f}%)"
            )

    print("\nPEFAC MASK DIAGNOSTICS")
    print("-" * 60)

    print(f"Voiced samples: {np.mean(voiced_mask) * 100:.2f}%")
    print(f"Mask transitions: {np.sum(np.diff(voiced_mask.astype(int)) != 0)}")

    print(f"Probability minimum: {np.min(pefac_probability):.4f}")
    print(f"Probability maximum: {np.max(pefac_probability):.4f}")
    print(f"Probability mean:    {np.mean(pefac_probability):.4f}")
    print(f"Probability std:     {np.std(pefac_probability):.4f}")

def inspect_spectral_energy(signal, fs):

    left = np.asarray(signal[0], dtype=np.float64)
    right = np.asarray(signal[1], dtype=np.float64)

    left = features_takawale_roy._ensure_one_second(left, fs)
    right = features_takawale_roy._ensure_one_second(right, fs)

    left_spectrum = np.fft.rfft(left, norm="forward")
    right_spectrum = np.fft.rfft(right, norm="forward")

    frequencies = np.fft.rfftfreq(len(left), 1 / fs)

    binaural_power = (
        np.abs(left_spectrum) ** 2 +
        np.abs(right_spectrum) ** 2
    ) / 2

    print("\nENERGY MASK")
    print("-" * 60)

    max_power = np.max(binaural_power)

    relative_threshold = 1e-10  # -100 dB in power
    energy_mask = binaural_power > max_power * relative_threshold

    for threshold_db in [-60, -80, -100, -120]:
        threshold = 10 ** (threshold_db / 10)
        mask = binaural_power > max_power * threshold

        print(
            f"{threshold_db:4d} dB: "
            f"{np.sum(mask)} / {len(mask)} bins "
            f"({100 * np.mean(mask):.2f}%)"
        )

    print("\nSPECTRAL ENERGY DIAGNOSTICS")
    print("-" * 60)

    relative_power_db = 10 * np.log10(
        binaural_power / np.max(binaural_power) + 1e-30
    )

    for threshold_db in [-40, -60, -80, -100, -120]:
        retained = relative_power_db >= threshold_db

        print(
            f"Threshold {threshold_db:4d} dB: "
            f"{np.sum(retained):5d} / {len(retained)} bins "
            f"({100 * np.mean(retained):6.2f}%)"
        )

    print("\nStrongest frequency bins:")

    indices = np.argsort(binaural_power)[-10:][::-1]

    for idx in indices:
        print(
            f"{frequencies[idx]:9.2f} Hz | "
            f"{relative_power_db[idx]:8.2f} dB"
        )

fs = 48000

# inspect_spectral_energy(binaural_signal, fs)
# compare_energy_thresholds(binaural_signal, fs)
# compare_energy_thresholds_with_pefac(binaural_signal, fs)
# compare_pefac_spectral_spreading(binaural_signal, fs)

def compare_pefac_masking_approaches(signal, fs, threshold=0.5):
    """
    Compare binary, probability-weighted, and smoothed binary PEFAC masks.
    """

    left = signal[0]
    right = signal[1]

    # Get PEFAC voicing probabilities and binary mask
    pefac_times, pefac_probability, voiced_mask = (
        features_takawale_roy._get_pefac_voicing(
            left, right, fs, threshold
        )
    )

    # Construct the three masking approaches
    binary_mask = voiced_mask.astype(np.float64)

    probability_mask = np.clip(
        voiced_mask.astype(np.float64) * 0 +
        np.interp(
            np.arange(left.size) / fs,
            pefac_times,
            pefac_probability,
            left=0.0,
            right=0.0,
        ),
        0.0,
        1.0,
    )

    smoothed_mask = gaussian_filter1d(
        binary_mask,
        sigma=fs * 0.002,
    )

    masks = {
        "Original": np.ones_like(binary_mask),
        "Binary": binary_mask,
        "Probability weighted": probability_mask,
        "Smoothed binary": smoothed_mask,
    }

    frequencies = np.fft.rfftfreq(left.size, d=1.0 / fs)

    results = {}

    for name, mask in masks.items():

        # Apply the same mask to both ears
        masked_left = left * mask
        masked_right = right * mask

        # Calculate RMS
        rms_left = np.sqrt(np.mean(masked_left ** 2))
        rms_right = np.sqrt(np.mean(masked_right ** 2))
        rms_binaural = np.sqrt(
            (rms_left ** 2 + rms_right ** 2) / 2
        )

        # Calculate spectra
        left_spectrum = np.fft.rfft(
            masked_left, norm="forward"
        )
        right_spectrum = np.fft.rfft(
            masked_right, norm="forward"
        )

        power = (
            np.abs(left_spectrum) ** 2 +
            np.abs(right_spectrum) ** 2
        ) / 2

        # Relative spectral distribution
        relative_power_db = 10 * np.log10(
            power / (np.max(power) + 1e-30) + 1e-30
        )

        # Absolute spectral energy
        total_spectral_power = np.sum(power)

        results[name] = {
            "mask": mask,
            "left_spectrum": left_spectrum,
            "right_spectrum": right_spectrum,
            "power": power,
            "rms": rms_binaural,
            "relative_power_db": relative_power_db,
            "total_power": total_spectral_power,
        }

    # Print summary
    print("\nPEFAC MASKING APPROACH COMPARISON")
    print("=" * 75)

    print(
        f"{'Approach':<23}"
        f"{'RMS':>14}"
        f"{'Total power':>16}"
        f"{'Mask mean':>14}"
    )

    for name, result in results.items():
        print(
            f"{name:<23}"
            f"{result['rms']:>14.6e}"
            f"{result['total_power']:>16.6e}"
            f"{np.mean(result['mask']):>14.4f}"
        )

    # Compare relative spectral distributions
    print("\nRELATIVE SPECTRAL DISTRIBUTION")
    print("=" * 75)
    print(f"{'Approach':<23}" + "".join(
        f"{db:>12} dB" for db in [-40, -60, -80, -100]
    ))

    for name, result in results.items():
        percentages = [
            np.mean(result["relative_power_db"] >= db) * 100
            for db in [-40, -60, -80, -100]
        ]

        print(
            f"{name:<23}" +
            "".join(f"{p:>12.2f}%" for p in percentages)
        )

    # Compare each approach against the original spectrum
    original_power = results["Original"]["power"]

    print("\nSPECTRAL DIFFERENCE FROM ORIGINAL")
    print("=" * 75)

    for name in ["Binary", "Probability weighted", "Smoothed binary"]:

        power = results[name]["power"]

        relative_error = (
            np.linalg.norm(power - original_power) /
            (np.linalg.norm(original_power) + 1e-30)
        )

        print(
            f"{name:<23}"
            f"Relative power-spectrum error: {relative_error:.6f}"
        )

    return frequencies, results

frequencies, results = compare_pefac_masking_approaches(
    binaural_signal,
    fs=48000,
    threshold=0.5,
)