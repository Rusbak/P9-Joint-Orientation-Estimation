
import numpy as np

from helper_scripts import features_takawale_roy


def generate_signal(fs, frequency=500.0, duration=1.0):
    """Generate a simple sinusoidal test signal."""
    t = np.arange(int(fs * duration)) / fs
    return np.sin(2 * np.pi * frequency * t)


def extract_features(left, right, fs, **kwargs):
    """Calculate spectra and extract the five features."""
    left_spectrum = np.fft.rfft(left, norm="forward")
    right_spectrum = np.fft.rfft(right, norm="forward")

    frequencies = np.fft.rfftfreq(len(left), d=1 / fs)

    features = features_takawale_roy._extract_binaural_features(
        left_spectrum,
        right_spectrum,
        frequencies,
        **kwargs
    )

    return features, frequencies


def print_feature_summary(features):
    names = [
        "ILD low",
        "ILD high",
        "ITD low",
        "ITD high",
        "Inverse convolution"
    ]

    print("\nFeature statistics:")

    for i, name in enumerate(names):
        feature = features[i]

        print(
            f"{name:22s} "
            f"min={np.min(feature):12.6f} "
            f"max={np.max(feature):12.6f} "
            f"mean={np.mean(feature):12.6f} "
            f"std={np.std(feature):12.6f}"
        )


def compare_features(features_a, features_b):
    """Return maximum absolute difference for each feature."""
    return np.max(np.abs(features_a - features_b), axis=1)


def main():

    fs = 48000
    frequency = 500.0
    duration = 1.0

    signal = generate_signal(fs, frequency, duration)

    print("=" * 70)
    print("TAKAWALE & ROY FEATURE EXTRACTION TESTS")
    print("=" * 70)

    # ---------------------------------------------------------
    # TEST 1: IDENTICAL LEFT AND RIGHT SIGNALS
    # ---------------------------------------------------------

    print("\nTEST 1: Identical binaural signals")
    print("-" * 70)

    features_identical, frequencies = extract_features(
        signal,
        signal,
        fs
    )

    print_feature_summary(features_identical)

    print("\nExpected:")
    print("ILD low       = 0 dB")
    print("ILD high      = 0 dB")
    print("ITD low       = 0 s")
    print("ITD high      = 0 s")
    print("Inverse conv. = 0")

    print("\nMeasured at 500 Hz:")
    idx = np.argmin(np.abs(frequencies - frequency))

    print(f"ILD low       = {features_identical[0, idx]:.8f} dB")
    print(f"ITD low       = {features_identical[2, idx]:.8e} s")

    # ---------------------------------------------------------
    # TEST 2: KNOWN AMPLITUDE DIFFERENCE
    # ---------------------------------------------------------

    print("\nTEST 2: Known amplitude difference")
    print("-" * 70)

    right = signal * 0.5

    features_gain, _ = extract_features(
        signal,
        right,
        fs
    )

    expected_ild = 20 * np.log10(2.0)
    measured_ild = features_gain[0, idx]

    print(f"Expected ILD = {expected_ild:.6f} dB")
    print(f"Measured ILD = {measured_ild:.6f} dB")

    print(f"Expected ITD = 0 s")
    print(f"Measured ITD = {features_gain[2, idx]:.8e} s")

    print(
        "PASS" if np.isclose(
            measured_ild, expected_ild, atol=1e-4
        ) else "FAIL"
    )

    # ---------------------------------------------------------
    # TEST 3: KNOWN TIME DELAY
    # ---------------------------------------------------------

    print("\nTEST 3: Known time delay")
    print("-" * 70)

    delay_samples = 3

    delayed_right = np.concatenate([
        np.zeros(delay_samples),
        signal[:-delay_samples]
    ])

    features_delay, _ = extract_features(
        signal,
        delayed_right,
        fs
    )

    expected_itd = delay_samples / fs
    measured_itd = features_delay[2, idx]

    print(f"Delay samples = {delay_samples}")
    print(f"Expected ITD  = {expected_itd:.10e} s")
    print(f"Measured ITD  = {measured_itd:.10e} s")

    print(f"Expected ITD  = {expected_itd * 1e6:.4f} us")
    print(f"Measured ITD  = {measured_itd * 1e6:.4f} us")

    print(
        "PASS" if np.isclose(
            measured_itd, expected_itd, atol=1e-7
        ) else "FAIL"
    )

    # ---------------------------------------------------------
    # TEST 4: SWAPPING LEFT AND RIGHT
    # ---------------------------------------------------------

    print("\nTEST 4: Swapping left and right channels")
    print("-" * 70)

    left = signal
    right = signal * 0.5

    features_original, _ = extract_features(
        left, right, fs
    )

    features_swapped, _ = extract_features(
        right, left, fs
    )

    print("Maximum absolute differences after swapping:")

    print(
        "ILD low sign reversal:",
        np.max(np.abs(
            features_original[0] + features_swapped[0]
        ))
    )

    print(
        "ILD high sign reversal:",
        np.max(np.abs(
            features_original[1] + features_swapped[1]
        ))
    )

    print(
        "ITD low sign reversal:",
        np.max(np.abs(
            features_original[2] + features_swapped[2]
        ))
    )

    print(
        "ITD high sign reversal:",
        np.max(np.abs(
            features_original[3] + features_swapped[3]
        ))
    )

    print(
        "Inverse convolution invariance:",
        np.max(np.abs(
            features_original[4] - features_swapped[4]
        ))
    )

    # ---------------------------------------------------------
    # TEST 5: COMMON AMPLITUDE SCALING
    # ---------------------------------------------------------

    print("\nTEST 5: Common amplitude scaling")
    print("-" * 70)

    features_normal, _ = extract_features(
        signal,
        signal * 0.5,
        fs
    )

    scale = 10.0

    features_scaled, _ = extract_features(
        signal * scale,
        signal * 0.5 * scale,
        fs
    )

    differences = compare_features(
        features_normal,
        features_scaled
    )

    names = [
        "ILD low",
        "ILD high",
        "ITD low",
        "ITD high",
        "Inverse convolution"
    ]

    for name, difference in zip(names, differences):
        print(f"{name:22s}: {difference:.10e}")

    # ---------------------------------------------------------
    # TEST 5B: INVESTIGATE AMPLITUDE SCALING ERROR
    # ---------------------------------------------------------

    print("\nTEST 5B: Investigating amplitude scaling")
    print("-" * 70)

    left = signal
    right = signal * 0.5

    features_1, frequencies = extract_features(
        left, right, fs
    )

    features_10, _ = extract_features(
        left * 10, right * 10, fs
    )

    difference = np.abs(features_1 - features_10)

    # Inspect the frequency bins with the largest differences.
    for feature_idx, name in enumerate([
        "ILD low",
        "ILD high",
        "ITD low",
        "ITD high",
        "Inverse convolution"
    ]):

        feature_difference = difference[feature_idx]

        largest_indices = np.argsort(
            feature_difference
        )[-5:][::-1]

        print(f"\n{name}")
        print("Largest differences:")

        for idx in largest_indices:
            print(
                f"Frequency: {frequencies[idx]:9.2f} Hz | "
                f"Original: {features_1[feature_idx, idx]:12.6f} | "
                f"Scaled: {features_10[feature_idx, idx]:12.6f} | "
                f"Difference: {feature_difference[idx]:12.6f}"
            )

    # Also inspect the actual spectral magnitudes.
    left_spectrum = np.fft.rfft(left, norm="forward")
    right_spectrum = np.fft.rfft(right, norm="forward")

    print("\nSpectral magnitudes at the largest ILD discrepancy:")

    indices = np.argsort(difference[0])[-5:][::-1]

    for idx in indices:
        print(
            f"Frequency: {frequencies[idx]:9.2f} Hz | "
            f"Left: {abs(left_spectrum[idx]):.4e} | "
            f"Right: {abs(right_spectrum[idx]):.4e}"
        )

    # ---------------------------------------------------------
    # TEST 6: INVERSE CONVOLUTION
    # ---------------------------------------------------------

    print("\nTEST 6: Inverse convolution")
    print("-" * 70)

    low_mask = frequencies <= 952.0
    high_mask = frequencies > 952.0

    ild_low = features_normal[0]
    ild_high = features_normal[1]
    inverse_conv = features_normal[4]

    print("ILD low nonzero bins:",
          np.count_nonzero(ild_low))

    print("ILD high nonzero bins:",
          np.count_nonzero(ild_high))

    print("Inverse convolution nonzero bins:",
          np.count_nonzero(inverse_conv))

    print("Inverse convolution maximum:",
          np.max(np.abs(inverse_conv)))

    print("Expected frequency separation:")
    print("ILD low  : frequencies <= 952 Hz")
    print("ILD high : frequencies > 952 Hz")

    print("ILD low outside its band:",
          np.max(np.abs(ild_low[high_mask])))

    print("ILD high outside its band:",
          np.max(np.abs(ild_high[low_mask])))

    # ---------------------------------------------------------
    # TEST 7: NUMERICAL STABILITY
    # ---------------------------------------------------------

    print("\nTEST 7: Numerical stability")
    print("-" * 70)

    for amplitude in [1.0, 0.1, 0.01, 0.001, 0.0001]:

        test_features, _ = extract_features(
            signal * amplitude,
            signal * amplitude * 0.5,
            fs
        )

        print(
            f"Amplitude={amplitude:8.4f} | "
            f"Finite={np.all(np.isfinite(test_features))} | "
            f"Max abs={np.max(np.abs(test_features)):.6f}"
        )

    # ---------------------------------------------------------
    # FINAL SUMMARY
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL FEATURE SUMMARY")
    print("=" * 70)

    print_feature_summary(features_normal)

    print("\nFeature shape:", features_normal.shape)
    print("Feature dtype:", features_normal.dtype)
    print("Frequency bins:", len(frequencies))
    print("Frequency resolution:", frequencies[1] - frequencies[0])

    print("\nAll tests completed.")


if __name__ == "__main__":
    main()
