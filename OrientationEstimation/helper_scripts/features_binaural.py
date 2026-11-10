import numpy as np
from scipy.signal import butter, sosfilt, correlate, correlation_lags


# ============================================================
# Helper functions
# ============================================================

def _validate_signals(left, right, fs):
    """Validate and convert binaural signals."""

    left = np.asarray(left, dtype=float).flatten()
    right = np.asarray(right, dtype=float).flatten()

    if left.size != right.size:
        raise ValueError("Left and right signals must have equal length.")

    if left.size == 0:
        raise ValueError("Signals cannot be empty.")

    if fs <= 0:
        raise ValueError("Sampling rate must be positive.")

    return left, right


def _remove_dc(signal):
    """Remove the mean value of a signal."""

    signal -= np.mean(signal)

    return signal


def _bandpass_filter(left, right, fs, frequency_band, order=4):
    """
    Apply the same Butterworth bandpass filter to both channels.

    Parameters
    ----------
    left, right : array_like
        Binaural signals.
    fs : float
        Sampling rate in Hz.
    frequency_band : tuple
        (low_frequency, high_frequency) in Hz.
    order : int
        Filter order.

    Returns
    -------
    left_filtered, right_filtered : ndarray
    """

    low_freq, high_freq = frequency_band

    if not (0 < low_freq < high_freq < fs / 2):
        raise ValueError(
            "Frequency band must satisfy 0 < low < high < Nyquist."
        )

    sos = butter(
        order,
        [low_freq, high_freq],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    left_filtered = sosfilt(sos, left)
    right_filtered = sosfilt(sos, right)

    return left_filtered, right_filtered


def _max_lag_samples(fs, max_lag):
    """Convert maximum lag in seconds to samples."""

    lag_samples = int(round(max_lag * fs))

    return lag_samples


def compute_features(signals, fs, feature_list, frequency_bands):
    '''
    As we introduce more features, we really need to do this better...
    There is a function called something like str2func, which would make this so much easier.
    '''
    
    all_features = []
    
    for i in range(len(signals)):
        left = signals[i, 0]
        right = signals[i, 1]

        left, right = _validate_signals(left, right, fs)

        computed_features = []

        # broadband features
        if 'broadband_itd' in feature_list:
            computed_features.append(
                broadband_itd(left, right, fs)
            )
        if 'broadband_ild' in feature_list:
            computed_features.append(
                broadband_ild(left, right, fs)
            )
        if 'broadband_iacc' in feature_list:
            computed_features.append(
                broadband_iacc(left, right, fs)
            )

        # frequency dependent features
        if 'frequency_dependent_itd' in feature_list:
            computed_features.append(
                frequency_dependent_itd(left, right, fs, frequency_bands)
            )
        if 'frequency_dependent_ild' in feature_list:
            computed_features.append(
                frequency_dependent_ild(left, right, fs, frequency_bands)
            )
        if 'frequency_dependent_iacc' in feature_list:
            computed_features.append(
                frequency_dependent_iacc(left, right, fs, frequency_bands)
            )
        if 'frequency_dependent_ipd' in feature_list:
            computed_features.append(
                frequency_dependent_ipd(left, right, fs, frequency_bands)
            )

        # Combine all features for this sample into one vector
        sample_features = np.concatenate([
            np.asarray(feature, dtype=np.float32).reshape(-1)
            for feature in computed_features
        ])

        all_features.append(sample_features)

    return np.asarray(all_features, dtype=np.float32)

# ============================================================
# Broadband Features
# ============================================================

# ------------------------------------------------------------
# ITD - Interaural Time Difference
# ------------------------------------------------------------
def broadband_itd(left, right, fs, max_lag=0.001):
    """
    Estimate broadband ITD using cross-correlation.

    Positive ITD means the sound arrives at the left ear first.

    Parameters
    ----------
    left, right : array_like
        Binaural signals.
    fs : float
        Sampling rate in Hz.
    max_lag : float
        Maximum absolute time lag in seconds.

    Returns
    -------
    itd : float
        Estimated ITD in seconds.
    """
    left = _remove_dc(left)
    right = _remove_dc(right)

    corr = correlate(right, left, mode="full", method="fft")
    lags = correlation_lags(len(right), len(left), mode="full")

    max_lag_samples = _max_lag_samples(fs, max_lag)

    valid = np.abs(lags) <= max_lag_samples

    corr = corr[valid]
    lags = lags[valid]

    peak_index = np.argmax(np.abs(corr))

    itd = lags[peak_index] / fs

    return itd


# ------------------------------------------------------------
# ILD - Interaural Level Difference
# ------------------------------------------------------------
def broadband_ild(left, right, fs, eps=1e-12):
    """
    Calculate broadband ILD using RMS levels.

    Positive ILD means the left ear receives a higher level.

    Returns
    -------
    ild : float
        ILD in dB.
    """
    left_rms = np.sqrt(np.mean(left ** 2))
    right_rms = np.sqrt(np.mean(right ** 2))

    ild = 20.0 * np.log10(
        (left_rms + eps) / (right_rms + eps)
    )

    return ild


# ------------------------------------------------------------
# IACC - Interaural Cross-Correlation Coefficient
# ------------------------------------------------------------

def broadband_iacc(left, right, fs, max_lag=0.001):
    """
    Calculate broadband IACC.

    Returns the maximum absolute normalized cross-correlation
    within the specified lag range.

    Returns
    -------
    iacc : float
        IACC between 0 and 1.
    """
    left = _remove_dc(left)
    right = _remove_dc(right)

    left_energy = np.sum(left ** 2)
    right_energy = np.sum(right ** 2)

    denominator = np.sqrt(left_energy * right_energy)

    if denominator == 0:
        return 0.0

    corr = correlate(right, left, mode="full", method="fft")
    lags = correlation_lags(len(right), len(left), mode="full")

    max_lag_samples = _max_lag_samples(fs, max_lag)

    valid = np.abs(lags) <= max_lag_samples

    corr = corr[valid]

    normalized_corr = corr / denominator

    iacc = np.max(np.abs(normalized_corr))

    clipped_iacc = float(np.clip(iacc, 0.0, 1.0))

    return clipped_iacc


# ============================================================
# Frequency-Dependent Features
# ============================================================

# ------------------------------------------------------------
# Frequency-Dependent ITD
# ------------------------------------------------------------
def frequency_dependent_itd(
    left,
    right,
    fs,
    frequency_bands,
    max_lag=0.001,
    filter_order=4
):
    """
    Calculate ITD independently for each frequency band.

    Returns
    -------
    itd_values : ndarray
        One ITD value in seconds per frequency band.
    """
    itd_values = []

    for band in frequency_bands:
        left_band, right_band = _bandpass_filter(
            left,
            right,
            fs,
            band,
            order=filter_order
        )

        itd = broadband_itd(
            left_band,
            right_band,
            fs,
            max_lag=max_lag
        )

        itd_values.append(itd)

    return np.asarray(itd_values)


# ------------------------------------------------------------
# Frequency-Dependent ILD
# ------------------------------------------------------------
def frequency_dependent_ild(
    left,
    right,
    fs,
    frequency_bands,
    filter_order=4,
    eps=1e-12
):
    """
    Calculate ILD independently for each frequency band.

    Returns
    -------
    ild_values : ndarray
        One ILD value in dB per frequency band.
    """
    ild_values = []

    for band in frequency_bands:

        left_band, right_band = _bandpass_filter(
            left,
            right,
            fs,
            band,
            order=filter_order
        )

        ild = broadband_ild(
            left_band,
            right_band,
            fs,
            eps=eps
        )

        ild_values.append(ild)

    return np.asarray(ild_values)


# ------------------------------------------------------------
# Frequency-Dependent IACC
# ------------------------------------------------------------
def frequency_dependent_iacc(
    left,
    right,
    fs,
    frequency_bands,
    max_lag=0.001,
    filter_order=4
):
    """
    Calculate IACC independently for each frequency band.

    Returns
    -------
    iacc_values : ndarray
        One IACC value per frequency band.
    """
    iacc_values = []

    for band in frequency_bands:

        left_band, right_band = _bandpass_filter(
            left,
            right,
            fs,
            band,
            order=filter_order
        )

        iacc = broadband_iacc(
            left_band,
            right_band,
            fs,
            max_lag=max_lag
        )

        iacc_values.append(iacc)

    return np.asarray(iacc_values)


# ============================================================
# Additional Features
# ============================================================

# ------------------------------------------------------------
# Frequency-Dependent Interaural Phase Difference (IPD)
# ------------------------------------------------------------
def frequency_dependent_ipd(
    left,
    right,
    fs,
    frequency_bands,
    filter_order=4
):
    """
    Calculate the interaural phase difference for each band.

    Uses the phase difference between the complex FFT
    coefficients at the dominant frequency in each band.

    Returns
    -------
    ipd_values : ndarray
        Phase differences in radians, in the range [-pi, pi].
    """
    ipd_values = []

    for band in frequency_bands:

        left_band, right_band = _bandpass_filter(
            left,
            right,
            fs,
            band,
            order=filter_order
        )

        left_fft = np.fft.rfft(left_band)
        right_fft = np.fft.rfft(right_band)

        frequencies = np.fft.rfftfreq(
            len(left_band),
            d=1 / fs
        )

        valid = (
            (frequencies >= band[0]) &
            (frequencies <= band[1])
        )

        if not np.any(valid):
            ipd_values.append(0.0)
            continue

        left_magnitude = np.abs(left_fft[valid])

        dominant_index = np.argmax(left_magnitude)

        left_phase = np.angle(left_fft[valid][dominant_index])
        right_phase = np.angle(right_fft[valid][dominant_index])

        ipd = np.angle(
            np.exp(1j * (left_phase - right_phase))
        )

        ipd_values.append(ipd)

    return np.asarray(ipd_values)


# ============================================================
# Features from Takawale and Roy (STFTs)
# ============================================================

# many of the features that we need for takawale and roy's model, are some that we also need to make for the talker orientation anyways, so it might make more sense to have a script for features related to the listener and a script related to the talker. but then again spectrogram features could also be used by the 2d cnn

# ------------------------------------------------------------
# Hi/Low ITD
# ------------------------------------------------------------

# ------------------------------------------------------------
# Hi/Low ILD
# ------------------------------------------------------------

# ------------------------------------------------------------
# Inverse Convolution
# ------------------------------------------------------------


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    fs = 48000

    # Example binaural signals
    duration = 0.25

    t = np.arange(int(fs * duration)) / fs

    left = np.sin(2 * np.pi * 1000 * t)
    right = np.sin(2 * np.pi * 1000 * t - 0.2)

    # Frequency bands
    frequency_bands = [
        (200, 500),
        (500, 1000),
        (1000, 2000),
        (2000, 4000)
    ]

    # Broadband features
    itd = broadband_itd(left, right, fs)
    ild = broadband_ild(left, right, fs)
    iacc = broadband_iacc(left, right, fs)

    print("Broadband ITD:", itd)
    print("Broadband ILD:", ild)
    print("Broadband IACC:", iacc)

    # Frequency-dependent features
    itd_bands = frequency_dependent_itd(
        left, right, fs, frequency_bands
    )

    ild_bands = frequency_dependent_ild(
        left, right, fs, frequency_bands
    )

    iacc_bands = frequency_dependent_iacc(
        left, right, fs, frequency_bands
    )

    ipd_bands = frequency_dependent_ipd(
        left, right, fs, frequency_bands
    )

    print("Frequency-dependent ITD:", itd_bands)
    print("Frequency-dependent ILD:", ild_bands)
    print("Frequency-dependent IACC:", iacc_bands)
    print("Frequency-dependent IPD:", ipd_bands)
