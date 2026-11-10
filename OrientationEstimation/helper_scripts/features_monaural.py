import numpy as np
from scipy.fft import dct

# --------------------------------------------------
# Configuration
# --------------------------------------------------

WINDOW_LENGTH = 0.025
HOP_LENGTH = 0.010
N_FFT = 2048

N_MELS = 64
N_MFCC = 13

# TODO: these values were just picked to fit the frequencies of the chirps used in data generation
F_MIN = 300.0
F_MAX = 7000.0

FREQUENCY_BANDS = [
    # (0, 500),
    (300, 500),
    (500, 1000),
    (1000, 2000),
    (2000, 4000),
    (4000, 7000),
    # (4000, 8000),
    # (8000, 12000),
    # (12000, 20000),
]

# --------------------------------------------------
# STFT
# --------------------------------------------------

def compute_stft(signal, fs):
    """
    Compute a centered, windowed STFT.

    Returns
    -------
    spectrum : complex ndarray
        Shape: (frequency_bins, time_frames)
    power : ndarray
        One-sided power spectral density.
    frequencies : ndarray
        Frequency corresponding to each bin.
    frames : ndarray
        Windowed signal frames before FFT.
    """

    signal = np.asarray(signal, dtype=np.float64).squeeze()

    if signal.ndim != 1:
        raise ValueError("Expected a mono, one-dimensional signal.")

    if signal.size == 0:
        raise ValueError("Signal cannot be empty.")

    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal contains NaN or infinite values.")

    if fs <= 0:
        raise ValueError("Sampling frequency must be positive.")

    win_length = round(WINDOW_LENGTH * fs)
    hop_length = round(HOP_LENGTH * fs)

    if N_FFT < win_length:
        raise ValueError("N_FFT must be >= window length.")

    # Center the frames using zero-padding.
    padded = np.pad(
        signal,
        (win_length // 2, win_length // 2)
    )

    frames = np.lib.stride_tricks.sliding_window_view(
        padded, win_length
    )[::hop_length]

    window = np.hanning(win_length)

    windowed_frames = frames * window[None, :]

    spectrum = np.fft.rfft(
        windowed_frames,
        n=N_FFT,
        axis=1
    ).T

    # One-sided power spectral density.
    power = np.abs(spectrum) ** 2
    power /= fs * np.sum(window ** 2)

    # Account for the omitted negative frequencies.
    if N_FFT % 2 == 0:
        power[1:-1] *= 2
    else:
        power[1:] *= 2

    frequencies = np.fft.rfftfreq(N_FFT, 1 / fs)

    return spectrum, power, frequencies, frames


# --------------------------------------------------
# Mel filterbank
# --------------------------------------------------

def _hz_to_mel(frequencies):
    mel = 2595.0 * np.log10(1.0 + frequencies / 700.0)

    return mel


def _mel_to_hz(mel):
    hz = 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    return hz


def _create_mel_filterbank(
    frequencies,
    n_mels=N_MELS,
    f_min=F_MIN,
    f_max=F_MAX
):
    """
    Create triangular mel filters.

    Returns
    -------
    filters : ndarray
        Shape: (n_mels, frequency_bins)
    """

    mel_min = _hz_to_mel(f_min)
    mel_max = _hz_to_mel(f_max)

    mel_points = np.linspace(
        mel_min, mel_max, n_mels + 2
    )

    hz_points = _mel_to_hz(mel_points)

    filters = np.zeros(
        (n_mels, len(frequencies)),
        dtype=np.float64
    )

    for i in range(n_mels):
        left = hz_points[i]
        center = hz_points[i + 1]
        right = hz_points[i + 2]

        rising = (
            (frequencies >= left) &
            (frequencies <= center)
        )

        falling = (
            (frequencies > center) &
            (frequencies <= right)
        )

        filters[i, rising] = (
            frequencies[rising] - left
        ) / (center - left)

        filters[i, falling] = (
            right - frequencies[falling]
        ) / (right - center)

    return filters


# --------------------------------------------------
# Spectrogram and MFCC
# --------------------------------------------------

def compute_log_stft(power, eps):
    '''
    Compute log-STFT

    Returns
    -------
    log_stft : ndarray
        Shape: (frequency_bins, time_frames)
    '''
    # Log power spectrogram.
    log_stft = 10.0 * np.log10(power + eps)

    return log_stft

def compute_log_mel(mel_filters, power, eps):
    '''
    Compute log-mel

    Returns
    -------
    log_mel : ndarray
        Shape: (n_mels, time_frames)
    '''
    # Mel-filtered power
    mel_power = mel_filters @ power

    log_mel = 10.0 * np.log10(mel_power + eps)

    return log_mel

def compute_mfcc(log_mel):
    '''
    Compute mfcc
    
    Returns
    -------
    mfcc : ndarray
        Shape: (n_mfcc, time_frames)
    '''
    # MFCC calculated from log-mel energies.
    mfcc = dct(
        log_mel,
        type=2,
        axis=0,
        norm="ortho"
    )[:N_MFCC]

    return mfcc

# --------------------------------------------------
# Broadband and frequency-band energy
# --------------------------------------------------

def compute_energy_features(
    power,
    frequencies,
    frequency_bands,
    frames,
    eps
):
    """
    Compute broadband and frequency-band energy statistics.

    Returns
    -------
    summary : ndarray
        One-dimensional feature vector.

    band_energy_db : ndarray
        Shape: (n_bands, time_frames)
    """

    freq_diff = frequencies[1] - frequencies[0]

    # Broadband power per frame.
    broadband_power = np.sum(power, axis=0) * freq_diff

    broadband_db = 10.0 * np.log10(
        broadband_power + eps
    )

    # Time-domain RMS per frame.
    frame_rms = np.sqrt(
        np.mean(frames ** 2, axis=1) + eps
    )

    log_rms = 20.0 * np.log10(frame_rms + eps)

    # Frequency-band power.
    band_energy_db = []

    for low, high in frequency_bands:
        mask = (
            (frequencies >= low) &
            (frequencies < high)
        )

        if not np.any(mask):
            raise ValueError(
                f"No FFT bins available for band {low}-{high} Hz."
            )

        band_power = np.sum(
            power[mask, :], axis=0
        ) * freq_diff

        band_db = 10.0 * np.log10(
            band_power + eps
        )

        band_energy_db.append(band_db)

    band_energy_db = np.stack(band_energy_db)

    # Summarize temporal variation.
    summary = [
        np.mean(broadband_db),
        np.std(broadband_db),
        np.mean(log_rms),
        np.std(log_rms),
    ]

    for band in band_energy_db:
        summary.extend([
            np.mean(band),
            np.std(band)
        ])

    return np.asarray(summary, dtype=np.float32), band_energy_db


# --------------------------------------------------
# Main feature extraction function
# --------------------------------------------------

def extract_vdp_features(signals, frequency_bands, fs):
    """
    Extract all features relevant to monaural
    VDP-based speaker orientation estimation.

    Parameters
    ----------
    signal : ndarray
        Monaural waveform.
    fs : int
        Sampling frequency.

    Returns
    -------
    features : dict
        Dictionary containing all feature representations.
    """

    eps = 1e-12

    spectrum, power, frequencies, frames = compute_stft(signals, fs)

    mel_filters = _create_mel_filterbank(frequencies)

    log_stft = compute_log_stft(power, eps)
    log_mel = compute_log_mel(mel_filters, power, eps)
    mfcc = compute_mfcc(log_mel)

    summary, band_energy_db = compute_energy_features(
        power, frequencies, frequency_bands, frames
    )

    return {
        "log_stft": log_stft[None, :, :].astype(np.float32),
        "log_mel": log_mel[None, :, :].astype(np.float32),
        "mfcc": mfcc.astype(np.float32),
        "summary": summary,
        "band_energy": band_energy_db.astype(np.float32),
        "frequencies": frequencies.astype(np.float32),
    }

def compute_features(signals, fs, feature_list, frequency_bands):
    '''
    As we introduce more features, we really need to do this better...
    There is a function called something like str2func, which would make this so much easier.
    '''
    eps = 1e-12

    all_features = []
    
    for signal in signals:
        computed_features = []
        log_mel = None

        spectrum, power, frequencies, frames = compute_stft(signal, fs)
        mel_filters = _create_mel_filterbank(frequencies)

        if 'log_stft' in feature_list:
            computed_features.append(
                compute_log_stft(power, eps)
            )
        if 'log_mel' in feature_list:
            log_mel = compute_log_mel(mel_filters, power, eps)
            computed_features.append(
                log_mel
            )
        if 'mfcc' in feature_list:
            if log_mel is None:
                log_mel = compute_log_mel(mel_filters, power, eps)
            computed_features.append(
                compute_mfcc(log_mel)
            )
        if 'summary' in feature_list or 'band_energy_db' in feature_list:
            summary, band_energy_db = compute_energy_features(power, frequencies, frequency_bands, frames, eps)
            if 'summary' in feature_list:
                computed_features.append(
                    summary
                )
            if 'band_energy_db' in feature_list:
                computed_features.append(
                    band_energy_db
                )
        if 'frequencies' in feature_list:
            computed_features.append(
                frequencies
            )

        # Sanity check
        if not computed_features:
            raise ValueError("No features were computed. Check feature_list.")

        # # Combine all features for this sample into one vector
        # sample_features = np.concatenate([
        #     np.asarray(feature, dtype=np.float32).reshape(-1)
        #     for feature in computed_features
        # ])
        
        sample_features = computed_features[0].astype(np.float32)

        all_features.append(sample_features)

    # return np.asarray(all_features, dtype=np.float32)
    return np.stack(all_features)
