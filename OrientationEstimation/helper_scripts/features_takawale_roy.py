
import numpy as np
from scipy.signal import stft, convolve


def extract_binaural_features(
    left,
    right,
    fs,
    n_fft=2048,
    hop_length=512,
    cutoff_hz=952.0,
    eps=0.1,
    max_inverse=10.0,
):
    """
    Extract the five binaural features used by Takawale & Roy.

    Parameters
    ----------
    left, right : ndarray
        Left and right microphone signals, shape (samples,).
    fs : int
        Sampling frequency.
    n_fft : int
        STFT window length.
    hop_length : int
        STFT hop length.
    cutoff_hz : float
        Low/high frequency separation.
    eps : float
        Minimum absolute ILD value before inversion, in dB.
    max_inverse : float
        Maximum absolute reciprocal value.

    Returns
    -------
    features : ndarray
        Shape (5, frequency_bins, time_frames).
        Channels: ILD_low, ILD_high, ITD_low,
                  ITD_high, inverse_convolution.
    """

    # 1. Calculate STFT
    frequencies, times, Y_left = stft(
        left,
        fs=fs,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary=None,
        padded=False,
    )

    _, _, Y_right = stft(
        right,
        fs=fs,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary=None,
        padded=False,
    )

    # 2. Frequency masks
    low_mask = frequencies <= cutoff_hz
    high_mask = frequencies > cutoff_hz

    # 3. ILD calculation (dB)
    magnitude_left = np.abs(Y_left)
    magnitude_right = np.abs(Y_right)

    ild = 20 * np.log10(
        (magnitude_left + 1e-10) /
        (magnitude_right + 1e-10)
    )

    ild_low = np.where(low_mask[:, None], ild, 0.0)
    ild_high = np.where(high_mask[:, None], ild, 0.0)

    # 4. ITD calculation (seconds)
    phase_difference = np.angle(Y_left / (Y_right + 1e-15))

    itd = np.zeros_like(phase_difference)

    nonzero_freq = frequencies > 0

    itd[nonzero_freq, :] = (
        phase_difference[nonzero_freq, :]
        / (2 * np.pi * frequencies[nonzero_freq, None])
    )

    itd_low = np.where(low_mask[:, None], itd, 0.0)
    itd_high = np.where(high_mask[:, None], itd, 0.0)

    # 5. Inverse convolution
    # Reciprocal of ILD_low with numerical stabilization
    denominator = np.where(
        np.abs(ild_low) < eps,
        np.where(ild_low < 0, -eps, eps),
        ild_low,
    )

    inverse_ild_low = np.where(
        low_mask[:, None],
        1.0 / denominator,
        0.0,
    )

    inverse_ild_low = np.clip(
        inverse_ild_low,
        -max_inverse,
        max_inverse,
    )

    inverse_conv = np.zeros_like(ild_high)

    # Convolve along frequency, independently for each time frame
    for t in range(ild_high.shape[1]):
        inverse_conv[:, t] = convolve(
            ild_high[:, t],
            inverse_ild_low[:, t],
            mode="same",
        )

    # 6. Stack the five feature channels
    features = np.stack(
        [
            ild_low,
            ild_high,
            itd_low,
            itd_high,
            inverse_conv,
        ],
        axis=0,
    )

    return features, frequencies, times
