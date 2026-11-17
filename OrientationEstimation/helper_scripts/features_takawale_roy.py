import numpy as np
from scipy.signal import stft, convolve
from pyvoicebox import v_fxpefac

def _ensure_one_second(signal, fs):
    """Trim or zero-pad a 1-D signal to exactly one second."""
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    target_length = int(fs)

    if target_length <= 0:
        raise ValueError("fs must be a positive sampling frequency.")

    if signal.size > target_length:
        return signal[:target_length]
    
    if signal.size < target_length:
        return np.pad(signal, (0, target_length - signal.size))
    
    return signal

def _get_pefac_voicing(left, right, fs, voicing_threshold):
    """
    Run PEFAC independently on both binaural channels and combine the
    probabilities with max(). Return PEFAC times, combined probabilities,
    and a sample-level shared voiced mask.
    """
    _, times_left, pv_left = v_fxpefac(left, fs)
    _, times_right, pv_right = v_fxpefac(right, fs)

    times_left = np.asarray(times_left, dtype=np.float64).reshape(-1)
    times_right = np.asarray(times_right, dtype=np.float64).reshape(-1)
    pv_left = np.asarray(pv_left, dtype=np.float64).reshape(-1)
    pv_right = np.asarray(pv_right, dtype=np.float64).reshape(-1)

    if not (
        times_left.size == pv_left.size
        and times_right.size == pv_right.size
    ):
        raise RuntimeError("PEFAC returned inconsistent time/probability lengths.")

    if times_left.size == 0 or times_right.size == 0:
        raise ValueError("PEFAC did not return any frames.")

    if not np.allclose(times_left, times_right):
        raise RuntimeError("PEFAC frame times differ between channels.")

    if not np.all(np.isfinite(pv_left)) or not np.all(np.isfinite(pv_right)):
        raise ValueError("PEFAC returned non-finite voicing probabilities.")

    pv = np.maximum(pv_left, pv_right)
    sample_times = np.arange(left.size, dtype=np.float64) / fs

    # Interpolate the frame-level PEFAC probabilities to individual samples.
    # Outside the PEFAC time range, samples are considered unvoiced.
    sample_voicing_probability = np.interp(
        sample_times, times_left, pv, left=0.0, right=0.0
    )
    voiced_sample_mask = sample_voicing_probability >= voicing_threshold

    if not np.any(voiced_sample_mask):
        raise ValueError("PEFAC did not detect any voiced samples.")

    return times_left, pv, voiced_sample_mask

def _estimate_frequency_threshold(
    left,
    right,
    fs,
    pefac_times,
    pefac_probability,
    stft_n_fft=2048,
    stft_hop_length=512,
    voicing_threshold=0.5,
):
    """
    Estimate a frequency-dependent power threshold from voiced STFT frames.

    For each frequency bin, calculate mean binaural power over voiced frames.
    The threshold is later interpolated onto the full-recording FFT grid.
    """
    if stft_n_fft > left.size:
        raise ValueError("stft_n_fft cannot exceed the one-second signal length.")
    if not 0 < stft_hop_length <= stft_n_fft:
        raise ValueError("stft_hop_length must be in [1, stft_n_fft].")

    frequencies, times, left_stft = stft(
        left,
        fs=fs,
        window="boxcar",
        nperseg=stft_n_fft,
        noverlap=stft_n_fft - stft_hop_length,
        boundary=None,
        padded=False,
        scaling="spectrum",
    )
    _, _, right_stft = stft(
        right,
        fs=fs,
        window="boxcar",
        nperseg=stft_n_fft,
        noverlap=stft_n_fft - stft_hop_length,
        boundary=None,
        padded=False,
        scaling="spectrum",
    )

    frame_voicing_probability = np.interp(
        times,
        pefac_times,
        pefac_probability,
        left=0.0,
        right=0.0,
    )
    voiced_frame_mask = frame_voicing_probability >= voicing_threshold

    if not np.any(voiced_frame_mask):
        raise ValueError("No voiced STFT frames available for frequency suppression.")

    binaural_power = (
        np.abs(left_stft) ** 2 + np.abs(right_stft) ** 2
    ) / 2.0

    mean_power_per_frequency = np.mean(
        binaural_power[:, voiced_frame_mask], axis=1
    )

    return frequencies, mean_power_per_frequency, voiced_frame_mask


def _extract_binaural_features(
    left_spectrum,
    right_spectrum,
    frequencies,
    cutoff_hz=952.0,
    eps=0.1,
    max_inverse=10.0,
):
    """
    Extract the five feature channels from one pair of full-recording spectra.

    Returns an array with shape (5, frequency_bins), ready for nn.Conv1d.
    """
    # frequency masks
    low_mask = frequencies <= cutoff_hz
    high_mask = frequencies > cutoff_hz

    # ILD calculation (dB)
    magnitude_left = np.abs(left_spectrum)
    magnitude_right = np.abs(right_spectrum)

    ild = 20.0 * np.log10(
        (magnitude_left + 1e-10) / 
        (magnitude_right + 1e-10)
    )
    ild_low = np.where(low_mask, ild, 0.0)
    ild_high = np.where(high_mask, ild, 0.0)

    # ITD Calculation (seconds)
    phase_difference = np.angle(
        left_spectrum * np.conj(right_spectrum)
    )
    itd = np.zeros_like(phase_difference, dtype=np.float64)

    nonzero_frequency = frequencies > 0
    itd[nonzero_frequency] = (
        phase_difference[nonzero_frequency]
        / (2.0 * np.pi * frequencies[nonzero_frequency])
    )
    itd_low = np.where(low_mask, itd, 0.0)
    itd_high = np.where(high_mask, itd, 0.0)

    # inverse convolution
    # Reciprocal of ILD_low with numerical stabilization
    denominator = np.where(
        np.abs(ild_low) < eps,
        np.where(ild_low < 0.0, -eps, eps),
        ild_low,
    )
    inverse_ild_low = np.where(
        low_mask, 
        1.0 / denominator, 
        0.0
    )
    inverse_ild_low = np.clip(
        inverse_ild_low, 
        -max_inverse, 
        max_inverse
    )

    inverse_convolution = convolve(
        ild_high, 
        inverse_ild_low, 
        mode="same", method="auto"
    )

    features = np.stack(
        [ild_low, ild_high, itd_low, itd_high, inverse_convolution],
        axis=0,
    )

    return features.astype(np.float32, copy=False)


def extract_takawale_roy_features(
    signal,
    fs,
    cutoff_hz=952.0,
    voicing_threshold=0.5,
    stft_n_fft=2048,
    stft_hop_length=512,
    eps=0.1,
    max_inverse=10.0,
    return_diagnostics=False,
):
    """
    Preprocess one binaural recording and extract its five frequency-domain
    feature channels.

    Parameters
    ----------
    signal : ndarray, shape (2, samples)
        Binaural recording. Row 0 is left; row 1 is right.
    fs : int
        Sampling frequency in Hz. The recording is trimmed/padded to 1 second.
    cutoff_hz : float
        Low/high frequency separation used for ILD and ITD.
    voicing_threshold : float
        PEFAC probability threshold.
    stft_n_fft, stft_hop_length : int
        STFT settings used only to estimate the frequency-dependent suppression
        threshold. The final model features come from one FFT over the
        one-second recording.
    eps, max_inverse : float
        Numerical stabilization parameters for the reciprocal ILD operation.
    return_diagnostics : bool
        If True, also return preprocessing masks and frequency axes.

    Returns
    -------
    features : ndarray, shape (5, fs // 2 + 1)
        Five feature channels in Conv1d's per-recording layout:
        [ILD_low, ILD_high, ITD_low, ITD_high, inverse_convolution].
    diagnostics : dict, optional
        Additional information useful for inspecting preprocessing.
    """
    signal = np.asarray(signal)

    if signal.ndim != 2 or signal.shape[0] != 2:
        raise ValueError("signal must have shape (2, samples): left and right.")

    left = _ensure_one_second(signal[0], fs)
    right = _ensure_one_second(signal[1], fs)

    # 1. PEFAC on the existing binaural channels.
    pefac_times, pefac_probability, voiced_sample_mask = _get_pefac_voicing(
        left, right, fs, voicing_threshold
    )

    # 2. Estimate a frequency-dependent threshold using voiced STFT frames.
    threshold_frequencies, mean_power, voiced_frame_mask = (
        _estimate_frequency_threshold(
            left,
            right,
            fs,
            pefac_times,
            pefac_probability,
            stft_n_fft=stft_n_fft,
            stft_hop_length=stft_hop_length,
            voicing_threshold=voicing_threshold,
        )
    )

    # 3. Apply the shared PEFAC sample mask to both channels.
    left_voiced = left * voiced_sample_mask
    right_voiced = right * voiced_sample_mask

    # 4. One full-recording, one-sided FFT per channel.
    # norm="forward" scales by 1/N. The same scale is used for both channels,
    # so ILD ratios and phase differences are unaffected by this normalization.
    left_spectrum = np.fft.rfft(left_voiced, norm="forward")
    right_spectrum = np.fft.rfft(right_voiced, norm="forward")
    frequencies = np.fft.rfftfreq(left_voiced.size, d=1.0 / fs)

    # 5. Interpolate the STFT-derived mean-power threshold to the full FFT grid.
    # The same frequency mask is applied to both ears to preserve binaural
    # relationships.
    mean_power_fft = np.interp(
        frequencies, threshold_frequencies, mean_power
    )
    binaural_fft_power = (
        np.abs(left_spectrum) ** 2 + np.abs(right_spectrum) ** 2
    ) / 2.0
    frequency_mask = binaural_fft_power >= mean_power_fft

    left_spectrum = np.where(frequency_mask, left_spectrum, 0.0)
    right_spectrum = np.where(frequency_mask, right_spectrum, 0.0)

    # 6. Calculate the five frequency-domain feature channels.
    features = _extract_binaural_features(
        left_spectrum,
        right_spectrum,
        frequencies,
        cutoff_hz=cutoff_hz,
        eps=eps,
        max_inverse=max_inverse,
    )

    if return_diagnostics:
        diagnostics = {
            "frequencies": frequencies,
            "pefac_times": pefac_times,
            "pefac_probability": pefac_probability,
            "voiced_sample_mask": voiced_sample_mask,
            "voiced_frame_mask": voiced_frame_mask,
            "frequency_mask": frequency_mask,
            "mean_power_threshold": mean_power_fft,
        }
        return features, diagnostics

    return features

## example for how to use
'''
batch_features = np.stack([
    extract_takawale_roy_features(signal, fs=48000)
    for signal in batch_signals
])

batch_features = torch.from_numpy(batch_features)

predictions = model(batch_features)
'''

# suggested first check
'''
features, diagnostics = extract_takawale_roy_features(
    signal,
    fs=48000,
    return_diagnostics=True
)

print(features.shape)
print(features.dtype)

print("Voiced samples:", np.mean(diagnostics["voiced_sample_mask"]))
print("Retained frequency bins:", np.mean(diagnostics["frequency_mask"]))
'''