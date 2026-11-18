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

def _get_pefac_voicing(left, right, fs):
    """
    Run PEFAC independently on both binaural channels and combine
    their voicing probabilities.

    Returns:
        times: PEFAC frame times
        pv: Combined frame-level probabilities
        sample_voicing_probability: Interpolated sample-level probabilities
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
        raise RuntimeError(
            "PEFAC returned inconsistent time/probability lengths."
        )

    if times_left.size == 0 or times_right.size == 0:
        raise ValueError("PEFAC did not return any frames.")

    if not np.allclose(times_left, times_right):
        raise RuntimeError("PEFAC frame times differ between channels.")

    if not np.all(np.isfinite(pv_left)) or not np.all(np.isfinite(pv_right)):
        raise ValueError("PEFAC returned non-finite voicing probabilities.")

    # Combine the two channel probabilities
    pv = np.maximum(pv_left, pv_right)

    # Interpolate frame-level probabilities to individual samples
    sample_times = np.arange(left.size, dtype=np.float64) / fs

    sample_voicing_probability = np.interp(
        sample_times,
        times_left,
        pv,
        left=0.0,
        right=0.0,
    )

    return times_left, pv, sample_voicing_probability


def _estimate_frequency_threshold(
    left,
    right,
    fs,
    pefac_times,
    pefac_probability,
    stft_n_fft=2048,
    stft_hop_length=512,
):
    """
    Estimate mean binaural power per frequency, weighted by PEFAC
    frame-level voicing probabilities.

    This function is retained for experimental analysis. It is not
    currently used in the main feature extraction pipeline.
    """
    if stft_n_fft > left.size:
        raise ValueError(
            "stft_n_fft cannot exceed the one-second signal length."
        )

    if not 0 < stft_hop_length <= stft_n_fft:
        raise ValueError(
            "stft_hop_length must be in [1, stft_n_fft]."
        )

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

    # Interpolate PEFAC probabilities onto the STFT frame times.
    frame_voicing_probability = np.interp(
        times,
        pefac_times,
        pefac_probability,
        left=0.0,
        right=0.0,
    )

    if np.sum(frame_voicing_probability) <= 0:
        raise ValueError(
            "No voiced STFT frames available for frequency estimation."
        )

    # Calculate mean binaural power for each frequency bin.
    binaural_power = (
        np.abs(left_stft) ** 2 +
        np.abs(right_stft) ** 2
    ) / 2.0

    # Probability-weighted average over time.
    mean_power_per_frequency = np.average(
        binaural_power,
        axis=1,
        weights=frame_voicing_probability,
    )

    return frequencies, mean_power_per_frequency



def _extract_binaural_features(
    left_spectrum,
    right_spectrum,
    frequencies,
    cutoff_hz=952.0,
    eps=0.1,
    max_inverse=10.0,
    valid_frequency_mask=None,
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
    if valid_frequency_mask is None:
        valid_frequency_mask = np.ones_like(frequencies, dtype=bool)

    # Reciprocal of ILD_low with numerical stabilization
    denominator = np.where(
        np.abs(ild_low) < eps,
        np.where(ild_low < 0.0, -eps, eps),
        ild_low,
    )

    inverse_ild_low = np.where(
        low_mask & valid_frequency_mask,
        1.0 / denominator,
        0.0,
    )

    inverse_ild_low = np.clip(
        inverse_ild_low,
        -max_inverse,
        max_inverse,
    )

    inverse_convolution = convolve(
        ild_high,
        inverse_ild_low,
        mode="same",
        method="auto",
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
    eps=0.1,
    max_inverse=10.0,
    return_diagnostics=False,
):
    """
    Preprocess one binaural recording and extract five frequency-domain
    feature channels.

    Returns:
        features: ndarray, shape (5, fs // 2 + 1)
        diagnostics: dict, optional
    """
    signal = np.asarray(signal)

    if signal.ndim != 2 or signal.shape[0] != 2:
        raise ValueError(
            "signal must have shape (2, samples): left and right."
        )

    left = _ensure_one_second(signal[0], fs)
    right = _ensure_one_second(signal[1], fs)

    # 1. Estimate shared binaural voicing probabilities.
    pefac_times, pefac_probability, sample_voicing_probability = (
        _get_pefac_voicing(left, right, fs)
    )

    # 2. Apply continuous probability weighting.
    left_weighted = left * sample_voicing_probability
    right_weighted = right * sample_voicing_probability

    # 3. Calculate one-sided FFTs.
    left_spectrum = np.fft.rfft(
        left_weighted, norm="forward"
    )
    right_spectrum = np.fft.rfft(
        right_weighted, norm="forward"
    )

    frequencies = np.fft.rfftfreq(
        left_weighted.size,
        d=1.0 / fs,
    )

    # 4. Extract the five features.
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
            "sample_voicing_probability": sample_voicing_probability,
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

if __name__ == '__main__':
    fs = 48000
    t = np.arange(fs) / fs

    signal = np.sin(2 * np.pi * 500 * t)

    original_left = signal
    original_right = signal

    original_spectrum_left = np.fft.rfft(original_left, norm="forward")
    original_spectrum_right = np.fft.rfft(original_right, norm="forward")

    frequencies = np.fft.rfftfreq(fs, 1/fs)

    original_features = _extract_binaural_features(
        original_spectrum_left,
        original_spectrum_left,
        frequencies
    )

    print('test 1:')
    print(original_features.shape)
    print('feature   |   value')
    print(f'ild_low  |   {original_features[0]}')
    print(f'ild_high |   {original_features[1]}')
    print(f'itd_low  |   {original_features[2]}')
    print(f'itd_high |   {original_features[3]}')
    print(f'inv_conv |   {original_features[4]}')

    '''output
    (5, 24001)
    feature   |   value
    ild_low  |   [0. 0. 0. ... 0. 0. 0.]
    ild_high |   [0. 0. 0. ... 0. 0. 0.]
    itd_low  |   [ 0.0000000e+00 -3.3342501e-18  1.4784865e-19 ...  0.0000000e+00
    0.0000000e+00  0.0000000e+00]
    itd_high |   [ 0.0000000e+00  0.0000000e+00  0.0000000e+00 ... -2.5498083e-24
    2.3672353e-22  0.0000000e+00]
    inv_conv |   [0. 0. 0. ... 0. 0. 0.]
    '''

    left = signal
    right = 0.5 * signal

    spectrum_left = np.fft.rfft(left, norm="forward")
    spectrum_right = np.fft.rfft(right, norm="forward")

    frequencies = np.fft.rfftfreq(fs, 1/fs)

    features = _extract_binaural_features(
        spectrum_left,
        spectrum_right,
        frequencies
    )

    print('test 2:')
    print(features.shape)
    print('feature   |   value')
    print(f'ild_low  |   {features[0]}')
    print(f'ild_high |   {features[1]}')
    print(f'itd_low  |   {features[2]}')
    print(f'itd_high |   {features[3]}')
    print(f'inv_conv |   {features[4]}')

    '''output
    test 2:
    (5, 24001)
    feature   |   value
    ild_low  |   [7.1197866e-07 1.6813609e-06 7.9587380e-07 ... 0.0000000e+00 0.0000000e+00
    0.0000000e+00]
    ild_high |   [0.0000000e+00 0.0000000e+00 0.0000000e+00 ... 3.2259011e-06 1.8535341e-06
    4.4053822e-06]
    itd_low  |   [ 0.0000000e+00 -3.3342501e-18  1.4784865e-19 ...  0.0000000e+00
    0.0000000e+00  0.0000000e+00]
    itd_high |   [ 0.0000000e+00  0.0000000e+00  0.0000000e+00 ... -2.5498083e-24
    2.3672353e-22  0.0000000e+00]
    inv_conv |   [ 5.9727699e-02  5.9245534e-02  5.9797738e-02 ... -9.3569414e-18
    1.5205031e-17 -1.6374647e-17]
    '''

    delay_samples = 3

    left = signal
    right = np.concatenate([
        np.zeros(delay_samples),
        signal[:-delay_samples]
    ])

    spectrum_left = np.fft.rfft(left, norm="forward")
    spectrum_right = np.fft.rfft(right, norm="forward")

    frequencies = np.fft.rfftfreq(fs, 1/fs)

    features = _extract_binaural_features(
        spectrum_left,
        spectrum_right,
        frequencies
    )

    print('test 3:')
    print(features.shape)
    print('feature   |   value')
    print(f'ild_low  |   {features[0]}')
    print(f'ild_high |   {features[1]}')
    print(f'itd_low  |   {features[2]}')
    print(f'itd_high |   {features[3]}')
    print(f'inv_conv |   {features[4]}')

    features_original = _extract_binaural_features(
        original_spectrum_left,
        original_spectrum_right,
        frequencies
    )

    features_swapped = _extract_binaural_features(
        original_spectrum_right,
        original_spectrum_left,
        frequencies
    )

    print('test 4:')
    print(features_swapped.shape)
    print('feature   |   value')
    print(f'ild_low  |   {features[0]}')
    print(f'ild_high |   {features[1]}')
    print(f'itd_low  |   {features[2]}')
    print(f'itd_high |   {features[3]}')
    print(f'inv_conv |   {features[4]}')

    left = 10 * signal
    right = 10 * signal

    spectrum_left = np.fft.rfft(left, norm="forward")
    spectrum_right = np.fft.rfft(right, norm="forward")

    frequencies = np.fft.rfftfreq(fs, 1/fs)

    features = _extract_binaural_features(
        spectrum_left,
        spectrum_right,
        frequencies
    )

    print('test 5:')
    print(features_swapped.shape)
    print('feature   |   value')
    print(f'ild_low  |   {features[0]}')
    print(f'ild_high |   {features[1]}')
    print(f'itd_low  |   {features[2]}')
    print(f'itd_high |   {features[3]}')
    print(f'inv_conv |   {features[4]}')