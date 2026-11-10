%% Inspect Pörschmann VDP SOFA file
% Inspects metadata, source positions, impulse responses,
% sampling rate, and directional coverage.

clear;
clc;
close all;

%% 1. Select SOFA file

pathname = 'D:\Python Projects\P9Datasets\PörschmannVDP_database\SOFA_Phonemes';
filename = 'Directivity_1_1.sofa';

sofaFilePath = fullfile(pathname, filename);

fprintf('\n========================================\n');
fprintf('INSPECTING VDP SOFA FILE\n');
fprintf('========================================\n');
fprintf('%s\n', sofaFilePath);

%% 2. Read SOFA metadata

info = ncinfo(sofaFilePath);

%% 3. Global attributes

fprintf('\n========== GLOBAL ATTRIBUTES ==========\n');

for k = 1:numel(info.Attributes)

    name = info.Attributes(k).Name;
    value = info.Attributes(k).Value;

    fprintf('%s: %s\n', name, string(value));

end

%% 4. Variables and dimensions

fprintf('\n========== VARIABLES ==========\n');

varNames = {info.Variables.Name};

for k = 1:numel(info.Variables)

    var = info.Variables(k);

    fprintf('\nVariable: %s\n', var.Name);

    fprintf('Dimensions: ');

    for d = 1:numel(var.Dimensions)

        fprintf('%s=%d ', ...
            var.Dimensions(d).Name, ...
            var.Dimensions(d).Length);

    end

    fprintf('\n');

    for a = 1:numel(var.Attributes)

        fprintf('  %s: %s\n', ...
            var.Attributes(a).Name, ...
            string(var.Attributes(a).Value));

    end

end

%% 5. Read important variables

fprintf('\n========== IMPORTANT VARIABLES ==========\n');

% Data.IR
irIdx = find(strcmpi(varNames, 'Data.IR'), 1);

if ~isempty(irIdx)

    ir = ncread(sofaFilePath, ...
        info.Variables(irIdx).Name);

    fprintf('\nData.IR size:\n');
    disp(size(ir));

    fprintf('Number of IR elements: %d\n', numel(ir));

else

    ir = [];
    warning('Data.IR not found.');

end

% Sampling rate
fsIdx = find(strcmpi(varNames, 'Data.SamplingRate'), 1);

if ~isempty(fsIdx)

    fs = double(ncread(sofaFilePath, ...
        info.Variables(fsIdx).Name));

    fprintf('\nSampling rate:\n');
    disp(fs);

else

    fs = [];
    warning('Data.SamplingRate not found.');

end

% SourcePosition
sourceIdx = find(strcmpi(varNames, 'SourcePosition'), 1);

if ~isempty(sourceIdx)

    sourcePosition = ncread(sofaFilePath, ...
        info.Variables(sourceIdx).Name);

    fprintf('\nSourcePosition size:\n');
    disp(size(sourcePosition));

    fprintf('First 10 entries:\n');
    disp(sourcePosition(:, 1:min(10, size(sourcePosition, 2))));

else

    sourcePosition = [];
    fprintf('\nSourcePosition not found.\n');

end

% ReceiverPosition
receiverIdx = find(strcmpi(varNames, 'ReceiverPosition'), 1);

if ~isempty(receiverIdx)

    receiverPosition = ncread(sofaFilePath, ...
        info.Variables(receiverIdx).Name);

    fprintf('\nReceiverPosition size:\n');
    disp(size(receiverPosition));

    disp(receiverPosition);

else

    receiverPosition = [];
    fprintf('\nReceiverPosition not found.\n');

end

% ListenerPosition
listenerIdx = find(strcmpi(varNames, 'ListenerPosition'), 1);

if ~isempty(listenerIdx)

    listenerPosition = ncread(sofaFilePath, ...
        info.Variables(listenerIdx).Name);

    fprintf('\nListenerPosition size:\n');
    disp(size(listenerPosition));

    disp(listenerPosition);

else

    listenerPosition = [];
    fprintf('\nListenerPosition not found.\n');

end

% Data.Delay
delayIdx = find(strcmpi(varNames, 'Data.Delay'), 1);

if ~isempty(delayIdx)

    delay = ncread(sofaFilePath, ...
        info.Variables(delayIdx).Name);

    fprintf('\nData.Delay size:\n');
    disp(size(delay));

    fprintf('Data.Delay values:\n');
    disp(delay);

else

    delay = [];
    fprintf('\nData.Delay not found.\n');

end

%% 6. Inspect coordinate information

fprintf('\n========== SOURCE POSITION ANALYSIS ==========\n');

if ~isempty(sourcePosition)

    % Interpret coordinate information using the attributes
    % printed above. SOFA commonly uses spherical coordinates
    % for SourcePosition, but this must be verified.

    fprintf('Unique values per coordinate row:\n');

    for k = 1:size(sourcePosition, 1)

        fprintf('Coordinate %d:\n', k);

        disp(unique(sourcePosition(k, :)).');

    end

    fprintf('Number of source positions: %d\n', ...
        size(sourcePosition, 2));

end

%% 7. Inspect impulse response statistics

fprintf('\n========== IR STATISTICS ==========\n');

if ~isempty(ir)

    fprintf('Minimum amplitude: %.6f\n', min(ir(:)));
    fprintf('Maximum amplitude: %.6f\n', max(ir(:)));
    fprintf('Mean amplitude: %.6f\n', mean(ir(:)));
    fprintf('RMS amplitude: %.6f\n', rms(ir(:)));

    fprintf('Number of zero elements: %d\n', ...
        sum(ir(:) == 0));

    fprintf('Number of non-zero elements: %d\n', ...
        sum(ir(:) ~= 0));

end

%% 8. Inspect SOFA conventions and dimensions

fprintf('\n========== SOFA CONVENTION ==========\n');

for k = 1:numel(info.Attributes)

    if contains(info.Attributes(k).Name, ...
            'Conventions', 'IgnoreCase', true)

        fprintf('%s: %s\n', ...
            info.Attributes(k).Name, ...
            string(info.Attributes(k).Value));

    end

end

fprintf('\n========== IR DIMENSION INTERPRETATION ==========\n');

if ~isempty(irIdx)

    var = info.Variables(irIdx);

    for k = 1:numel(var.Dimensions)

        fprintf('Dimension %d: %s = %d\n', ...
            k, ...
            var.Dimensions(k).Name, ...
            var.Dimensions(k).Length);

    end

end

%% 9. Plot impulse responses

if ~isempty(ir) && ~isempty(fs)

    % Initially select the first measurement.
    % We will adapt this once we know the dimension order.

    fprintf('\n========== PLOTTING ==========\n');

    figure;

    plot(squeeze(ir(:, 1, 1)));

    title('Example VDP Impulse Response');
    xlabel('Sample');
    ylabel('Amplitude');
    grid on;

end

fprintf('\n========================================\n');
fprintf('INSPECTION COMPLETE\n');
fprintf('========================================\n');

fprintf('\nVIEWING VDP SOFA FILE USING NDISP()\n')
ncdisp(sofaFilePath);


%% 10. Inspect frequency-domain transfer functions

fprintf('\n========== FREQUENCY DOMAIN ==========\n');

freq = double(ncread(sofaFilePath, 'N'));
freq = freq(:);

fprintf('Number of frequency bins: %d\n', numel(freq));
fprintf('Minimum frequency: %.2f Hz\n', min(freq));
fprintf('Maximum frequency: %.2f Hz\n', max(freq));

frequencySpacing = diff(freq);

fprintf('First frequency bins:\n');
disp(freq(1:min(10, numel(freq))).');

fprintf('Frequency spacing:\n');
fprintf('Minimum: %.6f Hz\n', min(frequencySpacing));
fprintf('Maximum: %.6f Hz\n', max(frequencySpacing));

fprintf('Uniform frequency spacing: %d\n', ...
    all(abs(frequencySpacing - frequencySpacing(1)) < 1e-6));

if abs(freq(1)) < 1e-6
    fprintf('Spectrum starts at DC.\n');
else
    fprintf('Spectrum does not start at DC.\n');
end

% Read complex transfer functions
realPart = double(ncread(sofaFilePath, 'Data.Real'));
imagPart = double(ncread(sofaFilePath, 'Data.Imag'));

H = realPart + 1i * imagPart;

fprintf('\nComplex transfer function size:\n');
disp(size(H));

% Magnitude and phase
magnitude = abs(H);
phase = angle(H);

fprintf('Magnitude range: %.6f to %.6f\n', ...
    min(magnitude(:)), max(magnitude(:)));

fprintf('Mean magnitude: %.6f\n', mean(magnitude(:)));

%% 11. Inspect emitter position

fprintf('\n========== EMITTER POSITION ==========\n');

emitterIdx = find(strcmpi(varNames, 'EmitterPosition'), 1);

if ~isempty(emitterIdx)

    emitterPosition = ncread(sofaFilePath, ...
        info.Variables(emitterIdx).Name);

    disp(emitterPosition);

else

    fprintf('EmitterPosition not found.\n');

end

%% 12. Inspect microphone angular distribution

fprintf('\n========== MICROPHONE DIRECTIONS ==========\n');

if ~isempty(receiverPosition)

    receiverPosition = squeeze(receiverPosition);

    azimuth = receiverPosition(1, :).';
    elevation = receiverPosition(2, :).';
    radius = receiverPosition(3, :).';

    fprintf('Number of microphones: %d\n', numel(azimuth));

    fprintf('Unique azimuth values:\n');
    disp(unique(round(azimuth, 3)).');

    fprintf('Unique elevation values:\n');
    disp(unique(round(elevation, 3)).');

    fprintf('Microphone radii (m):\n');
    disp(radius.');

    % Convert spherical coordinates to Cartesian coordinates.
    azRad = deg2rad(azimuth);
    elRad = deg2rad(elevation);

    x = radius .* cos(elRad) .* cos(azRad);
    y = radius .* cos(elRad) .* sin(azRad);
    z = radius .* sin(elRad);

    figure;

    scatter3(x, y, z, 80, elevation, 'filled');

    axis equal;
    grid on;

    xlabel('X (m)');
    ylabel('Y (m)');
    zlabel('Z (m)');

    title('VDP Microphone Distribution');
    colorbar;
    view(3);

end

%% 13. Plot magnitude and phase for one phoneme

% Select a phoneme and microphone.
phonemeIdx = 1;
receiverIdx = 1;

H_selected = squeeze(H(:, receiverIdx, phonemeIdx));

figure;

subplot(2, 1, 1);

plot(freq, 20*log10(abs(H_selected) + eps));

grid on;
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');

title(sprintf('VDP Magnitude - Measurement %d, Microphone %d', ...
    phonemeIdx, receiverIdx));

subplot(2, 1, 2);

plot(freq, unwrap(angle(H_selected)));

grid on;
xlabel('Frequency (Hz)');
ylabel('Phase (rad)');

title('VDP Phase Response');

%% 14. Compare directional magnitude patterns

% Select the phoneme and frequency closest to 1000 Hz.

targetFrequency = 1000;

[~, frequencyIdx] = min(abs(freq - targetFrequency));

phonemeIdx = 1;

directionalMagnitude = squeeze( ...
    magnitude(frequencyIdx, :, phonemeIdx));

directionalMagnitudeDB = 20*log10( ...
    directionalMagnitude + eps);

fprintf('\n========== DIRECTIONAL MAGNITUDE ==========\n');

fprintf('Selected frequency: %.2f Hz\n', freq(frequencyIdx));

fprintf('Magnitude range across microphones: %.2f dB\n', ...
    max(directionalMagnitudeDB) - min(directionalMagnitudeDB));

figure;

scatter3(x, y, z, 100, directionalMagnitudeDB, 'filled');

axis equal;
grid on;

xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');

title(sprintf('VDP Directivity Magnitude at %.0f Hz (Measurement %d)', ...
    freq(frequencyIdx), phonemeIdx));

colorbar;
view(3);

%% 15. Compare phoneme responses

% Compare average magnitude across all microphones.
% This is NOT yet a directivity comparison, but gives an
% initial impression of differences between phonemes.

meanMagnitude = squeeze(mean(magnitude, 2));

figure;

plot(freq, 20*log10(meanMagnitude + eps));

grid on;
xlabel('Frequency (Hz)');
ylabel('Mean Magnitude (dB)');

title('Mean Transfer Function Magnitude Across Phonemes');

legend(compose('Measurement %d', 1:size(H, 3)), ...
    'Location', 'bestoutside');

fprintf('\nAdditional inspection complete.\n');

%% 16. Reconstruct VDP impulse responses

fs = 48000;
nfft = 1024;

% Reconstruct negative-frequency spectrum.
% DC and Nyquist must not be duplicated.

Hfull = cat(1, ...
    H, ...
    conj(H(end-1:-1:2, :, :)));

fprintf('Full spectrum size:\n');
disp(size(Hfull));

% Check DC and Nyquist imaginary components.
fprintf('Maximum imaginary DC component: %.3e\n', ...
    max(abs(imag(H(1, :, :))), [], 'all'));

fprintf('Maximum imaginary Nyquist component: %.3e\n', ...
    max(abs(imag(H(end, :, :))), [], 'all'));

% Inverse FFT along frequency dimension.
hComplex = ifft(Hfull, nfft, 1);

fprintf('Maximum imaginary IFFT component: %.3e\n', ...
    max(abs(imag(hComplex(:)))));

hVDP = real(hComplex);

fprintf('Impulse response dimensions:\n');
disp(size(hVDP));

%% 17. Plot selected impulse responses

receiverIndices = [1 5 10 15];
phonemeIdx = 1;

t = (0:nfft-1)/fs * 1000;

figure;

for k = 1:numel(receiverIndices)

    r = receiverIndices(k);

    subplot(numel(receiverIndices), 1, k);

    plot(t, hVDP(:, r, phonemeIdx));

    grid on;
    xlabel('Time (ms)');
    ylabel('Amplitude');

    title(sprintf('Microphone %d', r));

end

sgtitle('Reconstructed VDP Impulse Responses');


%% 18. Verify frequency-domain reconstruction

H_reconstructed = fft(hVDP, 1024, 1);

% Compare against original frequency-domain data.
H_original = H;

H_check = H_reconstructed(1:513, :, :);

absoluteError = abs(H_original - H_check);

relativeError = absoluteError ./ (abs(H_original) + eps);

fprintf('\n========== FFT RECONSTRUCTION CHECK ==========\n');

fprintf('Maximum absolute error: %.6e\n', ...
    max(absoluteError(:)));

fprintf('Mean absolute error: %.6e\n', ...
    mean(absoluteError(:)));

fprintf('Maximum relative error: %.6e\n', ...
    max(relativeError(:)));

fprintf('Mean relative error: %.6e\n', ...
    mean(relativeError(:)));
