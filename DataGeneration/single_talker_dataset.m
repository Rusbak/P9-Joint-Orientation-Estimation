%% Single VDP Monaural Dataset Generator

% Signal generation:
%   1. Load a singular VDP. (Measured from phonetically balanced sentence)
%   2. Rotate the receiver pair according to talker orientation.
%   3. Generate clean source signals (chirps).
%   4. Apply free-field propagation delay and spherical spreading.
%   5. Save monaural signals and labels to HDF5.

clear;
clc;
close all;

%% 1. Configuration

cfg.fs = 48000;               % Sampling frequency [Hz]
cfg.signalDuration = 0.25;    % Chirp duration [s]
cfg.numClasses = 48;          % One class per measured azimuth if classification
cfg.labelMode = 'regression'; % 'regression' or 'classification'
cfg.signalDataType = 'single';
cfg.randomSeed = 42;

% cfg.outputFile = 'Datasets/single_talker_dataset_600.h5';

cfg.talkerSOFAFile =  ...
    'D:\Python Projects\P9Datasets\VDP_Porschmann_database\Sentence_Upsampled\DirectivitySentence_6_upsampled.sofa';


%% 2. Read SOFA measurements

% check if the file exist
if ~isfile(cfg.talkerSOFAFile)
    error('SOFA file not found: %s', cfg.talkerSOFAFile);
end

receiverPosition = ncread(cfg.talkerSOFAFile, 'ReceiverPosition');

% ReceiverPosition is [1 x 3 x R]
if size(receiverPosition, 2) ~= 3
    error('Expected ReceiverPosition to have size [1 x 3 x R]; got %s.', ...
        mat2str(size(sourcePosition)));
end

% Spherical ReceiverPosition:
% row 1 = azimuth [degrees], row 2 = elevation [degrees], row 3 = radius [m].
receiverPosition = squeeze(receiverPosition); 

azimuthDeg = mod(double(receiverPosition(1, :)), 360); % [0, 359]
elevationDeg = double(receiverPosition(2, :)); % [-90, 90]
radiusM = double(receiverPosition(3, :)); % constant 1 m

% Select only horizontal-plane measurements. Use a small tolerance for
% floating-point representation of zero elevation.
horizontalIdx = find(abs(elevationDeg) < 1e-6);
if isempty(horizontalIdx)
    error('No elevation=0 degree measurements found in the SOFA file.');
end

% Sort by azimuth for deterministic ordering.
[angleDegs, sortOrder] = sort(azimuthDeg(horizontalIdx));
measurementIdx = horizontalIdx(sortOrder);

% Confirm that the selected directions are unique and uniformly spaced.
if numel(unique(round(angleDegs, 6))) ~= numel(angleDegs)
    error('Duplicate horizontal azimuths found in the SOFA measurements.');
end
if numel(angleDegs) ~= 60
    warning('Expected 60 horizontal measurements, found %d.', numel(angleDegs));
end

% angular gaps are between [5, 7], due to the Lebedev 2702 pattern. (maybe
% we remove angles: 0, 90, 180 and 270), since those are 'in-betweeners'
circularGaps = diff([angleDegs, angleDegs(1) + 360]);
% if any(abs(circularGaps - 6.0) > 1e-4)
%     warning('Horizontal azimuths are not all separated by 6.0 degrees.');
% end

fprintf('Loaded talker file: %s\n', cfg.talkerSOFAFile);
% fprintf('Talker sampling rate: %.0f Hz\n', sofaFs);
% fprintf('Data.Real/Imag size [samples, receivers]: %s\n', ...
%     mat2str(size(dataReal)));
fprintf('Selected horizontal directions: %d\n', numel(angleDegs));
fprintf('Azimuths (degrees):\n');
disp(angleDegs);
fprintf('Receiver radii among selected directions (m):\n');
disp(unique(radiusM(measurementIdx)));


%% 3. Reconstruct full frequency spectrum

% Calculate the talker sampling rate
frequencyVector = ncread(cfg.talkerSOFAFile, 'N');
frequencyBins = numel(frequencyVector);

fprintf("Frequency range: %.1f - %.1f Hz\n", ...
    frequencyVector(1), frequencyVector(end));
fprintf("Frequency bins: %d\n", ...
    frequencyBins);
fprintf("Corresponds to a 1024-point FFT at 48kHz\n");

sofaFs = 2 * frequencyVector(end);

if abs(sofaFs - cfg.fs) > 1e-6
    error('SOFA sampling rate (%g Hz) does not match cfg.fs (%g Hz).', ...
        sofaFs, cfg.fs);
end

dataReal = double(ncread(cfg.talkerSOFAFile, 'Data.Real'));
dataImag = double(ncread(cfg.talkerSOFAFile, 'Data.Imag'));

% Layout of Data.Real/Imag is [N x R x M].
if size(dataReal, 1) ~= 513 || size(dataImag, 1) ~= 513
    error('Expected Data.Real/Imag to have 513 frequency bins; got %d.', ...
        size(dataReal, 1));
end

frequencyResponses = complex(dataReal.', dataImag.'); % i like having [R x N] instead

numReceivers = size(frequencyResponses,1);
numFrequencies = size(frequencyResponses,2);

if numFrequencies ~= frequencyBins
    error('Frequency response dimensions do not match frequency vector.');
end

% assuming positive frequencies include both DC and Nyquist
nfft = 2 * (numFrequencies - 1);

impulseResponses = zeros(numReceivers, nfft, ...
    'like', frequencyResponses); % keeps same input type

for m = 1:numel(measurementIdx)
    receiverIdx = measurementIdx(m); % this way we only compute for receivers on the horizontal plane

    % Select one receiver direction
    H = frequencyResponses(receiverIdx,:); % complex frequency response
    
    % Build a Hermitian-symmetric spectrum
    H_full = [H, conj(H(end-1:-1:2))]; % full frequency spectrum
    
    % Convert to time domain
    h = real(ifft(H_full,nfft)); % impulse response

    % Verify reconstruction
    H_check = fft(h,nfft);
    reconstructionError = max(abs(H_check-H_full));
    
    fprintf("Receiver %d: FFT reconstruction error: %.3e\n", ...
        receiverIdx, reconstructionError);

    % Store impulse response
    impulseResponses(receiverIdx,:) = h;
end


%% 4. Generate source signal (chirp)

rng(cfg.randomSeed);

numChirps = 10;

fs = cfg.fs;
numTimeSamples = round(cfg.signalDuration * fs);
t = (0:numTimeSamples-1)' / fs;

minFrequency = 300;
maxFrequency = 7000;
fadeDuration = 0.005;

sourceSignals = zeros(numTimeSamples, numChirps);

% Store chirp parameters for later reference.
chirpParameters = zeros(numChirps, 2);

for chirpIdx = 1:numChirps
    % Random start and end frequencies.
    f0 = minFrequency + ...
        rand * (maxFrequency - minFrequency);

    f1 = minFrequency + ...
        rand * (maxFrequency - minFrequency);

    % Avoid very small frequency sweeps.
    while abs(f1 - f0) < 500
        f1 = minFrequency + ...
            rand * (maxFrequency - minFrequency);
    end

    % Generate chirp.
    sourceSignal = chirp( ...
        t, f0, cfg.signalDuration, f1, "linear");

    % Apply fade.
    fadeLength = round(fadeDuration * fs);
    fadeLength = min(fadeLength, floor(numTimeSamples / 2));

    fade = sin(linspace(0, pi/2, fadeLength)').^2;

    sourceSignal(1:fadeLength) = ...
        sourceSignal(1:fadeLength) .* fade;

    sourceSignal(end-fadeLength+1:end) = ...
        sourceSignal(end-fadeLength+1:end) .* flipud(fade);

    % Normalize.
    sourceSignal = sourceSignal / max(abs(sourceSignal));

    % Store signal and parameters.
    sourceSignals(:, chirpIdx) = sourceSignal;

    chirpParameters(chirpIdx,:) = [f0, f1];
end


%% 5. Allocate arrays

N = numel(angleDegs);
numIrSamples = size(impulseResponses, 2);
numOutputSamples = numTimeSamples + numIrSamples - 1;

numSamples = N * numChirps;

signals = zeros(numOutputSamples, 1, numSamples, cfg.signalDataType); % receivers for this dataset are not ears

orientationDeg = zeros(numSamples, 1, 'single');
orientationRad = zeros(numSamples, 1, 'single');

classIndex = zeros(numSamples, 1, 'uint16');
classCenterDeg = zeros(numSamples, 1, 'single');

% Additional metadata
chirpLabels = zeros(numSamples, 1, 'uint16');
measurementLabels = zeros(numSamples, 1, 'uint16');


%% 6. Generate monaural samples

% Classification labels are assigned by sorted measured direction.
% This gives 48 classes if cfg.numClasses=48. Regression labels always
% retain the actual measured azimuth in [0, 360).
if cfg.numClasses < 1 || cfg.numClasses ~= round(cfg.numClasses)
    error('cfg.numClasses must be a positive integer.');
end

sampleIdx = 0;

for chirpIdx = 1:numChirps
    sourceSignal = sourceSignals(:, chirpIdx);

    for k = 1:N
        sampleIdx = sampleIdx + 1;

        m = measurementIdx(k);

        ir = double(impulseResponses(m,:));

        % Apply the measured talker VDP.
        emittedSignal = conv(sourceSignal, ir, 'full');

        signals(:, 1, sampleIdx) = ...
            cast(emittedSignal, cfg.signalDataType);

        % Regression labels.
        orientationDeg(sampleIdx) = single(angleDegs(k));
        orientationRad(sampleIdx) = single(deg2rad(angleDegs(k)));

        % Classification labels.
        bin = floor((k - 1) * cfg.numClasses / N);
        bin = min(max(bin, 0), cfg.numClasses - 1);

        classIndex(sampleIdx) = uint16(bin);

        classCenterDeg(sampleIdx) = ...
            single((bin + 0.5) * 360 / cfg.numClasses);

        % Metadata. meh
        chirpLabels(sampleIdx) = uint16(chirpIdx);
        measurementLabels(sampleIdx) = uint16(m);
    end
end


%% 6. Save HDF5 dataset

% make sure the folder exists
outputFolder = fileparts(cfg.outputFile);
if ~isempty(outputFolder) && ~isfolder(outputFolder)
    mkdir(outputFolder);
end

% keep newest generation
if isfile(cfg.outputFile)
    delete(cfg.outputFile);
end

% Acoustic signals
h5create(cfg.outputFile, '/signals', size(signals), ...
    'Datatype', cfg.signalDataType, ...
    'Deflate', 4, ...
    'ChunkSize', [numOutputSamples, 1, 1]);
h5write(cfg.outputFile, '/signals', signals);

% Labels
h5create(cfg.outputFile, '/orientation_deg', [numSamples, 1], 'Datatype', 'single');
h5write(cfg.outputFile, '/orientation_deg', orientationDeg);

h5create(cfg.outputFile, '/orientation_rad', [numSamples, 1], 'Datatype', 'single');
h5write(cfg.outputFile, '/orientation_rad', orientationRad);

h5create(cfg.outputFile, '/class_index', [numSamples, 1], 'Datatype', 'uint16');
h5write(cfg.outputFile, '/class_index', classIndex);

h5create(cfg.outputFile, '/class_center_deg', [numSamples, 1], 'Datatype', 'single');
h5write(cfg.outputFile, '/class_center_deg', classCenterDeg);


%% 8. Store configuration metadata

h5writeatt(cfg.outputFile, '/', 'sampling_rate_hz', cfg.fs);
h5writeatt(cfg.outputFile, '/', 'signal_duration_s', cfg.signalDuration);
h5writeatt(cfg.outputFile, '/', 'num_examples', N);
h5writeatt(cfg.outputFile, '/', 'num_classes', cfg.numClasses);
h5writeatt(cfg.outputFile, '/', 'label_mode', cfg.labelMode);
h5writeatt(cfg.outputFile, '/', 'source_type', 'clean_broadband_chirps');
h5writeatt(cfg.outputFile, '/', 'acoustic_condition', 'free_field_vdp');
% h5writeatt(cfg.outputFile, '/', 'listener_hrtf_enabled', true);
h5writeatt(cfg.outputFile, '/', 'talker_sofa_file', cfg.talkerSOFAFile);
h5writeatt(cfg.outputFile, '/', 'ir_convolution', 'full');
h5writeatt(cfg.outputFile, '/', 'source_distance_m', 1);
h5writeatt(cfg.outputFile, '/', 'source_elevation_deg', 0);
h5writeatt(cfg.outputFile, '/', 'orientation_convention', ...
    'Source azimuth relative to fixed talekr; SOFA azimuth normalized to [0, 360).');
h5writeatt(cfg.outputFile, '/', 'class_index_convention', ...
    'Zero-based class index assigned in sorted measured-azimuth order.');


%% 9. Report results

fprintf('\nHRTF dataset generation completed.\n');
fprintf('Output file: %s\n', cfg.outputFile);
fprintf('Number of examples: %d\n', N);
fprintf('Sampling frequency: %d Hz\n', fs);
fprintf('Chirp samples: %d\n', numTimeSamples);
fprintf('IR samples: %d\n', numIrSamples);
fprintf('Output samples per channel: %d\n', numOutputSamples);
fprintf('Label mode: %s\n', cfg.labelMode);

fprintf('HDF5 contents:\n');
h5disp(cfg.outputFile);
