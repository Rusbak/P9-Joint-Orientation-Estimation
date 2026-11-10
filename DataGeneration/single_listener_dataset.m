%% Single HRTF Binaural Dataset Generator

% Signal generation:
%   1. Load a singular HRTF set.
%   2. Rotate the receiver pair according to listener orientation.
%   3. Generate clean source signals (chirps).
%   4. Apply free-field propagation delay and spherical spreading.
%   5. Save binaural signals and labels to HDF5.

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

cfg.outputFile = 'Datasets/single_listener_dataset_480.h5';

cfg.listenerSOFAFile = ...
    'D:\Python Projects\P9Datasets\HRTF_OlHeaD_database\HRIRs_SOFA\VP_N6-ED.SOFA';


%% 2. Read SOFA measurements

% check if the file exist
if ~isfile(cfg.listenerSOFAFile)
    error('SOFA file not found: %s', cfg.listenerSOFAFile);
end

sourcePosition = ncread(cfg.listenerSOFAFile, 'SourcePosition');
hrir = ncread(cfg.listenerSOFAFile, 'Data.IR');
sofaFs = double(ncread(cfg.listenerSOFAFile, 'Data.SamplingRate'));

% SourcePosition is [3 x M] and
% Data.IR is [N x R x M], where R=1 is left and R=2 is right.
if size(sourcePosition, 1) ~= 3
    error('Expected SourcePosition to have size [3 x M]; got %s.', ...
        mat2str(size(sourcePosition)));
end
if size(hrir, 2) ~= 2
    error('Expected Data.IR to have 2 receiver channels; got %d.', ...
        size(hrir, 2));
end
if abs(sofaFs - cfg.fs) > 1e-6
    error('SOFA sampling rate (%g Hz) does not match cfg.fs (%g Hz).', ...
        sofaFs, cfg.fs);
end

% SimpleFreeFieldHRIR uses spherical SourcePosition:
% row 1 = azimuth [degrees], row 2 = elevation [degrees], row 3 = distance [m].
azimuthDeg = mod(double(sourcePosition(1, :)), 360);
elevationDeg = double(sourcePosition(2, :));
distanceM = double(sourcePosition(3, :));

% Select only horizontal-plane measurements. Use a small tolerance for
% floating-point representation of zero elevation.
horizontalIdx = find(abs(elevationDeg) < 1e-6);
if isempty(horizontalIdx)
    error('No elevation=0 degree measurements found in the SOFA file.');
end

% Sort by azimuth for deterministic ordering.
[anglesDeg, sortOrder] = sort(azimuthDeg(horizontalIdx));
measurementIdx = horizontalIdx(sortOrder);

% Confirm that the selected directions are unique and uniformly spaced.
if numel(unique(round(anglesDeg, 6))) ~= numel(anglesDeg)
    error('Duplicate horizontal azimuths found in the SOFA measurements.');
end
if numel(anglesDeg) ~= 48
    warning('Expected 48 horizontal measurements, found %d.', numel(anglesDeg));
end

circularGaps = diff([anglesDeg, anglesDeg(1) + 360]);
if any(abs(circularGaps - 7.5) > 1e-4)
    warning('Horizontal azimuths are not all separated by 7.5 degrees.');
end

fprintf('Loaded SOFA file: %s\n', cfg.listenerSOFAFile);
fprintf('SOFA sampling rate: %.0f Hz\n', sofaFs);
fprintf('HRIR size [samples, receivers, measurements]: %s\n', ...
    mat2str(size(hrir)));
fprintf('Selected horizontal directions: %d\n', numel(anglesDeg));
fprintf('Azimuths (degrees):\n');
disp(anglesDeg);
fprintf('Source distance(s) among selected directions (m):\n');
disp(unique(distanceM(measurementIdx)));


%% 3. Generate source signal (chirp)

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


%% 4. Allocate arrays

N = numel(anglesDeg);
numHrirSamples = size(hrir, 1);
numOutputSamples = numTimeSamples + numHrirSamples - 1;

numSamples = N * numChirps;

signals = zeros(numOutputSamples, 2, numSamples, cfg.signalDataType);

orientationDeg = zeros(numSamples, 1, 'single');
orientationRad = zeros(numSamples, 1, 'single');

classIndex = zeros(numSamples, 1, 'uint16');
classCenterDeg = zeros(numSamples, 1, 'single');

% Additional metadata
chirpLabels = zeros(numSamples, 1, 'uint16');
measurementLabels = zeros(numSamples, 1, 'uint16');


%% 5. Generate binaural samples

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

        % ReceiverDescription: R=1 is ED-l; R=2 is ED-r.
        hrirLeft = double(hrir(:, 1, m));
        hrirRight = double(hrir(:, 2, m));

        % Apply the measured listener HRTF.
        receivedLeft = conv(sourceSignal, hrirLeft, 'full');
        receivedRight = conv(sourceSignal, hrirRight, 'full');

        signals(:, 1, sampleIdx) = ...
            cast(receivedLeft, cfg.signalDataType);

        signals(:, 2, sampleIdx) = ...
            cast(receivedRight, cfg.signalDataType);

        % Regression labels.
        orientationDeg(sampleIdx) = single(anglesDeg(k));
        orientationRad(sampleIdx) = single(deg2rad(anglesDeg(k)));

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
    'ChunkSize', [numOutputSamples, 2, 1]);
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


%% 7. Store configuration metadata

h5writeatt(cfg.outputFile, '/', 'sampling_rate_hz', cfg.fs);
h5writeatt(cfg.outputFile, '/', 'signal_duration_s', cfg.signalDuration);
h5writeatt(cfg.outputFile, '/', 'num_examples', N);
h5writeatt(cfg.outputFile, '/', 'num_classes', cfg.numClasses);
h5writeatt(cfg.outputFile, '/', 'label_mode', cfg.labelMode);
h5writeatt(cfg.outputFile, '/', 'source_type', 'clean_broadband_chirps');
h5writeatt(cfg.outputFile, '/', 'acoustic_condition', 'free_field_hrtf');
% h5writeatt(cfg.outputFile, '/', 'listener_hrtf_enabled', true);
h5writeatt(cfg.outputFile, '/', 'listener_sofa_file', cfg.listenerSOFAFile);
h5writeatt(cfg.outputFile, '/', 'hrir_convolution', 'full');
h5writeatt(cfg.outputFile, '/', 'source_distance_m', 2.5);
h5writeatt(cfg.outputFile, '/', 'source_elevation_deg', 0);
h5writeatt(cfg.outputFile, '/', 'orientation_convention', ...
    'Source azimuth relative to fixed listener; SOFA azimuth normalized to [0, 360).');
h5writeatt(cfg.outputFile, '/', 'class_index_convention', ...
    'Zero-based class index assigned in sorted measured-azimuth order.');


%% 8. Report results

fprintf('\nHRTF dataset generation completed.\n');
fprintf('Output file: %s\n', cfg.outputFile);
fprintf('Number of examples: %d\n', N);
fprintf('Sampling frequency: %d Hz\n', fs);
fprintf('Chirp samples: %d\n', numTimeSamples);
fprintf('HRIR samples: %d\n', numHrirSamples);
fprintf('Output samples per channel: %d\n', numOutputSamples);
fprintf('Label mode: %s\n', cfg.labelMode);

fprintf('HDF5 contents:\n');
h5disp(cfg.outputFile);
