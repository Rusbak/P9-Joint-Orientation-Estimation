%% Simple Binaural Dataset Generator
% Signal generation:
%   1. Place two omnidirectional receivers at the ears.
%   2. Rotate the receiver pair according to listener orientation.
%   3. Generate a clean source signal (chirp).
%   4. Apply free-field propagation delay and spherical spreading.
%   5. Save binaural signals and labels to HDF5.

clear;
clc;
close all;

%% 1. Configuration

cfg.outputFile = 'Datasets/simple_listener_dataset.h5';

% Dataset size
cfg.numSamples = 3600;

% Acoustic parameters
cfg.fs = 48000;               % Sampling frequency [Hz]
cfg.signalDuration = 0.25;    % Duration of each signal [s]
cfg.soundSpeed = 343;         % Speed of sound [m/s]
cfg.earSpacing = 0.18;        % Distance between receivers [m]

% Coordinate system
cfg.listenerPosition = [0, 0];  % Listener head centre [x, y]
cfg.sourcePosition = [2.5, 0];  % Source position [x, y]

% Orientation convention:
% 0 degrees   = facing positive x
% 90 degrees  = facing positive y
% Positive angles are counterclockwise.
cfg.angleMin = -180;
cfg.angleMax = 180;
% we could also just use 0 - 360?

% Label mode:
% 'regression'    -> continuous orientation target
% 'classification'-> discrete orientation classes
cfg.labelMode = 'regression';
cfg.numClasses = 4;          % 90-degree classes if classification

% Reproducibility
cfg.randomSeed = 42;

% Save signals as single precision to reduce file size
cfg.signalDataType = 'single';


%% 2. Generate source signal (chirp)

rng(cfg.randomSeed);

% Chirp settings
cfg.chirpStartHz = 300;
cfg.chirpEndHz = 7000;
cfg.fadeDuration = 0.005;

fs = cfg.fs;
numTimeSamples = round(cfg.signalDuration * fs);
t = (0:numTimeSamples-1)' / fs;

% Clean broadband chirp: controlled excitation, not random noise.
sourceSignal = chirp(t, cfg.chirpStartHz, cfg.signalDuration, cfg.chirpEndHz);

% Apply a short fade at each end to reduce abrupt transitions.
fadeLength = round(cfg.fadeDuration * fs);
fadeLength = min(fadeLength, floor(numTimeSamples / 2));
fade = sin(linspace(0, pi/2, fadeLength)').^2;

sourceSignal(1:fadeLength) = ...
    sourceSignal(1:fadeLength) .* fade;

sourceSignal(end-fadeLength+1:end) = ...
    sourceSignal(end-fadeLength+1:end) .* flipud(fade);

% Normalize source amplitude.
sourceSignal = sourceSignal / max(abs(sourceSignal));


%% 3. Allocate dataset arrays

N = cfg.numSamples;

% MATLAB layout: [time samples, channels, examples]
% Channel 1 = left receiver
% Channel 2 = right receiver
signals = zeros(numTimeSamples, 2, N, cfg.signalDataType);

orientationDeg = zeros(N, 1, 'single');
orientationRad = zeros(N, 1, 'single');

classIndex = zeros(N, 1, 'uint16');
classCenterDeg = zeros(N, 1, 'single');


%% 4. Sample listener orientations

% Uniformly distributed orientations in the configured interval.
anglesDeg = cfg.angleMin + ...
    (cfg.angleMax - cfg.angleMin) * rand(N, 1); % replace random with interpolated angles based on numSamples

% Equally spaced orientations in the configured interval.
%anglesDeg = linspace(cfg.angleMin, cfg.angleMax, N + 1)';
%anglesDeg(end) = [];

anglesRad = deg2rad(anglesDeg);

% Classification bins cover the full angular interval.
classWidth = (cfg.angleMax - cfg.angleMin) / cfg.numClasses;


%% 5. Generate binaural signals

for k = 1:N

    theta = anglesRad(k);

    % Listener's local left direction in global coordinates.
    leftDirection = [-sin(theta), cos(theta)];

    % Receiver positions rotate around the listener centre.
    leftPosition = cfg.listenerPosition ...
        + (cfg.earSpacing / 2) * leftDirection;

    rightPosition = cfg.listenerPosition ...
        - (cfg.earSpacing / 2) * leftDirection;

    % Distances from the source to each receiver.
    rLeft = norm(cfg.sourcePosition - leftPosition);
    rRight = norm(cfg.sourcePosition - rightPosition);

    % Propagation delays in seconds.
    delayLeft = rLeft / cfg.soundSpeed;
    delayRight = rRight / cfg.soundSpeed;

    % Fractional-delay propagation.
    % Interpolation avoids restricting delays to whole samples.
    receivedLeft = interp1( ...
        t, sourceSignal, t - delayLeft, 'linear', 0);

    receivedRight = interp1( ...
        t, sourceSignal, t - delayRight, 'linear', 0);

    % Free-field spherical spreading.
    % Relative to a reference distance of 1 metre.
    receivedLeft = receivedLeft / rLeft;
    receivedRight = receivedRight / rRight;

    % Store the binaural example.
    signals(:, 1, k) = cast(receivedLeft, cfg.signalDataType);
    signals(:, 2, k) = cast(receivedRight, cfg.signalDataType);

    % Continuous regression labels.
    orientationDeg(k) = single(anglesDeg(k));
    orientationRad(k) = single(anglesRad(k));

    % Discrete classification label, if needed.
    bin = floor((anglesDeg(k) - cfg.angleMin) / classWidth);
    bin = min(max(bin, 0), cfg.numClasses - 1);

    classIndex(k) = uint16(bin);

    classCenterDeg(k) = single( ...
        cfg.angleMin + (bin + 0.5) * classWidth);
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
    'ChunkSize', [numTimeSamples, 2, 1]);
h5write(cfg.outputFile, '/signals', signals);

% Labels
h5create(cfg.outputFile, '/orientation_deg', [N, 1], 'Datatype', 'single');
h5write(cfg.outputFile, '/orientation_deg', orientationDeg);

h5create(cfg.outputFile, '/orientation_rad', [N, 1], 'Datatype', 'single');
h5write(cfg.outputFile, '/orientation_rad', orientationRad);

h5create(cfg.outputFile, '/class_index', [N, 1], 'Datatype', 'uint16');
h5write(cfg.outputFile, '/class_index', classIndex);

h5create(cfg.outputFile, '/class_center_deg', [N, 1], 'Datatype', 'single');
h5write(cfg.outputFile, '/class_center_deg', classCenterDeg);


%% 7. Store configuration metadata

h5writeatt(cfg.outputFile, '/', 'sampling_rate_hz', cfg.fs);
h5writeatt(cfg.outputFile, '/', 'sound_speed_m_s', cfg.soundSpeed);
h5writeatt(cfg.outputFile, '/', 'ear_spacing_m', cfg.earSpacing);
h5writeatt(cfg.outputFile, '/', 'signal_duration_s', cfg.signalDuration);
h5writeatt(cfg.outputFile, '/', 'num_examples', N);
h5writeatt(cfg.outputFile, '/', 'num_classes', cfg.numClasses);
h5writeatt(cfg.outputFile, '/', 'label_mode', cfg.labelMode);
h5writeatt(cfg.outputFile, '/', 'source_type', 'clean_broadband_chirp');
h5writeatt(cfg.outputFile, '/', 'acoustic_condition', 'free_field');
h5writeatt(cfg.outputFile, '/', 'source_distance_m', 2.5);
h5writeatt(cfg.outputFile, '/', 'source_elevation_deg', 0);
% h5writeatt(cfg.outputFile, '/', 'listener_hrtf_enabled', false);
% h5writeatt(cfg.outputFile, '/', 'source_directivity_enabled', false);
% h5writeatt(cfg.outputFile, '/', 'room_enabled', false);
% h5writeatt(cfg.outputFile, '/', 'noise_enabled', false);

h5writeatt(cfg.outputFile, '/', ...
    'orientation_convention', ...
    '0 deg = positive x; positive angles counterclockwise');

h5writeatt(cfg.outputFile, '/', ...
    'class_index_convention', ...
    'Zero-based; class centers span the configured angular interval');


%% 8. Report results

fprintf('\nHRTF dataset generation completed.\n');
fprintf('Output file: %s\n', cfg.outputFile);
fprintf('Number of examples: %d\n', N);
fprintf('Sampling frequency: %d Hz\n', fs);
fprintf('Output samples per channel: %d\n', numOutputSamples);
fprintf('Label mode: %s\n', cfg.labelMode);

fprintf('HDF5 contents:\n');
h5disp(cfg.outputFile);
