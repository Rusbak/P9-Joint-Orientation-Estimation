%% Comparison of receiver pairs between the sparse and the upsampled dataset
clear;
clc;
close all;

%% Configuration
sparseFolder = "D:\Python Projects\P9Datasets\VDP_Porschmann_database\Voice_Directivities_Sparse_2023";
upsampledFolder = "D:\Python Projects\P9Datasets\VDP_Porschmann_database\Voice_Directivities_Upsampled_2023";

subjectIdx = 6;
repeatIdx  = 1;
phonemeIdx = 1;  % 1 = a, 2 = e, ..., 23 = r

sparseFile = fullfile(sparseFolder, ...
    sprintf("Directivity_%d_%d.sofa", subjectIdx, repeatIdx));

upsampledFile = fullfile(upsampledFolder, ...
    sprintf("Directivity_%d_%d_upsampled.sofa", subjectIdx, repeatIdx));

phonemes = ["a","e","i","o","u","f","s","sh", ...
            "p","t","k","b","d","g","m","n","ng", ...
            "z","v","x","h","l","r"];

%% Load files
sofaSparse = sofaread(sparseFile);
sofaUpsampled = sofaread(upsampledFile);

fprintf("\nSubject %d, repetition %d, phoneme %s\n", ...
    subjectIdx, repeatIdx, phonemes(phonemeIdx));

%% Inspect dimensions
fprintf("\n--- Sparse file ---\n");
disp(size(sofaSparse.FrequencyResponse));
disp(size(sofaSparse.ReceiverPosition));
disp(size(sofaSparse.SourcePosition));

fprintf("\n--- Upsampled file ---\n");
disp(size(sofaUpsampled.FrequencyResponse));
disp(size(sofaUpsampled.ReceiverPosition));
disp(size(sofaUpsampled.SourcePosition));

%% Extract frequency responses
% Expected layout: phoneme x frequency x receiver

H_sparse = squeeze(sofaSparse.FrequencyResponse(phonemeIdx,:,:));
H_upsampled  = squeeze(sofaUpsampled.FrequencyResponse(phonemeIdx,:,:));

fprintf("\nExtracted response dimensions:\n");
fprintf("Sparse: %d receivers x %d frequencies\n", ...
    size(H_sparse,1), size(H_sparse,2));
fprintf("Upsampled:  %d receivers x %d frequencies\n", ...
    size(H_upsampled,1), size(H_upsampled,2));

%% Check frequency vectors
f_sparse = sofaSparse.N(:);
f_upsampled  = sofaUpsampled.N(:);

if ~isequal(size(f_sparse), size(f_upsampled)) || ...
        max(abs(f_sparse - f_upsampled)) > 1e-6
    error("Frequency vectors do not match.");
end

if size(H_sparse,2) ~= numel(f_sparse) || ...
        size(H_upsampled,2) ~= numel(f_upsampled)
    error("Unexpected FrequencyResponse dimension order.");
end

fprintf("Frequency vectors match (%d bins).\n", numel(f_sparse));

%% Extract receiver positions
% SOFA spherical coordinates: azimuth, elevation, radius.
% Reshape to one row per receiver.

pos_sparse = squeeze(sofaSparse.ReceiverPosition); % .'
pos_upsampled = squeeze(sofaUpsampled.ReceiverPosition); % .'

fprintf("\nReceiver position dimensions:\n");
fprintf("Sparse: %d x %d\n", size(pos_sparse,1), size(pos_sparse,2));
fprintf("Upsampled:  %d x %d\n", size(pos_upsampled,1), size(pos_upsampled,2));

if size(pos_sparse,2) ~= 3 || size(pos_upsampled,2) ~= 3
    error("Unexpected receiver-position dimensions.");
end

%% Match sparse directions to nearest upsampled directions
% Match using angular position, ignoring radius.
% This avoids small differences in stored radius affecting the match.

azS = deg2rad(pos_sparse(:,1));
elS = deg2rad(pos_sparse(:,2));

azU = deg2rad(pos_upsampled(:,1));
elU = deg2rad(pos_upsampled(:,2));

% Convert directions to unit vectors
xyzS = [cos(elS).*cos(azS), ...
        cos(elS).*sin(azS), ...
        sin(elS)];

xyzU = [cos(elU).*cos(azU), ...
        cos(elU).*sin(azU), ...
        sin(elU)];

numSparse = size(pos_sparse,1);
matchedIdx = zeros(numSparse,1);
angularError = zeros(numSparse,1);

for k = 1:numSparse
    % Angular separation from this sparse direction to all upsampled directions
    dotProducts = xyzU * xyzS(k,:)';
    dotProducts = max(-1, min(1, dotProducts));

    angles = acos(dotProducts);

    [angularError(k), matchedIdx(k)] = min(angles);
end

angularErrorDeg = rad2deg(angularError);

fprintf("\nMaximum angular matching error: %.6f degrees\n", ...
    max(angularErrorDeg));

fprintf("Mean angular matching error: %.6f degrees\n", ...
    mean(angularErrorDeg));

%% Compare complex frequency responses
H_upsampled_matched = H_upsampled(matchedIdx, :);

complexDifference = H_upsampled_matched - H_sparse;

magnitudeSparse = 20*log10(abs(H_sparse));
magnitudeUpsampled  = 20*log10(abs(H_upsampled_matched));

magnitudeDifference = magnitudeUpsampled - magnitudeSparse;

% Wrapped phase difference, in degrees
phaseDifference = rad2deg(angle(H_upsampled_matched .* conj(H_sparse)));

%% Summary statistics
fprintf("\n--- Comparison results ---\n");

fprintf("Mean absolute magnitude difference: %.4f dB\n", ...
    mean(abs(magnitudeDifference(:)), 'omitnan'));

fprintf("Maximum absolute magnitude difference: %.4f dB\n", ...
    max(abs(magnitudeDifference(:)), [], 'omitnan'));

fprintf("Mean absolute phase difference: %.4f degrees\n", ...
    mean(abs(phaseDifference(:)), 'omitnan'));

fprintf("Maximum absolute phase difference: %.4f degrees\n", ...
    max(abs(phaseDifference(:)), [], 'omitnan'));

fprintf("Mean absolute complex difference: %.6g\n", ...
    mean(abs(complexDifference(:)), 'omitnan'));

%% Plot magnitude difference
figure;
imagesc(1:numSparse, f_sparse, magnitudeDifference);
axis xy;
colorbar;
xlabel("Sparse receiver index");
ylabel("Frequency (Hz)");
title("Magnitude difference: upsampled - sparse (dB)");

%% Plot phase difference
figure;
imagesc(1:numSparse, f_sparse, phaseDifference);
axis xy;
colorbar;
xlabel("Sparse receiver index");
ylabel("Frequency (Hz)");
title("Phase difference: upsampled - sparse (degrees)");

%% Plot magnitude responses for one receiver
receiverIdx = 1;

figure;
plot(f_sparse, magnitudeSparse(receiverIdx, :), ...
    'LineWidth', 1.2);
hold on;
plot(f_upsampled, magnitudeUpsampled(receiverIdx, :), ...
    '--', 'LineWidth', 1.2);

xlabel("Frequency (Hz)");
ylabel("Magnitude (dB)");
legend("Sparse", "Upsampled");
title(sprintf("Receiver %d: %s", ...
    receiverIdx, phonemes(phonemeIdx)));
grid on;
