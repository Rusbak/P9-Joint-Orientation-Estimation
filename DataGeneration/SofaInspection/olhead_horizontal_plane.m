%% Inspect OlHeaD HRTF directions and impulse responses

clear;
clc;
close all;

%% Select SOFA file
% DiffFieldIRs_SOFA, FreeFieldIRs_SOFA, HpIRs_SOFA, HRIRs_SOFA
pathname = 'D:\Python Projects\P9Datasets\OlHeaDHRTF_database\HRIRs_SOFA';
filename = 'VP_N6-ED.SOFA';

sofaFilePath = fullfile(pathname, filename);

fprintf('\nInspecting SOFA file:\n%s\n\n', sofaFilePath);

%% Read data
info = ncinfo(sofaFilePath);

sourcePosition = ncread(sofaFilePath, 'SourcePosition');
ir = ncread(sofaFilePath, 'Data.IR');
fs = double(ncread(sofaFilePath, 'Data.SamplingRate'));

% SourcePosition is stored as C x M in this file.
azimuth = sourcePosition(1, :).';
elevation = sourcePosition(2, :).';
distance = sourcePosition(3, :).';

% Convert azimuth to [0, 360).
azimuth = mod(azimuth, 360);

%% Inspect Data.Delay metadata
fprintf('\n========== DATA.DELAY METADATA ==========\n');

varIdx = find(strcmp({info.Variables.Name}, 'Data.Delay'), 1);

if ~isempty(varIdx)
    delayInfo = info.Variables(varIdx);

    for k = 1:numel(delayInfo.Attributes)
        fprintf('%s: %s\n', ...
            delayInfo.Attributes(k).Name, ...
            string(delayInfo.Attributes(k).Value));
    end

    fprintf('Stored delay values:\n');
    disp(ncread(sofaFilePath, 'Data.Delay'));
else
    fprintf('Data.Delay was not found.\n');
end

%% Find horizontal-plane measurements
elevationTolerance = 1e-6;
horizontalIdx = find(abs(elevation) < elevationTolerance);

horizontalAzimuth = azimuth(horizontalIdx);

% Sort measurements by azimuth.
[sortedAzimuth, order] = sort(horizontalAzimuth);
sortedIdx = horizontalIdx(order);

fprintf('\n========== HORIZONTAL-PLANE COVERAGE ==========\n');
fprintf('Number of horizontal measurements: %d\n', ...
    numel(sortedAzimuth));

fprintf('Azimuths (degrees):\n');
disp(sortedAzimuth.');

if numel(sortedAzimuth) > 1
    circularGaps = diff([sortedAzimuth; sortedAzimuth(1) + 360]);

    fprintf('Circular angular gaps (degrees):\n');
    disp(circularGaps.');

    fprintf('Unique azimuth count: %d\n', ...
        numel(unique(sortedAzimuth)));

    fprintf('Minimum gap: %.4f degrees\n', min(circularGaps));
    fprintf('Maximum gap: %.4f degrees\n', max(circularGaps));
end

fprintf('Source distances (m):\n');
disp(unique(distance(horizontalIdx)).');

%% Plot horizontal-plane directions
figure;

polarplot(deg2rad(sortedAzimuth), ...
    ones(size(sortedAzimuth)), 'o');

title('Available Horizontal-Plane HRTF Directions');

%% Plot left and right impulse responses
% Select up to four directions distributed across the horizontal plane.
if ~isempty(sortedIdx)
    nToPlot = min(4, numel(sortedIdx));

    selectedPositions = round(linspace(1, numel(sortedIdx), nToPlot));
    selectedMeasurements = sortedIdx(selectedPositions);

    figure;

    for k = 1:nToPlot
        m = selectedMeasurements(k);

        % Data.IR dimensions in this file: N x R x M.
        leftIR = squeeze(ir(:, 1, m));
        rightIR = squeeze(ir(:, 2, m));

        t_ms = (0:numel(leftIR)-1) / fs * 1000;

        subplot(nToPlot, 1, k);

        plot(t_ms, leftIR, 'DisplayName', 'Left');
        hold on;
        plot(t_ms, rightIR, 'DisplayName', 'Right');

        grid on;
        xlabel('Time (ms)');
        ylabel('Amplitude');

        title(sprintf('Azimuth %.1f degrees', azimuth(m)));
        legend('Location', 'best');
    end

    sgtitle('OlHeaD Left and Right Impulse Responses');
end