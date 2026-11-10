
%% Inspect and listen to a generated binaural signal
% Simple free-field setup:
%   - One clean chirp source
%   - Two omnidirectional receivers representing the ears
%   - Listener orientation can be set manually
%   - No HRTF, source directivity, room reflections, or noise

clear;
clc;
close all;

%% 1. Configuration

fs = 48000;
signalDuration = 0.05;
soundSpeed = 343;
earSpacing = 0.18;

listenerPosition = [0, 0];
sourcePosition = [2.5, 0];

% Change this angle to inspect different orientations.
% 0 degrees   = facing positive x
% 90 degrees  = facing positive y
% Positive angles are counterclockwise.
orientationDeg = 0;

%% 2. Generate the clean source signal

numSamples = round(signalDuration * fs);
t = (0:numSamples-1)' / fs;

sourceSignal = chirp(t, 300, signalDuration, 7000);

% Apply short fade-in and fade-out.
fadeLength = round(0.005 * fs);
fade = sin(linspace(0, pi/2, fadeLength)').^2;

sourceSignal(1:fadeLength) = ...
    sourceSignal(1:fadeLength) .* fade;

sourceSignal(end-fadeLength+1:end) = ...
    sourceSignal(end-fadeLength+1:end) .* flipud(fade);

sourceSignal = sourceSignal / max(abs(sourceSignal));

%% 3. Calculate receiver positions

theta = deg2rad(orientationDeg);

% Local left direction, rotated into global coordinates.
leftDirection = [-sin(theta), cos(theta)];

leftPosition = listenerPosition ...
    + (earSpacing / 2) * leftDirection;

rightPosition = listenerPosition ...
    - (earSpacing / 2) * leftDirection;

%% 4. Simulate free-field propagation

rLeft = norm(sourcePosition - leftPosition);
rRight = norm(sourcePosition - rightPosition);

delayLeft = rLeft / soundSpeed;
delayRight = rRight / soundSpeed;

% Fractional delays with linear interpolation.
leftSignal = interp1( ...
    t, sourceSignal, t - delayLeft, 'linear', 0);

rightSignal = interp1( ...
    t, sourceSignal, t - delayRight, 'linear', 0);

% Spherical spreading, relative to 1 metre.
leftSignal = leftSignal / rLeft;
rightSignal = rightSignal / rRight;

% Match the stored signal format:
% columns = left and right channels.
binauralSignal = single([leftSignal, rightSignal]);

%% 5. Plot listener orientation and geometry

figure('Name', 'Binaural Geometry');

hold on;
grid on;
axis equal;

% Source and listener centre.
plot(sourcePosition(1), sourcePosition(2), ...
    'r*', 'MarkerSize', 12, 'LineWidth', 2);

plot(listenerPosition(1), listenerPosition(2), ...
    'ko', 'MarkerSize', 8, 'LineWidth', 2);

% Ear positions.
plot(leftPosition(1), leftPosition(2), ...
    'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'b');

plot(rightPosition(1), rightPosition(2), ...
    'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g');

% Head-centre-to-ear connections.
plot([listenerPosition(1), leftPosition(1)], ...
     [listenerPosition(2), leftPosition(2)], 'b-');

plot([listenerPosition(1), rightPosition(1)], ...
     [listenerPosition(2), rightPosition(2)], 'g-');

% Facing direction.
arrowLength = 0.5;
quiver(listenerPosition(1), listenerPosition(2), ...
    arrowLength*cos(theta), arrowLength*sin(theta), ...
    0, 'k', 'LineWidth', 2, 'MaxHeadSize', 0.5);

% Global reference direction: positive x.
quiver(listenerPosition(1), listenerPosition(2), ...
    0.65, 0, 0, '--k', 'LineWidth', 1);

text(sourcePosition(1), sourcePosition(2), '  Source');
text(leftPosition(1), leftPosition(2), '  Left ear');
text(rightPosition(1), rightPosition(2), '  Right ear');

xlabel('x [m]');
ylabel('y [m]');
title(sprintf('Listener orientation = %.1f degrees', ...
    orientationDeg));

legend('Source', 'Listener centre', 'Left ear', 'Right ear', ...
    'Location', 'best');

hold off;

%% 6. Plot the source and received signals

figure('Name', 'Signal Waveforms');

subplot(3,1,1);
plot(t, sourceSignal);
grid on;
xlabel('Time [s]');
ylabel('Amplitude');
title('Clean source signal');

subplot(3,1,2);
plot(t, leftSignal);
grid on;
xlabel('Time [s]');
ylabel('Amplitude');
title('Left receiver signal');

subplot(3,1,3);
plot(t, rightSignal);
grid on;
xlabel('Time [s]');
ylabel('Amplitude');
title('Right receiver signal');

%% 7. Listen to the binaural signal

% Play both channels together as stereo audio.
% Normalize for listening only; stored signals remain unchanged.
playbackSignal = double(binauralSignal);

peak = max(abs(playbackSignal(:)));

if peak > 0
    playbackSignal = 0.95 * playbackSignal / peak;
end

fprintf('Listener orientation: %.1f degrees\n', orientationDeg);
fprintf('Source-to-left-ear distance: %.3f m\n', rLeft);
fprintf('Source-to-right-ear distance: %.3f m\n', rRight);
fprintf('Left-ear propagation delay: %.3f ms\n', delayLeft*1000);
fprintf('Right-ear propagation delay: %.3f ms\n', delayRight*1000);

sound(playbackSignal, fs);

%% Optional: inspect the same signal with audioplayer
% player = audioplayer(playbackSignal, fs);
% playblocking(player);
