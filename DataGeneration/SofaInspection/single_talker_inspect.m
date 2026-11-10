%% Single VDP Binaural Dataset Generator

clear;
clc;
close all;

%% Configuration
filename = "D:\Python Projects\P9Datasets\VDP_Porschmann_database\Sentence_Upsampled\DirectivitySentence_6_upsampled.sofa";

phonemeIdx = 1;     % /a/
receiverIdx = 16;    % 0 azimuth, 0 elevation

Fs = 48000;

%% Load VDP
VDP = sofaread(filename);

% Expected layout: phoneme x receiver x frequency
H_all = squeeze(VDP.FrequencyResponse(phonemeIdx,:,:));

% Select one receiver direction
H = H_all(receiverIdx,:);

% Frequency vector
f = VDP.N(:);

fprintf("Selected phoneme: a\n");
fprintf("Selected receiver: %d\n", receiverIdx);
fprintf("Receiver position [azimuth, elevation, radius]:\n");
disp(VDP.ReceiverPosition(receiverIdx,:));

fprintf("Number of frequency bins: %d\n", numel(H));
fprintf("Frequency range: %.1f - %.1f Hz\n", f(1), f(end));

%% Reconstruct full frequency spectrum
nfft = 2*(numel(H)-1);

H_full = [H, conj(H(end-1:-1:2))];

% Convert to time domain
h = real(ifft(H_full,nfft));

t_h = (0:nfft-1)'/Fs;

% Verify reconstruction
H_check = fft(h,nfft);
reconstructionError = max(abs(H_check-H_full));

fprintf("FFT reconstruction error: %.3e\n", ...
    reconstructionError);

% Plot impulse response
figure;
plot(t_h*1000,h);
xlabel("Time (ms)");
ylabel("Amplitude");
title("VDP impulse response");
grid on;

%% Generate chirp
duration = 0.25;
t = (0:round(duration*Fs)-1)'/Fs;

x = chirp(t,300,t(end),7000);

% Apply 5 ms fade-in and fade-out
fadeDuration = 0.005;
fadeSamples = round(fadeDuration*Fs);

fade = sin(linspace(0,pi/2,fadeSamples)').^2;

x(1:fadeSamples) = x(1:fadeSamples).*fade;
x(end-fadeSamples+1:end) = ...
    x(end-fadeSamples+1:end).*flipud(fade);

% Convolve chirp with VDP impulse response
y = conv(x,h);

t_y = (0:numel(y)-1)'/Fs;

% Plot signals
figure;

subplot(2,1,1);
plot(t,x);
xlabel("Time (s)");
ylabel("Amplitude");
title("Original chirp");
grid on;

subplot(2,1,2);
plot(t_y,y);
xlabel("Time (s)");
ylabel("Amplitude");
title("Chirp after VDP filtering");
grid on;

%% Compare spectra
nfftPlot = 2^nextpow2(numel(y));

X = fft(x,nfftPlot);
Y = fft(y,nfftPlot);

fPlot = (0:nfftPlot/2)'*Fs/nfftPlot;

figure;
plot(fPlot,20*log10(abs(X(1:nfftPlot/2+1))+eps));
hold on;
plot(fPlot,20*log10(abs(Y(1:nfftPlot/2+1))+eps));

xlim([0 10000]);
xlabel("Frequency (Hz)");
ylabel("Magnitude (dB)");
legend("Input chirp","VDP-filtered chirp");
title("Input and output spectra");
grid on;
