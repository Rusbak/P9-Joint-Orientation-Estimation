%% Inspect a SOFA file
% Reports the metadata and directional coverage needed to prepare
% an HRTF dataset for binaural orientation estimation.

clear;
clc;
close all;

%% Select SOFA file
% DiffFieldIRs_SOFA, FreeFieldIRs_SOFA, HpIRs_SOFA, HRIRs_SOFA
pathname = 'D:\Python Projects\P9Datasets\OlHeaDHRTF_database\HRIRs_SOFA';
filename = 'VP_N6-ED.SOFA';

sofaFilePath = fullfile(pathname, filename);

fprintf('\nInspecting SOFA file:\n%s\n\n', sofaFilePath);

%% Display global attributes
info = ncinfo(sofaFilePath);

fprintf('========== GLOBAL ATTRIBUTES ==========\n');

for k = 1:numel(info.Attributes)
    name = info.Attributes(k).Name;
    value = info.Attributes(k).Value;

    fprintf('%s: %s\n', name, string(value));
end

%% Display variable names and dimensions
fprintf('\n========== VARIABLES ==========\n');

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

    if ~isempty(var.Attributes)
        for a = 1:numel(var.Attributes)
            attrName = var.Attributes(a).Name;
            attrValue = var.Attributes(a).Value;

            fprintf('  %s: %s\n', ...
                attrName, string(attrValue));
        end
    end
end

%% Read and summarize important variables
fprintf('\n========== IMPORTANT DATA ==========\n');

% Helper: locate a variable without assuming case sensitivity.
varNames = {info.Variables.Name};

% Source positions
sourceIdx = find(strcmpi(varNames, 'SourcePosition'), 1);

if ~isempty(sourceIdx)
    sourcePosition = ncread(sofaFilePath, ...
        info.Variables(sourceIdx).Name);

    fprintf('\nSourcePosition size: ');
    disp(size(sourcePosition));

    fprintf('First few source-position entries:\n');
    disp(sourcePosition(1:min(10, size(sourcePosition, 1)), :));

    % SourcePosition commonly has three coordinates per measurement.
    % Verify its dimension order and coordinate system in the
    % metadata printed above before interpreting these values.
else
    warning('SourcePosition was not found.');
end

% Sampling rate
fsIdx = find(strcmpi(varNames, 'Data.SamplingRate'), 1);

if ~isempty(fsIdx)
    fs = ncread(sofaFilePath, info.Variables(fsIdx).Name);

    fprintf('\nSampling rate:\n');
    disp(fs);
else
    warning('Data.SamplingRate was not found.');
end

% Impulse responses
irIdx = find(strcmpi(varNames, 'Data.IR'), 1);

if ~isempty(irIdx)
    ir = ncread(sofaFilePath, info.Variables(irIdx).Name);

    fprintf('\nData.IR size: ');
    disp(size(ir));

    fprintf('Total number of IR elements: %d\n', numel(ir));
else
    warning('Data.IR was not found.');
end

% Delay information
delayIdx = find(strcmpi(varNames, 'Data.Delay'), 1);

if ~isempty(delayIdx)
    delay = ncread(sofaFilePath, info.Variables(delayIdx).Name);

    fprintf('\nData.Delay size: ');
    disp(size(delay));

    fprintf('Data.Delay values:\n');
    disp(delay);
else
    fprintf('\nData.Delay was not found.\n');
end

% Listener position
listenerIdx = find(strcmpi(varNames, 'ListenerPosition'), 1);

if ~isempty(listenerIdx)
    listenerPosition = ncread(sofaFilePath, ...
        info.Variables(listenerIdx).Name);

    fprintf('\nListenerPosition size: ');
    disp(size(listenerPosition));
    disp(listenerPosition);
else
    fprintf('\nListenerPosition was not found.\n');
end

% Receiver position
receiverIdx = find(strcmpi(varNames, 'ReceiverPosition'), 1);

if ~isempty(receiverIdx)
    receiverPosition = ncread(sofaFilePath, ...
        info.Variables(receiverIdx).Name);

    fprintf('\nReceiverPosition size: ');
    disp(size(receiverPosition));
    disp(receiverPosition);
else
    fprintf('\nReceiverPosition was not found.\n');
end

fprintf('\nInspection complete.\n');