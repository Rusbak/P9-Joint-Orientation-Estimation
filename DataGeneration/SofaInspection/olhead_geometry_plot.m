clear; clc; close all;

%% load sofa file for testing
dataBasePath = 'D:\Python Projects\P9Datasets\OlHeaDHRTF_database\';

% DiffFieldIRs_SOFA, FreeFieldIRs_SOFA, HpIRs_SOFA, HRIRs_SOFA
dataSetPath = strcat(dataBasePath, 'HRIRs_SOFA\');

testType = 'VP_N6-ED';
%testType = 'VP_E12-BTE_mid';
sofaFilePath = strcat(dataSetPath, testType, '.SOFA');

sofaFile = sofaread(sofaFilePath);

%% look at the contents of the sofa file
ncdisp(sofaFilePath);

%ir = sofaFile.Numerator;
%size(ir)

%srcPositions = sofaFile.SourcePosition;
%size(srcPositions)

%% visualize the experiment
figure;
plotGeometry(sofaFile)

% plot geometry of all measurements combined
