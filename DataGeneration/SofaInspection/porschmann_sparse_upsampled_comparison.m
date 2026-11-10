%% Comparison between the dataset of sparse measurements (32) 
% and the dataset upsampled from 32 to 2702. 

clear;
clc;
close all;

%% 1. Select SOFA files

pathnameSparse = 'D:\Python Projects\P9Datasets\VDP_Porschmann_database\Voice_Directivities_Sparse_2023';
pathnameUpsampled = 'D:\Python Projects\P9Datasets\VDP_Porschmann_database\Voice_Directivities_Upsampled_2023';

talkerSparse = 'Directivity_9_1.sofa';
talkerUpsampled = 'Directivity_9_1_upsampled.sofa';

filePathSparse = fullfile(pathnameSparse, talkerSparse);
filePathUpsampled = fullfile(pathnameUpsampled, talkerUpsampled);


%% 2. Load the SOFA files

sofaSparse = sofaread(filePathSparse);
sofaUpsampled = sofaread(filePathUpsampled);


%% 3. Display file contents

ncdisp(filePathSparse);
ncdisp(filePathUpsampled);


%% 4. Plot geometry

% Extract coordinates
posS = sofaSparse.ReceiverPosition;
posU = sofaUpsampled.ReceiverPosition;

% Sanity check
fprintf('Sparse azimuth:   %.4f to %.4f\n', min(posS(:,1,:)), max(posS(:,1,:)));
fprintf('Sparse elevation: %.4f to %.4f\n', min(posS(:,2,:)), max(posS(:,2,:)));
fprintf('Sparse radius:    %.4f to %.4f\n', min(posS(:,3,:)), max(posS(:,3,:)));

fprintf('Upsampled azimuth:   %.4f to %.4f\n', min(posU(:,1,:)), max(posU(:,1,:)));
fprintf('Upsampled elevation: %.4f to %.4f\n', min(posU(:,2,:)), max(posU(:,2,:)));
fprintf('Upsampled radius:    %.4f to %.4f\n', min(posU(:,3,:)), max(posU(:,3,:)));

% Each row is one coordinate:
% Row 1: Azimuth
% Row 2: Elevation
% Row 3: Radius

azS = deg2rad(posS(:,1,:));
elS = deg2rad(posS(:,2,:));
rS  = posS(:,3,:);

azU = deg2rad(posU(:,1,:));
elU = deg2rad(posU(:,2,:));
rU  = posU(:,3,:);

% Convert to cartesian coordinates
xS = rS .* cos(elS) .* cos(azS);
yS = rS .* cos(elS) .* sin(azS);
zS = rS .* sin(elS);

xU = rU .* cos(elU) .* cos(azU);
yU = rU .* cos(elU) .* sin(azU);
zU = rU .* sin(elU);

% Plot
figure;
hold on;

scatter3(xS, yS, zS, 25, 'b', 'filled');
scatter3(xU, yU, zU, 12, 'r', 'filled');

axis equal;
grid on;

xlim([-1.2 1.2]);
ylim([-1.2 1.2]);
zlim([-1.2 1.2]);
% zlim([-.02 .02]);

xlabel('X [m]');
ylabel('Y [m]');
zlabel('Z [m]');

title('VDP Receiver Positions');
legend('Sparse measurements', 'Upsampled measurements');

view(3);