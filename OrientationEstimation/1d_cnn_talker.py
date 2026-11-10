import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
import os

from helper_scripts.model_architectures import OneDimensionalCNN
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Configuration
# --------------------------------------------------

# H5_PATH = fr"D:\Python Projects\P9-Joint-Orientation-Estimation\Matlab\Datasets\single_talker_dataset_600.h5"
H5_PATH = fr"D:\Python Projects\P9-Joint-Orientation-Estimation\Matlab\Datasets\single_talker_dataset_3600.h5"
# H5_PATH = fr"D:\Python Projects\P9-Joint-Orientation-Estimation\Matlab\Datasets\multiple_talker_dataset.h5"

scenario = 'scenario_0'
scenario_folder = fr'Python\Scenario Results\{scenario}'
os.makedirs(scenario_folder, exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-4
SEED = 144

save_plots = False
save_checkpoints = False

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Device:", device)


# --------------------------------------------------
# Load HDF5 dataset
# --------------------------------------------------

with h5py.File(H5_PATH, "r") as f:
    X = np.asarray(f["signals"], dtype=np.float32)

    angles = np.asarray(f["orientation_rad"], dtype=np.float32).reshape(-1)

print("Original signal shape:", X.shape)
print("Angle shape:", angles.shape)

N = len(angles)

# MATLAB/HDF5 dimension ordering can differ.
# Convert to (N, 2, T): examples, ears, time.
if X.shape[0] == N and X.shape[1] == 1:
    pass
elif X.shape[0] == N and X.shape[2] == 1:
    X = X.transpose(0, 2, 1)
elif X.shape[2] == N and X.shape[1] == 1:
    X = X.transpose(2, 1, 0)
else:
    raise ValueError(
        f"Unexpected signal shape {X.shape}; "
        f"expected {N} examples and 1 channels."
    )

print("Prepared signal shape:", X.shape)

# Normalize both ears with the SAME scale per example.
# This removes overall level while preserving relative
# amplitude differences between the ears.
scale = np.max(np.abs(X), axis=(1, 2), keepdims=True)
X = X / np.maximum(scale, 1e-8)

# Predict circular orientation using sin(theta), cos(theta).
Y = np.stack(
    [np.sin(angles), np.cos(angles)], axis=1
).astype(np.float32)


# --------------------------------------------------
# Train / validation / test split
# --------------------------------------------------

indices = np.arange(N)

train_idx, temp_idx = train_test_split(
    indices, test_size=0.30, random_state=SEED
)

val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.50, random_state=SEED
)

def make_loader(idx, shuffle):
    x = torch.from_numpy(X[idx])
    y = torch.from_numpy(Y[idx])

    return DataLoader(
        TensorDataset(x, y),
        batch_size=BATCH_SIZE,
        shuffle=shuffle
    )

train_loader = make_loader(train_idx, True)
val_loader = make_loader(val_idx, False)
test_loader = make_loader(test_idx, False)

print(
    f"Train: {len(train_idx)}, "
    f"Validation: {len(val_idx)}, "
    f"Test: {len(test_idx)}"
)


# --------------------------------------------------
# 1D CNN
# --------------------------------------------------

model = OneDimensionalCNN()

# change input to fit monaural signal
model.features[0] = nn.Conv1d(1, 16, kernel_size=15, padding=7)
model = model.to(device)

optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)
criterion = nn.MSELoss()

print("\n1d CNN architecture:")
print(model)


# --------------------------------------------------
# Training and validation
# --------------------------------------------------

def run_epoch(loader, training):
    model.train(training)
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(training):
            pred = model(x)
            loss = criterion(pred, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * len(x)

    return total_loss / len(loader.dataset)


best_val_loss = float("inf")
best_state = None
train_losses = []
val_losses = []

start_time = time.perf_counter()
print('Started Training')
for epoch in range(EPOCHS):
    train_loss = run_epoch(train_loader, True)
    val_loss = run_epoch(val_loader, False)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(
            f"Epoch {epoch + 1:3d}/{EPOCHS} | "
            f"Train loss: {train_loss:.6f} | "
            f"Val loss: {val_loss:.6f}"
        )
end_time = time.perf_counter() - start_time
print(f'Finished Training\n\t{end_time:.2f} seconds')

model.load_state_dict(best_state)


# --------------------------------------------------
# Test-set evaluation
# --------------------------------------------------

model.eval()
predictions = []
targets = []

with torch.no_grad():
    for x, y in test_loader:
        pred = model(x.to(device)).cpu().numpy()
        predictions.append(pred)
        targets.append(y.numpy())

predictions = np.concatenate(predictions)
targets = np.concatenate(targets)

# Convert predicted sin/cos values to angles.
pred_angles = np.arctan2(
    predictions[:, 0], predictions[:, 1]
)
true_angles = np.arctan2(
    targets[:, 0], targets[:, 1]
)

# Circular error in degrees, in [-180, 180].
error = np.rad2deg(
    np.arctan2(
        np.sin(pred_angles - true_angles),
        np.cos(pred_angles - true_angles),
    )
)

abs_error = np.abs(error)

print("\nTest results")
print(f"Mean absolute angular error: {abs_error.mean():.2f} deg")
print(f"Median angular error:        {np.median(abs_error):.2f} deg")
print(f"90th percentile error:       {np.percentile(abs_error, 90):.2f} deg")

if save_checkpoints: 
    torch.save(model.state_dict(), "listener_orientation_cnn.pt")


# --------------------------------------------------
# Plots
# --------------------------------------------------

plt.figure()
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.title("Loss graph")
plt.legend()
plt.grid()
plt.tight_layout()
if save_plots: 
    plt.savefig(fr'{scenario_folder}\loss_graph.png')

plt.figure()
plt.scatter(
    np.rad2deg(true_angles),
    np.rad2deg(pred_angles),
    alpha=0.5
)
limits = [-180, 180]
plt.plot(limits, limits, "r--")
plt.xlim(limits)
plt.ylim(limits)
plt.xlabel("True orientation (degrees)")
plt.ylabel("Predicted orientation (degrees)")
plt.title("Talker orientation estimation (1D CNN)")
plt.grid()
plt.tight_layout()
if save_plots: 
    plt.savefig(fr'{scenario_folder}\angular_prediction_error.png')

plt.figure()
plt.hist(abs_error, bins=36)
plt.xlabel("Absolute angular error (degrees)")
plt.ylabel("Number of examples")
plt.title("Test-set angular error")
plt.grid()
plt.tight_layout()
if save_plots: 
    plt.savefig(fr'{scenario_folder}\abs_angular_error.png')

plt.show()
