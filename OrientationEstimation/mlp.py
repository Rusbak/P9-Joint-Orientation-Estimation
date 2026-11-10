import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
import os

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Configuration
# --------------------------------------------------

# H5_PATH = fr"D:\Python Projects\P9-Joint-Orientation-Estimation\Matlab\Datasets\simple_listener_dataset.h5"
# H5_PATH = fr"D:\Python Projects\P9-Joint-Orientation-Estimation\Matlab\Datasets\single_listener_dataset.h5"
# H5_PATH = fr"D:\Python Projects\P9-Joint-Orientation-Estimation\Matlab\Datasets\dummy_listener_dataset.h5"
# H5_PATH = fr"D:\Python Projects\P9-Joint-Orientation-Estimation\Matlab\Datasets\multiple_listener_dataset.h5"
H5_PATH = fr"D:\Python Projects\P9-Joint-Orientation-Estimation\Matlab\Datasets\single_talker_dataset_600.h5"

scenario = 'scenario_1'
scenario_folder = fr'Python\Scenario Results\{scenario}'
os.makedirs(scenario_folder, exist_ok=True)

# BATCH_SIZE = 32
# EPOCHS = 16
# LEARNING_RATE = 1e-4
# SEED = 154
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-4
SEED = 154

save_plots = False
save_checkpoints = False

torch.manual_seed(SEED)
np.random.seed(SEED)

# Set to the actual dataset sampling frequency in Hz.
# Example: FS = 48000
# Use None to represent ITD in samples instead.
FS = 48000 # we should pick this value straight from thedataset instead

# Maximum cross-correlation lag in samples.
# 20 means lags from -20 to +20, giving 41 features.
MAX_LAG = 5

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# Load HDF5 dataset
# --------------------------------------------------

with h5py.File(H5_PATH, "r") as f:

    X = np.asarray(
        f["signals"],
        dtype=np.float32
    )

    angles = np.asarray(
        f["orientation_rad"],
        dtype=np.float32
    ).reshape(-1)


print("Original signal shape:", X.shape)
print("Angle shape:", angles.shape)

N = len(angles)


# --------------------------------------------------
# Convert signals to (N, 2, T)
# --------------------------------------------------

# N = examples
# 2 = ears
# T = time samples

if X.shape[0] == N and X.shape[1] == 2:
    pass
elif X.shape[0] == N and X.shape[2] == 2:
    X = X.transpose(0, 2, 1)
elif X.shape[2] == N and X.shape[1] == 2:
    X = X.transpose(2, 1, 0)
else:
    raise ValueError(
        f"Unexpected signal shape {X.shape}; "
        f"expected {N} examples and 2 channels."
    )


print("Prepared signal shape:", X.shape)


# --------------------------------------------------
# Normalize both ears with the same scale
# --------------------------------------------------

scale = np.max(
    np.abs(X),
    axis=(1, 2),
    keepdims=True
)

X = X / np.maximum(scale, 1e-8)


# --------------------------------------------------
# Feature extraction
# --------------------------------------------------

def extract_features(signals, max_lag=20, fs=None):
    """
    Extract binaural features from signals.

    Parameters
    ----------
    signals : ndarray, shape (N, 2, T)
        Binaural signals.

    max_lag : int
        Maximum absolute lag for cross-correlation.

    fs : float or None
        Sampling frequency in Hz.
        If provided, ITD is returned in seconds.
        Otherwise, ITD is returned in samples.

    Returns
    -------
    features : ndarray, shape (N, 2 + 2*max_lag + 1)
        Features:
            1 ITD
            1 ILD
            2*max_lag + 1 cross-correlation values
    """

    features = []

    eps = 1e-8

    for i in range(len(signals)):

        left = signals[i, 0]
        right = signals[i, 1]

        # ------------------------------------------
        # 1. Interaural Level Difference (ILD)
        # ------------------------------------------

        rms_left = np.sqrt(
            np.mean(left ** 2)
        )

        rms_right = np.sqrt(
            np.mean(right ** 2)
        )

        ild = 20.0 * np.log10(
            (rms_left + eps) /
            (rms_right + eps)
        )

        # ------------------------------------------
        # 2. Normalized cross-correlation
        # ------------------------------------------

        # Correlation at lags:
        # -max_lag, ..., 0, ..., +max_lag

        lags = np.arange(
            -max_lag,
            max_lag + 1
        )

        corr_values = []

        for lag in lags:

            if lag > 0:

                x_left = left[:-lag]
                x_right = right[lag:]

            elif lag < 0:

                x_left = left[-lag:]
                x_right = right[:lag]

            else:

                x_left = left
                x_right = right

            # Normalized correlation.
            numerator = np.sum(
                x_left * x_right
            )

            denominator = np.sqrt(
                np.sum(x_left ** 2) *
                np.sum(x_right ** 2)
            )

            corr = numerator / (
                denominator + eps
            )

            corr_values.append(corr)

        corr_values = np.asarray(
            corr_values,
            dtype=np.float32
        )

        # ------------------------------------------
        # 3. Interaural Time Difference (ITD)
        # ------------------------------------------

        # Select the lag with maximum correlation.
        best_idx = np.argmax(corr_values)

        itd_samples = lags[best_idx]

        if fs is not None:

            itd = itd_samples / fs

        else:

            itd = itd_samples

        # ------------------------------------------
        # Combine features
        # ------------------------------------------

        example_features = np.concatenate([
            np.asarray([itd, ild], dtype=np.float32),
            corr_values
        ])

        features.append(example_features)

    return np.asarray(
        features,
        dtype=np.float32
    )


print("\nExtracting binaural features...")

start_time = time.perf_counter()

X_features = extract_features(
    X,
    max_lag=MAX_LAG,
    fs=FS
)

feature_time = time.perf_counter() - start_time

print("Feature extraction complete.")
print("Feature shape:", X_features.shape)
print(f"Feature extraction time: {feature_time:.2f} s")


# Feature names for interpretation.
itd_unit = "seconds" if FS is not None else "samples"

feature_names = (
    [f"ITD ({itd_unit})", "ILD (dB)"] +
    [
        f"Cross-correlation lag {lag}"
        for lag in range(-MAX_LAG, MAX_LAG + 1)
    ]
)

print("Number of input features:", len(feature_names))


# --------------------------------------------------
# Prepare target: sin(theta), cos(theta)
# --------------------------------------------------

Y = np.stack(
    [
        np.sin(angles),
        np.cos(angles)
    ],
    axis=1
).astype(np.float32)


# --------------------------------------------------
# Train / validation / test split
# --------------------------------------------------

indices = np.arange(N)

train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.30,
    random_state=SEED
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=SEED
)


# --------------------------------------------------
# Standardize features
# --------------------------------------------------

# Fit the scaler ONLY on training data.
# This prevents information leakage from the
# validation and test sets.

feature_mean = X_features[train_idx].mean(
    axis=0,
    keepdims=True
)

feature_std = X_features[train_idx].std(
    axis=0,
    keepdims=True
)

feature_std = np.maximum(
    feature_std,
    1e-6
)

X_features = (
    X_features - feature_mean
) / feature_std


def make_loader(idx, shuffle):

    x = torch.from_numpy(
        X_features[idx]
    )

    y = torch.from_numpy(
        Y[idx]
    )

    return DataLoader(
        TensorDataset(x, y),
        batch_size=BATCH_SIZE,
        shuffle=shuffle
    )


train_loader = make_loader(
    train_idx, True
)

val_loader = make_loader(
    val_idx, False
)

test_loader = make_loader(
    test_idx, False
)


print(
    f"Train: {len(train_idx)}, "
    f"Validation: {len(val_idx)}, "
    f"Test: {len(test_idx)}"
)


# --------------------------------------------------
# MLP model
# --------------------------------------------------

class OrientationMLP(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_size, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.network(x)


model = OrientationMLP(
    input_size=X_features.shape[1]
).to(device)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

criterion = nn.MSELoss()


print("\nMLP architecture:")
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

            loss = criterion(
                pred,
                y
            )

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

print("\nStarted training")

for epoch in range(EPOCHS):

    train_loss = run_epoch(
        train_loader,
        True
    )

    val_loss = run_epoch(
        val_loader,
        False
    )

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


training_time = time.perf_counter() - start_time

print(
    f"Finished training in {training_time:.2f} seconds"
)


model.load_state_dict(best_state)


# --------------------------------------------------
# Test-set evaluation
# --------------------------------------------------

model.eval()

predictions = []
targets = []

with torch.no_grad():

    for x, y in test_loader:

        pred = model(
            x.to(device)
        ).cpu().numpy()

        predictions.append(pred)
        targets.append(y.numpy())


predictions = np.concatenate(
    predictions
)

targets = np.concatenate(
    targets
)


# --------------------------------------------------
# Convert sin/cos to angles
# --------------------------------------------------

pred_angles = np.arctan2(
    predictions[:, 0],
    predictions[:, 1]
)

true_angles = np.arctan2(
    targets[:, 0],
    targets[:, 1]
)


# Circular error in degrees, in [-180, 180].

error = np.rad2deg(
    np.arctan2(
        np.sin(pred_angles - true_angles),
        np.cos(pred_angles - true_angles)
    )
)

abs_error = np.abs(error)


print("\nTest results")

print(
    f"Mean absolute angular error: "
    f"{abs_error.mean():.2f} deg"
)

print(
    f"Median angular error: "
    f"{np.median(abs_error):.2f} deg"
)

print(
    f"90th percentile error: "
    f"{np.percentile(abs_error, 90):.2f} deg"
)


# --------------------------------------------------
# Save model and feature preprocessing
# --------------------------------------------------

if save_checkpoints:

    checkpoint = {

        "model_state_dict": model.state_dict(),

        "feature_mean": feature_mean,
        "feature_std": feature_std,

        "max_lag": MAX_LAG,
        "fs": FS,

        "feature_names": feature_names
    }

    torch.save(
        checkpoint,
        os.path.join(
            scenario_folder,
            "listener_orientation_mlp.pt"
        )
    )


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
plt.title("Listener orientation estimation (MLP)")
plt.grid()
plt.tight_layout()
if save_plots:
    plt.savefig(fr'{scenario_folder}\angular_prediction_error.png')

plt.figure()
plt.hist(abs_error, bins=36)
plt.xlabel("Absolute angular error (degrees)")
plt.ylabel("Number of examples")
plt.title("Test-set angular error (MLP)")
plt.grid()
plt.tight_layout()
if save_plots:
    plt.savefig(fr'{scenario_folder}\abs_angular_error.png')

plt.show()