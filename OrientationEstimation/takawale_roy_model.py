import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
import os

from helper_scripts.model_architectures import TakawaleRoyDNN
from dataset_takawale_roy import TakawaleRoyDataset
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

print('Imported Libraries!')

# --------------------------------------------------
# Configuration
# --------------------------------------------------

dataset_folder_path = fr"D:\Python Projects\P9-Joint-Orientation-Estimation\DataGeneration\Datasets"
# dataset_type = 'combined_dataset_20'
dataset_type = 'combined_dataset_commands'
dataset_path = fr'{dataset_folder_path}\{dataset_type}.h5'

scenario = 'scenario_9'
scenario_folder = fr'Python\Scenario Results\{scenario}'
os.makedirs(scenario_folder, exist_ok=True)

BATCH_SIZE = 8 # increase to 32 or 64 on ai-lab
EPOCHS = 5
LEARNING_RATE = 1e-4
SEED = 154

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

with h5py.File(dataset_path, "r") as f:
    num_samples = int(f.attrs["num_samples"].item())
    fs = int(f.attrs["sampling_rate_hz"].item())

print(f'Dataset "{dataset_type}" has:\n\tSamples: {num_samples}\n\tSampling Rate: {fs}')

# --------------------------------------------------
# Train / validation / test split
# --------------------------------------------------

train_dataset = TakawaleRoyDataset(dataset_path, "train", fs)
val_dataset = TakawaleRoyDataset(dataset_path, "val", fs)
test_dataset = TakawaleRoyDataset(dataset_path, "test", fs)

print('Created Datasets!')

# --------------------------------------------------
# Make data loader
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True, # True for training
    num_workers=0, # windows does not like workers in my experience
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print('Created Dataloaders!')

# --------------------------------------------------
# Standardize features
# --------------------------------------------------

def calculate_feature_statistics(dataloader):
    feature_sum = None
    feature_sq_sum = None
    num_values = 0

    for features, _ in dataloader:
        features = features.double()

        if feature_sum is None:
            feature_sum = torch.zeros(
                features.shape[1], dtype=torch.float64
            )
            feature_sq_sum = torch.zeros_like(feature_sum)

        feature_sum += features.sum(dim=(0, 2))
        feature_sq_sum += (features ** 2).sum(dim=(0, 2))

        num_values += features.shape[0] * features.shape[2]

    mean = feature_sum / num_values

    variance = feature_sq_sum / num_values - mean ** 2
    std = torch.sqrt(variance.clamp_min(0))

    # Avoid division by zero
    std = std.clamp_min(1e-8)

    return mean.float(), std.float()

feature_mean, feature_std = calculate_feature_statistics(train_loader)

print('Calculated Feature Statistics!')

# --------------------------------------------------
# Make the model
# --------------------------------------------------

model = TakawaleRoyDNN().to(device)

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

def run_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    feature_mean,
    feature_std,
    train=True
):

    model.train() if train else model.eval()

    total_loss = 0.0
    num_samples = 0

    for features, targets in dataloader:

        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Standardize features
        mean = feature_mean.to(device)
        std = feature_std.to(device)

        features = (
            features - mean[None, :, None]
        ) / std[None, :, None]

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):

            predictions = model(features)

            loss = criterion(predictions, targets)

            if train:
                loss.backward()
                optimizer.step()

        batch_size = features.size(0)

        total_loss += loss.item() * batch_size
        num_samples += batch_size

    return total_loss / num_samples

best_val_loss = float("inf")
best_state = None

loss_history = {
    "train_loss": [],
    "val_loss": []
}

start_time = time.perf_counter()
print("\nStarted training")

for epoch in range(EPOCHS):

    train_loss = run_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        device,
        feature_mean,
        feature_std,
        train=True
    )

    val_loss = run_epoch(
        model,
        val_loader,
        optimizer,
        criterion,
        device,
        feature_mean,
        feature_std,
        train=False
    )

    loss_history["train_loss"].append(train_loss)
    loss_history["val_loss"].append(val_loss)

    if (epoch + 1) % 1 == 0 or epoch == 0:
        print(
            f"Epoch [{epoch+1:3d}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss

        best_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

        if save_checkpoints:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "feature_mean": feature_mean,
                "feature_std": feature_std
            }, "best_takawale_roy.pt")

training_time = time.perf_counter() - start_time
print(f"Finished training in {training_time:.2f} seconds")

# --------------------------------------------------
# Test-set evaluation
# --------------------------------------------------

def test_model(
    model,
    dataloader,
    device,
    feature_mean,
    feature_std
):
    model.eval()

    listener_errors = []
    talker_errors = []

    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device)

            mean = feature_mean.to(device)
            std = feature_std.to(device)

            features = (
                features - mean[None, :, None]
            ) / std[None, :, None]

            predictions = model(features)

            # Convert predictions to angles
            listener_pred = torch.atan2(
                predictions[:, 0],
                predictions[:, 1]
            )

            talker_pred = torch.atan2(
                predictions[:, 2],
                predictions[:, 3]
            )

            # Convert targets to angles
            listener_true = torch.atan2(
                targets[:, 0],
                targets[:, 1]
            )

            talker_true = torch.atan2(
                targets[:, 2],
                targets[:, 3]
            )

            # Circular angular errors
            listener_error = torch.atan2(
                torch.sin(listener_pred - listener_true),
                torch.cos(listener_pred - listener_true)
            ).abs()

            talker_error = torch.atan2(
                torch.sin(talker_pred - talker_true),
                torch.cos(talker_pred - talker_true)
            ).abs()

            listener_errors.extend(listener_error.cpu().numpy())
            talker_errors.extend(talker_error.cpu().numpy())

    listener_mae = np.mean(listener_errors)
    talker_mae = np.mean(talker_errors)

    print(f"Listener MAE: {np.degrees(listener_mae):.3f} degrees")
    print(f"Talker MAE:   {np.degrees(talker_mae):.3f} degrees")

    return listener_mae, talker_mae

# load the best checkpoint
if save_checkpoints:
    checkpoint = torch.load(
        "best_takawale_roy.pt",
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    feature_mean = checkpoint["feature_mean"]
    feature_std = checkpoint["feature_std"]
else:
    model.load_state_dict(best_state)

print('Loaded Best Model!')

listener_mae, talker_mae = test_model(
    model,
    test_loader,
    device,
    feature_mean,
    feature_std
)

''' error for both the listener and the talker
print("\nTest results")
print(
    f"Mean absolute angular error: "
    f"{abs_error.mean():.2f} deg"
)
print(
    f"Mean signed error: "
    f"{error.mean():.2f} deg"
)
print(
    f"Median angular error: "
    f"{np.median(abs_error):.2f} deg"
)
print(
    f"90th percentile error: "
    f"{np.percentile(abs_error, 90):.2f} deg"
)
'''

# --------------------------------------------------
# Plots
# --------------------------------------------------

# we might need other plots, since many prediction gives many dots
# also we are predicting both listener and talker orientation now

plt.figure()
plt.plot(loss_history['train_loss'], label="Train")
plt.plot(loss_history["val_loss"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.title("Loss graph")
plt.legend()
plt.grid()
plt.tight_layout()
if save_plots:
    plt.savefig(fr'{scenario_folder}\loss_graph.png')

