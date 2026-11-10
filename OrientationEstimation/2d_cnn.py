
import numpy as np
import torch
import torch.nn as nn
import torchaudio

from torch.utils.data import Dataset, DataLoader


# ============================================================
# Configuration
# ============================================================

FS = 48000

N_FFT = 1024
WIN_LENGTH = 1024
HOP_LENGTH = 256
N_MELS = 128

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 50

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Log-mel spectrogram transformation
# ============================================================

class BinauralLogMelTransform(nn.Module):

    def __init__(self):

        super().__init__()

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=FS,
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            window_fn=torch.hann_window,
            n_mels=N_MELS,
            f_min=0.0,
            f_max=FS / 2,
            power=2.0,
            center=True
        )

        self.to_db = torchaudio.transforms.AmplitudeToDB(
            stype="power",
            top_db=80
        )

    def forward(self, signal):

        # signal shape: (2, samples)

        mel = self.mel_spectrogram(signal)

        # Convert power spectrogram to dB
        log_mel = self.to_db(mel)

        # Output shape: (2, n_mels, time_frames)

        return log_mel


# ============================================================
# Dataset
# ============================================================

class BinauralOrientationDataset(Dataset):

    def __init__(
        self,
        signals,
        angles,
        angle_classes,
        transform=None
    ):

        self.signals = signals
        self.angles = np.asarray(angles)

        self.angle_classes = np.asarray(angle_classes)

        self.transform = transform

        # Map each angle to a class index
        self.angle_to_class = {
            angle: index
            for index, angle in enumerate(self.angle_classes)
        }

        self.labels = np.array([
            self.angle_to_class[angle]
            for angle in self.angles
        ])

    def __len__(self):

        return len(self.signals)

    def __getitem__(self, index):

        signal = self.signals[index]

        # Expected signal shape: (2, samples)
        signal = torch.as_tensor(
            signal,
            dtype=torch.float32
        )

        if self.transform is not None:

            signal = self.transform(signal)

        label = torch.tensor(
            self.labels[index],
            dtype=torch.long
        )

        angle = torch.tensor(
            self.angles[index],
            dtype=torch.float32
        )

        return signal, label, angle


# ============================================================
# 2D CNN
# ============================================================

class BinauralSpectrogramCNN(nn.Module):

    def __init__(
        self,
        num_classes,
        dropout=0.3
    ):

        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(
                2, 32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(
                128, 256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# Training
# ============================================================

def train_model(
    model,
    train_loader,
    val_loader,
    epochs=EPOCHS
):

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(epochs):

        # -------------------------
        # Training
        # -------------------------

        model.train()

        train_loss = 0.0
        correct = 0
        total = 0

        for spectrograms, labels, _ in train_loader:

            spectrograms = spectrograms.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(spectrograms)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        train_accuracy = correct / total

        # -------------------------
        # Validation
        # -------------------------

        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for spectrograms, labels, _ in val_loader:

                spectrograms = spectrograms.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(spectrograms)

                loss = criterion(outputs, labels)

                val_loss += loss.item()

                predictions = outputs.argmax(dim=1)

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += labels.size(0)

        val_accuracy = val_correct / val_total

        print(
            f"Epoch {epoch + 1:03d}/{epochs} | "
            f"Train Loss: {train_loss / len(train_loader):.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss / len(val_loader):.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

    return model


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(
    model,
    test_loader,
    angle_classes
):

    model.eval()

    all_predictions = []
    all_angles = []

    with torch.no_grad():

        for spectrograms, _, angles in test_loader:

            spectrograms = spectrograms.to(DEVICE)

            outputs = model(spectrograms)

            predictions = outputs.argmax(dim=1)

            predicted_angles = np.asarray(
                angle_classes
            )[predictions.cpu().numpy()]

            all_predictions.extend(predicted_angles)

            all_angles.extend(
                angles.numpy()
            )

    all_predictions = np.asarray(all_predictions)
    all_angles = np.asarray(all_angles)

    # Circular angular error
    errors = (
        all_predictions - all_angles + 180
    ) % 360 - 180

    absolute_errors = np.abs(errors)

    accuracy = np.mean(
        all_predictions == all_angles
    )

    mean_error = np.mean(absolute_errors)

    median_error = np.median(absolute_errors)

    percentile_90 = np.percentile(
        absolute_errors,
        90
    )

    print("\nTest Results")

    print(f"Accuracy: {accuracy:.4f}")

    print(f"Mean Angular Error: {mean_error:.2f} degrees")

    print(f"Median Angular Error: {median_error:.2f} degrees")

    print(f"90th Percentile Error: {percentile_90:.2f} degrees")

    return all_predictions, all_angles



# ================================================


# ------------------------------------------------------------
# Data
# ------------------------------------------------------------

# Replace these with your actual datasets
#
# train_signals, train_angles
# val_signals, val_angles
# test_signals, test_angles


# ------------------------------------------------------------
# Transform
# ------------------------------------------------------------

transform = BinauralLogMelTransform()


# ------------------------------------------------------------
# Datasets
# ------------------------------------------------------------

train_dataset = BinauralOrientationDataset(
    train_signals,
    train_angles,
    angle_classes,
    transform
)

val_dataset = BinauralOrientationDataset(
    val_signals,
    val_angles,
    angle_classes,
    transform
)

test_dataset = BinauralOrientationDataset(
    test_signals,
    test_angles,
    angle_classes,
    transform
)


# ------------------------------------------------------------
# DataLoaders
# ------------------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

model = BinauralSpectrogramCNN(
    num_classes=len(angle_classes)
).to(DEVICE)


# ------------------------------------------------------------
# Train
# ------------------------------------------------------------

model = train_model(
    model,
    train_loader,
    val_loader
)


# ------------------------------------------------------------
# Evaluate
# ------------------------------------------------------------

predictions, true_angles = evaluate_model(
    model,
    test_loader,
    angle_classes
)