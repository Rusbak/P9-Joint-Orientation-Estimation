import torch
import torch.nn as nn

# 1D-CNN for raw signal
class OneDimensionalCNN(nn.Module):
    '''One-dimensional CNN for raw signals'''

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(
                in_channels=2, 
                out_channels=16, 
                kernel_size=15, 
                padding=7
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),

            # Block 2
            nn.Conv1d(
                in_channels=16, 
                out_channels=32, 
                kernel_size=9, 
                padding=4
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),

            # Block 3
            nn.Conv1d(
                in_channels=32, 
                out_channels=64, 
                kernel_size=9, 
                padding=4
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.head = nn.Sequential(
            nn.Flatten(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.head(self.features(x))
    

# 2D-CNN for spectrograms
# use for:
# log-magnitude STFT spectrogram (main baseline)
# linear magnitude STFT stectrogram (useful comparison)
# mel spectrogram (optional experiment)
# the study in related work used log-mel spectrogram
class TwoDimensionalCNN(nn.Module):
    """Two-dimensional CNN for spectrograms"""

    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 2
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 3
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 4
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Make the output independent of input size
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


# MLP for pure feature inputs
class FeatureMLP(nn.Module):
    '''MLP for computed features'''

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

# combined CNN-MLP for combined signal and features
class FusionModel(nn.Module):
    '''Combined model with two branches: 1D-CNN and MLP'''

    def __init__(self, input_size):
        super().__init__()

        # Make each of the models
        self.cnn = OneDimensionalCNN()
        self.mlp = FeatureMLP(
            input_size=input_size
        )

        # Remove prediction heads
        self.cnn.head = self.cnn.head[:-1]
        self.mlp.network = self.mlp.network[:-1]

        # Shared prediction head
        cnn_embeddings = self.cnn.head[-2].out_features
        mlp_embeddings = self.mlp.network[-2].out_features

        self.fusion_head = nn.Sequential(
            nn.Linear(cnn_embeddings + mlp_embeddings, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 2)
        )

    def forward(self, raw_signal, computed_features):
        # -=- CNN branch -=- #
        cnn_embedding = self.cnn.features(raw_signal)
        cnn_embedding = self.cnn.head(
            cnn_embedding
        )

        # -=- MLP branch -=- #
        mlp_embedding = self.mlp.network(
            computed_features
        )

        # -=- Feature fusion -=- #
        combined_embedding = torch.cat(
            [
                cnn_embedding,
                mlp_embedding
            ],
            dim=1
        )

        # -=- Final prediction -=- #
        output = self.fusion_head(
            combined_embedding
        )

        return output


class TakawaleRoyDNN(nn.Module):
    '''
    This model is supposed to resemble the DNN used by Takawale and Roy.

    The model takes in 5 inputs: ILD_low, ILD_high, ITD_low, ITD_high and ratio of energies(ILD_low * 1/ILD_low). * = convolution.

    Each input is a vector or values computed from a moving window (STFT).

    The model then outputs 4 predicted estimates: sin(theta) and cos(theat) for both the speaker and listener orietations.
    '''

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1 + 2
            nn.Conv1d(
                in_channels=5, 
                out_channels=128, 
                kernel_size=5, 
                padding=2
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=128, 
                out_channels=128, 
                kernel_size=5, 
                padding=2
            ),
            nn.ReLU(),

            # Block 3 + 4
            nn.Conv1d(
                in_channels=128, 
                out_channels=256, 
                kernel_size=5, 
                padding=2
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=256, 
                out_channels=256, 
                kernel_size=5, 
                padding=2
            ),
            nn.ReLU(),

            # Block 5 + 6
            nn.Conv1d(
                in_channels=256, 
                out_channels=512, 
                kernel_size=5, 
                padding=2
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=512, 
                out_channels=512, 
                kernel_size=5, 
                padding=2
            ),
            nn.ReLU(),

            # Block 7 + 8
            nn.Conv1d(
                in_channels=512, 
                out_channels=1024, 
                kernel_size=5, 
                padding=2
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=1024, 
                out_channels=1024, 
                kernel_size=5, 
                padding=2
            ),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.head = nn.Sequential( # they use dropout regularization before the fc layers
            nn.Flatten(),
            nn.Dropout1d(p=0.5),

            nn.Linear(1024, 2048),
            nn.ReLU(),

            nn.Linear(2048, 1024),
            nn.ReLU(),

            nn.Linear(1024, 4),
        )
