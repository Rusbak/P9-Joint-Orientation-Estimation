import h5py
import numpy as np
import matplotlib.pyplot as plt

H5_PATH = fr"D:\Python Projects\P9-Joint-Orientation-Estimation\Matlab\Datasets\binaural_listener_dataset.h5"

with h5py.File(H5_PATH, "r") as f:
    X = np.asarray(f["signals"], dtype=np.float32)

    angles = np.asarray(f["orientation_rad"], dtype=np.float32).reshape(-1)

print("Original signal shape:", X.shape)
print("Angle shape:", angles.shape)

Y = np.stack(
    [np.sin(angles), np.cos(angles)], axis=1
).astype(np.float32)

plt.figure()
plt.scatter(
    np.rad2deg(angles),
    np.zeros(len(angles)),
    alpha=0.5
)
limits = [-180, 180]
# plt.plot(limits, limits, "r--")
plt.xlim(limits)
plt.ylim(limits)
plt.grid()
plt.tight_layout()
plt.show()