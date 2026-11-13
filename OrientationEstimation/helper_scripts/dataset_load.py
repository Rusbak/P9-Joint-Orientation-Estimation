import h5py
import numpy as np

def load_dataset(path, requests):
    data = {}

    with h5py.File(path, "r") as file:
        for request in requests:
            dataset = file[request]

            # Retrieve the dataset
            if h5py.check_string_dtype(dataset.dtype) is not None:
                entry = dataset.asstr()[()]
            else:
                entry = dataset[()]

            type_converted_entry = _convert_entry_to_type(entry, request)

            # Store the request in dictionary
            data[request] = type_converted_entry

    return data

# Dataset entry types
orientations = ['rel_deg', 'rel_rad', 'abs_deg', 'abs_rad']
float_arrays = ['signals', 'class_center']
int_arrays = ['class_index']
str_arrays = ['names', 'origins']
floats = ['duration', 'spacing', 'distance', 'elevation']
ints = ['hz', 'num']
strs = ['mode', 'type', 'convention']

def _convert_entry_to_type(entry, entry_name):
    entry_name = entry_name.lower()

    if any(label in entry_name for label in orientations):
        entry = np.asarray(
            entry,
            dtype=np.float32
        ).reshape(-1)

    elif any(label in entry_name for label in float_arrays):
        entry = np.asarray(
            entry,
            dtype=np.float32
        )

    elif any(label in entry_name for label in int_arrays):
        entry = np.asarray(
            entry,
            dtype=np.uint16
        )

    elif any(label in entry_name for label in str_arrays):
        entry = np.asarray(
            entry,
            dtype=np.str_
        )

    elif any(label in entry_name for label in floats):
        entry = float(entry)

    elif any(label in entry_name for label in ints):
        entry = int(entry)

    elif any(label in entry_name for label in strs):
        entry = str(entry)

    return entry

def display_entries(path):
    entries = []

    with h5py.File(path, "r") as file:
        def collect_entries(name, obj):
            if isinstance(obj, h5py.Dataset):
                entries.append({
                    "name": name,
                    "dtype": str(obj.dtype),
                    "shape": str(obj.shape),
                    "size": obj.size
                })

        file.visititems(collect_entries)

    # Print table
    headers = ["Entry", "Datatype", "Shape", "Elements"]

    widths = [
        max(len(headers[0]), max(len(e["name"]) for e in entries)),
        max(len(headers[1]), max(len(e["dtype"]) for e in entries)),
        max(len(headers[2]), max(len(e["shape"]) for e in entries)),
        max(len(headers[3]), max(len(str(e["size"])) for e in entries))
    ]

    row_format = (
        f"{{:<{widths[0]}}}  "
        f"{{:<{widths[1]}}}  "
        f"{{:<{widths[2]}}}  "
        f"{{:>{widths[3]}}}"
    )

    print(row_format.format(*headers))
    print("-" * (sum(widths) + 6))

    for entry in entries:
        print(row_format.format(
            entry["name"],
            entry["dtype"],
            entry["shape"],
            entry["size"]
        ))
