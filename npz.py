import numpy as np
import os
folder_path = '../data/intrinsics'

# Get a list of file names in the folder
file_names = os.listdir(folder_path)
print(file_names)

# Initialize a dictionary to store the arrays and their corresponding names
arrays_dict = {}

# Loop through each file in the folder and load the data into arrays
for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)
    array_data = np.fromfile(file_path, dtype=np.float32)
    arrays_dict[file_name] = array_data

# Save all arrays into a single NPZ file
output_file = "/Users/khushboo/Desktop/AGV/data"
np.savez(output_file, **arrays_dict)

print("Folder converted to NPZ file successfully.")
