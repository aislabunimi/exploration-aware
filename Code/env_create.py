import os
import json

# Define the path to the main folder
main_folder_path = '/media/aislab/EXTERNAL_USB/slampbench/runs/training/gmapping/realistic'

# Initialize an empty dictionary
subdirectories_dict = {}

# Loop through the items in the folder
for item in os.listdir(main_folder_path):
    item_path = os.path.join(main_folder_path, item)
    if os.path.isdir(item_path):  # Check if the item is a directory
        subdirectories_dict[item] = "large"
new_arr = []
for key, value in subdirectories_dict.items():
    new_arr.append(str(key))

# Define the path to the output txt file
output_file_path = 'subdirectories_mit_kth_old.txt'

# Write the dictionary to the txt file
with open(output_file_path, 'w') as file:
    file.write(json.dumps(subdirectories_dict, indent=4))

print(f"Subdirectory names have been saved to {output_file_path}")
print(new_arr)
