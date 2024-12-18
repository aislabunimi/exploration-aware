import os
import shutil


def create_new_structure(base_dirs, destination_dir):
    """
    Reorganize files based on the env and run extracted from filenames and create a new directory structure.
    
    Parameters:
        base_dirs (dict): A dictionary containing the source directories for explored and non-explored files.
        destination_dir (str): The path of the destination directory where the new structure will be created.
    """
    all_categories = ["EXPLORED_E", "EXPLORED_NE", "NOT_EXPLORED_E", "NOT_EXPLORED_NE"]

    for base_category, base_path in base_dirs.items():
        for root, _, files in os.walk(base_path):
            for file in files:
                if '@' in file:
                    # Split the file name into its components
                    env, run, filename = file.split('@')

                    # Create the target directory path for the environment and run
                    base_target_dir = os.path.join(destination_dir, env, run)

                    # Create all the label folders inside the run directory (even if empty)
                    for category in all_categories:
                        target_dir = os.path.join(base_target_dir, category)
                        os.makedirs(target_dir, exist_ok=True)

                    # Copy the file to the corresponding folder
                    target_dir = os.path.join(base_target_dir, base_category)
                    source_file = os.path.join(root, file)
                    destination_file = os.path.join(target_dir, file)
                    shutil.copy2(source_file, destination_file)
                    print(f"Copied {file} to {destination_file}")


if __name__ == "__main__":
    # Define the base directories for explored and non-explored files
    base_dirs = {
        "EXPLORED_E": "Results/GradCAM_predictions/KTH+MIT/EXPLORED_E",
        "EXPLORED_NE": "Results/GradCAM_predictions/KTH+MIT/EXPLORED_NE",
        "NOT_EXPLORED_E": "Results/GradCAM_predictions/KTH+MIT/NOT_EXPLORED_E",
        "NOT_EXPLORED_NE": "Results/GradCAM_predictions/KTH+MIT/NOT_EXPLORED_NE"
    }

    # Define the destination directory where the new structure will be created
    destination_dir = "Results/GradCAM_predictions/labels_for_metrics_KTH+MIT"

    # Call the function to create the new structure
    create_new_structure(base_dirs, destination_dir)
