
# Dataset Labelling and Classification Pipeline

This repository contains scripts for dataset labelling, classification using CNN with EfficientNet and Grad-CAM, and regression. The instructions below will guide you through setting up the environment, labelling datasets, running classification and regression models, and gathering results.

## Setup and Installation

### 1. Cloning the Repository
First, clone this repository to your local machine using:
```bash
git clone https://github.com/aislabunimi/robot-aware-exploration.git
cd robot-aware-exploration
```

### 2. Installing Dependencies

You can install the required dependencies using either `pip` or `conda`. Follow one of the methods below:

#### Method 1: Using `pip`
```bash
conda create --name new_env_name python=3.x
conda activate new_env_name
pip install -r requirements.txt
```

#### Method 2: Using `conda`
```bash
conda env create -f environment.yml
conda activate your_env_name
```

## Dataset Labelling

### 3. Connecting Your Dataset
To connect your dataset, modify the path in `label_dataset.py`:
```python
dataset_dir = 'your path to dataset'
```

### 4. Changing Test Environment List
If you need to modify the test environments, update the `test_env_list` in `label_dataset.py` accordingly:
```python
test_env_list = ['your', 'test', 'environments']
```

### 5. Running the Labelling Script

Before running the labelling script, make sure to set the path to your dataset in the `env_create` script by updating the `main_folder_path` variable:
```python
main_folder_path = 'your/dataset/path'
```

Then, run the labelling script:
```bash
python label_dataset.py
```

After running this, labelled data will be available in the `Datasets/Labelled_Data` directory.

## Classification

### 6. Running the Classification Model

To run the classification model, use the Jupyter notebook `CNN_EfficientNet_GradCAM.ipynb`:
```bash
jupyter notebook CNN_EfficientNet_GradCAM.ipynb
```

This notebook will train a CNN model using EfficientNet architecture with Grad-CAM visualization.

### 7. Metrics Collection

Once the model has finished training, run the following scripts to gather and restructure the metrics:
```bash
python restructure_files_for_metrics.py
python get_saved_time.py
```

- **Final classification metrics** will be saved in `Results/Confusion_Matrices`.
- **Saved time metrics** will be available in `Results/Saved_time_offline`.

## Regression

### 8. Running the Regression Model

To run the regression model, use the Jupyter notebook `CNN_EfficientNet_regression.ipynb`:
```bash
jupyter notebook CNN_EfficientNet_regression.ipynb
```

Once completed, the final regression metrics will be stored in `Results/regression_results.csv`.

## Directory Structure

Below is a summary of the key directories and where outputs will be saved:

- **Datasets/Labelled_Data**: Contains the labelled dataset after running `label_dataset.py`.
- **Results/Confusion_Matrices**: Contains the final classification metrics.
- **Results/Saved_time_offline**: Stores metrics on saved time after classification.
- **Results/regression_results.csv**: Stores the final regression metrics.

## Summary of Commands

1. Install dependencies:
    - With `pip`: 
      ```bash
      conda create --name new_env_name python=3.x
      conda activate new_env_name
      pip install -r requirements.txt
      ```
    - Or with `conda`: 
      ```bash
      conda env create -f environment.yml
      conda activate your_env_name
      ```

2. Set dataset path:
    - Update `dataset_dir` in `label_dataset.py`.
    - Update `main_folder_path` in `env_create`.

3. Run labelling script:
    ```bash
    python label_dataset.py
    ```

4. Run classification:
    ```bash
    jupyter notebook CNN_EfficientNet_GradCAM.ipynb
    ```

5. Collect metrics:
    ```bash
    python restructure_files_for_metrics.py
    python get_saved_time.py
    ```

6. Run regression:
    ```bash
    jupyter notebook CNN_EfficientNet_regression.ipynb
    ```
## Paper

```
@misc{luperto2024estimatingmapcompletenessrobot,
      title={Estimating Map Completeness in Robot Exploration}, 
      author={Matteo Luperto and Marco Maria Ferrara and Giacomo Boracchi and Francesco Amigoni},
      year={2024},
      eprint={2406.13482},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2406.13482}, 
}
```

## Acknowledgments

The authors want to acknowledge [Valerii Stakanov](https://github.com/are1ove) for its contribution to the experimental evalution of this work and for refactoring the code, and providing the documentation for future uses.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
