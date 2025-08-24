<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

# Alternative backbones for map completeness estimation

Alternative feature extractors for predicting map completeness during exploration. The tested backbones are:
- ResNet18
- Swin-V2-T
- ViT-T
- ViT-B
  
## Setup and installation

After cloning the repo, initialize a virtual environment and configure the package from `pyproject.toml` using
```bash
$ pip install -e .
```

**Alternatively**, there is a `make` target for environment setup and dependency installation:

```bash
$ make create_environment 
```

The dataset paths and output folders should be set in the `src/config.py` file.

## Usage
### Data preprocessing
Place the images and the corresponding CSV files in the relative folders, then run 
```bash
$ python3 src/script/csv_from_raw_data.py
``` 
to store and preprocess the data in the format expected by our custom Dataset.

### Hyperparameter optimization
In `hyperopt.py` specify the models for optimization, then run 
```bash
$ python3 src/script/hyperopt.py
```
The script optimizes the specified backbones and stores the optimization history under the `studies/` folder.