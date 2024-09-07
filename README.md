# Project Documentation

## Overview

This project provides a framework for training and evaluating text classification models. It supports both the SetFit and RoBERTa models and includes functionality for data preprocessing, model training, prediction, and result evaluation.

## Project Structure

The project directory is organized as follows:
- **config.yaml**: Configuration file for model training and evaluation.
- **config_roberta**: Configuration file for RoBERTa model training and evaluation.
- **config_setfit**: Configuration file for SetFit model training and evaluation.
- **data/**: Contains datasets
  -**.keep**: Placeholder file to ensure the `data` directory is included in version control.
- **data_processing/** scripts.
  - **dataset_utils.py**: Functions for loading and preprocessing datasets.
  - **text_processing.py**: Utility functions for handling text data, including label distribution printing.
  - **label_mapping.py**: Handles text column creation and validation.
- **model/**: Contains scripts for model training and prediction.
  - **model_training.py**: Functions for training SetFit and RoBERTa models.
  - **model_prediction.py**: Functions for making predictions with the trained models.
- **evaluation/**: Contains scripts for saving results and generating classification reports.
  - **result_saving.py**: Functions for saving results and generating classification reports.
- **output/**: Directory where trained models and results are saved.
  - **.keep**: Placeholder file to ensure the `output` directory is included in version control.
- **logs/**: Directory for logging information during training and evaluation.
  - **.keep**: Placeholder file to ensure the `logs` directory is included in version control.
- **main.py**: Main script to run the entire process, from data preprocessing to result saving.
- **requirements.txt**: List of required packages for the project.

## Configuration File: `config.yaml`

The `config.yaml` file allows you to configure various aspects of the model training and evaluation. Below is an explanation of each parameter:

- **base_model**: Specifies the model(s) to use for training. Can be a single model or a list of models. **Note:** This parameter is only used for the `setfit` model type.
- **output_path**: Directory where the trained models and results will be saved.
- **random_seed**: Seed for random number generators to ensure reproducibility.
- **train_path**: Path to the training dataset (CSV file).
- **test_path**: Path to the test dataset (CSV file).
- **just_predict**: If `True`, the script will skip training and only perform predictions with a pre-trained model. **Note:** This option only supports the `setfit` model type.
- **add_train**: Path to additional training data (CSV file) to merge with the primary training data.
- **add_labels**: List of labels to include from the additional training data.
- **merge_train**: If `True`, merge additional training data with the primary training data.
- **map_labels**: If `True`, apply label mapping as specified in `label_mapping`.
- **label_mapping**: Dictionary for mapping original labels to target labels (used if `map_labels` is `True`).
- **label_to_int**: Dictionary for mapping labels to integers and vice versa.
- **model_type**: Specifies the model type to use. Options are `"setfit"` and `"roberta"`.

### Example Configuration

```yaml
model_type: "roberta"
output_path: "output"
random_seed: 42
train_path: "data/nasa_train_sample.csv"
test_path: "data/nasa_test_sample.csv"
just_predict: False
label_to_int:
  bug: 0
  non-bug: 1
merge_train: False
add_train: "data/nlbse_23_train.csv"
add_labels: ["bug", "feature", "documentation", "question"]
map_labels: True
label_mapping:
  bug: bug
  documentation: non-bug
  feature: non-bug
  question: non-bug
```

## Usage

### Prerequisites
Python 3.11 or higher is required to run this project. You can check your Python version by running:

```bash
python --version
```
Wandb is used for logging and tracking model performance. To use Wandb, you will need to sign up for an account at [https://wandb.ai/](https://wandb.ai/). After signing up, you will receive an API key that you can use to authenticate your account.

The API key has to be stored in a file named `.env`.

An example `.env` file is shown below:

```YAML
WANDB_API_KEY=your-api-key
```

Add to the config.yaml file the following lines:

```YAML
wandb:
  project: "your-project-name"
  entity: "your-username"
  mode: "online" # or "offline"
```

If the mode is set to "offline", the logs will be saved locally in the logs directory.

You can then sync the runs to the Wandb server by running:

```bash
wandb sync --sync-all
```

### 1. **Install Dependencies**

Ensure all required packages are installed. Use `pip` to install dependencies:

```bash
pip install -r requirements.txt
```

### 2. **Prepare Configuration**

Edit the `config.yaml` file to set the desired parameters for training and prediction.

### 3. **Run the Script**

Execute the `main.py` script to start the training and prediction process:

```bash
python main.py
```

### 4. **View Results**

Results will be saved in the directory specified by `output_path`. Classification reports and metrics will be available in `results.json`.

## File Descriptions

- **`config.yaml`**: Configuration file for setting model parameters, data paths, and other settings.
- **`config_roberta.yaml`**: Configuration file for RoBERTa model training and evaluation.
- **`config_setfit.yaml`**: Configuration file for SetFit model training and evaluation.
- **`data/dataset_utils.py`**: Functions for loading and preprocessing datasets.
- **`data/text_processing.py`**: Utility functions for text data, including label distribution printing.
- **`model/model_training.py`**: Functions for training SetFit and RoBERTa models.
- **`model/model_prediction.py`**: Functions for making predictions with trained models.
- **`evaluation/result_saving.py`**: Functions for saving results and generating classification reports.
- **`main.py`**: Orchestrates the workflow including data preprocessing, model training, prediction, and result saving.

## Troubleshooting

If you encounter issues:
- Verify that the `config.yaml` file is correctly formatted and contains valid paths.
- Ensure all dependencies are installed properly.
- Confirm that the datasets are correctly formatted and accessible in the data folder.