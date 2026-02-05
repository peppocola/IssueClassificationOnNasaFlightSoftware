# Project Documentation

This is the replication package for the paper: "Issue Classification with LLMs: an Empirical Study of the NASA Flight Software Systems", submitted to Journal of Systems and Software (JSS) - In Practice 

Authors:

- Giuseppe Colavito (University of Bari)
- Filippo Lanubile (University of Bari)
- Nicole Novielli (University of Bari)
- Christopher Arreza (NASA Goddard Space Flight Center)
- Ying Shi (NASA Goddard Space Flight Center)

## Overview

This project provides a framework for training and evaluating text classification models. It supports SetFit, RoBERTa, and LLM models (including OpenAI models) and includes functionality for data preprocessing, model training, prediction, and result evaluation.

## Project Structure

The project directory is organized as follows:
- **config/**: Contains configuration files
  - **config.yaml**: Main configuration file for model training and evaluation.
  - **config_roberta.yaml**: Configuration file for RoBERTa model training and evaluation.
  - **config_setfit.yaml**: Configuration file for SetFit model training and evaluation.
  - **config_llm.yaml**: Configuration file for LLM model (if applicable).
- **data/**: Contains datasets
  - **.keep**: Placeholder file to ensure the `data` directory is included in version control.
- **data_processing/**: Contains scripts for data processing.
  - **dataset_utils.py**: Functions for loading and preprocessing datasets.
  - **text_processing.py**: Utility functions for handling text data, including label distribution printing.
  - **label_mapping.py**: Handles text column creation and validation.
  - **prompt_builder.py**: Generates prompts for LLM-based classification (if applicable).
- **model/**: Contains scripts for model training and prediction.
  - **model_training.py**: Functions for training SetFit and RoBERTa models.
  - **model_prediction.py**: Functions for making predictions with the trained models.
  - **model_prompting.py**: Functions for LLM-based classification.
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
- **model_type**: Specifies the model type to use. Options are `"setfit"`, `"roberta"`, and `"llm"`.
- **text_columns**: List of column names in the dataset to use as input text.
- **merged_text_column**: Name of the column to store the merged text from `text_columns`.
- **config_roberta_path**: Path to the RoBERTa-specific configuration file.
- **config_setfit_path**: Path to the SetFit-specific configuration file.
- **config_llm_path**: Path to the LLM-specific configuration file.
- **prompts_path**: Path to the JSON file containing prompts for LLM-based classification.
- **responses_dir**: Directory to store LLM responses.
- **is_openai**: Boolean flag to indicate if using OpenAI models.
- **openai_organization**: Your OpenAI organization ID (if using OpenAI models).
- **wandb**: Configuration for logging and tracking model performance with Wandb.
  - **project**: Name of the project in Wandb.
  - **entity**: Username of the Wandb account.
  - **mode**: Logging mode, either `"online"` or `"offline"`.
- **save_predictions**: If `True`, save the predictions to a CSV file.
- **save_predictions_path**: Path to save the predictions CSV file.
- **balance_data**: If `True`, balances the dataset by undersampling the majority class.
- **use_validation**: If `True`, uses a validation set during training.
- **validation_split**: The proportion of the training data to use as validation data when `use_validation` is `True`.
- **wandb.log_model**: If `True`, logs the model to Weights & Biases.

### Example Configuration

```yaml
model_type: "roberta"
output_path: "output"
random_seed: 42
train_path: "data/nasa_train_sample.csv"
test_path: "data/nasa_test_sample.csv"
just_predict: False
text_columns:
  - "title"
  - "body"
merged_text_column: "text"
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
config_roberta_path: "config/config_roberta.yaml"
config_setfit_path: "config/config_setfit.yaml"
config_llm_path: "config/config_llm.yaml"
```

## Usage

### Reproducibility with Docker (Recommended)

This project provides a fully reproducible environment using Docker. This is the **recommended approach** for ensuring consistent results across different machines.

#### Prerequisites for Docker
- Docker (version 20.10 or higher)
- Docker Compose (version 1.29 or higher, or Docker Compose V2 integrated with Docker)
- NVIDIA Docker runtime (optional, for GPU support)

**Note**: Commands in this guide use `docker-compose` (V1). If you have Docker Compose V2, use `docker compose` (without hyphen) instead.

#### Setup with Docker

**1. Clone the repository**

```bash
git clone https://github.com/peppocola/NasaExperiments.git
cd NasaExperiments
```

Don't forget to initialize and update the submodules:

```bash
git submodule init
git submodule update
```

**2. Create `.env` file**

Create a `.env` file in the root directory with your API keys:

```bash
WANDB_API_KEY=your-wandb-api-key
OPENAI_API_KEY=your-openai-api-key
```

**3. Build the Docker image**

```bash
docker-compose build
```

This builds a Docker image with Python 3.11.6 and all pinned dependencies.

**4. Run training and testing**

```bash
docker-compose up
```

This will run the main.py script inside the container using the configuration in `config/config.yaml`.

**5. Access results**

Results will be available in the `output/` directory on your host machine, and logs in the `logs/` directory.

#### Advanced Docker Usage

**Run with custom configuration:**

```bash
docker-compose run nasa-classifier python main.py
```

**Run with GPU support:**

Uncomment the `deploy` section in `docker-compose.yml` and ensure you have NVIDIA Docker runtime installed:

```bash
docker-compose up
```

**Interactive shell inside container:**

```bash
docker-compose run nasa-classifier bash
```

**Validate Docker setup:**

```bash
./validate_docker.sh
```

This script checks that Docker is properly configured and all dependencies are correctly installed.

**Clean up Docker resources:**

```bash
docker-compose down
docker image rm nasa-issue-classifier:latest
```

### Manual Installation (Alternative)

If you prefer not to use Docker, you can set up the environment manually.

#### Prerequisites
Python 3.11.x is recommended to run this project. For best reproducibility, use Python 3.11.6 (as specified in the Docker environment). You can check your Python version by running:

```bash
python --version
```

Wandb is used for logging and tracking model performance. To use Wandb, you will need to sign up for an account at [https://wandb.ai/](https://wandb.ai/). After signing up, you will receive an API key that you can use to authenticate your account.

The API keys have to be stored in a file named `.env`.

An example `.env` file is shown below:

```YAML
WANDB_API_KEY=your-wandb-api-key
OPENAI_API_KEY=your-openai-api-key
```

If you're using OpenAI models, make sure to include your OpenAI API key in the `.env` file as shown above.
The organization ID has to be set in the config_llm.yaml file.

Add to the config.yaml file the following lines:

```YAML
wandb:
  project: "your-project-name"
  entity: "your-username"
  mode: "online" # or "offline"

# If using OpenAI models
is_openai: true
openai_organization: "YOUR_ORGANIZATION_ID"
```

If the mode is set to "offline", the logs will be saved locally in the logs directory.

You can then sync the runs to the Wandb server by running:

```bash
wandb sync --sync-all
```

For OpenAI models, ensure that the `is_openai` flag is set to `true` in your configuration, and that you've provided your OpenAI organization ID in the `openai_organization` field.

### 0. **Clone the repository**

```bash
git clone https://github.com/peppocola/NasaExperiments.git
```

Don't forget to initialize and update the submodules:

```bash
git submodule init
git submodule update
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

### LLM-based Classification

If you want to use LLM-based classification:

1. Ensure that the `config.yaml` file has the `model_type` set to "llm".
2. Prepare your prompts in a JSON file and specify its path in the `prompts_path` parameter.
3. Set up the LLM-specific configuration in the file specified by `config_llm_path`.
4. Run the script as usual:

```bash
python main.py
```

The LLM responses will be saved in the directory specified by `responses_dir`.

## File Descriptions

- **`config/config.yaml`**: Main configuration file for setting model parameters, data paths, and other settings.
- **`config/config_roberta.yaml`**: Configuration file for RoBERTa model training and evaluation.
- **`config/config_setfit.yaml`**: Configuration file for SetFit model training and evaluation.
- **`config/config_llm.yaml`**: Configuration file for LLM-based classification, including model settings, quantization options, and paths for prompts and responses.
- **`data_processing/dataset_utils.py`**: Functions for loading and preprocessing datasets.
- **`data_processing/text_processing.py`**: Utility functions for text data, including label distribution printing.
- **`data_processing/prompt_builder.py`**: Generates prompts for LLM-based classification (if applicable).
- **`model/model_training.py`**: Functions for training SetFit and RoBERTa models.
- **`model/model_prediction.py`**: Functions for making predictions with trained models.
- **`model/model_prompting.py`**: Functions for LLM-based classification, including setup, prompt processing, and inference.
- **`evaluation/result_saving.py`**: Functions for saving results and generating classification reports.
- **`main.py`**: Orchestrates the workflow including data preprocessing, model training, prediction, and result saving.

## Troubleshooting

If you encounter issues:
- Verify that the `config/config.yaml` file and other configuration files are correctly formatted and contain valid paths.
- Ensure all dependencies are installed properly.
- Confirm that the datasets are correctly formatted and accessible in the data folder.
- Check that the model-specific configuration files (`config_roberta.yaml`, `config_setfit.yaml`, `config_llm.yaml`) are present in the `config/` directory and properly formatted.
- For OpenAI models, make sure your API key and organization ID are correctly set in the `.env` file and the configuration.

## Reproducibility

This project is designed to be fully reproducible across different machines and environments.

### Environment Specifications

- **Python Version**: 3.11.6 (pinned)
- **All dependencies**: Version-pinned in `requirements.txt`
- **Random seed**: Configurable in `config.yaml` (default: 42)

### Ensuring Reproducibility

1. **Use Docker (Recommended)**: The Docker setup guarantees a consistent environment with exact Python and library versions.

2. **Manual setup**: If not using Docker, ensure you use Python 3.11.6 and install dependencies from the pinned `requirements.txt`:
   ```bash
   python3.11 -m pip install -r requirements.txt
   ```

3. **Fixed random seed**: The `random_seed` parameter in `config.yaml` ensures reproducible results across runs.

4. **Version control**: All dependency versions are explicitly pinned to ensure compatibility.

### Known Issues and Compatibility

The dependencies have been tested and verified to work together with Python 3.11.6. Newer versions of some libraries may introduce breaking changes. The Docker environment ensures all versions are correct.

### Verifying Your Setup

To verify your environment is correctly set up, run a quick test:

```bash
# With Docker
docker-compose run nasa-classifier python -c "import transformers; print(transformers.__version__)"

# Without Docker
python -c "import transformers; print(transformers.__version__)"
```

Expected output: `4.39.0`

### Security Considerations

**Important**: Some dependencies have known security vulnerabilities. Where possible, we've upgraded to patched versions while maintaining compatibility. However, some vulnerabilities remain due to compatibility requirements specified in the original issue.

#### Vulnerabilities Patched in This Release

✅ **protobuf** - Upgraded from 4.25.3 to 4.25.8
   - Fixed: Critical DoS vulnerabilities in the 4.x series (CVE-specific to versions < 4.25.8)

✅ **sentencepiece** - Upgraded from 0.2.0 to 0.2.1
   - Fixed: Heap overflow issue (CVE-specific to versions < 0.2.1)

#### Known Remaining Vulnerabilities

⚠️ **protobuf 4.25.8** - CANNOT BE UPGRADED (Patched version: 6.33.5)
   - **Issue**: JSON recursion depth bypass (affects ALL versions <= 6.33.4)
   - **Attempted fix**: Upgrade to 6.33.5 attempted but failed
   - **Blocking dependency**: wandb 0.16.6 requires protobuf<5, creating dependency conflict
   - **Error**: `wandb 0.16.6 depends on protobuf!=4.21.0, <5 and >=3.19.0`
   - **Options to fix**:
     1. Remove wandb entirely (breaks model tracking)
     2. Upgrade wandb to version that supports protobuf 6.x (may break compatibility)
     3. Accept the vulnerability with mitigations
   - **Mitigation**: Avoid processing untrusted JSON/protobuf data; run in isolated Docker container

⚠️ **transformers 4.39.0** (Patched version: 4.48.0)
   - Issue: Deserialization of untrusted data vulnerability
   - Reason kept: Required for codebase compatibility as specified in issue
   - Mitigation: Only load models from trusted sources (e.g., official Hugging Face repos)

⚠️ **torch 2.2.2** (Patched version: 2.6.0)
   - Issue: `torch.load` with `weights_only=True` can lead to RCE; deserialization vulnerability
   - Reason kept: Required for compatibility with transformers 4.39.0 and numpy 1.26.4
   - Mitigation: Code does not use `torch.load` directly; only use trusted model files
   - Note: One advisory for this has been withdrawn

⚠️ **wandb 0.16.6** (Advisory withdrawn)
   - Issue: SSRF vulnerability (advisory has been withdrawn)
   - Status: Advisory withdrawn - likely false positive or issue resolved
   - Note: Not a genuine security concern

#### Security Best Practices

When using this repository:
1. **Do not load untrusted model files or pickled data**
2. **Only use models from trusted sources** (e.g., official Hugging Face repositories)
3. **Run in isolated environment** (Docker container strongly recommended)
4. **Avoid processing untrusted user inputs** directly
5. **Do not expose this application directly to the internet**
6. **Monitor for security updates** in the dependencies

#### For Production Use

**⚠️ Warning**: These pinned versions are for **research reproducibility only**. 

For production deployments:
- Upgrade to patched versions: transformers >= 4.48.0, torch >= 2.6.0, protobuf >= 5.29.5 or >= 6.33.5
- Test thoroughly for compatibility with your use case
- Implement additional security controls (input validation, sandboxing, etc.)
- Security must take priority over exact version matching

#### Security Summary

| Dependency | Current | Patched | Status | Risk Level |
|------------|---------|---------|--------|------------|
| protobuf | 4.25.8 | 6.33.5 | Advisory for 6.x series | Low-Medium |
| sentencepiece | 0.2.1 | ✅ Latest | Patched | None |
| transformers | 4.39.0 | 4.48.0 | Known vuln | High (if loading untrusted models) |
| torch | 2.2.2 | 2.6.0 | Known vuln | High (if using torch.load on untrusted data) |
| wandb | 0.16.6 | N/A | Advisory withdrawn | None |

**Note**: The pinned versions are primarily for research reproducibility and replicating published results. This setup should **only be used in trusted, isolated environments** for research purposes.
