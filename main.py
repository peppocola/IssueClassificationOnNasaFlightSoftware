import os
import wandb
from config.config_loader import load_config
from data_processing.dataset_utils import preprocess_dataset, print_label_distribution
from model.model_training import train_setfit_model, train_roberta_model
from model.model_prediction import predict_setfit, predict_roberta
from evaluation.result_saving import save_results, generate_classification_report, save_csv_results
from setfit import SetFitModel

def get_hyperparameters(config, model_type):
    """Extract relevant hyperparameters based on model type."""
    common_params = {
        "model_type": model_type,
        "random_seed": config.get('random_seed'),
        "base_model": config.get('base_model'),
    }
    
    if model_type == 'setfit':
        return {
            **common_params,
            "batch_size": config.get('batch_size'),
            "num_iterations": config.get('num_iterations'),
            "num_epochs": config.get('num_epochs'),
        }
    elif model_type == 'roberta':
        return {
            **common_params,
            "learning_rate": config.get('learning_rate'),
            "num_train_epochs": config.get('num_train_epochs'),
            "per_device_train_batch_size": config.get('per_device_train_batch_size'),
            "weight_decay": config.get('weight_decay'),
            "use_custom_loss": config.get('use_custom_loss'),
            "class_weights": config.get('class_weights'),
        }
    else:
        return common_params

def process_model(config, model_type, train_set, test_set):
    """Process a single model for training or prediction."""
    output_path = os.path.join(config['output_path'], config['base_model'].split('/')[-1])
    os.makedirs(output_path, exist_ok=True)

    predict_mapping = config.get('label_to_int', {})

    # Extract hyperparameters
    hyperparameters = get_hyperparameters(config, model_type)

    # Initialize wandb run
    with wandb.init(project=config['wandb']['project'], 
                    entity=config['wandb']['entity'], 
                    config=hyperparameters,
                    mode=config['wandb']['mode'],
                    dir='./logs'
                    ):  # Log hyperparameters
        if config.get('just_predict', False):
            if model_type == 'setfit':
                model = SetFitModel.from_pretrained(config['base_model'])
                references, predictions, prediction_time = predict_setfit(model, test_set, predict_mapping)
            else:
                raise ValueError("'just_predict' can only be used with the 'setfit' model type.")
        else:
            if model_type == 'setfit':
                model, training_time = train_setfit_model(config, train_set)
                references, predictions, prediction_time = predict_setfit(model, test_set, predict_mapping)
            elif model_type == 'roberta':
                model, training_time = train_roberta_model(config, train_set, predict_mapping, val_data=test_set)
                references, predictions, prediction_time = predict_roberta(model, test_set, predict_mapping)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            training_time_sec = training_time.total_seconds()
            wandb.log({"training_time": training_time_sec})

        results = generate_classification_report(references, predictions)
        results['prediction_time'] = prediction_time.total_seconds()
        if not config.get('just_predict', False):
            results['training_time'] = training_time_sec

        # Log results to wandb so they can be tracked
        # Unnest the results dictionary for logging
        result_log = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    result_log[f"{key}_{k}"] = v
            else:
                result_log[key] = value
        wandb.log(result_log)

        save_results(results, output_path, config['base_model'])
        print(f"Results for {config['base_model']}:")
        print(results)

        # Log model to wandb if specified
        if config['wandb']['log_model']:
            wandb.save(os.path.join(output_path, "*"))
        
        # Save the input text with ground truth and predicted labels
        if config['save_csv_results']:
            save_csv_results(test_set, references, predictions, output_path)

def main():
    main_config = load_config("config.yaml")
    if not main_config:
        return

    model_type = main_config.get('model_type', 'setfit')

    if main_config.get('just_predict', False) and model_type != 'setfit':
        raise ValueError("'just_predict' can only be used with the 'setfit' model type.")

    # Load model-specific config
    model_config = load_config(f"config_{model_type}.yaml")
    if not model_config:
        return

    # Merge main config with model-specific config, prioritizing model-specific settings
    config = {**main_config, **model_config}

    # Prepare datasets
    train_set, test_set = preprocess_dataset(config)

    print("Training set label distribution:")
    print_label_distribution(train_set)

    if test_set:
        print("Test set label distribution:")
        print_label_distribution(test_set)

    # Process the model
    process_model(config, model_type, train_set, test_set)

if __name__ == "__main__":
    main()