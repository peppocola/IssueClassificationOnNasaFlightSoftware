import os
import wandb
from config.config_loader import load_config
from data_processing.dataset_utils import preprocess_dataset, print_label_distribution, map_labels_in_dataset
from model.model_training import train_setfit_model, train_roberta_model
from model.model_prediction import predict_setfit, predict_roberta
from model.model_prompting import process_llm_prompts
from evaluation.result_saving import save_results, generate_classification_report, save_csv_results
from setfit import SetFitModel
from evaluation.llm_response_eval import LLMEvaluator
from data_processing.prompt_builder import PromptGenerator

def load_and_merge_configs():
    """Load and merge main configuration with model-specific configuration."""
    main_config = load_config("config/config.yaml")
    if not main_config:
        return None

    model_type = main_config.get('model_type', 'setfit')

    # Load model-specific config
    model_config_path = main_config.get(f'config_{model_type}_path', f'config/config_{model_type}.yaml')
    model_config = load_config(model_config_path)
    if not model_config:
        return None

    # Merge main config with model-specific config, prioritizing model-specific settings
    return {**main_config, **model_config}

def flatten_metrics(metrics):
    """Flatten a nested dictionary of metrics."""
    result_log = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            for k, v in value.items():
                result_log[f"{key}_{k}"] = v
        else:
            result_log[key] = value
    return result_log

def process_model(config, model_type, train_set, test_set):
    """Process a single model for training or prediction."""
    output_path = config['output_path']
    predict_mapping = config.get('label_to_int', {})

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
        result_log = flatten_metrics(results)
        wandb.log(result_log)

        save_results(results, output_path, config['base_model'])
        print(f"Results for {config['base_model']}:")
        print(results)

        # Log model to wandb if specified
        if config['wandb']['log_model']:
            wandb.save(os.path.join(output_path, "*"))
        
        # Save the input text with ground truth and predicted labels
        if config['save_predictions']:
            save_csv_results(test_set, references, predictions, config['save_predictions_path'])

def eval_llm_model(config):
    """Process and evaluate the LLM model."""
    evaluator = LLMEvaluator(config, single_model=config['model_name'])
    metrics = evaluator.evaluate_model()
        
    model_metrics = metrics.get(config['model_name'], {})
    
    # Log flattened metrics to wandb
    wandb.log(flatten_metrics(model_metrics))
    

def main():
    config = load_and_merge_configs()
    if not config:
        return

    model_type = config.get('model_type', 'setfit')
    
    with wandb.init(project=config['wandb']['project'], 
                    entity=config['wandb']['entity'], 
                    config=config,
                    mode=config['wandb']['mode'],
                    dir='./logs'):

        if model_type == 'llm':
            if config.get('rebuild_prompts', False):
                prompt_generator = PromptGenerator(config)
                prompt_generator.run()
            process_llm_prompts(config)
            eval_llm_model(config)
        else:
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