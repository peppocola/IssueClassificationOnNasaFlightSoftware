import os
import sys
import argparse
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

def parse_args():
    """Parse command line arguments for config overrides."""
    parser = argparse.ArgumentParser(description='NASA Issue Classification')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to custom config file (e.g., config/config_llm_sample.yaml)')
    parser.add_argument('--config-override', nargs='*', default=[],
                       help='Override config values in format key=value (e.g., model_type=setfit just_predict=true)')
    return parser.parse_args()

def apply_config_overrides(config, overrides):
    """Apply command-line config overrides to the configuration."""
    if not overrides:
        return config
    
    for override in overrides:
        if '=' not in override:
            print(f"Warning: Invalid override format '{override}'. Expected key=value")
            continue
        
        key, value = override.split('=', 1)
        key = key.strip()
        value = value.strip()
        
        # Convert string values to appropriate types
        if value.lower() == 'true':
            value = True
        elif value.lower() == 'false':
            value = False
        elif value.isdigit():
            value = int(value)
        elif value.replace('.', '', 1).isdigit():
            value = float(value)
        
        config[key] = value
        print(f"Config override: {key} = {value}")
    
    return config

def load_and_merge_configs(cli_overrides=None, custom_config_path=None):
    """Load and merge main configuration with model-specific configuration."""
    # If custom config is provided, use it directly without merging
    if custom_config_path:
        print(f"Loading custom config from: {custom_config_path}")
        config = load_config(custom_config_path)
        if not config:
            return None
        # Apply CLI overrides to custom config
        if cli_overrides:
            config = apply_config_overrides(config, cli_overrides)
        return config
    
    main_config = load_config("config/config.yaml")
    if not main_config:
        return None
    
    # Apply CLI overrides to main config first to determine model_type
    temp_config = main_config.copy()
    if cli_overrides:
        temp_config = apply_config_overrides(temp_config, cli_overrides)

    model_type = temp_config.get('model_type', 'setfit')

    # Load model-specific config
    model_config_path = main_config.get(f'config_{model_type}_path', f'config/config_{model_type}.yaml')
    model_config = load_config(model_config_path)
    if not model_config:
        return None

    # Merge main config with model-specific config, prioritizing model-specific settings
    merged_config = {**main_config, **model_config}
    
    # Apply CLI overrides AFTER merging to ensure they take precedence
    if cli_overrides:
        merged_config = apply_config_overrides(merged_config, cli_overrides)
    
    return merged_config

def flatten_metrics(metrics):
    """Flatten a nested dictionary of metrics."""
    flattened = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flattened[f"{key}_{subkey}"] = subvalue
        else:
            flattened[key] = value
    return flattened

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
        if config['save_predictions']:
            save_csv_results(test_set, references, predictions, config['save_predictions_path'])

def process_llm_model(config):
    """Process and evaluate the LLM model."""
    evaluator = LLMEvaluator(config['config_path'], single_model=config['base_model'])
    metrics = evaluator.evaluate_model()
        
    model_metrics = metrics.get(config['base_model'], {})
    
    # Log flattened metrics to wandb
    wandb.log(flatten_metrics(model_metrics))
    
    # Save the Excel report
    output_dir = os.path.join(config['responses_dir'], config['base_model'])
    evaluator.create_excel_table({config['base_model']: model_metrics}, output_dir)
    

def main():
    print("="*50)
    print("MAIN.PY STARTING")
    print("="*50)
    args = parse_args()
    print(f"CLI Arguments parsed: {args}")
    if args.config:
        print(f"Custom config file: {args.config}")
    print(f"Config overrides: {args.config_override}")
    print("="*50)
    
    config = load_and_merge_configs(args.config_override, args.config)
    if not config:
        return

    model_type = config.get('model_type', 'setfit')
    print(f"\n{'='*50}")
    print(f"FINAL CONFIGURATION:")
    print(f"  Model type: {model_type}")
    print(f"  Just predict: {config.get('just_predict', False)}")
    print(f"  Random seed: {config.get('random_seed', 'not set')}")
    print(f"{'='*50}\n")
    
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